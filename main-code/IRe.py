from dataset.build_dict import Dict
from dataset import utils
from dataset.DataManager import RecDatasetManager
from modules.GNNModule import GNNModule
from modules.SemanticModule import SemanticModule
from modules.IReGNN import IReGNN
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import time
import os
import pickle
import copy
import math

from IReEncoder import Word2VecEncoder,ConceptNetEncoder,LSTM_encoder,GRU_decoder

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class IRe(nn.Module):
    def __init__(self, opt):
        super(IRe, self).__init__()
        self.device = opt.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dict = opt["dict"]
        self.toolkits = opt["toolkits"]
        self.null_ind = self.toolkits.tok2ind(self.dict.null_tok)
        self.vocab_len = len(opt.get("dict"))
        self.w2v_emb_size = opt.get("w2v_emb_size", 300)
        self.wv2_weight_path = opt.get("w2v_weight_path", None)
        self.conceptnet_len = opt.get("conceptnet_len")
        self.conceptnet_emb_size = opt.get("conceptnet_emb_size", 300)
        self.conceptnet_emb_path = opt.get("conceptnet_emb_path")
        self.conceptnet_neibors_emb_path = opt.get("conceptnet_neibors_emb_path")
        self.hidden_size = opt.get("hidden_size")
        self.words_topk = opt.get("words_topk", 10)
        self.hidden_size = opt.get("hidden_size")
        self.words_topk = opt.get("words_topk", 10)
        self.bilstm_hidden_size = opt.get("bilstm_hidden_size", 256)
        self.bilstm_num_layers = opt.get("bilstm_num_layers", 2)
        self.dropout = opt.get("dropout", 0)
        self.num_heads = opt.get("num_heads", 2)
        self.max_text_len = opt.get("max_text_len", 64)
        self.max_sent_len = opt.get("max_sent_len", 16)
        with open(opt["relations_path"], "rb") as f_relations:
            self.user_ne_items, self.item_ne_users, self.user_ne_users, self.item_ne_items, \
                    self.user_item_review, self.pair2ind = pickle.load(f_relations)
        with open(opt["graph_info_path"], "rb") as f_ginfo:
            self.user2text_vectors, self.item2text_vectors, self.review2text_vectors = pickle.load(f_ginfo)

        self.word2vec_encoder = Word2VecEncoder(self.w2v_emb_size, self.vocab_len, self.wv2_weight_path)
        
        self.conceptnet_encoder = ConceptNetEncoder(self.conceptnet_emb_size, self.hidden_size, self.dropout, self.conceptnet_len,
                                                        self.conceptnet_emb_path, self.conceptnet_neibors_emb_path, self.words_topk, self.device)
        self.multi_head_att = MultiHeadAttention(n_heads=self.num_heads, qdim=4 * self.bilstm_hidden_size, kdim=2 * self.bilstm_hidden_size,
                                                    vdim=2*self.bilstm_hidden_size , hdim=self.hidden_size, dropout=self.dropout)
        self.text_bilstm = LSTM_encoder(in_size=self.w2v_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)
        self.sent_bilstm = LSTM_encoder(in_size=self.w2v_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)
        self.conceptnet_bilstm = LSTM_encoder(in_size=self.conceptnet_emb_size, hidden_size=self.bilstm_hidden_size,
                                    num_layers=self.bilstm_num_layers, dropout=self.dropout, device=self.device, biflag=True)

        self.decoder = GRU_decoder(
            embedding_layer=self.word2vec_encoder.w2v_emb,
            word_dict=self.dict,
            toolkits=self.toolkits,
            max_tip_len=opt["max_tip_len"],
            max_copy_len=opt["max_copy_len"],
            hidden_size=opt["hidden_size"],
            num_layers=opt["gru_num_layers"],
            device=self.device,
            dropout=0.,
            use_copy=False
        )

        self.h0_layer = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.h0_norm = nn.LayerNorm(self.hidden_size)
        nn.init.xavier_normal_(self.h0_layer.weight)

        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=self.null_ind, reduction='sum')

    def loss_gen(self, scores, y_hat, y_true):
        nll_loss = self.gen_criterion(scores.view(-1, scores.size(-1)), y_true.view(-1))
        notnull = y_true.ne(self.toolkits.tok2ind(self.dict.null_tok))
        target_tokens = notnull.long().sum().item()
        correct = ((y_true == y_hat) * notnull).sum().item()
        loss_per_tok = nll_loss / target_tokens
        return loss_per_tok, nll_loss, target_tokens, correct
    
    def get_review_vectors(self, ind1, ind2, ntype="user"):
        edge = (ind1, ind2) if ntype == "user" else (ind2, ind1)
        return copy.deepcopy(self.review2text_vectors[edge])

    def get_user_vectors(self, uind):
        return copy.deepcopy(self.user2text_vectors[uind])

    def get_item_vectors(self, iind):
        return copy.deepcopy(self.item2text_vectors[iind])
    
    def get_review_text(self, ind1, ind2, ntype="user"):
        edge = (ind1, ind2) if ntype == "user" else (ind2, ind1)
        return self.user_item_review[edge]
    
    def sem_out(self, hn, biflag=True):
        batch_size = hn.shape[1]
        num_layers = self.bilstm_num_layers
        num_directions = 2 if biflag else 1
        hn = hn.transpose(0, 1).reshape(batch_size, num_layers, -1)[:, -1, :]
        return hn
    
    def forward(self, users_ind, items_ind, ys=None):
        user_vectors = [self.get_user_vectors(uind) for uind in users_ind]
        user_vectors = self.toolkits.batch_vectors(user_vectors)
        users_pref = self.encoder(*user_vectors)

        item_vectors = [self.get_item_vectors(iind) for iind in items_ind]
        item_vectors = self.toolkits.batch_vectors(item_vectors)
        items_pref = self.encoder(*item_vectors)
        relations_pref = users_pref * items_pref
        h0 = torch.tanh(self.h0_layer(torch.cat((users_pref, items_pref, relations_pref), dim=-1))).unsqueeze(0)
        scores, predicts = self.decoder(None, h0, ys)
        return scores, predicts

    def sent_embedding(self, sent_vec):
        sent_emb = self.word2vec_encoder(sent_vec)
        return sent_emb.mean(dim=-2)

    def encoder(self, text_vec, text_lens, sent_vec, sent_lens, conceptnet_text_vec):
        w2v_text_emb = self.word2vec_encoder(text_vec)
        w2v_sent_emb = self.sent_embedding(sent_vec)
        conceptnet_emb = self.conceptnet_encoder(conceptnet_text_vec)

        text_out, text_hn = self.text_bilstm(w2v_text_emb)
        sent_out, sent_hn = self.sent_bilstm(w2v_sent_emb)
        conceptnet_out, _ = self.conceptnet_bilstm(conceptnet_emb)

        sem_g = self.sem_out(text_hn)           
        sem_s = self.sem_out(sent_hn)           
        sem_c = torch.cat([sem_g, sem_s], dim=-1)         
        mask = (text_vec != self.null_ind)
        sem_out = self.multi_head_att(sem_c.unsqueeze(dim=1), text_out, conceptnet_out, mask)
        return sem_out

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_normal_(self.a.data)
        nn.init.xavier_normal_(self.b.data)

    def forward(self, h):
        N = h.shape[-2]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).transpose(-1, -2)
        attention = F.softmax(e, dim=-1)
        return torch.matmul(attention, h).squeeze(-2)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, qdim, kdim, vdim, hdim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = hdim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(qdim, hdim)
        self.k_lin = nn.Linear(kdim, hdim)
        self.v_lin = nn.Linear(vdim, hdim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(hdim, hdim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, hao, qing, bo, mask=None):
        batch_size, hao_len, qdim = hao.size()
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        def neginf(dtype):
            if dtype is torch.float16:
                return -NEAR_INF_FP16
            else:
                return -NEAR_INF

        if qing is None and bo is None:
            qing = bo = hao
        elif bo is None:
            bo = qing
        _, qing_len, kdim = qing.size()

        q = prepare_head(self.q_lin(hao))
        k = prepare_head(self.k_lin(qing))
        v = prepare_head(self.v_lin(bo))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, qing_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, hao_len, qing_len)
            .view(batch_size * n_heads, hao_len, qing_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(hao)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(hao)
            .view(batch_size, n_heads, hao_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, hao_len, self.dim)
        )

        out = self.out_lin(attentioned)

        return out.squeeze(1)


