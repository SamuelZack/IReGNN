import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import numpy as np
import os
import math

from SemanticEncoder import Word2VecEncoder,ConceptNetEncoder,LSTM_encoder
from SemanticLayer import MultiHeadAttention

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class SemanticModule(nn.Module):
    def __init__(self, opt):
        super(SemanticModule, self).__init__()
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
        self.bilstm_hidden_size = opt.get("bilstm_hidden_size", 256)
        self.bilstm_num_layers = opt.get("bilstm_num_layers", 2)
        self.dropout = opt.get("dropout", 0)
        self.num_heads = opt.get("num_heads", 2)
        self.max_text_len = opt.get("max_text_len", 64)
        self.max_sent_len = opt.get("max_sent_len", 16)
        self.emb_norm1 = nn.LayerNorm(self.w2v_emb_size)
        self.emb_norm2 = nn.LayerNorm(self.w2v_emb_size)
        self.emb_norm3 = nn.LayerNorm(self.conceptnet_emb_size)
        self.out_norm1 = nn.LayerNorm(2 * self.bilstm_hidden_size)
        self.out_norm2 = nn.LayerNorm(2 * self.bilstm_hidden_size)
        self.out_norm3 = nn.LayerNorm(2 * self.bilstm_hidden_size)
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
    
    def sent_embedding(self, sent_vec):
        sent_emb = self.word2vec_encoder(sent_vec)
        return sent_emb.mean(dim=-2)
    
    def sem_out(self, hn, biflag=True):
        batch_size = hn.shape[1]
        num_layers = self.bilstm_num_layers
        num_directions = 2 if biflag else 1
        hn = hn.transpose(0, 1).reshape(batch_size, num_layers, -1)[:, -1, :]

        return hn

    def forward(self, text_vec, text_lens, sent_vec, sent_lens, conceptnet_text_vec):
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
