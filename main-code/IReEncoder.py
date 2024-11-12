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

from IRe import SelfAttentionLayer

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class Word2VecEncoder(nn.Module):
    def __init__(self, emb_size, vocab_len, wv2_weight_path):
        super(Word2VecEncoder, self).__init__()
        if os.path.exists(wv2_weight_path):
            print("Load word2vec weight from exist file {}".format(wv2_weight_path))
            weight = torch.from_numpy(np.load(wv2_weight_path)).float()
            self.w2v_emb = nn.Embedding.from_pretrained(weight, freeze=False)

        else:
            print("Can not find pretrained word2vec weight file, and it will init randomly")
            self.w2v_emb = nn.Embedding(vocab_len, emb_size)
            nn.init.xavier_normal_(self.w2v_emb.weight)
    def forward(self, text_vec):
        emb = self.w2v_emb(text_vec)
        return emb

class ConceptNetEncoder(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout, conceptnet_len, conceptnet_emb_path, conceptnet_neibors_emb_path, topk, device):
        super(ConceptNetEncoder, self).__init__()
        self.emb_size = emb_size
        self.conceptnet_len = conceptnet_len
        self.topk = topk
        if os.path.exists(conceptnet_emb_path):
            print("Load conceptnet numberbatch weight from exist file {}".format(conceptnet_emb_path))
            weight = torch.from_numpy(np.load(conceptnet_emb_path))
            self.conceptnet_emb = nn.Embedding.from_pretrained(weight, freeze=True)
        else:
            print("Can not find pretrained conceptnet numberbatch weight file, and it will init randomly")
            self.conceptnet_emb = nn.Embedding(self.conceptnet_len, self.emb_size)
            nn.init.xavier_normal_(self.conceptnet_emb.weight)

        if os.path.exists(conceptnet_neibors_emb_path):
            print("Load conceptnet neighbor weight from exist file {}".format(conceptnet_neibors_emb_path))
            self.neighbors = torch.from_numpy(np.load(conceptnet_neibors_emb_path)).to(device)
        else:
            raise("no neighbors")
        
        self.self_att = SelfAttentionLayer(self.emb_size, self.emb_size, dropout=dropout)

    def forward(self, conceptnet_text_vec):
        bs = conceptnet_text_vec.size(0)
        neighbors = self.neighbors[conceptnet_text_vec]
        att_emb = self.self_att(self.conceptnet_emb(neighbors))
        return att_emb

class LSTM_encoder(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, dropout, device, biflag=True):
        super(LSTM_encoder, self).__init__()
        self.device = device
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.biflag = biflag
        self.num_directions = 2 if self.biflag else 1
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, dropout=self.dropout, bidirectional=self.biflag)
        
    def init_hidden_state(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device))

    def forward(self, input_data):
        if isinstance(input_data, torch.Tensor):
            batch_size = input_data.shape[0]
        else:
            batch_size = input_data.batch_sizes[0].item()
        h_c = self.init_hidden_state(batch_size)
        out, (hn, cn) = self.lstm(input_data, h_c)
        return out, hn

class GRU_decoder(nn.Module):
    def __init__(self, embedding_layer, word_dict, toolkits, max_tip_len, max_copy_len, hidden_size, num_layers, device, dropout=0., use_copy=True):
        super(GRU_decoder, self).__init__()
        self.use_copy = use_copy
        self.dict = word_dict
        self.toolkits = toolkits
        self.start_ind = toolkits.tok2ind(word_dict.start_tok)
        self.end_ind = toolkits.tok2ind(word_dict.end_tok)
        self.null_ind = toolkits.tok2ind(word_dict.null_tok)
        self.max_tip_len = max_tip_len
        self.max_copy_len = max_copy_len
        self.device = device
        self.hidden_size = hidden_size
        self.emb_size = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer
        self.dropout_layer = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(self.emb_size)
        self.gru_decoder = nn.GRU(self.emb_size, hidden_size, batch_first=True, dropout=dropout)
        self.out_linear = nn.Linear(hidden_size, self.emb_size)
        self.norm2 = nn.LayerNorm(self.emb_size)
        self.copy_trans = nn.Linear(2 * hidden_size, self.emb_size)
        self.norm3 = nn.LayerNorm(self.emb_size)

    def neginf(self, dtype):
            if dtype is torch.float16:
                return -NEAR_INF_FP16
            else:
                return -NEAR_INF

    def decode_forced(self, masks, h0, ys):
        bs = ys.shape[0]
        starts = torch.LongTensor([self.start_ind]).repeat((bs, 1)).to(self.device)
        seqlen = ys.shape[1]
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat((starts, inputs), 1)
        x = self.embedding_layer(inputs)
        hs, hn = self.gru_decoder(x, h0)
        x1_scores = self.out_linear(hs)
        x1_scores = F.relu(F.linear(x1_scores, self.embedding_layer.weight))
        if not self.use_copy:
            _, predicts = x1_scores.max(dim=-1)
            return x1_scores, predicts
        masks = (masks == 0).unsqueeze(1).expand(-1, seqlen, -1)
        h0 = h0.view(bs, 1, -1).repeat(1, seqlen, 1)
        x2_scores = self.copy_trans(torch.cat((hs, h0), dim=-1))
        x2_scores = F.relu(F.linear(x2_scores, self.embedding_layer.weight))
        x2_scores = x2_scores.masked_fill_(masks, 0.0)
        
        scores = x1_scores + x2_scores
        _, predicts = scores.max(dim=-1)
        return scores, predicts
        #*****************

    def decode_greedy(self, masks, h0):
        bs = h0.shape[1]
        x = torch.LongTensor([self.start_ind]).repeat((bs, 1)).to(self.device)
        if masks is not None:
            masks = (masks == 0)
        scores = []
        hn = h0
        for i in range(self.max_tip_len):
            h = self.embedding_layer(x)
            hs, hn = self.gru_decoder(h, hn)
            hs = hs[:, -1, :]
            x1_scores =  self.out_linear(hs)
            x1_scores = F.linear(x1_scores, self.embedding_layer.weight)
            if self.use_copy:
                x2_scores = self.copy_trans(torch.cat((hs, h0), dim=-1))
                x2_scores = F.relu(F.linear(x2_scores, self.embedding_layer.weight))
                x2_scores = x2_scores.masked_fill_(masks, 0.0)
                score = x1_scores + x2_scores
            else:
                score = x1_scores
            _, predicts = score.max(dim=-1)
            scores.append(score.unsqueeze(1))
            x = torch.cat((x, predicts.unsqueeze(1)), dim=1)
            all_finished = ((x == self.end_ind).sum(dim=1) > 0).sum().item() == bs
            if all_finished:
                break

        return torch.cat(scores, dim=1), x[:, 1:]

    def forward(self, masks, h0, ys=None):
        if ys is not None:
            scores, predicts = self.decode_forced(masks, h0, ys)
        else:
            scores, predicts = self.decode_greedy(masks, h0)

        return scores, predicts


