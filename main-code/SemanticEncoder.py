import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import numpy as np
import os
import math

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

from SemanticLayer import SelfAttentionLayer
def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
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
            self.conceptnet_emb = nn.Embedding.from_pretrained(weight)
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

