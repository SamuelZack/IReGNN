import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import numpy as np
import os
import math

from SemanticEncoder import Word2VecEncoder,ConceptNetEncoder,LSTM_encoder

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

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
        bb = prepare_head(self.v_lin(bo))
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
