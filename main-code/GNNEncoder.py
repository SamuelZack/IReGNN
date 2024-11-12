import torch
from torch import nn
import torch.nn.functional as F

def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class NodeEncoder(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, dropout=0.1):
        super(NodeEncoder, self).__init__()
        self.user_emb = nn.Embedding(num_users, hidden_size)
        self.item_emb = nn.Embedding(num_items, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, text_tensors, inds, ntype, len_diff):
        diff_emb_layer = self.item_emb if ntype == "user" else self.user_emb
        same_emb_layer = self.user_emb if ntype == "user" else self.item_emb
        diff_inds = inds[:len_diff]
        same_inds = inds[len_diff:]
        diff_emb = diff_emb_layer(diff_inds)
        same_emb = same_emb_layer(same_inds)
        emb = torch.cat((diff_emb, same_emb), dim=0)

        if text_tensors is None:
            return emb

        out = torch.cat((text_tensors, emb), dim=1)
        out = F.relu(self.linear1(out))
        out = _normalize(self.dropout(out), self.norm2)
        out = F.relu(self.linear2(out))
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Attention, self).__init__()
        self.att1 = nn.Linear(hidden_size * 3, hidden_size)
        self.att2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.att1.weight)
        nn.init.xavier_normal_(self.att2.weight)

    def forward(self, this_node, relations, ne_nodes):
        this_nodes = this_node.expand(ne_nodes.shape)
        x = torch.cat((this_nodes, relations, ne_nodes), dim=1)
        x = F.relu(self.att1(x))
        x = self.dropout(x)
        x = F.relu(self.att2(x))
        att = F.softmax(x, dim=1)
        out = torch.mm(att.t(), ne_nodes).squeeze(0)
        return out
