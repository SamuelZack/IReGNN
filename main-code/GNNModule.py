import torch
from torch import nn
import torch.nn.functional as F
import pickle
import copy
from GNNEncoder import NodeEncoder, Attention

def _normalize(tensor, norm_layer):
    size = tensor.size()
    return norm_layer(tensor.reshape(-1, size[-1])).reshape(size)

class GATModule(nn.Module):
    def __init__(self, opt, semantic_encoder=None):
        super(GATModule, self).__init__()
        self.use_gnn = opt["use_gnn"]
        self.use_relation = opt["use_relation"]
        if self.use_gnn:
            with open(opt["relations_path"], "rb") as f_relations:
                self.user_ne_items, self.item_ne_users, self.user_ne_users, self.item_ne_items, \
                    _, _, self.pair2ind = pickle.load(f_relations)
        with open(opt["graph_info_path"], "rb") as f_ginfo:
            self.user2text_vectors, self.item2text_vectors, _, self.review2text_vectors = pickle.load(f_ginfo)

        self.toolkits = opt["toolkits"]
        self.device = opt["device"]
        self.max_text_len = opt["max_text_len"]
        self.max_sent_len = opt["max_sent_len"]
        self.num_users = opt["num_users"]
        self.num_items = opt["num_items"]
        self.hidden_size = opt["hidden_size"]
        self.dropout = opt["dropout"]
        self.max_neighbors = opt["max_neighbors"]
        self.semantic_encoder = semantic_encoder
        if self.semantic_encoder is None:
            self.review_embedding = nn.Embedding(len(self.pair2ind), self.hidden_size)
            nn.init.xavier_normal_(self.review_embedding.weight)
        self.node_encoder = NodeEncoder(self.num_users, self.num_items, self.hidden_size, self.dropout)
        self.node_cache = {}
        self.att = Attention(self.hidden_size, self.dropout)
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def get_review_vectors(self, ind1, ind2, ntype="user"):
        edge = (ind1, ind2) if ntype == "user" else (ind2, ind1)
        return copy.deepcopy(self.review2text_vectors[edge])

    def get_user_vectors(self, uind):
        return copy.deepcopy(self.user2text_vectors[uind])

    def get_item_vectors(self, iind):
        return copy.deepcopy(self.item2text_vectors[iind])

    def node_preference(self, ind, ntype="user"):
        diff_type = "item" if ntype == "user" else "user"
        node_ne_same_nodes = getattr(self, "{}_ne_{}s".format(ntype, ntype))
        node_ne_diff_nodes = getattr(self, "{}_ne_{}s".format(ntype, diff_type))
        # same_ind2text = getattr(self, "get_{}_text".format(ntype))
        # diff_ind2text = getattr(self, "get_{}_text".format(diff_type))
        same_ind2vectors = getattr(self, "get_{}_vectors".format(ntype))
        diff_ind2vectors = getattr(self, "get_{}_vectors".format(diff_type))

        ne_diff_nodes = node_ne_diff_nodes[ind][:self.max_neighbors]
        ne_same_nodes = []
        if self.use_relation:
            ne_same_nodes = node_ne_same_nodes[ind][:self.max_neighbors]
        len_diff = len(ne_diff_nodes)
        len_same = len(ne_same_nodes)

        ne_nodes = ne_diff_nodes + ne_same_nodes
        all_nodes = ne_nodes + [ind]
        nodes_text_tensor = None
        if self.semantic_encoder is not None:
            diff_nodes_vectors = [diff_ind2vectors(i) for i in all_nodes[:len_diff]]
            same_nodes_vectors = [same_ind2vectors(i) for i in all_nodes[len_diff:]]
            batch_vectors = diff_nodes_vectors + same_nodes_vectors
            review_vectors = [self.get_review_vectors(ind, ind2, ntype) for ind2 in ne_diff_nodes]
            batch_vectors.extend(review_vectors)
            batch_vectors = self.toolkits.batch_vectors(batch_vectors)
            # time_use1 = time.time() - s
            # s = time.time()
            all_tensors = self.semantic_encoder(*batch_vectors)
            # time_use2 = time.time() - s
            # print("one node_{:06d}'s neighbors num {:03d}: get vectors use {:.6f}s, vectors2semantic_pref use {:.6f}s".format(ind, len(reset_sorted_index), time_use1, time_use2))
            nodes_text_tensor = all_tensors[:len(all_nodes)]

        all_nodes_tensors = self.node_encoder(nodes_text_tensor, torch.LongTensor(all_nodes).to(self.device), ntype,
                                              len_diff)

        ne_nodes_tensors = all_nodes_tensors[:-1, :]
        this_node_tensor = all_nodes_tensors[-1, :]
        if self.use_relation:
            same_relations_tensors = ne_nodes_tensors[len_diff:, :] * this_node_tensor
        else:
            same_relations_tensors = torch.randn([0, self.hidden_size]).to(self.device)
        if self.semantic_encoder:
            diff_relations_tensors = all_tensors[(len_diff + len_same) + 1:, :]
        else:
            user_inds = [ind] * len_diff if ntype == "user" else ne_diff_nodes
            item_inds = [ind] * len_diff if ntype == "item" else ne_diff_nodes
            diff_relations_tensors = self.get_relations_pref(user_inds, item_inds)
        relations_tensors = torch.cat((diff_relations_tensors, same_relations_tensors), dim=0)
        node_preference = self.att(this_node_tensor, relations_tensors, ne_nodes_tensors)
        return torch.cat((this_node_tensor, node_preference)).view(1, -1)

    def transform(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = self.dropout_layer(x)
        x = _normalize(x, self.norm1)
        x = F.relu(self.linear2(x))
        x = _normalize(x, self.norm2)
        return x

    def get_relations_pref(self, users_ind, items_ind):
        # TODO: deal none review text user-item pairs
        if self.semantic_encoder is not None:
            review_vectors = [self.get_review_vectors(ind1, ind2, ntype="user") for ind1, ind2 in
                              zip(users_ind, items_ind)]
            review_vectors = self.toolkits.batch_vectors(review_vectors)
            prefs = self.semantic_encoder(*review_vectors)
        else:
            review_inds = torch.LongTensor(
                [self.pair2ind[(ind1, ind2)] for ind1, ind2 in zip(users_ind, items_ind)]).to(self.device)
            prefs = self.review_embedding(review_inds)
        return prefs

    def forward(self, users_ind, items_ind):
        batch_size = len(users_ind)
        users_pref = []
        items_pref = []
        # relations_pref = self.get_relations_pref(users_ind, items_ind)
        # s = time.time()
        if self.use_gnn:
            for i, (uind, iind) in enumerate(zip(users_ind, items_ind)):
                if uind in self.node_cache:
                    u_pref = self.node_cache[uind]
                else:
                    u_pref = self.node_preference(uind, "user")
                    self.node_cache[uind] = u_pref
                if iind in self.node_cache:
                    i_pref = self.node_cache[iind]
                else:
                    i_pref = self.node_preference(iind, "item")
                    self.node_cache[iind] = i_pref
                users_pref.append(u_pref)
                items_pref.append(i_pref)
            users_pref = torch.cat(users_pref, 0)
            items_pref = torch.cat(items_pref, 0)
            users_pref = self.transform(users_pref)
            items_pref = self.transform(items_pref)
            self.node_cache.clear()

        else:

            user_vectors = [self.get_user_vectors(uind) for uind in users_ind]
            user_vectors = self.toolkits.batch_vectors(user_vectors)
            users_pref = self.semantic_encoder(*user_vectors)
            item_vectors = [self.get_item_vectors(iind) for iind in items_ind]
            item_vectors = self.toolkits.batch_vectors(item_vectors)
            items_pref = self.semantic_encoder(*item_vectors)
        # e = time.time()
        # print("get semantic prefs of a batch users and itmes use {:.6f}s".format(e - s))
        relations_pref = users_pref * items_pref
        return users_pref, items_pref, relations_pref
