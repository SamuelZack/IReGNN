
import torch
import torch.nn.functional as F
import numpy as np
from hqb import ReviewEncoder,LatentFactor

class NarreModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.toolkits = opt["toolkits"]
        self.user_nums = len(self.toolkits.user2ind_dict)
        self.item_nums = len(self.toolkits.item2ind_dict)
        weight = torch.from_numpy(np.load(opt["w2v_weight_path"])).float()
        self.word_embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)

        self.user_review_layer = ReviewEncoder(opt, self.user_nums+1, self.item_nums+1)
        self.item_review_layer = ReviewEncoder(opt, self.item_nums+1, self.user_nums+1)

        self.predict_linear = LatentFactor(opt, self.user_nums, self.item_nums)

    def forward(self, user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review):

        user_review_emb = self.word_embedding(user_review)
        user_feature, user_tip_index = self.user_review_layer(user_review_emb, user_id, item_id_per_review)

        item_review_emb = self.word_embedding(item_review)
        item_feature, item_tip_index = self.item_review_layer(item_review_emb, item_id, user_id_per_review)

        predict = self.predict_linear(user_feature, user_id, item_feature, item_id)
        return predict.squeeze(), user_review.gather(1, user_tip_index).squeeze(), item_review.gather(1, item_tip_index).squeeze()
