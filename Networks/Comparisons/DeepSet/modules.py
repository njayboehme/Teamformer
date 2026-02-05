import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Code adapted from https://github.com/christopher-hsu/scalableMARL
'''

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        # This is for the enc
        if type(Q) == list:
            Q, mask = Q
            K = K[0]
        # This is for the decoder
        else:
            mask = None

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)


        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if mask is None:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V) + mask, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if mask is None:
            return O
        else:
            return [O, mask]


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
























# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# __author__ = 'Christopher D Hsu'
# __copyright__ = ''
# __credits__ = ['Christopher D Hsu']
# __license__ = ''
# __version__ = '0.0.1'
# __maintainer__ = 'Christopher D Hsu'
# __email__ = 'chsu8@seas.upenn.edu'
# __status__ = 'Dev'

# """
# @InProceedings{lee2019set,
#     title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
#     author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
#     booktitle={Proceedings of the 36th International Conference on Machine Learning},
#     pages={3744--3753},
#     year={2019}
# }
# """

# import itertools
# from ScalableCVaR.params import ACT_MASK_NORMAL_VAL

# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# class MAB(nn.Module):
#     def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, max_num_caps=0, is_actor=False):
#         super(MAB, self).__init__()
#         self.dim_V = dim_V
#         self.num_heads = num_heads
#         self.fc_q = nn.Linear(dim_Q, dim_V)
#         self.fc_k = nn.Linear(dim_K, dim_V)
#         self.fc_v = nn.Linear(dim_K, dim_V)
#         if ln:
#             self.ln0 = nn.LayerNorm(dim_V)
#             self.ln1 = nn.LayerNorm(dim_V)
#         self.fc_o = nn.Linear(dim_V, dim_V)
#         self.max_num_caps = max_num_caps
#         self.is_actor = is_actor

#     def forward(self, Q, K, action_mask, completed_mask, split_pairs, cur_team_index): # I need the mask for masking completed targets
#         # Q = self.fc_q(Q) # This is the embedding (team)
#         if self.is_actor:
#             Q = self.get_possible_team_embeddings(Q, action_mask, split_pairs, cur_team_index)
#         else:
#             Q = self.fc_q(Q)
#         K, V = self.fc_k(K), self.fc_v(K) # This is the embedding (target)

#         dim_split = self.dim_V // self.num_heads
#         Q_ = torch.cat(Q.split(dim_split, 2), 0)
#         K_ = torch.cat(K.split(dim_split, 2), 0)
#         V_ = torch.cat(V.split(dim_split, 2), 0)

#         # t = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V)
#         completed_mask = torch.repeat_interleave(completed_mask, self.num_heads, dim=0).unsqueeze(1)
#         completed_mask = completed_mask.repeat(1, Q.shape[1], 1)
#         # tt = t + completed_mask
#         # Add the mask to remove attn to completed targets
#         A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V) + completed_mask, 2) # This is where masking would need to be done
#         O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
#         O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
#         O = O + F.relu(self.fc_o(O))
#         O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
#         return O


# class SAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, ln=False):
#         super(SAB, self).__init__()
#         self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

#     def forward(self, X):
#         return self.mab(X, X)


# class ISAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
#         super(ISAB, self).__init__()
#         self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
#         nn.init.xavier_uniform_(self.I)
#         self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
#         self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

#     def forward(self, X):
#         H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
#         return self.mab1(X, H)


# class PMA(nn.Module):
#     def __init__(self, dim, num_heads, num_seeds, ln=False):
#         super(PMA, self).__init__()
#         self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
#         nn.init.xavier_uniform_(self.S)
#         self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

#     def forward(self, X):
#         return self.mab(self.S.repeat(X.size(0), 1, 1), X)