import torch
import torch.nn as nn

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CapabilityEmbedding(nn.Module):
    def __init__(self, num_caps, d_model):
        super().__init__()
        self.cap_embed = nn.Linear(num_caps, d_model, bias=False)
        self.d_model = d_model
    
    def forward(self, x, do_vec=True):
        '''
        @param x - The capabilities
        @param do_vec - If x is a vector or indices
        '''
        if do_vec:
            return self.cap_embed(x)
        else:
            return torch.transpose(torch.index_select(torch.cat((self.cap_embed.weight, torch.zeros((self.d_model, 1), device=dev)), dim=1), 1, x), 0, 1).unsqueeze(1)