import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dims, max_team_size=0, rows=0, cols=0):
        super().__init__()
        self.dims = dims        
        if dims == 1:
            pos_enc = PositionalEncoding1D(d_model)
            pos_end_1D = pos_enc(torch.zeros(1, max_team_size, d_model)).squeeze(0).to(dev)
            # A team of len 0 should just be all zeros
            pos_end_1D[0] = 0.
            self.register_buffer('pos_enc_1D', pos_end_1D)
        if dims == 2:
            pos_enc = PositionalEncoding2D(d_model)
            pos_enc_2D = pos_enc(torch.zeros(1, rows, cols, d_model)).squeeze(0)
            # Add zeros at the end to index with -1, -1
            pos_enc_2D = torch.cat((pos_enc_2D, torch.zeros(rows, 1, d_model)), dim=1).to(dev)
            self.register_buffer('pos_enc_2D', pos_enc_2D)
    
    def forward(self, pos):
        '''
        @param to_add - the tensor to add the position embedding to
        @param pos - The positions we want the embedding for
        @return tensor - A tensor with to_add summed with positional embeddings
        '''
        if self.dims == 1:
            return self.pos_enc_1D[pos]
        elif self.dims == 2:
            return self.pos_enc_2D[pos[:,:,0], pos[:,:,1]]