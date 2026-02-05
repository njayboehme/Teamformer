import numpy as np
import torch
import torch.nn as nn
from Networks.Comparisons.DeepSet.modules import *

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
Code adapted from https://github.com/christopher-hsu/scalableMARL
'''
class DeepSetAttention(nn.Module):
    """ Written by Christopher Hsu:

    """
    def __init__(self, dim_input, dim_output, num_outputs=1,
                        dim_hidden=128, num_heads=4, ln=True, 
                        max_num_caps=0, is_actor=False):
        super().__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, obs):
        '''
        @param N - The number of agents
        '''
        v = self.enc(obs)[0]
        # The sum should only include the current team and all of the targets
        v = v.sum(dim=1, keepdim=True)  #pooling mechanism: sum, mean, max
        v = self.dec(v).squeeze()
        return v


class DeepSetmodel(nn.Module):

    def __init__(self, observation_space, action_space, dim_hidden=128, num_heads=4, max_num_caps=0):
        super().__init__()
        obs_dim = observation_space
        act_dim = action_space
        self.max_num_caps = max_num_caps
        self.num_heads = num_heads

        # build policy and value functions
        self.q1 = DeepSetAttention(dim_input=obs_dim, dim_output=act_dim, dim_hidden=dim_hidden, num_heads=num_heads, max_num_caps=max_num_caps)
    
    def forward(self, x):
        obs, _, team_obs, target_obs, team_mask, local_target_mask = x
        batch, N, _ = team_obs.shape
        # Add in zeros for the target indicators
        team_obs = torch.cat((team_obs, torch.zeros((batch, N, 2), device=dev)), dim=2)
        # Don't include the target complete indicator
        target_obs = target_obs[torch.arange(batch), :, :-1]
        # Add team indicators to the correct location
        target_obs = torch.cat((target_obs[:,:,:2 + self.max_num_caps], torch.zeros((batch, target_obs.shape[1], 2), device=dev), target_obs[:, :, -2:]), dim=2)

        fin_in = torch.cat((team_obs, target_obs), dim=1)

        msk = torch.full((batch, fin_in.shape[1], fin_in.shape[1]), -1e9, device=dev)
        msk[:, torch.arange(fin_in.shape[1]), torch.arange(fin_in.shape[1])] = 0.0

        # Mask the padded teams and targets and make sure the pad tokens don't see anything else
        imp_team_mask = torch.where(team_mask[:, 0] == -1e9, False, True)
        local_target_mask = torch.where(local_target_mask[:, :-1] == 0, False, True)
        imp_team_target_mask = torch.cat((imp_team_mask, local_target_mask), dim=1)

        index_mask = imp_team_target_mask.unsqueeze(-1).repeat_interleave(fin_in.shape[1], dim=-1)
        fill_mat = torch.where(imp_team_target_mask.unsqueeze(1).repeat_interleave(fin_in.shape[1], dim=1) == True, 0.0, -1e9)
        msk[index_mask] = fill_mat[index_mask]
        
        msk = torch.repeat_interleave(msk, repeats=self.num_heads, dim=0)

        out = self.q1([fin_in, msk])
        return out