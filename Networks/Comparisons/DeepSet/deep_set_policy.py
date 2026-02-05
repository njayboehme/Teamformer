from Networks.Comparisons.DeepSet.core import DeepSetmodel
from Networks.mlp import MLP

import torch
import torch.nn as nn

class DeepSetPolicy(nn.Module):
    def __init__(self, net_arch={}) -> None:
        super().__init__()

        obs_space_shape = net_arch['inp_dim']
        act_dim = net_arch['out_dim']
        hidden_dim = net_arch['hidden_dim']
        num_heads = net_arch['num_heads']
        self.rows, self.cols, self.max_num_caps, _ = net_arch['general_info']
        mlp_hidden, mlp_layers = net_arch['mlp_info']

        self.latent_dim_pi = 1
        self.latent_dim_vf = hidden_dim

        # The dim of output will be hidden dim
        self.net = DeepSetmodel(obs_space_shape, 
                                  action_space=hidden_dim, 
                                  dim_hidden=hidden_dim, 
                                  num_heads=num_heads,
                                  max_num_caps=self.max_num_caps
                                  )
        
        # This will take hidden dim and transform to action space
        self.action_mlp = MLP(hidden_dim, act_dim - 1, mlp_hidden, mlp_layers)
        # This will take hidden dim and transform to value space
        self.value_mlp = MLP(hidden_dim, 1, mlp_hidden, mlp_layers)
                

    def forward_actor(self, features):
        obs, action_mask, team_obs, _, _, _ = features
        agent_team_head = obs[:, 2].long()
        out = self.net(features)
        action_out = self.action_mlp(out)
        fin_logits = action_out + action_mask[:, :-1]
        # Add in the dummy mask
        fin_logits = torch.cat([fin_logits, action_mask[:, -1:]], -1)
        if self.training:
            return [fin_logits, None]
        else:
            return [fin_logits, agent_team_head.unsqueeze(0)]

    def forward_critic(self, features):
        '''
        Centralized critic
        '''
        out = self.net(features)
        return out

    def forward(self, x, deterministic=False, actions=None):
        return self.forward_actor(x), self.forward_critic(x)