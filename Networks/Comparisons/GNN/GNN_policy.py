import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import LayerNorm
from torch_geometric.data import Batch
from torch_geometric.data import Data

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GNN(nn.Module):
  def __init__(self, *args, **kwargs):
    net_arch = kwargs.pop('net_arch')
    super().__init__(*args, **kwargs)

    inp_dim = net_arch['inp_dim']
    self.out_dim = net_arch['out_dim']
    feature_dim = net_arch.get('feature_dim', 128)
    self.edge_dim = net_arch.get('edge_dim', None)
    num_heads = net_arch.get('num_heads', 1)
    self.num_agents = net_arch['num_agents']
    self.max_num_caps = net_arch['max_num_caps']
    self.max_team_obs_size = net_arch['max_team_obs_size']
    self.max_targ_ob_size = net_arch['max_targ_obs_size']

    # Embed the state (capabilities, locs, etc)
    self.state_embedder = nn.Linear(inp_dim, feature_dim)

    if self.edge_dim is not None:
      # for a team or target
      self.edge_embedder = nn.Linear(2, feature_dim)
      self.gnn_layer_1 = GATv2Conv(feature_dim, feature_dim // num_heads, edge_dim=self.edge_dim, heads=num_heads)
      self.gnn_layer_2 = GATv2Conv(feature_dim, feature_dim // num_heads, edge_dim=self.edge_dim, heads=num_heads)
      self.gnn_layer_3 = GATv2Conv(feature_dim, self.out_dim, edge_dim=self.edge_dim)
    else:
      self.gnn_layer_1 = GATv2Conv(feature_dim, feature_dim // num_heads, heads=num_heads)
      self.gnn_layer_2 = GATv2Conv(feature_dim, feature_dim // num_heads, heads=num_heads)
      self.gnn_layer_3 = GATv2Conv(feature_dim, self.out_dim)
    
    self.ln_1 = LayerNorm(feature_dim)
    self.ln_2 = LayerNorm(feature_dim)
    
  def forward(self, x):
    obs, _, team_obs, target_obs, team_mask, local_target_mask = x
    batch, N, _ = team_obs.shape
    # Add in target indicators
    team_obs = torch.cat((team_obs, torch.zeros((batch, N, 2), device=dev)), dim=2)
    
    # Don't include the target complete indicator
    target_obs = target_obs[torch.arange(batch), :, :-1]
    # Add in team indicators
    target_obs = torch.cat((target_obs[:,:,:2 + self.max_num_caps], torch.zeros((batch, target_obs.shape[1], 2), device=dev), target_obs[:, :, -2:]), dim=2)

    fin_in = torch.cat((team_obs, target_obs), dim=1)
    state_embeds = self.state_embedder(fin_in)

    mask = torch.full((batch, N + target_obs.shape[1], N + target_obs.shape[1]), False, device=dev)
    imp_team_mask = torch.where(team_mask[:, 0] == -1e9, False, True)
    local_target_mask = torch.where(local_target_mask[:, :-1] == 0, False, True)
    imp_team_target_mask = torch.cat((imp_team_mask, local_target_mask), dim=1)
    
    index_mask = imp_team_target_mask.unsqueeze(-1).repeat_interleave(fin_in.shape[1], dim=-1)
    fill_mat = torch.where(imp_team_target_mask.unsqueeze(1).repeat_interleave(fin_in.shape[1], dim=1) == True, True, False)
    mask[index_mask] = fill_mat[index_mask]
    mask[:, torch.arange(mask.shape[1]), torch.arange(mask.shape[2])] = False
    if self.edge_dim is not None:
      edge_features = None
      edge_embeds = self.edge_embedder(edge_features)
      graph_batch = self.create_batch(state_embeds, mask, edge_embeds)

      # Process graph data using GCN layers
      out = F.relu(self.ln_1(self.gnn_layer_1(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)))
      out = F.relu(self.ln_2(self.gnn_layer_2(out, graph_batch.edge_index, graph_batch.edge_attr)))
      out = self.gnn_layer_3(out, graph_batch.edge_index, graph_batch.edge_attr)
    else:
      graph_batch = self.create_batch(state_embeds, mask)

      # Process graph data using GCN layers
      out = F.relu(self.ln_1(self.gnn_layer_1(graph_batch.x, graph_batch.edge_index)))
      out = F.relu(self.ln_2(self.gnn_layer_2(out, graph_batch.edge_index)))
      out = self.gnn_layer_3(out, graph_batch.edge_index)

    out = out.reshape((batch, -1, self.out_dim))
    return out

  def create_batch(self, inp_embeds, mask, batched_edge_features=None):
    to_batch = []
    if batched_edge_features != None:
      for embs, mask, edge_feats in zip(inp_embeds, batched_edge_features):
        to_batch.append(Data(x=embs, edge_index=self.get_edge_indices(mask), edge_attr=edge_feats))
    else:
      for embs, mask in zip(inp_embeds, mask):
        to_batch.append(Data(x=embs, edge_index=self.get_edge_indices(mask)))
    fin_batch = Batch.from_data_list(to_batch)
    return fin_batch
    
  def get_edge_indices(self, mask):
    # mask should be # agents x (# agents + # targets)
    begin_inds = torch.arange(mask.shape[0], device=dev).unsqueeze(1).repeat(1, mask.shape[1])
    dest_inds = torch.arange(mask.shape[1], device=dev).repeat(mask.shape[0], 1)
    fin = torch.cat((begin_inds[mask].unsqueeze(0), dest_inds[mask].unsqueeze(0)), dim=0)
    return fin


class GNNPolicy(nn.Module):
  def __init__(self, net_arch={}):
    super(GNNPolicy, self).__init__()

    net_arch_actor = net_arch['actor']
    net_arch_critic = net_arch['critic']

    self.latent_dim_pi = 1
    self.latent_dim_vf = net_arch_actor['max_team_obs_size'] + net_arch_actor['max_targ_obs_size']

    self.actor = GNN(net_arch=net_arch_actor)
    self.critic = GNN(net_arch=net_arch_critic)

  def forward_actor(self, x):
    obs, action_mask, _, _, _, _ = x
    agent_team_head = obs[:, 2].long()
    # Grab the current team's output
    out = self.actor(x)[:, 0]
    fin_out = out + action_mask[:, :-1]
    fin_out = torch.cat([fin_out, action_mask[:, -1:]], dim=-1)

    if self.training:
      return [fin_out, None]
    else:
      return [fin_out, agent_team_head.unsqueeze(0)]
  
  def forward_critic(self, x):
    return self.critic(x).squeeze(-1) 

  def forward(self, x, deterministic=False, actions=None):
    return self.forward_actor(x), self.forward_critic(x)