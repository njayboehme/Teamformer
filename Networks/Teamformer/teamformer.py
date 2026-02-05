import torch
import torch.nn as nn
import torch.nn.functional as F

from Networks.mlp import MLP
from Networks.Teamformer.positional_embedding import PositionalEmbedding
from Networks.Teamformer.capability_embedding import CapabilityEmbedding
from Utils.mask import get_cur_mask

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
Teamformer Policy for scalable heterogeneous team formation
'''

class Teamformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        net_arch = kwargs.pop('net_arch', {})
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(p=0.1)
        rows, cols, self.max_num_caps, self.max_num_agents = net_arch.get('general_info')
        self.scale = torch.tensor([rows, cols], device=dev)
        self.do_informed = net_arch.get('do_informed', False)
        enc_dim, enc_num_heads, enc_num_layers = net_arch.get('enc_info')
        dec_dim, dec_num_heads, dec_num_layers = net_arch.get('dec_info')
        mlp_hid_dim, mlp_layers = net_arch.get('mlp_info')
        self.max_seq_len = self.max_num_agents + 1

        # required members
        self.latent_dim_pi = 1
        self.latent_dim_vf = self.max_num_agents

        # Embeddings
        # +2 because I need a SOS token and EOS token
        # Padding index at 0, EOS is index 1 and SOS is index 2.
        self.token_embedder = nn.Embedding(3, dec_dim, padding_idx=0)
        self.pos_emb_1d = PositionalEmbedding(enc_dim, 1, max_team_size=self.max_num_agents + 1)
        self.pos_emb_2d = PositionalEmbedding(enc_dim, 2, rows=rows, cols=cols)
        self.location_embedding = nn.Linear(2, enc_dim, bias=False)
        self.capability_layer = CapabilityEmbedding(self.max_num_caps, enc_dim)

        # Encoder Params
        self.enc_num_heads = enc_num_heads
        encoder_layer = nn.TransformerEncoderLayer(enc_dim, enc_num_heads, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, enc_num_layers)

        # Decoder Params
        self.dec_num_heads = dec_num_heads
        decoder_layer = nn.TransformerDecoderLayer(dec_dim, dec_num_heads, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(decoder_layer, dec_num_layers)

        # MLPs
        self.value_mlp = MLP(enc_dim, 1, mlp_hid_dim, mlp_layers)
        # This must be able to predict all caps plus an EOS token
        self.cap_mlp = MLP(dec_dim, self.max_num_caps + 1, mlp_hid_dim, mlp_layers)

        # Indicators for current team, nearby team, assigned task, and tasks
        self.team_info_embedder = nn.Linear(2, enc_dim, bias=False)
        self.target_info_embedder = nn.Linear(2, enc_dim, bias=False)


        ################################################################################

    def reinit(self, rows, cols, num_agents, d_model):
        self.scale = torch.tensor([rows, cols], device=dev)
        self.max_num_agents = num_agents
        self.max_seq_len = num_agents + 1
        self.pos_emb_1d = PositionalEmbedding(d_model, 1, max_team_size=self.max_num_agents + 1)


    # Run through the encoder
    def get_global_state(self, features, do_central):
        obs, _, team_obs, target_obs, team_mask, local_target_mask = features
        batch, N, _ = team_obs.shape

        # Team
        team_locs, team_caps, team_info = torch.split(team_obs, [2, self.max_num_caps, 2], dim=2)
        team_caps_embed = self.capability_layer(team_caps)
        team_locs_embeds = self.location_embedding(team_locs)
        team_info_embeds = self.team_info_embedder(team_info)
        # The last part of below is an embedding for the length of each team
        team_embeds = self.dropout(team_caps_embed + team_locs_embeds + team_info_embeds + self.pos_emb_1d(torch.sum(team_caps, dim=-1).to(int)))

        # Target
        target_locs, target_reqs, target_info, target_complete_mask = torch.split(target_obs, [2, self.max_num_caps, 2, 1], dim=2)
        target_reqs_embed = self.capability_layer(target_reqs)
        target_locs_embed = self.location_embedding(target_locs)
        target_info_embed = self.target_info_embedder(target_info)
        target_in = self.dropout(target_reqs_embed + target_locs_embed + target_info_embed)

        fin_in = torch.cat((team_embeds, target_in), dim=1)
        msk = torch.full((batch, fin_in.shape[1], fin_in.shape[1]), -1e9, device=dev)
        msk[:, torch.arange(fin_in.shape[1]), torch.arange(fin_in.shape[1])] = 0.0

        # Mask the padded teams and targets and make sure the pad tokens don't see anything else
        imp_team_mask = torch.where(team_mask[:, 0] == -1e9, False, True)
        local_target_mask = torch.where(local_target_mask[:, :-1] == 0, False, True)
        imp_team_target_mask = torch.cat((imp_team_mask, local_target_mask), dim=1)

        index_mask = imp_team_target_mask.unsqueeze(-1).repeat_interleave(fin_in.shape[1], dim=-1)
        fill_mat = torch.where(imp_team_target_mask.unsqueeze(1).repeat_interleave(fin_in.shape[1], dim=1) == True, 0.0, -1e9)
        msk[index_mask] = fill_mat[index_mask]
        
        msk = torch.repeat_interleave(msk, repeats=self.enc_num_heads, dim=0)
        return self.enc(fin_in, mask=msk), msk[:, [0], :]

    def forward_critic(self, features):
        # don't need to estimate value of the target sequences
        gs, _ = self.get_global_state(features, True)
        gs = gs[:, :self.max_num_agents]
        out = self.value_mlp(gs)
        return out.squeeze(-1)
    
    # Get the global state and run through decoder
    def forward_actor(self, features, deterministic=True, actions=None, testing=False):
        obs, action_mask, team_obs, target_obs, team_mask, local_target_mask = features
        og_team_len = obs[:, 0].long().unsqueeze(1)
        agent_team_head = obs[:, 2].long()
        gs, context_mask = self.get_global_state(features, False)
        # With the new state space, the current team's state is always the first one
        if not self.do_informed:
            gs = gs[torch.arange(obs.shape[0]), 0]
        if self.training:
            _, action_logs, ent = self.dec_masked_forward(gs, action_mask, actions, og_team_len, context_mask)
            return (_, action_logs, ent), None
        else:
            actions, action_logs, ent = self.dec_autoregressive_forward(gs, action_mask, deterministic, og_team_len, testing, context_mask)
            return (actions, action_logs, ent), agent_team_head.unsqueeze(0)

    # Causal mask when training
    def dec_masked_forward(self, glob_state, action_mask, ordered_actions, og_team_len, context_mask):
        batch = glob_state.shape[0]
        if not self.do_informed:
            glob_state = glob_state.unsqueeze(1)

        # Add in the final forced EOS
        ordered_actions = torch.cat((ordered_actions, torch.full((batch, 1), self.max_num_caps, device=dev)), dim=1)
        EOS_indices = torch.argmax(ordered_actions, dim=1)
        order_caps_embeds = self.capability_layer(torch.flatten(ordered_actions.to(int)), False).reshape(batch, self.max_num_agents + 1, -1)
        order_caps_embeds[torch.arange(batch), EOS_indices] = order_caps_embeds[torch.arange(batch), EOS_indices] + self.token_embedder(torch.tensor([1], device=dev))
        # Add in pos embeddings
        msk = ordered_actions != self.max_num_caps
        pos_emb = torch.cat((self.pos_emb_1d(torch.arange(1, self.max_num_agents + 1)).repeat(batch, 1, 1), torch.zeros((batch, 1, order_caps_embeds.shape[-1]), device=dev)), dim=1)
        order_caps_embeds[msk] = order_caps_embeds[msk] + pos_emb[msk]
        # Add in the SOS 
        order_caps_embeds = torch.cat((self.token_embedder(torch.full((batch, 1), 2, device=dev)), order_caps_embeds), dim=1)
        fin_context_mask = context_mask.repeat(1, order_caps_embeds.shape[1], 1)
        logits = self.cap_mlp(self.dec(order_caps_embeds, glob_state, tgt_mask=nn.Transformer.generate_square_subsequent_mask(order_caps_embeds.shape[1]), tgt_is_causal=True, memory_mask=fin_context_mask))

        ###
        chosen_caps = torch.full((batch, self.max_seq_len), self.max_num_caps, device=dev)
        # get the log prob for each action
        action_logs = torch.zeros((batch, self.max_seq_len), device=dev)
        # get the entropy for each action
        ents = torch.zeros((batch, self.max_seq_len), device=dev)
        # Need to generate the mask
        for i in range(self.max_seq_len):
            cur_action = ordered_actions[:, i].to(int)
            logit = logits[:, i]
            mask = get_cur_mask(action_mask[:, 0], chosen_caps, action_mask[:, 1:], self.max_num_caps, i, og_team_len, self.max_seq_len)
            logit = logit + mask

            d = torch.distributions.Categorical(logits=logit)
            action_logs[:, i] = d.log_prob(cur_action)
            ents[:, i] = d.entropy()
            chosen_caps[:, i] = cur_action
        return None, action_logs, ents

    # Autoregressive when not training
    def dec_autoregressive_forward(self, glob_state, action_mask, deterministic, og_team_len, testing, context_mask):
        '''
        @param glob_state - batch x d_model
        '''
        batch = glob_state.shape[0]
        if not self.do_informed:
            glob_state = glob_state.unsqueeze(1)
        # Get the SOS token
        action_embeds = self.token_embedder(torch.full((batch, 1), 2,  dtype=int ,device=dev))
        
        chosen_caps = torch.full((batch, self.max_seq_len), self.max_num_caps, device=dev)
        # holds each action
        actions = torch.zeros((batch, self.max_num_caps + 1), device=dev)
        # holds each action's lob_prob
        action_logs = torch.zeros((batch, self.max_seq_len), device=dev)
        # glob_state = glob_state.unsqueeze(1)
        ordered_actions = torch.tensor([], device=dev)
        cur_context_mask = context_mask
        # Begin Autoregression
        for i in range(self.max_seq_len):
            logit = self.cap_mlp(self.dec(action_embeds, glob_state, tgt_mask=nn.Transformer.generate_square_subsequent_mask(action_embeds.shape[1]), tgt_is_causal=True, memory_mask=cur_context_mask)[:, -1, :])
            mask = get_cur_mask(action_mask[:, 0], chosen_caps, action_mask[:, 1:], self.max_num_caps, i, og_team_len, self.max_seq_len)
            logit = logit + mask

            d = torch.distributions.Categorical(logits=logit)
            action = d.mode if deterministic else d.sample()
            action_log = d.log_prob(action)
            ordered_actions = torch.cat((ordered_actions, action.unsqueeze(1)), dim=-1)

            actions[torch.arange(batch), action] = actions[torch.arange(batch), action] + 1
            action_logs[:, i] = action_log
            chosen_caps[:, i] = action

            # When EOS would be predicted again, instead predict a pad token, otherwise give normal eos
            tokens = torch.where(actions[:, -1:] > 1, 0, actions[:, -1:]).to(int)
            # if EOS was predicted, the position emb will be zero, otherwise give the current position of the capability
            cur_pos = torch.where(actions[:, -1:] > 0, 0, i + 1).to(int)
            cur_act_embed = self.capability_layer(action, False) + self.token_embedder(tokens) + self.pos_emb_1d(cur_pos)
        
            # Add the action to the decoder input for the next iteration
            action_embeds = torch.cat((action_embeds, cur_act_embed), dim=1)
            cur_context_mask = torch.cat((cur_context_mask, context_mask), dim=1)

            if ordered_actions.shape[1] < self.max_seq_len and torch.all(ordered_actions[:, -1] == self.max_num_caps):
                ordered_actions = F.pad(ordered_actions, (0, self.max_seq_len - ordered_actions.shape[1], 0, 0), value=self.max_num_caps)
                break
        # If we are collecting rollouts, use below
        if not testing:
            return ordered_actions[:, :-1].to(int), action_logs, None
        else:
            return (actions[:, :-1].to(int),)

        

    def forward(self, x, deterministic=True, actions=None):
        return self.forward_actor(x, deterministic, actions), self.forward_critic(x)
    