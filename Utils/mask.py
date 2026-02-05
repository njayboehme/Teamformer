import torch

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
def get_cur_mask(cur_team_caps, chosen_caps, join_team_caps, EOS_index, index, og_team_len, max_seq_len):
    '''
    Use this for the autoregressive actions to generate the mask on the fly
    @param cur_team_caps - The capabilities of the team, in order
    @param chosen_caps - The capabilities that have been chosen so far, in order
    @param join_team_caps - The capabilities of the teams available for a team
    @param index - The current index being predicted
    @param og_team_len - The original team length of each team
    '''
    # Start with everything masked
    mask = torch.full((cur_team_caps.shape[0], EOS_index + 1), -1e9, device=dev)

    # If this is the first index, just unmask the first cap
    if index == 0:
        first_cap = cur_team_caps[:, 0].to(int)
        mask[torch.arange(mask.shape[0]), first_cap] = 0.0
        return mask
    # If the max seq length has been reached, force and EOS token
    elif index == max_seq_len - 1:
        mask[:, -1] = 0.0
        return mask

    # If i < len of cur team
    split_masks = torch.any(cur_team_caps < chosen_caps[:, :-1], dim=1)
    # if i == len of cur team
    same_masks = torch.all(cur_team_caps == chosen_caps[:, :-1], dim=1)
    # if i > len of cur team
    join_masks = torch.any(cur_team_caps > chosen_caps[:, :-1], dim=1)
    # If eos has been chosen, force it
    eos_masks = chosen_caps[:, index - 1] == EOS_index

    # For split
    # Get the next cap in the team and concat with EOS
    unmask_inds = torch.cat((cur_team_caps[:, index].unsqueeze(1), torch.tensor([EOS_index], device=dev).repeat(cur_team_caps.shape[0]).unsqueeze(1)), dim=1).to(int)
    # Select the indices at split mask to unmask (i.e. the next cap and EOS)
    mask[split_masks, torch.transpose(unmask_inds, 0, 1)[:, split_masks]] = 0.0

    # For planning
    unmask_inds_same = torch.cat((join_team_caps[:, :, 0], torch.tensor([EOS_index], device=dev).repeat(join_team_caps.shape[0]).unsqueeze(1)), dim=1).to(int)
    mask[same_masks, torch.transpose(unmask_inds_same, 0, 1)[:, same_masks]] = 0.0

    # For join
    cur_team_caps_ext = cur_team_caps.unsqueeze(1).repeat(1, join_team_caps.shape[1], 1)
    # This should be batch x max_num_teams - 1 x max_num_teams
    # Create all possible teams from the og team caps
    pot_teams = torch.empty(cur_team_caps_ext.shape, device=dev)
    cur_team_mask = cur_team_caps_ext != EOS_index
    indices_to_shift = torch.arange(join_team_caps.shape[-1], device=dev)
    shifted_indices = (indices_to_shift - og_team_len) % (max_seq_len - 1) # I think this should be modded by max_num_caps
    shifted_indices = shifted_indices.unsqueeze(1).repeat(1, join_team_caps.shape[1], 1)
    shifted_join_caps = torch.gather(join_team_caps, dim=-1, index=shifted_indices)
    pot_teams[cur_team_mask] = cur_team_caps_ext[cur_team_mask]
    pot_teams[~cur_team_mask] = shifted_join_caps[~cur_team_mask]
    
    # Get all possible caps to join
    join_caps = pot_teams[:, :, index].to(int)
    # Create mask for the invalid teams
    pot_cap_mask = torch.all(pot_teams[:, :, :index] == chosen_caps[:, :index].unsqueeze(1).repeat(1, join_team_caps.shape[1], 1), dim=-1)
    # Mask batches that are not joining
    pot_cap_mask[~join_masks] = False
    # Get the batch indices of where the mask is true
    batch_indices = torch.nonzero(pot_cap_mask)
    # Unmask all the viable potential caps
    mask[batch_indices[:, 0], join_caps[pot_cap_mask]] = 0.0

    # For sequences that have ended
    eos_mask = torch.full((EOS_index + 1,), -1e9, device=dev)
    eos_mask[-1] = 0.0
    mask[eos_masks] = eos_mask

    return mask
    
