import torch
import numpy as np

from Environment.Population import Population
from Environment.Team import Team
from Environment.Target import Target

def get_actions(population : Population, cur_dist_mat, targets: list[Target], do_optimal=False):
    actions = {}
    if do_optimal:
        for team in population.population:
            prev_team = population.get_prev_team(team.team[0])
            # take a plan action
            _set_agent_actions(team, prev_team.total_caps, actions)
        return actions

    for team in population.population:
        prev_team = population.get_prev_team(team.team[0])
        cur_targ_index = team.cur_assignment if team.cur_assignment != -1 else team.cur_contrib
        if cur_targ_index != -1:
            targ = targets[cur_targ_index]
            # If the team is at the target position
            if team.pos == [targ.x, targ.y]:
                # If all reqs are met
                if np.all(team.total_caps >= targ.reqs):
                    # take a plan action
                    _set_agent_actions(team, prev_team.total_caps, actions)
                else:
                    near_teams = _get_nearby_teams(population, team, cur_dist_mat)
                    if len(near_teams) > 0:
                        # join a team that has a capability the current team does not have
                        for n_t in near_teams:
                            n_t_targ_index = n_t.cur_assignment if n_t.cur_assignment != -1 else n_t.cur_contrib
                            if cur_targ_index != n_t_targ_index:
                                continue
                            needed_reqs = np.clip(targ.reqs - team.total_caps, 0, None)
                            needed_reqs = np.nonzero(needed_reqs)[0]
                            nearby_useful_caps = n_t.total_caps[needed_reqs]
                            if np.sum(nearby_useful_caps) > 0:
                                _set_agent_actions(team, team.total_caps + n_t.total_caps, actions)
                                break
                        
                        # If there are no nearby teams that are new, plan
                        if len(actions.get(team.team[0], [])) == 0:
                            _set_agent_actions(team, prev_team.total_caps, actions)
                    else:
                        # take the plan action since we can't join
                        _set_agent_actions(team, prev_team.total_caps, actions)
            # If the team is not at the target position, split down to single agents. Once at single agents, plan
            else:
                # If the team is bigger than 1, split
                if len(team.team) > 1:
                    mid_team = len(team.team) // 2
                    caps_inds = team.caps_ordered[:mid_team]
                    new_caps = np.zeros(len(team.total_caps))
                    np.add.at(new_caps, caps_inds, 1)
                    _set_agent_actions(team, new_caps, actions)
                # Once the team is size 1, plan
                else:
                    _set_agent_actions(team, prev_team.total_caps, actions)
        
        # If the team has nothing to do, take the plan actions
        else:
            _set_agent_actions(team, prev_team.total_caps, actions)
    return actions

def _get_nearby_teams(population: Population, cur_team: Team, cur_dist_mat):
    agent_to_team = population.agent_to_team
    p = np.array(population.population)
    dist = cur_dist_mat

    # Find what teams are within the join radius
    mask = dist[cur_team.team[0].label] <= 0.0
    sensed_team_indices = torch.unique(agent_to_team[mask], sorted=False)
    if len(sensed_team_indices) > 1:
        nearby_teams = p[sensed_team_indices]
        # Remove the cur_team
        m = nearby_teams != p[agent_to_team[cur_team.team[0].label].item()]
        return nearby_teams[m]
    else:
        return []

def _set_agent_actions(team: Team, action, action_dict: dict):
    for agent in team.team:
        action_dict[agent] = action