import functools
import copy
import itertools
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import torch
import torch.nn.functional as F
from pettingzoo import ParallelEnv

from Environment.Target import Target
from Environment.Population import Population
from Environment.Team import Team
from Environment.Agent import Agent
from Planners.task_allocator import TaskAllocator
from Planners.motion_planner import Motion_Planner
from Planners.tsp_policy import get_actions

class Environment(ParallelEnv):

    metadata = {
        "name": "custom_parallel_environment_v0",
    }
    render_mode = "None"

    def __init__(self, **kwargs):
        super().__init__()
        self.rows = kwargs.get('rows', 10)
        self.cols = kwargs.get('cols', 10)

        self.testing = kwargs.get('testing', False)
        other_num_caps = kwargs.get('other_num_caps', None)
        # These are only set during testing
        self.num_reqs_per_target = kwargs.get('num_reqs_per_target', None)
        max_team_spawn_size = kwargs.get('max_team_spawn_size', None)
        self.sensing_radius = kwargs.get('sensing_radius', 0.0)
        spawn_radius = kwargs.get('spawn_radius', 0.0)

        num_agents, agent_caps, max_num_caps, agent_start_locations, split_percent = kwargs.get('population_info', [0, np.array([]), None, np.array([])])
        self.pop = Population(num_agents, agent_caps, max_num_caps, agent_start_locations, self.rows, self.cols, split_percent, self.testing, other_num_caps, max_team_spawn_size, sensing_radius=self.sensing_radius, spawn_radius=spawn_radius)
        self.num_teams = len(self.pop.population)
        self.max_agents = num_agents
        self.agents = self.pop.get_agents()
        self.num_caps = self.pop.max_num_caps
        # This is used when the join action is taken
        self.min_join_dist = kwargs.get('min_join_dist', 0.0)

        # This is just added for the API and should never change. self.agents should change
        self.possible_agents = copy.copy(self.agents) + self.pop.dummy_agents

        self.shared_team_space = {}
        # team_split_pairs is set in action masking
        self.team_split_pairs = {}

        self.max_num_targets = kwargs.get('max_num_targets')
        self.targets = kwargs.get('targets', None)
        self.do_random_targets = kwargs.get('do_random_targets', True)
        self.test_min_num_targs = kwargs.get('test_min_num_targs', 1)
        self.completed_targets = np.full(self.max_num_targets, False)
        # This keeps track of what teams are assigned to what targets (note, assigned means the team can complete the target)
        self.assigned_targets = np.full((self.max_num_agents), -1)

        self.num_eps = kwargs.get('episodes', 100)
        self.time_step = 0

        self.eval = kwargs.get('eval', False)
        self.eval_agent_info = kwargs.get('eval_agent_info', None)
        self.eval_target_info = kwargs.get('eval_target_info', None)
        self.eval_ctr = 0

        self.is_split_env = False

        self.use_strict_reward = kwargs.get('use_strict_reward')
        self.ta_method = kwargs.get('ta_method', None)

        self.motion_planner = Motion_Planner('grid', **{'rows':self.rows, 'cols':self.cols})
        self.task_allocator = TaskAllocator(strict=self.use_strict_reward)
        self.max_targ_obs_size = kwargs.get('max_targ_obs_size')
        self.max_team_obs_size = kwargs.get('max_team_obs_size')

        self.targ_rew_dict = {}
        self.do_tsp_policy = kwargs.get('do_tsp_policy', False)
        self.do_optimal = kwargs.get('do_optimal', False)
        self.do_dynamic_targets = kwargs.get('do_dynamic_targets', False)
        self.base_num_targs = kwargs.get('base_num_targs', None)
        self.dynamic_TA_method = kwargs.get('dynamic_TA_method', 'TSP')
        self.do_normal_state = kwargs.get('do_normal_state', False)

        # This is used for action masking
        self.join_radius = kwargs.get('join_radius', 0.0)
        # Reward terms
        self.target_complete_reward = kwargs.get('target_complete_reward', 0.0)
        self.alignment_scale = kwargs.get('alignment_scale', 0.0)
        self.capability_scale = kwargs.get('capability_scale', 0.0)
        self.timestep_penalty = kwargs.get('timestep_penalty', 0.0)

        if self.do_tsp_policy:
            self.do_same_env = kwargs.get('do_same_env', False)
            
        # For normal action space (i.e. other policies that can't use the reduced action space)
        self.use_normal_action_space = kwargs.get('use_normal_action_space', False)
        if self.use_normal_action_space:
            self.num_actions = 2**(self.num_caps + 1)
            self.num_join_actions = 2**self.num_caps 
            self.dummy_action_mask = np.full(self.num_actions, -1e9)
            # Allow only the dummy action to be taken
            self.dummy_action_mask[-1] = 0.0
            self.action_index_to_caps = {}
            self.caps_to_action_index = {}
            for i, caps in enumerate(itertools.product([0., 1.], repeat=self.num_caps)):
                if i == 0: continue
                cap_inds = np.nonzero(caps)[0]
                self.caps_to_action_index[tuple(cap_inds)] = i
                self.action_index_to_caps[i] = cap_inds
            
            self.do_same_env = kwargs.get('do_same_env')


    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        
        if self.use_normal_action_space and self.do_same_env:
            return self.same_env_reset()
        
        if self.do_tsp_policy and self.do_same_env:
            return self.same_env_reset()
        
        if not self.eval:
            self.shared_team_space, self.cur_locs, present_capabilities, is_split_env, spawn_point = self.pop.reset()
            self.is_split_env = is_split_env
        # If this is an evaluation env
        else:
            self.is_split_env = False
            cur_agent_eval_info = self.eval_agent_info[self.eval_ctr]
            num_agents = cur_agent_eval_info['num_agents']
            team_sizes = cur_agent_eval_info['team_sizes']
            team_locs = cur_agent_eval_info['team_locs']
            team_caps = cur_agent_eval_info['team_caps']
            self.shared_team_space, self.cur_locs, _, _, spawn_point = self.pop.reset(num_agents, team_sizes, team_locs, team_caps)
            cur_target_info = self.eval_target_info[self.eval_ctr]
            self.targets = copy.deepcopy(cur_target_info['targets'])
            self.num_targs = len(self.targets)
            self.target_obs = cur_target_info['target_obs']
            if self.eval_ctr + 1 == len(self.eval_agent_info):
                self.eval_ctr = 0
            else:
                self.eval_ctr += 1

        self.cur_dist_mat = torch.cdist(self.cur_locs, self.cur_locs)
        self.prev_dist_mat = None
        self.team_split_pairs = {}

        self.num_teams = len(self.pop.population)
        self.num_caps = self.pop.max_num_caps

        self.agents = self.pop.get_agents()
        self.possible_agents = copy.copy(self.agents) + self.pop.dummy_agents

        # This keeps track of what teams are assigned to what targets
        self.assigned_targets = np.full((self.max_num_agents), -1)

        self.time_step = 0
        
        obs = {}
        info = {}

        # Note, the target information does not change throughout the episode, only on a reset
        if self.do_random_targets:
            self.targets, self.target_obs, self.num_targs = Target.generate_targets(self.max_num_targets, present_capabilities, 
                                                                                    self.num_caps, bounds=[self.cols, self.rows], 
                                                                                    is_split_env=is_split_env, testing=self.testing, 
                                                                                    min_num_targets=self.test_min_num_targs, num_reqs_per_target=self.num_reqs_per_target)
        # Need to reset the target.complete when not generating random targets or doing an eval env
        elif not self.eval:
            self.target_obs = np.zeros((len(self.targets), 2 + self.num_caps))
            for i, t in enumerate(self.targets):
                t.reset([self.cols, self.rows])
                self.target_obs[i] = np.append([t.x, t.y], t.reqs)

        self.completed_targets = np.full(self.max_num_targets, False)
        self.completed_targets[self.num_targs:] = True

        if not self.do_dynamic_targets:
            self.task_allocator.create_target_order(self.target_obs[~self.completed_targets, :2], spawn_point)
            self.ordered_targets_obs = self.target_obs[self.task_allocator.target_order]
        else:
            # the completed targets should be just the base number of targets and then add in the newly spawned targets
            self.completed_targets = np.full(self.base_num_targs, False)
            self.extra_targs = self.targets[self.base_num_targs:]
            self.targets = self.targets[:self.base_num_targs]

            self.extra_targs_obs = self.target_obs[self.base_num_targs:]
            self.target_obs = self.target_obs[:self.base_num_targs]

            self.task_allocator.create_target_order(self.target_obs[:, :2], spawn_point)
            self.ordered_targets_obs = self.target_obs[self.task_allocator.target_order]
            self.spawn_point = spawn_point
            # The time step to add more targets
            self.target_add_step = np.random.randint(10, 25)
        
        team_mask = torch.where(self.cur_dist_mat <= self.sensing_radius, 0.0, -1e9)
        # ignore the dummy agents
        team_mask[:, len(self.agents):] = -1e9
        norm_team_obs = self.pop.get_normalized_team_obs()
        for a in self.agents:
            team = self.pop.get_team(a)

            if not self.do_normal_state:
                tgt_obs, tgt_mask, comp_targs, next_x_reqs = self.get_relevant_targets(team)
                tm_obs, tm_mask, tm_indices = self.get_relevant_teams(team, next_x_reqs, norm_team_obs)
            else:
                tgt_obs, tgt_mask, comp_targs, next_x_reqs = self.get_targ_states(team)
                tm_obs, tm_mask, tm_indices = self.get_team_states(team, norm_team_obs)

            obs[a] = {"observation": [len(team.team), self.pop.agent_to_team[a.label], team.team[0].label]}
            if not self.use_normal_action_space:
                obs[a].update({"action_mask":self.action_masking(a, [0], tm_obs, next_x_reqs, tm_indices, True)})
            else:
                obs[a].update({"action_mask":self.action_masking_normal_space(a, 0, tm_obs, next_x_reqs, tm_indices)})

            obs[a].update({'team_obs':tm_obs})
            obs[a].update({"target_obs": np.concatenate((tgt_obs, np.expand_dims(comp_targs, axis=1)), axis=1)})
            obs[a].update({'target_mask': tgt_mask})
            obs[a].update({'team_mask': tm_mask})


            info[a] = {}
            self.observation_space(a).seed(seed)
            self.action_space(a).seed(seed)

        dum_team_mask = torch.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
        # This let's the dummy agents attend to themselves since they always appear at the end of a sequence
        dum_team_mask[-1, -1] = 0.0
        for a in self.pop.dummy_agents:
            obs[a] = {"observation": [0, -1, a.label]}
            if not self.use_normal_action_space:
                obs[a].update({"action_mask":np.full((self.max_team_obs_size, self.max_agents), self.num_caps)})
            else:
                obs[a].update({"action_mask":self.dummy_action_mask})
                
            obs[a].update({'team_obs': np.zeros((self.max_team_obs_size, self.num_caps + 4))})
            obs[a].update({'target_obs': np.zeros((self.max_targ_obs_size, self.num_caps + 5))})
            obs[a].update({'team_mask': dum_team_mask})
            obs[a].update({'target_mask': np.concatenate((np.full(self.max_targ_obs_size, False), [True]))})

            info[a] = {}

        self.init_team_obs = copy.deepcopy(self.pop.all_team_info)
        self.targ_rew_dict = {}
        return obs, info
    
    def same_env_reset(self):
        '''
        This is used when comparing against another method. This will make the same env here as the other env.
        '''
        self.shared_team_space = {}
        for t in self.pop.population:
            x, y = t.pos
            if (x, y) in self.shared_team_space.keys():
                self.shared_team_space[(x,y)].append(t)
            else:
                self.shared_team_space[(x, y)] = [t]

        self.cur_dist_mat = torch.cdist(self.cur_locs, self.cur_locs)
        self.prev_dist_mat = None
        self.team_split_pairs = {}

        self.num_teams = len(self.pop.population)
        self.num_caps = self.pop.max_num_caps

        self.agents = self.pop.get_agents()
        self.possible_agents = copy.copy(self.agents) + self.pop.dummy_agents

        # This keeps track of what teams are assigned to what targets
        self.assigned_targets = np.full((self.max_num_agents), -1)

        self.time_step = 0
        
        obs = {}
        info = {}
        
        if self.do_optimal:
            self.join_time = 0
            spawn = [int(torch.mean(self.cur_locs[:, 0]).item()), int(torch.mean(self.cur_locs[:, 1]).item())]
            self.task_allocator.create_optimal_order(self.agents, self.targets, spawn)
            self.ordered_targets_obs = self.target_obs[self.task_allocator.target_order]

        if not self.do_dynamic_targets:
            self.completed_targets = np.full(self.max_num_targets, False)
            self.completed_targets[self.num_targs:] = True
        else:
            self.completed_targets = np.full(len(self.targets), False)
            
        team_mask = torch.where(self.cur_dist_mat <= self.sensing_radius, 0.0, -1e9)
        # ignore the dummy agents
        team_mask[:, len(self.agents):] = -1e9
        norm_team_obs = self.pop.get_normalized_team_obs()
        for a in self.agents:
            team = self.pop.get_team(a)

            tgt_obs, tgt_mask, comp_targs, next_x_reqs = self.get_relevant_targets(team)
            tm_obs, tm_mask, tm_indices = self.get_relevant_teams(team, next_x_reqs, norm_team_obs)

            obs[a] = {"observation": [len(team.team), self.pop.agent_to_team[a.label], team.team[0].label]}
            if not self.use_normal_action_space:
                obs[a].update({"action_mask":self.action_masking(a, [0], tm_obs, next_x_reqs, tm_indices, True)})
            else:
                obs[a].update({"action_mask":self.action_masking_normal_space(a, 0, tm_obs, next_x_reqs, tm_indices)})
            obs[a].update({'team_obs':tm_obs})
            obs[a].update({"target_obs": np.concatenate((tgt_obs, np.expand_dims(comp_targs, axis=1)), axis=1)})
            obs[a].update({'target_mask': tgt_mask})
            obs[a].update({'team_mask': tm_mask})

            info[a] = {}

        dum_team_mask = torch.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
        # This let's the dummy agents attend to themselves since they always appear at the end of a sequence
        dum_team_mask[-1, -1] = 0.0
        for a in self.pop.dummy_agents:
            obs[a] = {"observation": [0, -1, a.label]}
            if not self.use_normal_action_space:
                obs[a].update({"action_mask":np.full((self.max_team_obs_size, self.max_agents), self.num_caps)})
            else:
                obs[a].update({"action_mask":self.dummy_action_mask})
            obs[a].update({'team_obs':  np.zeros((self.max_team_obs_size, self.num_caps + 4))})
            obs[a].update({'target_obs': np.zeros((self.max_targ_obs_size, self.num_caps + 5))})
            obs[a].update({'team_mask': dum_team_mask})
            obs[a].update({'target_mask': np.concatenate((np.full(self.max_targ_obs_size, False), [True]))})

            info[a] = {}

        return obs, info
    
    def get_tsp_steps(self):
        return self.task_allocator.tsp_steps

    def _seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_normalized_target_obs(self, do_ordered=False):
        if do_ordered:
            return np.concatenate((self.ordered_targets_obs[:, :2] / np.array([self.cols - 1, self.rows - 1]), self.ordered_targets_obs[:, 2:]), axis=1)
        else:
            return np.concatenate((self.target_obs[:, :2] / np.array([self.cols - 1, self.rows - 1]), self.target_obs[:, 2:]), axis=1)
        
    def take_action(self, cur_team, action, do_update, is_prev_head, prev_team):
        '''
        @param a - team to take action
        @param action - The new caps the team should have
        @param do_update - A bool indicating if shared_team_space should be updated
        @param is_prev_head - A bool indicating if the agent was at the head of a chain last time step
        '''
        # If an agent just got split, it shouldn't be able to do anything OR if the agent is not the head (current or prev), don't do anything
        if (do_update and not is_prev_head) or (not do_update and not is_prev_head):
            return
        x, y = cur_team.pos

        # Do planner
        if np.all(action == prev_team.total_caps):
            self.check_finished_targets(cur_team)
            
            self.allocate_task(cur_team)
            cur_team.pos = list(cur_team.get_next_pos())
            mask = self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()
            self.cur_locs[mask] = torch.tensor(cur_team.pos, dtype=torch.float)
            if do_update or is_prev_head:
                self.pop.update_all_team_info(self.pop.agent_to_team[cur_team.team[0].label].item(), cur_team.pos)
                self.update_shared_team_space(cur_team, (x, y))

        # Join Action
        elif np.any(action > prev_team.total_caps):
            if do_update or is_prev_head:
                # Based on the join action, join a team that meets those requirements
                team1_ind = self.pop.agent_to_team[cur_team.team[0].label].item()
                join_team = self.find_join_team(prev_team, action)
                cur_join_ind = self.pop.agent_to_team[join_team.team[0].label].item()
                cur_join_team = self.pop.population[cur_join_ind]

                # If the cur_join_team is too far away, move towards it
                if np.linalg.norm(np.array(prev_team.pos) - np.array(join_team.pos)) > self.min_join_dist:
                    # Move towards where the current join team is located
                    next_loc = self.motion_planner.get_next_loc(tuple(cur_team.pos), tuple(cur_join_team.pos), algo='a_star')
                    cur_team.pos = list(next_loc)
                    # Update cur_locs
                    mask = self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()
                    self.cur_locs[mask] = torch.tensor(cur_team.pos, dtype=torch.float)
                    self.pop.update_all_team_info(team1_ind, cur_team.pos)
                    self.update_shared_team_space(cur_team, (x, y))

                    # In case the team hasn't already done TA, we need to add this here
                    cur_team.reset_TA()
                    self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment

                    self.allocate_task(cur_team)

                # Else, join the team
                else:
                    # Can't join yourself
                    if cur_team == cur_join_team:
                        return
                    
                    cur_team.reset_TA()
                    self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
                    
                    # Move the team to the current join team position
                    cur_team.pos = cur_join_team.pos
                    self.cur_locs[self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()] = torch.tensor(cur_team.pos, dtype=torch.float)
                    self.pop.update_all_team_info(team1_ind, cur_team.pos)
                    # Update shared team space
                    self.update_shared_team_space(cur_team, (x, y))
                    cur_x, cur_y = cur_team.pos # this was cur_x, cur_y = cur_join_team.pos

                    self.pop.combine(team1_ind, cur_join_ind)
                    self.assigned_targets[cur_join_team.agent_labels] = cur_team.cur_assignment
                    self.shared_team_space[(cur_x, cur_y)].remove(cur_join_team)
                    self.num_teams -= 1

                    # Do TA with the new team
                    self.allocate_task(cur_team)
        # Split Action
        else:
            # If this team has not taken an action yet
            if do_update and is_prev_head:
                # Anytime the split action is taken, the current team's assignment and motion plan need to be reset
                cur_team.reset_TA()
                # Get all the agents that are a part of the same team
                self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment

                # Need to pick a point in the team that will create a team that satisfies the action
                des_team_caps = action
                split_ind = self.find_split_team(cur_team, des_team_caps, True)
                # Based on the split action, split the team such that a new team is created that has the desired capabilities
                self.pop.split(self.pop.agent_to_team[cur_team.team[0].label].item(), split_ind)
                # Add the new team from split to the shared_team_space
                self.shared_team_space[(x, y)].append(self.pop.population[-1])
                self.num_teams += 1

                # Do TA for both teams
                self.allocate_task(cur_team)
                self.allocate_task(self.pop.population[-1])
                
    
    def take_action_normal_space(self, cur_team, action, do_update, is_prev_head):
        '''
        This is used for policies that cannot use the reduced action space
        @param a - agent to take action
        @param action - Index indicating the action to take
        @param do_update - A bool indicating if shared_team_space should be updated
        @param is_prev_head - A bool indicating if the agent was at the head of a chain last time step
        '''
        # If an agent just got split, it shouldn't be able to do anything OR if the agent is not the head (current or prev), don't do anything
        if (do_update and not is_prev_head) or (not do_update and not is_prev_head):
            return
        x, y = cur_team.pos

        # Do planner
        if action == 0:
            self.check_finished_targets(cur_team)

            target_to_mask = self.check_same_assignments(cur_team)
            if target_to_mask is None:
                target_to_mask = self.check_contribs(cur_team)
            if cur_team.cur_assignment != -1 and np.sum(self.assigned_targets == cur_team.cur_assignment) > 1:
                # Note, prev_dist_mat == cur_dist_mat right now since cur_dist_mat is updated after all actions have been taken
                sensed_teams_mask = self.prev_dist_mat[cur_team.team[0].label] <= self.sensing_radius
                # The indices of the nearby teams
                sensed_teams_indices = torch.unique(self.pop.prev_agent_to_team[sensed_teams_mask], sorted=False)
                # If there are nearby teams
                if len(sensed_teams_indices) > 1:
                    *nearby_team_labels, = map(lambda x: x.team[0].label, np.array(self.pop.prev_population)[sensed_teams_indices])
                    nearby_team_labels = np.array(nearby_team_labels)
                    # Get the assignments of the nearby teams
                    nearby_team_assigns = self.assigned_targets[nearby_team_labels]
                    same_assigns_mask = self.assigned_targets[cur_team.team[0].label] == nearby_team_assigns
                    # If two teams are assigned to the same team
                    if np.sum(same_assigns_mask) > 1:
                        targ_loc = self.target_obs[self.assigned_targets[cur_team.team[0].label], :2]
                        filt_nearby_team_labels = nearby_team_labels[same_assigns_mask]
                        team_locs = self.prev_locs[filt_nearby_team_labels]
                        dists = np.linalg.norm(team_locs - targ_loc, axis=1)
                        closest_team_index = np.argmin(dists)
                        # If the current team is the closest team, don't change the assignment
                        if filt_nearby_team_labels[closest_team_index] == cur_team.team[0].label:
                            target_to_mask = None
                        else:
                            target_to_mask = self.assigned_targets[cur_team.team[0].label]
                            cur_team.reset_TA()
                            self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
            # Doing this means a team won't flip flop between targets
            if cur_team.motion_plan == None:
                targ_obs_with_dummy = np.append(np.expand_dims(np.append(np.array([x, y]), np.zeros(self.num_caps)), 0), self.target_obs, axis=0)
                completed_with_dummy = np.expand_dims(np.append(np.array([False]), self.completed_targets), 1)
                if target_to_mask is not None:
                    completed_with_dummy[target_to_mask + 1] = True
                des_targ_loc = self.task_allocator.select_target(cur_team, targ_obs_with_dummy, completed_with_dummy, selection_criterion=self.ta_method)
                cur_team.motion_plan = self.motion_planner.get_path(tuple(cur_team.pos), (des_targ_loc[0], des_targ_loc[1]), algo='a_star')
            cur_team.pos = list(cur_team.get_next_pos())
            self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
            mask = self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()
            self.cur_locs[mask] = torch.tensor(cur_team.pos, dtype=torch.float)
            if do_update or is_prev_head:
                self.pop.update_all_team_info(self.pop.agent_to_team[cur_team.team[0].label].item(), cur_team.pos)
                self.update_shared_team_space(cur_team, (x, y))

        # Join Action
        elif action < self.num_join_actions:
            if do_update or is_prev_head:

                # Based on the join action, join a team that meets those requirements
                team1_ind = self.pop.agent_to_team[cur_team.team[0].label].item()
                des_join_team_caps = self.convert_action_to_caps(action)

                join_team = self.find_join_team(cur_team, des_join_team_caps)
                cur_join_ind = self.pop.agent_to_team[join_team.team[0].label].item()
                cur_join_team = self.pop.population[cur_join_ind]

                # If the cur_join_team is too far away, move towards it
                if np.linalg.norm(np.array(cur_team.pos) - np.array(cur_join_team.pos)) > self.min_join_dist:
                    next_loc = self.motion_planner.get_next_loc(tuple(cur_team.pos), tuple(cur_join_team.pos), algo='a_star')
                    cur_team.pos = list(next_loc)
                    # Update cur_locs
                    mask = self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()
                    self.cur_locs[mask] = torch.tensor(cur_team.pos, dtype=torch.float)
                    self.pop.update_all_team_info(team1_ind, cur_team.pos)
                    self.update_shared_team_space(cur_team, (x, y))

                    cur_team.reset_TA()
                    self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment

                    self.allocate_task(cur_team)
                # Else, join the team
                else:
                    # Can't join yourself
                    if cur_team == cur_join_team:
                        return
                    
                    cur_team.reset_TA()
                    self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
                    
                    # Move the team to the current join team position
                    cur_team.pos = cur_join_team.pos
                    self.cur_locs[self.pop.agent_to_team == self.pop.agent_to_team[cur_team.team[0].label].item()] = torch.tensor(cur_team.pos, dtype=torch.float)
                    # This is different than OG...
                    self.pop.update_all_team_info(team1_ind, cur_team.pos)
                    # Update shared team space
                    self.update_shared_team_space(cur_team, (x, y))
                    cur_x, cur_y = cur_join_team.pos

                    self.pop.combine(team1_ind, cur_join_ind)
                    self.assigned_targets[cur_join_team.agent_labels] = cur_team.cur_assignment
                    self.shared_team_space[(cur_x, cur_y)].remove(cur_join_team)
                    self.num_teams -= 1

                    self.allocate_task(cur_team)
        # Split Action
        else:
            # If this team has not taken an action yet
            if do_update and is_prev_head:
                # Anytime the split action is taken, the current team's assignment and motion plan need to be reset\
                cur_team.reset_TA()
                self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
                # Need to pick a point in the team that will create a team that satisfies the action
                des_team_caps = self.convert_action_to_caps(action - self.num_join_actions + 1)
                split_ind = self.find_split_team(cur_team, des_team_caps)
                # Based on the split action, split the team such that a new team is created that has the desired capabilities
                self.pop.split(self.pop.agent_to_team[cur_team.team[0].label].item(), split_ind)
                # Add the new team from split to the shared_team_space
                self.shared_team_space[(x, y)].append(self.pop.population[-1])
                self.num_teams += 1

                self.allocate_task(cur_team)
                self.allocate_task(self.pop.population[-1])
    
    def allocate_task(self, cur_team: Team):
        '''
        Assign a team to a task
        '''
        cur_x, cur_y = cur_team.pos
        target_to_mask = self.check_same_assignments(cur_team)
        if target_to_mask is None:
            target_to_mask = self.check_contribs(cur_team)
    
        if cur_team.motion_plan == None or (len(cur_team.motion_plan) == 0 and cur_team.cur_assignment == -1 and cur_team.cur_contrib == -1):
            targ_obs_with_dummy = np.append(np.expand_dims(np.append(np.array([cur_x, cur_y]), np.zeros(self.num_caps)), 0), self.target_obs, axis=0)
            completed_with_dummy = np.expand_dims(np.append(np.array([False]), self.completed_targets), 1)
            if target_to_mask is not None:
                completed_with_dummy[target_to_mask + 1] = True
            des_targ_loc = self.task_allocator.select_target(cur_team, targ_obs_with_dummy, completed_with_dummy, selection_criterion=self.ta_method)
            cur_team.motion_plan = self.motion_planner.get_path(tuple(cur_team.pos), (des_targ_loc[0], des_targ_loc[1]), algo='a_star')
        self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
        

    def check_finished_targets(self, cur_team):
        '''
        Check if a team is assigned to a target that has been completed
        '''
        if cur_team.cur_assignment != -1:
            if self.completed_targets[cur_team.cur_assignment]:
                cur_team.reset_TA()
                self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
        if cur_team.cur_contrib != -1:
            if self.completed_targets[cur_team.cur_contrib]:
                cur_team.reset_TA()

    def check_same_assignments(self, cur_team: Team):
        '''
        Check to see if a team is assigned to a target that another team can complete AND the current team can sense the other team
        '''
        target_to_mask = None
        if cur_team.cur_assignment != -1 and np.sum(self.assigned_targets == cur_team.cur_assignment) > 1:
            # Note, prev_dist_mat == cur_dist_mat right now since cur_dist_mat is updated after all actions have been taken
            sensed_teams_mask = self.prev_dist_mat[cur_team.team[0].label] <= self.sensing_radius
            # The indices of the nearby teams
            sensed_teams_indices = torch.unique(self.pop.prev_agent_to_team[sensed_teams_mask], sorted=False)
            # If there are nearby teams
            if len(sensed_teams_indices) > 1:
                *nearby_team_labels, = map(lambda x: x.team[0].label, np.array(self.pop.prev_population)[sensed_teams_indices])
                nearby_team_labels = np.array(nearby_team_labels)
                # Get the assignments of the nearby teams
                nearby_team_assigns = self.assigned_targets[nearby_team_labels]
                same_assigns_mask = self.assigned_targets[cur_team.team[0].label] == nearby_team_assigns
                # If two teams are assigned to the same team
                if np.sum(same_assigns_mask) > 1:
                    targ_loc = self.target_obs[self.assigned_targets[cur_team.team[0].label], :2]
                    filt_nearby_team_labels = nearby_team_labels[same_assigns_mask]
                    team_locs = self.prev_locs[filt_nearby_team_labels]
                    dists = np.linalg.norm(team_locs - targ_loc, axis=1)
                    closest_team_index = np.argmin(dists)
                    # If the current team is the closest team, don't change the assignment
                    if filt_nearby_team_labels[closest_team_index] == cur_team.team[0].label:
                        target_to_mask = None
                    else:
                        target_to_mask = self.assigned_targets[cur_team.team[0].label]
                        cur_team.reset_TA()
                        self.assigned_targets[cur_team.agent_labels] = cur_team.cur_assignment
        return target_to_mask
    
    def check_contribs(self, cur_team: Team):
        '''
        Check to see if a team is contributing to a target that another team can complete AND the current team can sense the other team
        '''
        target_to_mask = None
        # Only do this if the current team is contributing to a target
        if cur_team.cur_assignment == -1 and cur_team.cur_contrib != -1:
            sensed_teams_mask = self.prev_dist_mat[cur_team.team[0].label] <= self.sensing_radius
            # The indices of the nearby teams
            sensed_teams_indices = torch.unique(self.pop.prev_agent_to_team[sensed_teams_mask], sorted=False)
            sensed_teams_indices = np.array(sensed_teams_indices)
            if len(sensed_teams_indices) > 1:
                *nearby_team_info, = map(lambda x: np.concatenate((np.array([x.cur_assignment, x.cur_contrib, x.team[0].label]), np.clip(x.total_caps, 0, 1))), np.array(self.pop.prev_population)[sensed_teams_indices])
                nearby_team_info = np.array(nearby_team_info)
                # See what teams have the same target as an assignment or contributing
                same_mask = nearby_team_info[:, :2] == cur_team.cur_contrib
                # If any nearby team is assigned to the current target, the current team should move on
                if np.any(same_mask[:, 0]):
                    target_to_mask = cur_team.cur_contrib
                    cur_team.reset_TA()
                # If another team has the same cur_contrib
                elif np.sum(same_mask[:, 1]) > 1:
                    team_labels = nearby_team_info[:, 2][same_mask[:, 1]]
                    targ_reqs_indices = np.nonzero(self.target_obs[cur_team.cur_contrib, 2:])[0]
                    same_conts = np.all(nearby_team_info[same_mask[:, 1], 3:][:, targ_reqs_indices] == np.clip(cur_team.total_caps, 0, 1)[targ_reqs_indices], axis=1)
                    if np.sum(same_conts) > 1:
                        targ_loc = self.target_obs[cur_team.cur_contrib, :2]
                        team_locs = self.prev_locs[team_labels[same_conts]]
                        dists = np.linalg.norm(team_locs - targ_loc, axis=1)
                        closest_team_index = np.argmin(dists)
                        # If the closest team is not the cur_team, mask
                        if team_labels[same_conts][closest_team_index] != cur_team.team[0].label:
                            target_to_mask = cur_team.cur_contrib
                            cur_team.reset_TA()
        return target_to_mask



    def update_shared_team_space(self, team, prev_loc):
        '''
        @param team - The current team
        @param prev_loc - The previous location of team. Must be a tuple of the form (x, y)
        '''
        self.shared_team_space[prev_loc].remove(team)
        # Remove a key if the team will not be there anymore
        if prev_loc != team.pos and len(self.shared_team_space[prev_loc]) == 0:
            del self.shared_team_space[prev_loc]

        # If there is another team at this postion, add the current team
        if tuple(team.pos) in self.shared_team_space.keys():
            self.shared_team_space[tuple(team.pos)].append(team)
        else:
            self.shared_team_space[tuple(team.pos)] = [team]
    
    def convert_action_to_caps(self, action):
        '''
        @param action - The current action being taken
        @return to_ret- An array of size num_caps with ones at the positions indicated by action_to_caps
        '''
        to_ret = np.zeros(self.num_caps)
        cap_indices = self.action_index_to_caps[action]
        to_ret[cap_indices] += 1
        return to_ret
            
    def find_join_team(self, prev_team, desired_team_caps):
        '''
        @param cur_team - The team from the beginning of this time step
        @param desired_team_caps - The capabilities of the team the current team wants to join
        @return join_team - The team in prev_pop to join with
        '''
        nearby_teams = self.get_nearby_teams(prev_team, True)
        # Find the valid teams from last time step and use the caps from last time step for the current team
        *valid_teams, = map(lambda x: np.all(x.total_caps + prev_team.total_caps >= desired_team_caps), nearby_teams)
        valid_team_indices = np.nonzero(valid_teams)
        return self.pick_join_team(nearby_teams, valid_team_indices)
    
    def pick_join_team(self, nearby_teams, indices, method='first'):
        '''
        @param indices - indices to pick a team from
        @param method - The method to choose a team from indices
        @return join_team - The team to join with
        '''
        if method == 'first':
            return nearby_teams[indices[0][0]]
        elif method == 'random':
            return nearby_teams[indices[np.random.randint(len(indices))][0]]
        
    def find_split_team(self, cur_team, desired_team_caps, do_forward=False):
        '''
        @param cur_team - The team that is going to be split
        @param desired_team_caps - The capabilities of the team that will be created after splitting
        @return index - The index in the current team to split at
        '''
        cur_caps = np.zeros(self.num_caps)
        if do_forward:
            for i, cap in enumerate(cur_team.caps_ordered):
                cur_caps[cap] += 1
                if np.all(cur_caps == desired_team_caps):
                    return i + 1
        else:
            for i in range(-1, -len(cur_team.team), -1):
                # If we haven't encountered this capability before
                if cur_caps[cur_team.team[i].cap] == 0:
                    cur_caps[cur_team.team[i].cap] += 1
                # If the needs of the split are met, keep track of the index
                if np.all(cur_caps == desired_team_caps):
                    # Return the first team to meet the requirements
                    return len(cur_team.team) + i

    def action_masking(self, a: Agent, action, team_obs, next_x_reqs, team_indices, reset=False):
        '''
        @param a - current agent
        @param action - The action the team took
        @return cur_action_mask - current action mask of the given agent
        '''
        cur_team = self.pop.get_team(a)
        mask = np.full((self.max_team_obs_size, self.max_agents), self.num_caps)
        mask[0, :len(cur_team.caps_ordered)] = cur_team.caps_ordered
        # If the split action was just taken, don't allow the join action to be taken
        if not self.do_tsp_policy:
            act = self.convert_action(action)
        elif self.do_tsp_policy:
            return mask
        
        if not reset and np.any(act < self.pop.get_prev_team(a).total_caps):
            return mask
        # join action masking
        if len(team_indices) > 0:
            new_dist_mat = torch.cdist(team_obs[0, :2].unsqueeze(0) * torch.tensor([self.cols - 1, self.rows - 1]), 
                                       team_obs[1:, :2] * torch.tensor([self.cols - 1, self.rows - 1]))
            dist_mask = (new_dist_mat <= self.join_radius).squeeze(0)
            nearby_team_indices = team_indices[dist_mask[:len(team_indices)]]
            if len(nearby_team_indices) > 0:
                if not self.do_normal_state:
                    # Shift the requirements by 2 to ignore locations in team_obs
                    next_x_reqs += 2
                    relevant_mask = torch.any(team_obs[1:][dist_mask, np.expand_dims(next_x_reqs, axis=1)] > 0, dim=0)
                    relevant_team_indices = nearby_team_indices[relevant_mask[:len(nearby_team_indices)]]
                else:
                    relevant_team_indices = nearby_team_indices
                mask[1:len(relevant_team_indices) + 1] = *map(lambda x: x.caps_ordered, np.take(self.pop.population, relevant_team_indices)),
        return mask
    
    def action_masking_normal_space(self, a, action, team_obs, next_x_reqs, team_indices):
        '''
        @param a - current agent
        @param action - The action the team took
        @return cur_action_mask - current action mask of the given agent
        '''
        cur_team = self.pop.get_team(a)
        # If the agent has failed, mask all actions
        if a.failed:
            return np.full(self.num_actions, -1e9)
        
        cur_action_mask = np.full(self.num_actions, -1e9)
        # Always allow the planner
        cur_action_mask[0] = 0.0

        if action >= self.num_join_actions:
            pass
        else:
            if len(team_indices) > 0:
                new_dist_mat = torch.cdist(team_obs[0, :2].unsqueeze(0) * torch.tensor([self.cols - 1, self.rows - 1]), 
                                       team_obs[1:, :2] * torch.tensor([self.cols - 1, self.rows - 1]))
                dist_mask = (new_dist_mat <= self.join_radius).squeeze(0)
                nearby_team_indices = team_indices[dist_mask[:len(team_indices)]]
                if not self.do_normal_state:
                    # shift requirements by 2 to ignore locations in team_obs
                    next_x_reqs += 2
                    relevant_mask = torch.any(team_obs[1:][dist_mask, np.expand_dims(next_x_reqs, axis=1)] > 0, dim=0)
                    relevant_team_indices = nearby_team_indices[relevant_mask[:len(nearby_team_indices)]]
                    relevant_teams = np.take(self.pop.population, relevant_team_indices)
                    if len(relevant_teams) > 0:
                        all_team_caps = self.get_join_team_caps(cur_team, relevant_teams)
                        cur_action_mask[[self.caps_to_action_index[tuple(team_cap[0])] for team_cap in all_team_caps]] = 0.0
                else:
                    if len(nearby_team_indices) > 0:
                        nearby_teams = np.take(self.pop.population, nearby_team_indices)
                        all_team_caps = self.get_join_team_caps(cur_team, nearby_teams)
                        cur_action_mask[[self.caps_to_action_index[tuple(team_cap[0])] for team_cap in all_team_caps]] = 0.0

        # split action masking
        if cur_team in self.team_split_pairs.keys():
            split_pairs = self.team_split_pairs[cur_team]
        else:
            split_pairs = self.get_split_pairs(cur_team)
            self.team_split_pairs[cur_team] = split_pairs
        # We just want to unmask the splits that can be created, i.e. we still want to mask the resultant team that comes from the actual split
        to_mask = split_pairs[np.arange(0, len(split_pairs), step=2)]
        mask = to_mask != -1
        cur_action_mask[2**self.num_caps + to_mask[mask]] = 0.0
        return cur_action_mask
    
    def get_nearby_teams(self, cur_team, use_prev):
        '''
        @param cur_team - The current team to reference nearby teams
        @param use_prev - Bool indicating whether to use current location/population info or previous
        @return nearby_team - A list of all the nearby teams
        '''
        if use_prev:
            agent_to_team = self.pop.prev_agent_to_team
            p = np.array(self.pop.prev_population)
            dist = self.prev_dist_mat
        else:
            agent_to_team = self.pop.agent_to_team
            p = np.array(self.pop.population)
            dist = self.cur_dist_mat

        # Find what teams are within the join radius
        mask = dist[cur_team.team[0].label] <= self.join_radius
        sensed_team_indices = torch.unique(agent_to_team[mask], sorted=False)
        if len(sensed_team_indices) > 1:
            nearby_teams = p[sensed_team_indices]
            # Remove the cur_team
            m = nearby_teams != p[agent_to_team[cur_team.team[0].label].item()]
            return nearby_teams[m]
        else:
            return []

    def get_join_team_caps(self, team, nearby_teams):
        return map(lambda x: np.nonzero(x.total_caps + team.total_caps), nearby_teams)
    
    def get_split_pairs(self, team):
        '''
        @param team - The team to find all split pairs of
        @return indices - A list of indices [new_team_index, old_team_without_new_team_capability_index]. Filled with -1 for padding.
        '''
        indices = np.full((self.num_caps * 2), -1)
        total_caps = copy.deepcopy(team.total_caps)
        # Remove the first agent's capability
        total_caps[team.team[0].cap] -= 1
        # Find which capabilities are present
        available_caps = np.where(total_caps > 0, 1, 0)

        new_team_caps = []
        accumulated_caps = np.zeros(self.num_caps, dtype=int)
        team_index = -1
        index = 0
        while sum(available_caps) != 0:
            cap = team.team[team_index].cap
            accumulated_caps[cap] += 1
            # If we encounter a new capability, we have a new team that could be created
            if available_caps[cap] != 0:
                available_caps[cap] -= 1
                new_team_caps.append(cap)
                # Subtract one so we are starting at the 0-th index
                indices[index] = self.caps_to_action_index[tuple(sorted(new_team_caps))] - 1
                indices[index + 1] = self.caps_to_action_index[tuple(np.nonzero(team.total_caps - accumulated_caps)[0])] - 1
                index += 2

            team_index -= 1
        return indices

    def update_targets(self, team, team_to_sensed_targs):
        sensed_targets = []
        for t in self.targets:
            if team.pos[0] == t.x and team.pos[1] == t.y:
                if self.do_optimal:
                    nearby_team_mask = torch.all(self.pop.all_team_info[:, :2] == torch.tensor([team.pos]), dim=1)
                    nearby_caps = self.pop.all_team_info[nearby_team_mask, 2:2+self.num_caps]
                    total_caps = torch.sum(nearby_caps, dim=0).numpy()
                    if np.all(total_caps >= t.reqs) and not t.completed:
                        sensed_targets.append(t)
                        t.completed = True
                        self.completed_targets[t.label] = True
                        # Once a target is completed, the team needs to reset TA
                        team.reset_TA()
                        self.assigned_targets[team.agent_labels] = team.cur_assignment
                        self.targ_rew_dict[team] = self.target_complete_reward

                        if team.motion_plan == None:
                            x, y = team.pos
                            targ_obs_with_dummy = np.append(np.expand_dims(np.append(np.array([x, y]), np.zeros(self.num_caps)), 0), self.target_obs, axis=0)
                            completed_with_dummy = np.expand_dims(np.append(np.array([False]), self.completed_targets), 1)
                            des_targ_loc = self.task_allocator.select_target(team, targ_obs_with_dummy, completed_with_dummy, selection_criterion=self.ta_method)
                            team.motion_plan = self.motion_planner.get_path(tuple(team.pos), (des_targ_loc[0], des_targ_loc[1]), algo='a_star')
                        self.assigned_targets[team.agent_labels] = team.cur_assignment
                        self.join_time += sum(t.reqs) - 1
                    continue
                # This relaxes the exact constraint when there are redundant capabilities in a team
                reduced_team_caps = (team.total_caps > 0).astype(int)
                # If the agent chain has all the capabilities to sense the target AND 
                # the target has not been sensed, give a reward
                if not self.use_strict_reward and np.all(team.total_caps >= t.reqs) and not t.completed:
                    sensed_targets.append(t)
                    t.completed = True
                    self.completed_targets[t.label] = True
                    # Once a target is completed, the team needs to reset TA
                    team.reset_TA()
                    self.assigned_targets[team.agent_labels] = team.cur_assignment
                    # In case the team hasn't already done TA, we need to add this here
                    self.targ_rew_dict[team] = self.target_complete_reward

                    target_to_mask = self.check_same_assignments(team)
                    if target_to_mask is None:
                        target_to_mask = self.check_contribs(team)

                    if team.motion_plan == None:
                        x, y = team.pos
                        targ_obs_with_dummy = np.append(np.expand_dims(np.append(np.array([x, y]), np.zeros(self.num_caps)), 0), self.target_obs, axis=0)
                        completed_with_dummy = np.expand_dims(np.append(np.array([False]), self.completed_targets), 1)
                        if target_to_mask is not None:
                            completed_with_dummy[target_to_mask + 1] = True
                        des_targ_loc = self.task_allocator.select_target(team, targ_obs_with_dummy, completed_with_dummy, selection_criterion=self.ta_method)
                        team.motion_plan = self.motion_planner.get_path(tuple(team.pos), (des_targ_loc[0], des_targ_loc[1]), algo='a_star')
                    self.assigned_targets[team.agent_labels] = team.cur_assignment

                elif self.use_strict_reward and np.all(reduced_team_caps == t.reqs) and not t.completed:
                    sensed_targets.append(t)
                    t.completed = True
                    self.completed_targets[t.label] = True
                    # Once a target is completed, the team needs to reset TA
                    team.reset_TA()
                    self.assigned_targets[team.agent_labels] = team.cur_assignment

                    # In case the team hasn't already done TA, we need to add this here
                    self.targ_rew_dict[team] = self.target_complete_reward

                    target_to_mask = self.check_same_assignments(team)
                    if target_to_mask is None:
                        target_to_mask = self.check_contribs(team)

                    if team.motion_plan == None:
                        x, y = team.pos
                        targ_obs_with_dummy = np.append(np.expand_dims(np.append(np.array([x, y]), np.zeros(self.num_caps)), 0), self.target_obs, axis=0)
                        completed_with_dummy = np.expand_dims(np.append(np.array([False]), self.completed_targets), 1)
                        if target_to_mask is not None:
                            completed_with_dummy[target_to_mask + 1] = True
                        des_targ_loc = self.task_allocator.select_target(team, targ_obs_with_dummy, completed_with_dummy, selection_criterion=self.ta_method)
                        team.motion_plan = self.motion_planner.get_path(tuple(team.pos), (des_targ_loc[0], des_targ_loc[1]), algo='a_star')
                    self.assigned_targets[team.agent_labels] = team.cur_assignment

        team_to_sensed_targs[team] = sensed_targets

    def convert_action(self, action):
        '''
        Create the desired capabilities and remove EOS
        '''
        z = np.zeros((self.num_caps + 1))
        np.add.at(z, action, 1)
        return z[:-1]
    
    def get_relevant_teams(self, cur_team, next_x_reqs, norm_team_obs):
        '''
        Filter out all_team_information to be size max_team_obs_size
        @param next_x_reqs - The next x required caps for the next x targets (these are indices)
        @param nearby_team_mask - Nearby team mask
        '''
        nearby_agent_mask = torch.bitwise_and(self.cur_dist_mat[cur_team.team[0].label] <= self.sensing_radius, self.pop.agent_to_team != self.pop.agent_to_team[cur_team.team[0].label])
        nearby_team_indices = self.pop.agent_to_team[nearby_agent_mask]
        nearby_team_indices = torch.unique(nearby_team_indices, sorted=False)
        nearby_team_obs = norm_team_obs[nearby_team_indices]
        if len(nearby_team_indices) > 0 and next_x_reqs is not None:
            # Shift next_x_reqs
            next_x_reqs += 2
            relevant_mask_raw = nearby_team_obs[:, next_x_reqs] > 0
            relevant_mask = torch.any(relevant_mask_raw, dim=1)
            relevant_team_indices = nearby_team_indices[relevant_mask]
            # If no nearby teams contribute to the next_x_reqs, treat it like there are no nearby teams (mask all other teams)
            if len(relevant_team_indices) == 0:
                team_obs = F.pad(norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), (0, 0, 0, self.max_team_obs_size - 1))
                # Add in the current team and nearby team indicator
                extra_team_info = torch.zeros((self.max_team_obs_size, 2))
                # Add in the current team indicator
                extra_team_info[0, 0] = 1.
                team_obs = torch.cat((team_obs, extra_team_info), dim=1)

                mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
                mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
                relevant_team_indices = torch.tensor([])
            # If the number of teams that can contribute to the next_x_reqs is less than the max_team_obs_size
            elif len(relevant_team_indices) < self.max_team_obs_size:
                team_obs = torch.cat((norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), nearby_team_obs[relevant_mask]), dim=0)
                team_obs = F.pad(team_obs, (0, 0, 0, self.max_team_obs_size - len(team_obs)))
                # Add in the current team and nearby team indicator
                extra_team_info = torch.zeros((self.max_team_obs_size, 2))
                extra_team_info[0, 0] = 1.
                extra_team_info[1:len(relevant_team_indices) + 1, 1] = 1.
                team_obs = torch.cat((team_obs, extra_team_info), dim=1)

                mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
                mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
                mask[:len(relevant_team_indices) + 1, :len(relevant_team_indices) + 1] = 0.0
                relevant_team_indices = nearby_team_indices[relevant_mask]
            # If there are more than max_team_obs_size teams that can contribute nearby, need to pick the teams that contribute the most to next_x_reqs
            else:
                # Find the teams that contribute to the next_x_reqs. Note, argmax will prioritize teams at the top.
                teams_to_keep_inds = np.argmax(relevant_mask_raw, axis=0)
                contributing_teams_mask = relevant_mask_raw[teams_to_keep_inds, np.arange(relevant_mask_raw.shape[1])]
                contributing_teams = teams_to_keep_inds[contributing_teams_mask]
                fin_contributing_teams = torch.unique(contributing_teams)
                relevant_team_indices = nearby_team_indices[fin_contributing_teams][:self.max_team_obs_size - 1]

                # Remove the teams that will be added and see if there are more teams that could be relevant
                relevant_mask[fin_contributing_teams] = False
                # Find the number of teams we need to add
                pad_num = max(self.max_team_obs_size - len(relevant_team_indices) - 1, 0)
                # Add in the other relevant teams
                relevant_team_indices = torch.cat((relevant_team_indices, nearby_team_indices[relevant_mask][:pad_num]))
                team_obs = torch.cat((norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), norm_team_obs[relevant_team_indices]), dim=0)
                # Create the mask for the current team_obs, unmasking everything up to the current size
                mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
                mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
                mask[:len(team_obs), :len(team_obs)] = 0.0
                # Add the final padding
                team_obs = F.pad(team_obs, (0, 0, 0, self.max_team_obs_size - len(team_obs)))
                # Add in the current team and nearby team indicator
                extra_team_info = torch.zeros((self.max_team_obs_size, 2))
                extra_team_info[0, 0] = 1.
                extra_team_info[1:len(relevant_team_indices) + 1, 1] = 1.
                team_obs = torch.cat((team_obs, extra_team_info), dim=1)

                
        # If there are no nearby teams or next_x_reqs is None, only do attn for the current team
        else:
            team_obs = F.pad(norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), (0, 0, 0, self.max_team_obs_size - 1))
            # Add in the current team and nearby team indicator
            extra_team_info = torch.zeros((self.max_team_obs_size, 2))
            extra_team_info[0, 0] = 1.
            team_obs = torch.cat((team_obs, extra_team_info), dim=1)

            mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
            mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
            relevant_team_indices = torch.tensor([])

        return team_obs, mask, relevant_team_indices

    def get_relevant_targets(self, cur_team: Team):
        '''
        Filter target information
        '''
        cur_team_cap_indices = np.nonzero(cur_team.total_caps)[0]
        if cur_team.cur_assignment != -1:
            index = np.nonzero(cur_team.cur_assignment == self.task_allocator.target_order)[0][0]
        elif cur_team.cur_contrib != -1:
            index = np.nonzero(cur_team.cur_contrib == self.task_allocator.target_order)[0][0]
        else:
            index = None

        # Only look at target beyond index (include index)
        if index is not None:
            ordered_completed_targets = ~self.completed_targets[self.task_allocator.target_order[index:]]
            filt_target_info = self.get_normalized_target_obs(True)[index:][ordered_completed_targets]
            filt_comp_targs = self.completed_targets[self.task_allocator.target_order[index:]][ordered_completed_targets]
        else:
            ordered_completed_targets = ~self.completed_targets[self.task_allocator.target_order]
            filt_target_info = self.get_normalized_target_obs(True)[ordered_completed_targets]
            filt_comp_targs = self.completed_targets[self.task_allocator.target_order][ordered_completed_targets]
        
        relevant_targs = np.any(filt_target_info[:, 2 + cur_team_cap_indices] > 0, axis=1)
        num_rel_targs = np.sum(relevant_targs)
        # If there are relevant targets
        if num_rel_targs > 0:
            # The targ_obs to be sent to the transformer
            targ_obs = filt_target_info[relevant_targs][:self.max_targ_obs_size]
            comp_targs = filt_comp_targs[relevant_targs][:self.max_targ_obs_size]
            targ_reqs = np.clip(np.sum(targ_obs[:, 2:], axis=0), 0, 1)
            targ_reqs[cur_team_cap_indices] -= 1
            next_x_reqs = np.nonzero(targ_reqs)[0]
            # This is technically only needed while training. Otherwise, padding would work fine.
            if len(targ_obs) < self.max_targ_obs_size:
                targ_obs = np.pad(targ_obs, ((0, self.max_targ_obs_size - len(targ_obs)), (0, 0)))
                comp_targs = np.pad(comp_targs, ((0, self.max_targ_obs_size - len(comp_targs))), constant_values=True)
            mask = np.full((self.max_targ_obs_size + 1), False)
            # unmask the relevant targets
            mask[:num_rel_targs] = True

            # Add in extra target info
            cur_target = np.zeros((self.max_targ_obs_size, 2))
            cur_target[:num_rel_targs, 0] = 1.
            if cur_team.cur_assignment != -1 or cur_team.cur_contrib != -1:
                t_ind = cur_team.cur_assignment if cur_team.cur_assignment != -1 else cur_team.cur_contrib
                cur_targ_obj = self.targets[t_ind]
                cur_targ_index = np.nonzero(np.all(targ_obs == np.concatenate((np.array([cur_targ_obj.x / (self.cols - 1), cur_targ_obj.y / (self.rows - 1)]), cur_targ_obj.reqs), axis=0), axis=1))[0]
                # If the length is 0, that means that the target the current team is assigned to has been completed.
                # If the length is greater than 1, there are multiple targets at the same location that can be completed.
                if len(cur_targ_index) > 0:
                    cur_target[cur_targ_index[0], 1] = 1.

            targ_obs = np.concatenate((targ_obs, cur_target), axis=1)

        # If there are no relevant targets, mask all remaining targets except the dummy target
        else:
            targ_obs = torch.zeros((self.max_targ_obs_size, self.target_obs.shape[1]))
            # Add in target and current target values (I will set them to all zero since these aren't real targets)
            targ_obs = torch.cat((targ_obs, torch.zeros((self.max_targ_obs_size, 2))), dim=1)
            comp_targs = self.completed_targets[:self.max_targ_obs_size]
            mask = np.full((self.max_targ_obs_size + 1), False)
            mask[-1] = True # unmask the dummy target
            next_x_reqs = None
        return targ_obs, mask, comp_targs, next_x_reqs
    
    def get_cur_target(self, cur_team: Team) -> Target:
        if cur_team.cur_assignment != -1:
            return self.targets[cur_team.cur_assignment]
        elif cur_team.cur_contrib != -1:
            return self.targets[cur_team.cur_contrib]
        elif list(cur_team.get_next_pos()) == cur_team.pos:
            return None
        else:
            raise ValueError(f"Cur team assignment and contrib equal to -1. Next pos: {cur_team.get_next_pos()}, current pos: {cur_team.pos}")

    def calculate_team_reward(self, cur_team: Team):
        '''
        Calculate the reward for a team given its capabilities and the task it is currently assigned to
        '''
        targ = self.get_cur_target(cur_team)
        targ_rew = self.targ_rew_dict[cur_team]
        if targ is not None:
            req_inds = np.nonzero(targ.reqs)[0]
            num_met_reqs = np.sum(cur_team.total_caps[req_inds] > 0)
            extra_caps = np.sum(np.clip(cur_team.total_caps - targ.reqs, 0, None))
            alignment_rew = self.alignment_scale * ((self.capability_scale * num_met_reqs - self.capability_scale * extra_caps) / (self.capability_scale * targ.num_reqs))
            rew = targ_rew + alignment_rew + self.timestep_penalty
        # When a team has nothing else to do, just give timestep penalty (which is already negative!!!!)
        else:
            rew = targ_rew + self.timestep_penalty
        return rew
    
    def get_targ_states(self, cur_team: Team):
        if cur_team.cur_assignment != -1:
            index = np.nonzero(cur_team.cur_assignment == self.task_allocator.target_order)[0][0]
        elif cur_team.cur_contrib != -1:
            index = np.nonzero(cur_team.cur_contrib == self.task_allocator.target_order)[0][0]
        else:
            index = None

        # Only look at targets beyond index (include index)
        if index is not None:
            ordered_completed_targets = ~self.completed_targets[self.task_allocator.target_order[index:]]
            filt_target_info = self.get_normalized_target_obs(True)[index:][ordered_completed_targets]
            filt_comp_targs = self.completed_targets[self.task_allocator.target_order[index:]][ordered_completed_targets]
        else:
            ordered_completed_targets = ~self.completed_targets[self.task_allocator.target_order]
            filt_target_info = self.get_normalized_target_obs(True)[ordered_completed_targets]
            filt_comp_targs = self.completed_targets[self.task_allocator.target_order][ordered_completed_targets]
        
        targ_obs = filt_target_info[:self.max_targ_obs_size]
        num_rel_targs = len(targ_obs)
        targ_obs = np.pad(targ_obs, ((0, self.max_targ_obs_size - len(targ_obs)), (0, 0)))
        comp_targs = filt_comp_targs[:self.max_targ_obs_size]
        comp_targs = np.pad(comp_targs, ((0, self.max_targ_obs_size - len(comp_targs))), constant_values=True)
        mask = np.full((self.max_targ_obs_size + 1), False)
        
        # unmask dummy targ
        if num_rel_targs == 0:
            targ_obs = np.concatenate((targ_obs, np.zeros((self.max_targ_obs_size, 2))), axis=1)
            mask[-1] = True
            next_x_reqs = None
        # unmask the relevant targets
        else:
            cur_target = np.zeros((self.max_targ_obs_size, 2))
            cur_target[:num_rel_targs, 0] = 1.
            if cur_team.cur_assignment != -1 or cur_team.cur_contrib != -1:
                t_ind = cur_team.cur_assignment if cur_team.cur_assignment != -1 else cur_team.cur_contrib
                cur_targ_obj = self.targets[t_ind]
                cur_targ_index = np.nonzero(np.all(targ_obs == np.concatenate((np.array([cur_targ_obj.x / (self.cols - 1), cur_targ_obj.y / (self.rows - 1)]), cur_targ_obj.reqs), axis=0), axis=1))[0]
                # If the length is 0, that means that the target the current team is assigned to has been completed.
                # If the length is greater than 1, there are multiple targets at the same location that can be completed. So far, I just select the first one, could do all though...
                if len(cur_targ_index) > 0:
                    cur_target[cur_targ_index[0], 1] = 1.
            targ_obs = np.concatenate((targ_obs, cur_target), axis=1)

            mask[:num_rel_targs] = True
            targ_reqs = np.clip(np.sum(targ_obs[:, 2:], axis=0), 0, 1)
            next_x_reqs = np.nonzero(targ_reqs)[0]
        return targ_obs, mask, comp_targs, next_x_reqs
        

    def get_team_states(self, cur_team: Team, norm_team_obs):
        '''
        For a given team, return the current team's state as well as the states of all nearby teams
        '''
        nearby_agent_mask = torch.bitwise_and(self.cur_dist_mat[cur_team.team[0].label] <= self.sensing_radius, self.pop.agent_to_team != self.pop.agent_to_team[cur_team.team[0].label])
        nearby_team_indices = self.pop.agent_to_team[nearby_agent_mask]
        nearby_team_indices = torch.unique(nearby_team_indices, sorted=False)
        if len(nearby_team_indices) > 0:
            nearby_team_obs = norm_team_obs[nearby_team_indices]
            team_obs = torch.cat((norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), nearby_team_obs[:self.max_team_obs_size - 1]), dim=0)
            team_obs = F.pad(team_obs, (0, 0, 0, self.max_team_obs_size - len(team_obs)))
            # Add in the current team and nearby team indicator
            extra_team_info = torch.zeros((self.max_team_obs_size, 2))
            extra_team_info[0, 0] = 1.
            extra_team_info[1:len(nearby_team_obs) + 1, 1] = 1.
            team_obs = torch.cat((team_obs, extra_team_info), dim=1)

            mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
            mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
            # mask[0, :len(relevant_team_indices) + 1] = 0.0
            mask[:len(nearby_team_obs) + 1, :len(nearby_team_obs) + 1] = 0.0
            if len(nearby_team_indices) >= self.max_team_obs_size:
                team_indices = nearby_team_indices[:self.max_team_obs_size - 1]
            else:
                team_indices = nearby_team_indices
        else:
            team_obs = F.pad(norm_team_obs[self.pop.agent_to_team[cur_team.team[0].label]].unsqueeze(0), (0, 0, 0, self.max_team_obs_size - 1))
            # Add in the current team and nearby team indicator
            extra_team_info = torch.zeros((self.max_team_obs_size, 2))
            extra_team_info[0, 0] = 1.
            team_obs = torch.cat((team_obs, extra_team_info), dim=1)

            mask = np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
            mask[np.arange(self.max_team_obs_size), np.arange(self.max_team_obs_size)] = 0.0
            team_indices = torch.tensor([])
        return team_obs, mask, team_indices
    

    def step(self, actions):
        '''
        @param - actions: a dictionary of agent: action pairs.
        '''
        # Add targets if adding new targets during a trial
        if self.do_dynamic_targets and self.time_step == self.target_add_step:
            # Add in the new targets
            self.targets = self.targets + self.extra_targs
            self.target_obs = np.concatenate((self.target_obs, self.extra_targs_obs))  
            self.completed_targets = np.concatenate((self.completed_targets, np.full(len(self.extra_targs), False)))
            
            if self.do_optimal:
                spawn = [int(torch.mean(self.cur_locs[:, 0]).item()), int(torch.mean(self.cur_locs[:, 1]).item())]
                targs = np.array(self.targets)[~self.completed_targets].tolist()
                self.task_allocator.create_optimal_order(self.agents, targs, spawn)
                self.task_allocator.target_order = np.arange(len(self.targets))
            elif self.dynamic_TA_method == 'TSP':
                self.task_allocator.update_target_order(self.target_obs, spawn_point=self.spawn_point, completed_targs=self.completed_targets)
            else:
                self.task_allocator.update_target_order(self.target_obs, completed_targs=self.completed_targets)
            
            self.ordered_targets_obs = self.target_obs[self.task_allocator.target_order]
            for tm in self.pop.population:
                tm.reset_TA()
        if self.do_tsp_policy:
            actions = get_actions(self.pop, self.cur_dist_mat, self.targets, do_optimal=self.do_optimal)
        
        observations = {}
        rewards = {a:0 for a in self.agents}    
        info = {a:{} for a in self.agents}
        self.team_split_pairs = {}
        self.targ_rew_dict = {}

        # Update previous distances before moving
        self.prev_dist_mat = self.cur_dist_mat
        self.prev_locs = copy.deepcopy(self.cur_locs)
        
        if not self.use_normal_action_space:
            for a, acts in actions.items():
                # If we are training, the actions are in order
                if not self.do_tsp_policy:
                    real_actions = self.convert_action(acts)
                else:
                    real_actions = actions[a]
                team = self.pop.get_team(a)
                prev_team = self.pop.get_prev_team(a)
                
                # Below is taking an action at a team level
                self.take_action(team, real_actions, a == team.team[0], a.label == prev_team.team[0].label, prev_team)
        else:
            for a, acts in sorted(actions.items(), key=lambda item: item[1], reverse=True):
                team = self.pop.get_team(a)
                prev_team = self.pop.get_prev_team(a)            
                # Below is taking an action at a team level
                self.take_action_normal_space(team, acts, a == team.team[0], a.label == prev_team.team[0].label)

        # Now that locations have changed, update current distance matrix
        self.cur_dist_mat = torch.cdist(self.cur_locs, self.cur_locs)

        # After performing all the actions, update target info
        team_to_sensed_targs = {}
        for a in self.agents:
            team = self.pop.get_team(a)
            # If the team hasn't been added, add it
            if self.targ_rew_dict.get(team, None) == None:
                self.targ_rew_dict[team] = 0.
            if team_to_sensed_targs.get(team, None) is None:
                self.update_targets(team, team_to_sensed_targs)
        
        # Terminated (All targets sensed OR agent fails)
        # The docs on PettingZoo say that agent failure should be in terminated (https://pettingzoo.farama.org/content/basic_usage/#variable-numbers-of-agents-death)
        terminated = {a:a.failed for a in self.agents}
        # If all of the targets are completed
        if np.all(np.array([t.completed for t in self.targets])):
            terminated = {a: True for a in self.agents}

        # Truncated (episode length reached)
        # If the episode is over, all agents have been truncated
        truncated = {a:False for a in self.agents}
        if self.time_step >= self.num_eps:
            truncated = {a: True for a in self.agents}

        # Update act_mask, shared_team_space, and agent_to_team all after stepping the agents
        # Note, the reward must be calculated after all agent actions have taken place
        team_mask = torch.where(self.cur_dist_mat <= self.sensing_radius, 0.0, -1e9)
        # ignore the dummy agents
        team_mask[:, len(self.agents):] = -1e9
        norm_team_obs = self.pop.get_normalized_team_obs()
        for a in self.agents:
            team = self.pop.get_team(a)
            prev_team = self.pop.get_prev_team(a)
            
            if not self.do_normal_state:
                tgt_obs, tgt_mask, comp_targs, next_x_reqs = self.get_relevant_targets(team)
                tm_obs, tm_mask, tm_indices = self.get_relevant_teams(team, next_x_reqs, norm_team_obs)
            else:
                tgt_obs, tgt_mask, comp_targs, next_x_reqs = self.get_targ_states(team)
                tm_obs, tm_mask, tm_indices = self.get_team_states(team, norm_team_obs)

            observations[a] = {"observation": [len(team.team), self.pop.agent_to_team[a.label], team.team[0].label]}
            if not self.use_normal_action_space:
                observations[a].update({"action_mask":self.action_masking(a, actions[a], tm_obs, next_x_reqs, tm_indices)})
            else:
                observations[a].update({"action_mask":self.action_masking_normal_space(a, actions[a], tm_obs, next_x_reqs, tm_indices)})

            observations[a].update({'team_obs':tm_obs})
            observations[a].update({"target_obs": np.concatenate((tgt_obs, np.expand_dims(comp_targs, axis=1)), axis=1)})
            observations[a].update({'target_mask': tgt_mask})
            observations[a].update({'team_mask': tm_mask})
            
            # Team based Reward based on team caps and target reqs
            rewards[a] = self.calculate_team_reward(team)

        # Update the population's previous information
        self.pop.update()
        
        # Now "step" for dummy agents
        dum_team_mask = torch.full((self.max_team_obs_size, self.max_team_obs_size), -1e9)
        dum_team_mask[len(self.agents):, -1] = 0.0
        for a in self.pop.dummy_agents:
            observations[a] = {"observation": [0, -1, a.label]}
            if not self.use_normal_action_space:
                observations[a].update({"action_mask": np.full((self.max_team_obs_size, self.max_agents), self.num_caps)})
            else:
                observations[a].update({"action_mask":self.dummy_action_mask})
                
            observations[a].update({'team_obs': np.zeros((self.max_team_obs_size, self.num_caps + 4))})
            observations[a].update({'target_obs': np.zeros((self.max_targ_obs_size, self.num_caps + 5))})
            observations[a].update({'team_mask': dum_team_mask})
            observations[a].update({'target_mask': np.concatenate((np.full(self.max_targ_obs_size, False), [True]))})

            info[a] = {}
            terminated[a] = True
            truncated[a] = False
            if self.eval:
                terminated[a] = terminated[self.agents[0]]
                truncated[a] = truncated[self.agents[0]]

        self.time_step += 1
        return observations, rewards, terminated, truncated, info
            
    def render(self):
        grid = np.full((self.rows, self.cols), " ")
        for t in self.targets:
            grid[t.y, t.x] = f'T{t.label, t.reqs}'
        for a in self.agents:
            team = self.pop.get_team(a)
            grid[team.pos[1], team.pos[0]] = f'{a.label}'
        print(f"{grid} \n")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        t = [self.num_caps] * self.num_caps
        t.append(1)
        t.insert(0, self.rows)
        t.insert(0, self.cols)
        t.append(1)
        t.append(1)
        if not self.use_normal_action_space:
            return spaces.Dict({
                                # Agent label, current team, and agent to team head
                                'observation': spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.max_agents, self.max_agents, self.max_agents])),
                                'action_mask': spaces.Box(low=np.zeros((self.max_team_obs_size, self.max_agents)), high=np.full((self.max_team_obs_size, self.max_agents), self.num_caps)),
                                'team_obs': spaces.Box(low=np.full((self.max_team_obs_size, self.num_caps + 4), [0] * (self.num_caps + 4)), high=np.full((self.max_team_obs_size, self.num_caps + 4), [self.cols, self.rows] + [self.num_caps - 1] * self.num_caps + [1, 1])),
                                'target_obs': spaces.Box(low=np.zeros((self.max_targ_obs_size, 5 + self.num_caps), dtype=float), high=np.full((self.max_targ_obs_size, 5 + self.num_caps), t)),
                                'team_mask': spaces.Box(low=np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9), high=np.zeros((self.max_team_obs_size, self.max_team_obs_size))),
                                'target_mask': spaces.Box(low=np.zeros(self.max_targ_obs_size + 1), high=np.ones(self.max_targ_obs_size + 1)),
                            })
        else:
            return spaces.Dict({
                            # Below is being used for split pairs
                            # Agent label, current team, and agent to team head
                            'observation': spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.max_agents, self.max_agents, self.max_agents])),
                            'action_mask': spaces.Box(low=-1e9, high=0.0, shape=(self.num_actions,), dtype=np.float32),
                            # team_obs keeps track of nearby team_caps and team_rel
                            'team_obs': spaces.Box(low=np.full((self.max_team_obs_size, self.num_caps + 4), [0] * (self.num_caps + 4)), high=np.full((self.max_team_obs_size, self.num_caps + 4), [self.cols, self.rows] + [self.num_caps - 1] * self.num_caps + [1, 1])),
                            # Target states: location - x, y - (size 2), requirements (size num_caps)
                            'target_obs': spaces.Box(low=np.zeros((self.max_targ_obs_size, 5 + self.num_caps), dtype=float), high=np.full((self.max_targ_obs_size, 5 + self.num_caps), t)),
                            'team_mask': spaces.Box(low=np.full((self.max_team_obs_size, self.max_team_obs_size), -1e9), high=np.zeros((self.max_team_obs_size, self.max_team_obs_size))),
                            'target_mask': spaces.Box(low=np.zeros(self.max_targ_obs_size + 1), high=np.ones(self.max_targ_obs_size + 1))
                           })


    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.use_normal_action_space:
            return spaces.Discrete(2**self.num_caps + 2**self.num_caps - 1)
        else:
            return spaces.MultiDiscrete(np.full(self.max_agents, 2))
    
