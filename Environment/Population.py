import logging
import copy
import numpy as np
import torch
from Environment.Team import Team
from Environment.Agent import Agent

'''
This class keeps track of all the teams and the agents within them
'''
class Population:
    def __init__(self, num_agents=0, 
                 capability_indices=[], 
                 max_num_caps=0, 
                 init_pos=None, 
                 rows=0, 
                 cols=0, 
                 split_percent=0., 
                 testing=False, 
                 other_num_caps=None, 
                 max_team_spawn_size=None,
                 sensing_radius=1.,
                 spawn_radius=1.):
        '''
        @param num_agents - The total number of agents that are going to be considered
        @param capability_indices - The indices of the individual agent capabilities. The len of this must be the same as num_agents or must be None.
        @param init_pos - The initial positions of each agent. Should be an array like: [(a1_x, a1_y), (a2_x, a2_y), ...]
        @param rels - A list of the agent reliabilities. Must be the same length as num_agents
        '''
        self.population : list[Team] = []
        self.num_agents = num_agents
        self.max_num_agents = num_agents
        self.init_pos = init_pos
        self.caps_indices = capability_indices
        self.max_num_caps = max_num_caps
        self.rows = rows
        self.cols = cols
        self.split_percent = split_percent
        self.testing = testing
        self.other_num_caps = other_num_caps
        self.max_team_spawn_size = max_team_spawn_size
        self.sensing_radius = sensing_radius
        self.spawn_radius = spawn_radius

        # Keeps track of where an agent is in its team
        self.agent_team_pos = torch.zeros(self.max_num_agents, dtype=torch.int)
        # Keeps track of what agent is in what team
        self.agent_to_team = torch.zeros(self.max_num_agents, dtype=torch.int)     
        
        self.all_team_info = torch.zeros(self.max_num_agents, 2 + self.max_num_caps)

        self.reset()

    def combine(self, team1_ind, team2_ind):
        '''
        @param team1_ind - The index of the team that team2 is being added to. Team1 will be the new head.
        @param team2_ind - The index of the team that is being added to team1. Team2 will be appended to team1 and then removed from the population.
        '''
        prev_team_len = len(self.population[team1_ind].team)
        # Add team2 to team1
        self.population[team1_ind].add(self.population[team2_ind])
        # Update team1 with the new caps
        self.all_team_info[team1_ind] = torch.tensor(self.population[team1_ind].get_team_obs())
        # Update agent_to_team
        update_mask_team_2 = self.agent_to_team == team2_ind
        self.agent_to_team[update_mask_team_2] = team1_ind
        dec_mask = self.agent_to_team > team2_ind
        self.agent_to_team[dec_mask] -= 1

        # I just want to update the positions of the newly added team based on how long the team was before adding the new team
        self.agent_team_pos[update_mask_team_2] += prev_team_len

        # Remove team2 from the population
        self.population.pop(team2_ind)
        # Update all_team_info by removing the old team and padding to self.num_agents
        self.all_team_info = torch.cat((self.all_team_info[:team2_ind], self.all_team_info[team2_ind + 1:], torch.zeros(1, 2 + self.max_num_caps)))

    def split(self, team_ind, break_index):
        '''
        @param team_ind - The index for .population that retrieves the team we want to split
        @param break_index - The index in the desired team where the split will occur
        '''
        try:
            team = self.population[team_ind]
            # This is the part of the chain splitting from team
            leftover = team.team[break_index:]
            # Remove leftover from team
            team.remove(break_index)
            self.all_team_info[team_ind] = torch.tensor(self.population[team_ind].get_team_obs())
            # Add leftover back into the population
            t = Team(leftover, self.max_num_caps, pos=copy.deepcopy(team.pos), sensing_radius=self.sensing_radius, max_num_agents=self.max_num_agents)
            for a in t.team:
                self.agent_to_team[a.label] = len(self.population)
                # Subtract the length of the team it was just removed from to get the current position
                self.agent_team_pos[a.label] -= len(team.team)

            # Add the new team info to the correct spot
            self.all_team_info[len(self.population)] = torch.tensor(t.get_team_obs())
            self.population.append(t)
        except IndexError:
            logging.error(f"Tried splitting team {self.population[team_ind]} of length {len(self.population[team_ind].team)} at index {break_index}")


    def get_team(self, agent: Agent) -> Team:
        return self.population[self.agent_to_team[agent.label]]

    def get_prev_team(self, agent: Agent) -> Team:
        return self.prev_population[self.prev_agent_to_team[agent.label]]
    
    def update_all_team_info(self, ind, cur_loc):
        '''
        @param ind - The index of the team to update
        @param cur_loc - The location of the team [x, y]
        @return - all_team_info with the correct location of the team
        '''
        self.all_team_info[ind, :2] = torch.tensor(cur_loc)
    
    def get_normalized_team_obs(self):
        return torch.cat((self.all_team_info[:, :2] / torch.tensor([self.cols - 1, self.rows - 1]), self.all_team_info[:, 2:]), dim=1)
    
    def update(self):
        '''
        This should be called after the step function to bring the prev members to the current time step
        '''
        self.prev_population = copy.deepcopy(self.population)
        self.prev_agent_to_team = copy.deepcopy(self.agent_to_team)

    def get_agents(self):
        '''
        Get the agents within a team
        '''
        agents = []
        for tm in self.population:
            agents += tm.team
        return agents

    def reset(self, num_agents=None, team_sizes=None, team_locs=None, team_caps=None):
        '''
        Reset all teams and agents within the population. Can do a random reset or a reset to a predefined scenario.
        '''
        self.population = []
        self.dummy_agents = []
        shared_space = {}
        cur_locs = torch.zeros(self.max_num_agents, 2)

        # On a reset, since the number of agents and teams can change, I need to change things like all_team_info, agent_team_pos, etc
        self.agent_team_pos = torch.zeros(self.max_num_agents, dtype=torch.int)
        self.agent_to_team = torch.zeros(self.max_num_agents, dtype=torch.int)
        self.all_team_info = torch.zeros(self.max_num_agents, 2 + self.max_num_caps)
    
        if num_agents is None:
            num_agents, shared_space, cur_locs, used_cap_indices, is_split_env, spawn_point = self.random_reset()
        else:
            shared_space, cur_locs, used_cap_indices, is_split_env, spawn_point = self.eval_reset(num_agents, team_sizes, team_locs, team_caps)

        if num_agents < self.max_num_agents:
            self.dummy_agents = [Agent(i + num_agents) for i in range(self.max_num_agents - num_agents)]
        
        self.update()
        return shared_space, cur_locs, used_cap_indices, is_split_env, spawn_point

    def random_reset(self):
        '''
        Generate random teams and capabilities. Can create environments that are better for taking the split action if desired.
        '''
        shared_space = {}
        cur_locs = torch.zeros(self.max_num_agents, 2)
        is_split_env = False

        # Split env (i.e. an env that is better to split in)
        if np.random.uniform() < self.split_percent:
            # Force there to be a team of size two or more
            rand_num_agents = np.random.randint(2, self.max_num_agents + 1)
            # Force the capabilities in a team to be different. Note, for this to work max_num_caps >= max_num_agents
            if self.max_num_caps >= rand_num_agents:
                rand_cap_indices = np.random.choice(self.max_num_caps, size=rand_num_agents, replace=False)
            else:
                rand_cap_indices = np.random.choice(self.max_num_caps, size=rand_num_agents, replace=True)
            is_split_env = True
            
        # Completely random
        else:
            # rand_num_agents = np.random.randint(2, self.max_num_agents + 1)
            rand_num_agents = self.max_num_agents
            if not self.testing:
                rand_cap_indices = np.random.randint(self.max_num_caps, size=rand_num_agents) # This allows redundancy
                # Force there to be different capabilities during training
                if self.max_num_caps >= rand_num_agents:
                    rand_cap_indices = np.random.choice(self.max_num_caps, size=rand_num_agents, replace=False)
                else:
                    rand_cap_indices = np.random.choice(self.max_num_caps, size=rand_num_agents)
            else:
                rand_cap_indices = np.random.randint(self.max_num_caps, size=rand_num_agents) # This allows redundancy
        
        if self.testing:
            rand_num_agents = self.max_num_agents
            # If testing, choose the minimum number of caps between the two methods
            if self.other_num_caps is not None:
                num_caps = min(self.max_num_caps, self.other_num_caps)
            else:
                num_caps = self.max_num_caps
            if num_caps >= rand_num_agents:
                rand_cap_indices = np.random.choice(num_caps, size=rand_num_agents, replace=False)
            else:
                rand_cap_indices = np.tile(np.arange(num_caps), rand_num_agents // num_caps)
                rand_cap_indices = np.append(rand_cap_indices, rand_cap_indices[:rand_num_agents - len(rand_cap_indices)])
        
        spawn_point_x, spawn_point_y = np.random.randint([0, 0], [self.cols, self.rows])
        spawn_point_lows = [max(0, spawn_point_x - self.spawn_radius), max(0, spawn_point_y - self.spawn_radius)]
        spawn_point_highs = [min(self.cols, spawn_point_x + self.spawn_radius), min(self.rows, spawn_point_y + self.spawn_radius)]

        cur_agents_left = rand_num_agents
        ctr = 0
        cur_team_index = 0
        while cur_agents_left != 0:
            if is_split_env:
                # force a team to be created with different capabilities
                cur_team_size = rand_num_agents
            else:
                if self.max_team_spawn_size is None:
                    # generate a random team size
                    cur_team_size = np.random.randint(1, cur_agents_left + 1)
                else:
                    cur_team_size = self.max_team_spawn_size
            agents = [Agent(ctr + i, cap=rand_cap_indices[ctr + i]) for i in range(cur_team_size)]
            x, y = np.random.randint(spawn_point_lows, spawn_point_highs)
            t = Team(agents, num_caps=self.max_num_caps, pos=[x, y], sensing_radius=self.sensing_radius, max_num_agents=self.max_num_agents)


            cur_locs[ctr: ctr + cur_team_size] = torch.tensor([x, y])
            # The individual agent information never changes
            self.all_team_info[cur_team_index] = torch.tensor(t.get_team_obs())
            # Colocation
            if (x, y) in shared_space.keys():
                shared_space[(x,y)].append(t)
            else:
                shared_space[(x, y)] = [t]
            self.agent_team_pos[ctr: ctr + cur_team_size] = torch.arange(0, cur_team_size)
            self.agent_to_team[ctr: ctr + cur_team_size] = cur_team_index
            self.population.append(t)


            ctr += cur_team_size
            cur_agents_left -= cur_team_size
            cur_team_index += 1
        return rand_num_agents, shared_space, cur_locs, np.unique(rand_cap_indices), is_split_env, [spawn_point_x, spawn_point_y]

    def eval_reset(self, num_agents, team_sizes, team_locs, team_caps):
        '''
        @param num_agents - An integer indicating the number of agents for the evaluation env
        @param team_sizes - A list with the size of each team (the sum must equal num_agents)
        @param team_locs - A list with the [x, y] of each team
        @param team_caps - A list with the capabilities of each team
        '''
        is_split_env = False
        shared_space = {}
        cur_locs = torch.zeros(self.max_num_agents, 2)
        agent_rels = np.ones(num_agents, dtype=float)
        ctr = 0
        for cur_team_index, cur_team_size in enumerate(team_sizes):
            agents = [Agent(ctr + i, cap=team_caps[ctr + i], rel=agent_rels[ctr + i]) for i in range(cur_team_size)]
            x, y = team_locs[cur_team_index]
            t = Team(agents, num_caps=self.max_num_caps, pos=[x, y], sensing_radius=self.sensing_radius, max_num_agents=self.max_num_agents)

            cur_locs[ctr: ctr + cur_team_size] = torch.tensor([x, y])
            # The individual agent information never changes
            self.all_team_info[cur_team_index] = torch.tensor(t.get_team_obs())
            # Colocation
            if (x, y) in shared_space.keys():
                shared_space[(x,y)].append(t)
            else:
                shared_space[(x, y)] = [t]
            self.agent_team_pos[ctr: ctr + cur_team_size] = torch.arange(0, cur_team_size)
            self.agent_to_team[ctr: ctr + cur_team_size] = cur_team_index
            self.population.append(t)

            ctr += cur_team_size
            num_agents -= cur_team_size
            
        return shared_space, cur_locs, None, is_split_env, self.population[0].pos

