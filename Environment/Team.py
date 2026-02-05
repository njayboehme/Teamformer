import numpy as np
from Environment.Agent import Agent

# This class keeps track of the agents in a team, their order, total capabilities, etc
class Team:
    def __init__(self, agents=[], num_caps=0, pos=[0, 0], sensing_radius=0., max_num_agents=0):
        '''
        @param agents - A list of agents to create a team out of
        @param num_caps - The total number of capabilities
        @param pos - The [x, y] position of the team
        '''
        self.max_num_agents = max_num_agents
        self.num_caps = num_caps
        self.team = agents
                
        # This holds the total number for each capability
        self.total_caps = np.zeros(num_caps)
        # This holds the order of the caps
        self.caps_ordered = np.full(self.max_num_agents, self.num_caps)
        self.agent_labels = np.array([], dtype=int)
        for i, a in enumerate(self.team): 
            self.total_caps[a.cap] += 1
            self.caps_ordered[i] = a.cap
            self.agent_labels = np.concatenate((self.agent_labels, [a.label]))

        self.pos = pos
        self.sensing_radius = sensing_radius
        # If a team can complete the target they picked, set this to the index of that target
        self.cur_assignment = -1
        # The current target the team is contributing to
        self.cur_contrib = -1
        self.motion_plan = None

    def add(self, to_add):
        '''
        @param to_add - A Team object to be added to the current team.
        '''
        # Add the team to the current team
        self.caps_ordered[len(self.team):len(self.team) + len(to_add.team)] = to_add.caps_ordered[:len(to_add.team)]
        self.team += to_add.team
        self.total_caps += to_add.total_caps
        self.agent_labels = np.concatenate((self.agent_labels, to_add.agent_labels))


    def remove(self, to_remove_index):
        '''
        @param to_remove_index - The index of the agent to remove. All agents behind this agent are removed as well.
        '''
        self.caps_ordered[to_remove_index:len(self.team)] = self.num_caps
        for a in self.team[to_remove_index:]: 
            self.total_caps[a.cap] -= 1
        self.team = self.team[0:to_remove_index]
        self.agent_labels = self.agent_labels[:to_remove_index]
    
    def get_team_obs(self):
        '''
        return team_info - A np array holding the position of the team and the total capabilities of the team
        '''
        return np.append(self.pos, self.total_caps, )
    
    def get_next_pos(self):
        if len(self.motion_plan) > 1:
            pos = self.motion_plan[0]
            self.motion_plan = self.motion_plan[1:]
            return pos
        else:
            # If a team has no more places to move, that could indicate that the target it was going to was completed by another team
            pos = self.motion_plan[0]
            return pos
    
    def reset_TA(self):
        self.cur_assignment = -1
        self.cur_contrib = -1
        self.motion_plan = None