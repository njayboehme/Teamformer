import networkx as nx

class Motion_Planner:
    def __init__(self, world_type, **kwargs) -> None:
        if world_type == 'grid':
            self.world = nx.grid_2d_graph(kwargs.get('cols'), kwargs.get('rows'))
        
    def get_next_loc(self, cur_loc, des_loc, algo='a_star'):
        if algo == 'a_star':
            return self.a_star(cur_loc, des_loc)
    
    def get_path(self, cur_loc, des_loc, algo='a_star'):
        if algo == 'a_star':
            path = nx.astar_path(self.world, cur_loc, des_loc)
            if len(path) > 1:
                return path[1:]
            else:
                return path

    def a_star(self, cur_loc, des_loc):
        '''
        @param cur_loc - A tuple holding x and y position of team
        @param des_loc - A tuple holding x and y position of target
        @return end_loc - The final location of the team
        '''
        path = nx.astar_path(self.world, cur_loc, des_loc)
        if len(path) > 1:
            return path[1]
        else:
            return path[0]