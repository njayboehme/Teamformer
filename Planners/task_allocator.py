import numpy as np
import fast_tsp
from sklearn.metrics import pairwise_distances
from Planners.optimal_planner import do_normal_mrta_routing

class TaskAllocator:
    def __init__(self, 
                 eps=1e-8, 
                 max_run_time=0.5, 
                 world_type='grid', 
                 strict=False,
                 do_random_target_order=False) -> None:
        self.eps = eps
        self.tsp_run_time = max_run_time
        self.world_type = world_type
        self.strict = strict
        self.do_random_target_order = do_random_target_order

    def select_target(self, team, targets, completed_targets, selection_criterion='greedy'):
        if selection_criterion == 'greedy':
            return self.greedy(team, targets, completed_targets)
        elif selection_criterion == 'heuristic':
            return self.heuristic(team, targets, completed_targets)
        elif selection_criterion == 'exact':
            return self.exact_match(team, targets, completed_targets)
        elif selection_criterion == 'all_tsp':
            return self.all_tsp(team, targets, completed_targets)
        elif selection_criterion == 'mixed':
            return self.mixed(team, targets, completed_targets)
        elif selection_criterion == 'mod_tsp':
            return self.all_tsp_w_complete(team, targets, completed_targets)
        elif selection_criterion == 'optimal':
            return self.optimal(team, targets, completed_targets)
        
    def greedy(self, team, target_infos, completed_targets):
        team_caps = np.expand_dims(team.total_caps, axis=1)
        scores = target_infos[:, 2:] @ team_caps
        best_target = np.argmax(scores)
        return target_infos[best_target, :2]
    
    def heuristic(self, team, target_infos, completed_targets):
        '''
        Pick a target based on distance, ability to complete, and whether or not the target has been completed
        @param team - The team that is doing task allocation right now
        @param target_infos - An array of size num_targets x [x, y, num_reqs]
        @param completed_targets - A Bool array of size num_targets x 1. True indicates the target has been completed.
        @return loc - Return the location of the selected target
        '''
        team_caps = np.expand_dims(team.total_caps, axis=1)
        # dot the requirments and team_caps
        cap_scores = target_infos[:, 2:] @ team_caps
        normalize = np.expand_dims(np.sum(target_infos[:, 2:], axis=1), axis=1)
        normalize[normalize==0] = 1
        cap_scores = cap_scores / (normalize)
        team_loc = np.tile(team.pos, (target_infos.shape[0], 1))
        # get distances from the team to all of the targets
        dists = np.linalg.norm(team_loc - target_infos[:, :2], axis=1).reshape(cap_scores.shape)
        scores = (1 / (dists + self.eps)) * (cap_scores)
        # Make sure to negate the completed_targets
        fin_scores = ~completed_targets * scores
        best_target = np.argmax(fin_scores)
        if (team.total_caps == target_infos[best_target, 2:]).all():
            team.cur_assignment = best_target - 1
        return target_infos[best_target, :2]

    def exact_match(self, team, target_infos, completed_targets):
        '''
        Prioritize targets that have the reqs matching the current team's capabilities
        '''
        avail_targs = target_infos[~completed_targets.squeeze(axis=1)]
        priority = avail_targs[np.all(avail_targs[:, 2:] == team.total_caps, axis=1)]
        # If the team cannot exactly complete a target, go to the next target in the order and wait
        if len(priority) == 0:
            # This means there are no targets the current team can complete
            if len(avail_targs) == 1:
                return avail_targs[0, :2]
            else:
                # Remove the dummy target
                ordered_mask = completed_targets[1:][self.target_order]
                ordered_targets = target_infos[1:][self.target_order]
                filtered_targs = ordered_targets[~ordered_mask[:, 0]]
                pot_targs = (team.total_caps @ filtered_targs[:, 2:].T) > 0
                best_targ_index = np.argmax(pot_targs)
                to_ret = filtered_targs[best_targ_index, :2]

                return to_ret
        # Move to the closest target the team can complete
        else:
            team_loc = np.tile(team.pos, (priority.shape[0], 1))
            # get distances from the team to all of the targets
            dists = np.linalg.norm(team_loc - priority[:, :2], axis=1)
            # pick the closest one
            best_target = np.argmin(dists)
            return priority[best_target, :2]
        
    def all_tsp(self, team, target_infos, completed_targets):
        '''
        Follow the order from tsp strictly. Have agents move to the next target they can most complete in the order.
        '''
        # Order the completed mask based on current target order
        ordered_mask = completed_targets[1:][self.target_order]
        ordered_targets = target_infos[1:][self.target_order]
        og_indices = self.target_order[~ordered_mask[:, 0]]
        # Get the non-completed targets in the correct order
        filtered_targs = ordered_targets[~ordered_mask[:, 0]]
        # Find what targets the current team can complete
        pot_targs = (team.total_caps @ filtered_targs[:, 2:].T) > 0
        if pot_targs.size == 0:
            team.cur_assignment = -1
            team.cur_contrib = -1
            return target_infos[0, :2]
        # Grab the first target this team can contribute to
        best_target = np.argmax(pot_targs)
        # If none of the targets can be completed, return the dummy target
        if not pot_targs[best_target]:
            team.cur_assignment = -1
            team.cur_contrib = -1
            return target_infos[0, :2]
        team.cur_contrib = og_indices[best_target]
        return filtered_targs[best_target, :2]
    
    def all_tsp_w_complete(self, team, target_infos, completed_targets):
        '''
        Follow the order from tsp, allowing for minor deviations when training. 
        Have agents move to the next target they can most complete in the order.
        '''
        # Order the completed mask based on current target order
        ordered_mask = completed_targets[1:][self.target_order]
        ordered_targets = target_infos[1:][self.target_order]
        og_indices = self.target_order[~ordered_mask[:, 0]]
        # Get the non-completed targets in the correct order
        filtered_targs = ordered_targets[~ordered_mask[:, 0]]
        # Find what targets the current team can contribute to
        pot_targs = (team.total_caps @ filtered_targs[:, 2:].T) > 0
        # Find the targets the team can complete
        comp_targs = np.all(team.total_caps >= filtered_targs[:, 2:], axis=1)
        # If all targets have been completed
        if pot_targs.size == 0:
            team.cur_assignment = -1
            team.cur_contrib = -1
            return target_infos[0, :2]
        # Grab the first target this team can contribute to
        best_target = np.argmax(pot_targs)
        # If a team is currently at a completeable target, don't move
        same_loc = np.all(team.pos == filtered_targs[:, :2], axis=1)
        same_loc_ind = np.argmax(same_loc)
        # If none of the targets can be contributed to, return the dummy target
        if not pot_targs[best_target]:
            team.cur_assignment = -1
            team.cur_contrib = -1
            return target_infos[0, :2]
        # If we are at a completeable target, don't move
        if same_loc[same_loc_ind] and comp_targs[same_loc_ind]:
            team.cur_assignment = og_indices[same_loc_ind]
            return filtered_targs[same_loc_ind, :2]
        # If the next target to go to can be completed by this team
        elif pot_targs[best_target] == comp_targs[best_target]: 
            team.cur_assignment = og_indices[best_target]
            return filtered_targs[best_target, :2]
        # If comp_targs is empty or all False, need to return the target the current team can contribute to
        elif not np.any(comp_targs) or comp_targs.size == 0:
            team.cur_contrib = og_indices[best_target]
            team.cur_assignment = -1
            return filtered_targs[best_target, :2]
        else:
            # Use below if you don't want to enable a team to stray from the TSP order
            # team.cur_contrib = og_indices[best_target]
            # team.cur_assignment = -1
            # return filtered_targs[best_target, :2]

            # This allows a team to ignore the current order if it can complete tasks faster
            # The requirements the team doesn't fulfill for the current target
            not_pres_reqs = np.nonzero(np.clip(filtered_targs[best_target, 2:] - team.total_caps, 0, np.inf))[0]
            # Find the distances between the current team to all the comp targs
            # Then find the distances between the comp targs to the best_target
            team_to_comp_targs_dists = np.sum(np.abs(np.expand_dims(team.pos, axis=0) - filtered_targs[comp_targs, :2]), axis=1)
            comp_targs_to_best_dists = np.sum(np.abs(filtered_targs[comp_targs, :2] - np.expand_dims(filtered_targs[best_target, :2], axis=0)), axis=1)
            total_comp_dists = team_to_comp_targs_dists + comp_targs_to_best_dists
            # Pick the comp target that has the minimum distance
            comp_targ = np.argmin(total_comp_dists)

            # Now, find the targets that the other requirements of best_target need to go to
            # Only look at targets up to and including the best target in the order
            # Add 2 since target infos starts with locations
            not_pres_targs = filtered_targs[:best_target + 1, not_pres_reqs + 2]
            # Get the locations of the targets that need to be completed for each capability
            locs_to_complete = np.where(np.any(not_pres_targs[:, np.arange(len(not_pres_reqs))] > 0, axis=1, keepdims=True), 
                                        filtered_targs[:best_target + 1, :2], 
                                        np.array([-1, -1]))
            repeated_locs = np.tile(locs_to_complete, (len(not_pres_reqs), 1, 1))
            mask = (not_pres_targs > 0).T
            final_locs = repeated_locs[mask]
            fin_dists = self.get_distances(final_locs)
            # Get the distances 1 away from the main diagonal
            diag_dists = np.diagonal(fin_dists, 1)
            steps = np.sum(mask, axis=1)
            prev = 0
            max_dist = -np.inf
            # If no targets have been completed, then we need to add the distance from spawn to the first target for each missing capability
            add_spawn = ~np.any(completed_targets[1:len(self.target_order) + 1])
            to_add = 0
            for cur in steps:
                if add_spawn:
                    to_add = np.sum(np.abs(final_locs[prev] - self.spawn))
                max_dist = max(max_dist, np.sum(diag_dists[prev:cur + prev - 1]) + to_add)
                prev += cur

            # If another target can be completed before other required teams reaching best_targ, go to the target you can complete
            # Grab the first target that has not been completed
            beginning_index = np.where(self.target_order == og_indices[0])[0][0]
            end_index = np.where(self.target_order == og_indices[comp_targs][comp_targ])[0][0]
            indices = self.target_order[beginning_index:end_index + 1]
            # Calculate the cost of a section of the original cycle. Includes already completed targets (sometimes)
            og_cycle_len = self.calculate_cost(indices + 1, add_spawn)
            # If the time to complete relevant targets up to comp_targ is less than the og cycle, go to comp_targ
            if max(total_comp_dists[comp_targ], max_dist) <= og_cycle_len:
                team.cur_assignment = og_indices[comp_targs][comp_targ]
                team.cur_contrib = -1
                return filtered_targs[comp_targs][comp_targ, :2]

            # If there are no targets that can be completed before the next target needs to be done, 
            # return the target the current team can contribute to
            team.cur_contrib = og_indices[best_target]
            team.cur_assignment = -1
            return filtered_targs[best_target, :2]

    def mixed(self, team, target_infos, completed_targets):
        '''
        Move to the closest target you can complete. 
        During training, move to the closest target a team can complete exactly (if doing strict reward). 
        During execution move to the closest target you can complete. 
        If the current team cannot complete a target, go to the next target in the order
        '''
        complete_mask = ~completed_targets.squeeze(axis=1)
        avail_targs = target_infos[complete_mask]
        og_indices = np.arange(len(completed_targets) - 1)[complete_mask[1:]]
        # Find what targets the team can do. Do not include the dummy target.
        if self.strict:
            # This removes redundancy when checking if a team can complete a target
            reduced_team_total_caps = (team.total_caps > 0).astype(int)
            caps_mask = np.all(avail_targs[1:, 2:] == reduced_team_total_caps, axis=1)
            potential_targets = avail_targs[1:][caps_mask]
            og_indices = og_indices[caps_mask]
        else:
            t = team.total_caps >= avail_targs[:, 2:]
            caps_mask = np.all(team.total_caps >= avail_targs[1:, 2:], axis=1)
            potential_targets = avail_targs[1:][caps_mask]
            og_indices = og_indices[caps_mask]
        
        # If there are targets the team can complete
        if len(potential_targets) > 0:
            team_loc = np.tile(team.pos, (potential_targets.shape[0], 1))
            # get distances from the team to all of the targets
            dists = np.linalg.norm(team_loc - potential_targets[:, :2], axis=1)
            # pick the closest one
            best_target = np.argmin(dists)
            team.cur_assignment = og_indices[best_target]
            return potential_targets[best_target, :2]
        # If the team cannot complete any targets, wait at the next target it can contribute to based on the order
        else:
            ordered_mask = completed_targets[1:][self.target_order]
            ordered_targets = target_infos[1:][self.target_order]
            filtered_targs = ordered_targets[~ordered_mask[:, 0]]
            potential_targets = (team.total_caps @ filtered_targs[:, 2:].T) > 0
            # If there are potential targets
            if len(potential_targets) > 0:
                best_target = np.argmax(potential_targets)
            # If all targets are on the way to completion, do nothing
            else:
                return target_infos[0, :2]
            # If none of the targets can be contributed to, return the dummy target
            if not potential_targets[best_target]:
                return target_infos[0, :2]
            else:
                return filtered_targs[best_target, :2]
    
    def optimal(self, team, target_infos, completed_targets):
        targ_inds = self.agent_to_task[team.team[0].label]
        # True is completed
        comp_targs = ~completed_targets[1:][targ_inds].squeeze(1)
        targ_infos = target_infos[1:][targ_inds]
        avail_targs = targ_infos[comp_targs]
        if len(avail_targs) > 0:
            team.cur_assignment = -1
            team.cur_contrib = np.array(targ_inds)[comp_targs][0]
            return avail_targs[0, :2]
        else:
            team.cur_assignment = -1
            team.cur_contrib = -1
            return target_infos[0, :2]
            
    def calculate_cost(self, indices, add_spawn):
        cost = 0
        if add_spawn:
            cost += self.d[self.start_ind, indices[0]]

        for i in range(len(indices) - 1):
            cost += self.d[indices[i], indices[i + 1]]
        return cost
            
    def get_distances(self, locs):
        if self.world_type == 'grid':
            d = pairwise_distances(locs, metric='manhattan').astype(int)
        else:
            d = np.ceil(pairwise_distances(locs), dtype=int)
        return d

    def create_target_order(self, targets_locs, spawn_point, do_optimal=False):
        '''
        Compute the distance between all non-dummy targets
        @param targets - Target observations for active targets
        @param spawn_point - The point at which all teams spawn around
        @return target_order - The order of targets
        '''
        self.spawn = np.array(spawn_point)
        spawn_with_locs = np.concatenate((np.expand_dims(spawn_point, axis=0), targets_locs[:, :2]), axis=0)
        self.d = self.get_distances(spawn_with_locs)

        # If there is more than one target, find the order
        if len(targets_locs) > 1:
            order = np.array(fast_tsp.find_tour(self.d.tolist(), duration_seconds=self.tsp_run_time))
            self.tsp_steps = fast_tsp.compute_cost(order, self.d) - self.d[order[-1], order[0]]
            # Index zero is the spawn point, which we want to remove
            self.start_ind = np.where(order == 0)[0][0]
            # Remove the spawn point as a target and adjust all indices accordingly
            self.target_order = np.concatenate((order[:self.start_ind], order[self.start_ind + 1:])) - 1
            if self.do_random_target_order:
                order_rand = np.arange(len(targets_locs))
                np.random.shuffle(order_rand)
                self.target_order = order_rand
        else:
            self.tsp_steps = self.d[0, 1]
            self.target_order = np.array([0])
    
    def create_optimal_order(self, agents, targets, spawn):
        num_steps = None
        # increase the time_steps until the problem is solved
        while True:
            num_steps, self.target_order, self.agent_to_task, self.solver_status = do_normal_mrta_routing(agents, targets, spawn, num_time_steps=num_steps)
            if num_steps == 0:
                print('solution found')
                break
            num_steps += 2
    
    def update_target_order(self, all_target_info, spawn_point=None, completed_targs=None):
        if spawn_point != None:
            todo_targ_locs = all_target_info[~completed_targs, :2]

            spawn_with_locs = np.concatenate((np.expand_dims(spawn_point, axis=0), todo_targ_locs), axis=0)
            d = self.get_distances(spawn_with_locs)

            order = np.array(fast_tsp.find_tour(d.tolist(), duration_seconds=self.tsp_run_time))
            # Index zero is the spawn point, which we want to remove
            start_ind = np.where(order == 0)[0][0]
            # Remove the spawn point as a target and adjust all indices accordingly
            temp_target_order = np.concatenate((order[:start_ind], order[start_ind + 1:])) - 1
            
            # Now add in the completed targets 
            num_comp_targs = sum(completed_targs)
            completed_targ_ord = np.arange(num_comp_targs)
            temp_target_order += num_comp_targs
            self.target_order = np.zeros(len(completed_targs), dtype=int)
            self.target_order[completed_targs] = completed_targ_ord
            self.target_order[~completed_targs] = temp_target_order
        else:
            # Get the initial ordering
            num_og_targs = len(self.target_order)
            # Add the new targets into the order based on each nearest neighbor
            for i, new_loc in enumerate(all_target_info[num_og_targs:, :2]):
                dist_mat = pairwise_distances(np.concatenate((np.expand_dims(new_loc, axis=0), all_target_info[:num_og_targs + i, :2]), axis=0), metric='manhattan')
                # Get the label of the nearest target 
                smallest_dist_targ_label = np.argmin(dist_mat[0, 1:])
                # Find where that target is in the order
                order_index = np.where(self.target_order == smallest_dist_targ_label)[0][0]
                # Add the current target after that one...
                self.target_order = np.insert(self.target_order, order_index + 1, num_og_targs + i)
                