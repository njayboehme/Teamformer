import numpy as np
import copy


class Target:
    do_group_targets = False
    def __init__(self, x, y, label, reqs, num_reqs=0):
        self.x = x
        self.og_x = x
        self.y = y
        self.og_y = y
        self.label = label
        # reqs must be the same length as the number of capabilities
        self.reqs = reqs
        self.completed = False
        self.num_reqs = num_reqs

    def reset(self, bounds=None):
        '''
        @param bounds - A list holding the x_bound and y_bound of the environment
        '''
        if bounds is None:
            self.x = self.og_x
            self.y = self.og_y
        else:
            x_bound, y_bound = bounds
            self.x = np.random.randint(x_bound)
            self.y = np.random.randint(y_bound)
        self.completed = False

    @staticmethod
    def generate_targets(max_num_targets, present_caps, num_caps, bounds, is_split_env, testing, min_num_targets=1, num_reqs_per_target=None):
        '''
        @param max_num_targets - The maximum number of targets allowed in a given environment
        @param present_caps - The capabilities that are currently present in the environment
        @param num_caps - The number of capabilities
        @param bounds - A list holding the x_bound and y_bound of the environment
        @param is_split_env - A Bool indicating if the current env is a split env
        @param testing - A Bool indicating if a policy is being tested
        @param min_num_targets - The minimum number of targets to spawn
        @param num_reqs_per_target - The number of requirements to have per target when testing
        @return targs - A list of Target objects
        '''
        targs = []
        used_reqs = np.full(num_caps, -1)
        used_reqs[present_caps] += 1

        og_caps = copy.deepcopy(present_caps)

        partitions = []

        if is_split_env:
            # Force 2 or more targets
            num_targs = np.random.randint(2, max_num_targets + 1)
        else:
            num_targs = np.random.randint(1, max_num_targets + 1)
        
        if testing:
            num_targs = np.random.randint(min_num_targets, max_num_targets + 1)
            if Target.do_group_targets:
                targs, targ_obs, num_targs = Target.generate_grouped_targets(num_targs, num_reqs_per_target, bounds, present_caps, num_caps)
                return targs, targ_obs, num_targs

        targ_obs = np.zeros((max_num_targets, 2 + num_caps))
        for i in range(num_targs):
            # Get the number of requirements for the current target
            # Below makes sure all present_caps are present in the generated targets by forcing the last target to have the leftover capabilities
            if i + 1 == num_targs and not np.all(used_reqs != 0):
                reqs_indices = np.nonzero(used_reqs == 0)[0]
                num_reqs = len(reqs_indices)
            else:
                if is_split_env:
                    if len(present_caps) > 0:
                        # don't allow all caps to be used (unless the len of present caps is 1 - this is what the max is for)
                        if num_reqs_per_target is None:
                            num_reqs = np.random.randint(1, max(2, len(present_caps)))
                        else:
                            num_reqs = num_reqs_per_target
                        # Take the caps from the end of the chain
                        reqs_indices = present_caps[-num_reqs:]
                        partitions.append(reqs_indices)
                        # remove the just used caps
                        present_caps = present_caps[:-num_reqs]
                        used_reqs[reqs_indices] += 1
                    else:
                        # pick an available partition
                        cur_part = partitions[np.random.randint(0, len(partitions))]
                        # pick reqs from it
                        if num_reqs_per_target is None:
                            num_reqs = np.random.randint(1, len(cur_part) + 1)
                        else:
                            num_reqs = num_reqs_per_target
                        reqs_indices = cur_part[-num_reqs:]

                else:
                    if num_reqs_per_target is None:
                        num_reqs = np.random.randint(1, len(present_caps) + 1)
                        reqs_indices= np.random.choice(present_caps, num_reqs, replace=False)[:num_reqs]  
                    else:
                        num_reqs = num_reqs_per_target
                        if len(present_caps) < num_reqs:
                            # If no caps are present, just sample any caps
                            if len(present_caps) == 0:
                                reqs_indices = np.random.choice(og_caps, num_reqs, replace=False)[:num_reqs]
                            else:
                                mask = og_caps == np.expand_dims(present_caps, axis=1)
                                avail_caps = og_caps[np.sum(mask, axis=0) == 0]
                                reqs_indices = np.concatenate((present_caps, np.random.choice(avail_caps, num_reqs - len(present_caps), replace=False)))
                                present_caps = np.array([])
                        else:
                            reqs_indices = np.random.choice(present_caps, num_reqs, replace=False)[:num_reqs]
                            mask = present_caps == np.expand_dims(reqs_indices, axis=1)
                            present_caps = present_caps[np.sum(mask, axis=0) == 0]
            used_reqs[reqs_indices] += 1
            reqs = np.zeros(num_caps)
            reqs[reqs_indices] += 1
            x, y = np.random.randint([0,0], bounds)
            targs.append(Target(x, y, i, reqs, num_reqs=num_reqs))
            targ_obs[i] = np.append([x, y], reqs)
        
        return targs, targ_obs, num_targs
    
    @staticmethod
    def generate_grouped_targets(num_targets, num_reqs_per_group, bounds, present_caps, num_caps):
        '''
        @param num_targets - The number of targets to create
        @param num_groups - The number of groups to create where targets in that group only have num_reqs within that group (all the same set of reqs)
        @param num_reqs_per_group - The number of reqs each group should have
        '''
        assert len(present_caps) >= num_reqs_per_group
        num_groups = np.ceil(len(present_caps) / num_reqs_per_group).astype(int)
        assert num_targets >= num_groups

        targs = []
        targ_obs = np.zeros((num_targets, 2 + num_caps))
        targets_per_group = num_targets // num_groups
        remaining_targs = num_targets % num_groups
        num_targets_per_group = [targets_per_group] * num_groups
        for i in range(remaining_targs): num_targets_per_group[i] += 1
        np.random.shuffle(present_caps)
        reqs_per_group = np.array_split(present_caps, num_groups)
        cur_group = 0
        cur_group_len = num_targets_per_group[cur_group]
        cur_reqs = reqs_per_group[0]
        used_reqs = set()
        do_any = False
        for i in range(num_targets):
            if i >= cur_group_len:
                cur_group += 1
                cur_reqs = reqs_per_group[cur_group]
                cur_group_len += num_targets_per_group[cur_group]
                used_reqs = set()
                do_any = False
            
            # If all reqs in the group have been generated, just create whatever type of targets
            if do_any:
                num_reqs = np.random.randint(1, len(cur_reqs) + 1)
                reqs_indices = np.random.choice(cur_reqs, num_reqs, replace=False)
            else:
                # If we are about to be at the end of the group or are about to finish the loop, force all unused caps to be used
                if i + 1 == cur_group_len:
                    reqs_indices = cur_reqs[[req not in used_reqs for req in cur_reqs]]      
                    num_reqs = len(reqs_indices)              
                else:
                    num_reqs = num_reqs = np.random.randint(1, len(cur_reqs) + 1)
                    reqs_indices = np.random.choice(cur_reqs, num_reqs, replace=False)
                used_reqs.update(reqs_indices)
                do_any = len(used_reqs) == len(cur_reqs)

            x, y = np.random.randint([0, 0], bounds)
            reqs = np.zeros(num_caps)
            reqs[reqs_indices] += 1
            targs.append(Target(x, y, i, reqs, num_reqs=num_reqs))
            targ_obs[i] = np.append([x, y], reqs)

        return targs, targ_obs, num_targets
            