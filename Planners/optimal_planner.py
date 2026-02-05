from ortools.sat.python import cp_model
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import manhattan_distances
import random

def extended_create_scenario(num_robots, num_caps_per_robot, num_heteros, num_tasks, num_reqs_per_task):
  robots = []
  robot_capabilities = {}
  used_caps = np.zeros(num_heteros)
  for r in range(num_robots):
    robots.append(r)
    new_caps = np.nonzero(used_caps == 0)[0].tolist()
    if len(new_caps) > 0:
      robot_capabilities[r] = random.sample(new_caps, k=num_caps_per_robot)
    else:
      robot_capabilities[r] = random.sample(range(num_heteros), k=num_caps_per_robot)
    used_caps[robot_capabilities[r]] += 1

  potential_reqs = np.nonzero(used_caps)[0].tolist()

  tasks = []
  task_requirements = {}
  steps_per_req = np.zeros(num_heteros)
  for t in range(num_tasks):
    num_reqs_cur = random.randint(1, num_reqs_per_task)
    tasks.append(t)
    task_requirements[t] = random.sample(potential_reqs, k=num_reqs_cur) # np.random.choice(num_heteros, size=num_reqs_per_task, replace=False)
    steps_per_req[task_requirements[t]] += 1

  locations = []
  for i in range(num_tasks + 1):
    locations.append(i)

  num_locations = len(locations)
  location_map = {loc: i for i, loc in enumerate(locations)}
  start_loc = location_map[locations[0]]
  task_locations = {}
  for task, task_loc in zip(tasks, locations[1:]):
    task_locations[task] = location_map[task_loc]

  rand_targ_locs = np.random.uniform(size=(num_tasks + 1, 2))
  # Should be num_targs + 1 x num_targs + 1 with zeros on diag
  distance_matrix = np.round(pairwise_distances(rand_targ_locs) * 10).astype(int).tolist()
  return robots, robot_capabilities, tasks, task_requirements, locations, num_locations, start_loc, task_locations, distance_matrix, steps_per_req


def extended_create_variables(model, robots, robot_capabilities, tasks, task_requirements, all_time_steps, num_locations):
  # Task and time variable
  z = {(t, time): model.NewBoolVar(f'z_{t}_{time}') for t in tasks for time in all_time_steps}
  # Robots to tasks with time
  y = {(r, t, time): model.NewBoolVar(f'y_{r}_{t}_{time}') for r in robots for t in tasks for time in all_time_steps}
  # x variables are created based on capabilities
  x = {} 
  for r in robots:
      for t in tasks:
          for req in task_requirements[t]:
              if req in robot_capabilities[r]:
                  for time in all_time_steps:
                      x[(r, t, req, time)] = model.NewBoolVar(f'x_{r}_{t}_{req}_{time}')

  # State-tracking and cost variables for travel ---
  robot_loc = {}
  for r in robots:
      for time in all_time_steps:
          robot_loc[(r, time)] = model.NewIntVar(0, num_locations - 1, f'loc_{r}_{time}')

  leg_costs = {}
  for r in robots:
      for time in all_time_steps:
          leg_costs[(r, time)] = model.NewIntVar(0, 1000, f'cost_{r}_{time}')
  return x, y, z, robot_loc, leg_costs


def extended_create_constraints(model, robots, tasks, task_requirements,
                                x, y, z, all_time_steps,
                                start_loc, robot_loc, task_locations,
                                num_locations, flat_distance_matrix, leg_costs):
  # Each task must be completed
  for t in tasks:
    model.AddExactlyOne(z[(t, time)] for time in all_time_steps)
  # Robots must fulfill tasks requirements
  for r in robots:
      for t in tasks:
          for time in all_time_steps:
              reqs = [x[(r, t, req, time)] for req in task_requirements[t] if (r, t, req, time) in x]
              # if the robot is assigned to the task at time t, it must fulfill at least one of the target's requirements
              if reqs:
                  model.Add(sum(reqs) > 0).OnlyEnforceIf(y[(r, t, time)])
                  model.Add(sum(reqs) == 0).OnlyEnforceIf(y[(r, t, time)].Not())
              else: model.Add(y[(r, t, time)] == 0)
  # A robot can be assigned to at most one task at a time
  for r in robots:
      for time in all_time_steps:
        model.AddAtMostOne(y[(r, t, time)] for t in tasks)
  # Make sure all tasks are completed
  for t in tasks:
      for time in all_time_steps:
          for req in task_requirements[t]:
            model.Add(sum(x.get((r, t, req, time),0) for r in robots) == z[(t, time)])

  # Update robot locations based on task assignments.
  for r in robots:
      for time in all_time_steps:
          # If robot is idle, it stays in its previous location
          is_idle = model.NewBoolVar(f'idle_{r}_{time}')
          model.Add(sum(y[(r, t, time)] for t in tasks) == 0).OnlyEnforceIf(is_idle)

          previous_loc = robot_loc[(r, time - 1)] if time > 0 else start_loc
          model.Add(robot_loc[(r, time)] == previous_loc).OnlyEnforceIf(is_idle)

          # If robot is assigned a task, its location becomes the task's location
          for t in tasks:
              model.Add(robot_loc[(r, time)] == task_locations[t]).OnlyEnforceIf(y[(r, t, time)])

  # Calculate the travel cost for each robot's leg of the journey.
  for r in robots:
      for time in all_time_steps:
          previous_loc = robot_loc[(r, time - 1)] if time > 0 else start_loc
          current_loc = robot_loc[(r, time)]
          index = model.NewIntVar(0, num_locations * num_locations - 1, f'idx_{r}_{time}')
          model.Add(index == previous_loc * num_locations + current_loc)
          model.AddElement(index, flat_distance_matrix, leg_costs[(r, time)])

def extended_create_objective(model, leg_costs):
  total_travel_cost = model.NewIntVar(0, 1000000, 'total_travel_cost')
  model.Add(total_travel_cost == sum(leg_costs.values()))
  model.Minimize(total_travel_cost)


def do_random_mrta_routing(num_robots, num_caps_per_robot, num_heteros, num_tasks, num_reqs_per_task, num_time_steps=None, max_solver_time=60.):
  """
  Solves the MRTA problem with dynamic, state-dependent travel costs.
  """
  
  robots, robot_capabilities, tasks, task_requirements, locations, num_locations, start_loc, task_locations, distance_matrix, req_freqs = extended_create_scenario(num_robots, num_caps_per_robot,
                                                                                                                                                        num_heteros, num_tasks, num_reqs_per_task)
  if num_time_steps == None:
    num_time_steps = np.max(req_freqs).astype(int)
  all_time_steps = range(num_time_steps)
  flat_distance_matrix = [dist for row in distance_matrix for dist in row]

  model = cp_model.CpModel()
  x, y, z, robot_loc, leg_costs = extended_create_variables(model, robots, robot_capabilities, tasks, task_requirements, all_time_steps, num_locations)
  extended_create_constraints(model, robots, tasks, task_requirements,
                              x, y, z, all_time_steps,
                              start_loc, robot_loc, task_locations,
                              num_locations, flat_distance_matrix, leg_costs)

  extended_create_objective(model, leg_costs)


  solver = cp_model.CpSolver()
  solver.parameters.max_time_in_seconds = max_solver_time
  solver.parameters.num_search_workers = 64
  status = solver.Solve(model)

  target_order = []
  rob_locs = {}
  # Every robot begins at the start
  for r in robots:
     rob_locs[r] = [0]

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(status)
    print(f'Total minimum travel cost: {solver.ObjectiveValue()}\n')
    print('--- Optimal Route Plan ---')
    for time in all_time_steps:
      print(f'\n## Time Step {time}:')
      for r in robots:
        prev_loc_val = solver.Value(robot_loc.get((r, time - 1), start_loc))
        curr_loc_val = solver.Value(robot_loc[(r, time)])

        # Get the last location of the robot
        t_loc = rob_locs[r][-1]
        cur_loc = solver.Value(robot_loc[(r, time)])
        # If the location right now is different than before, add it in
        if t_loc != cur_loc:
          rob_locs[r].append(cur_loc)

        if prev_loc_val != curr_loc_val:
          prev_loc_name = locations[prev_loc_val]
          curr_loc_name = locations[curr_loc_val]
          cost = solver.Value(leg_costs[(r, time)])
          print(f'  - Robot {r} travels from {prev_loc_name} to {curr_loc_name} (Cost: {cost}).')
          # Find which task this corresponds to
          for t in tasks:
            if solver.Value(y[(r, t, time)]) == 1:
              print(f'    - At {curr_loc_name}, performs Task {t}.')
              target_order.append(t)
        elif prev_loc_val == curr_loc_val:
          for t in tasks:
            if solver.Value(y[(r, t, time)]) == 1:
              print(f'    - At {curr_loc_name}, performs Task {t}.')
              target_order.append(t)
        else:
          if time == 0:
            print(f'  - Robot {r} remains at Start.')
    return 0
  else:
    print('No feasible solution found.')
    return num_time_steps + 1
  
def setup(agents, targets, spawn):
  robots = []
  robot_capabilities = {}
  for r in range(len(agents)):
    robots.append(r)
    robot_capabilities[r] = [agents[r].cap]

  tasks = []
  task_requirements = {}
  steps_per_req = np.zeros(len(targets[0].reqs))
  targ_locs = np.zeros((len(targets), 2), dtype=int)
  for t in range(len(targets)):
    tasks.append(t)
    req_inds = np.nonzero(targets[t].reqs)[0]
    task_requirements[t] = req_inds.tolist()
    targ_locs[t] = [targets[t].x, targets[t].y]
    steps_per_req[req_inds] += 1

  locations = []
  for i in range(len(targets) + 1):
    locations.append(i)

  num_locations = len(locations)
  location_map = {loc: i for i, loc in enumerate(locations)}
  start_loc = location_map[locations[0]]
  task_locations = {}
  for task, task_loc in zip(tasks, locations[1:]):
    task_locations[task] = location_map[task_loc]

  distance_matrix = manhattan_distances(np.concatenate(([spawn], targ_locs), axis=0)).astype(int)  
  return robots, robot_capabilities, tasks, task_requirements, locations, num_locations, start_loc, task_locations, distance_matrix, steps_per_req

def do_normal_mrta_routing(agents, targets, spawn, num_time_steps=None, max_solver_time=60.):
  robots, robot_capabilities, tasks, task_requirements, locations, num_locations, start_loc, task_locations, distance_matrix, req_freqs = setup(agents, targets, spawn)

  if num_time_steps == None:
    num_time_steps = np.max(req_freqs).astype(int) + 5
  all_time_steps = range(num_time_steps)
  flat_distance_matrix = [dist for row in distance_matrix for dist in row]
  model = cp_model.CpModel()
  x, y, z, robot_loc, leg_costs = extended_create_variables(model, robots, robot_capabilities, tasks, task_requirements, all_time_steps, num_locations)

  extended_create_constraints(model, robots, tasks, task_requirements,
                              x, y, z, all_time_steps,
                              start_loc, robot_loc, task_locations,
                              num_locations, flat_distance_matrix, leg_costs)

  extended_create_objective(model, leg_costs)

  solver = cp_model.CpSolver()
  solver.parameters.max_time_in_seconds = max_solver_time
  solver.parameters.num_search_workers = 64

  status = solver.Solve(model)

  target_order = []
  rob_locs = {}
  rob_task_order = {}
  # Every robot begins at the start
  for r in robots:
     rob_locs[r] = [0]
     rob_task_order[r] = []

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for time in all_time_steps:
      for r in robots:
        prev_loc_val = solver.Value(robot_loc.get((r, time - 1), start_loc))
        curr_loc_val = solver.Value(robot_loc[(r, time)])
        if prev_loc_val != curr_loc_val:
          # Find which task this corresponds to
          for t in tasks:
            if solver.Value(y[(r, t, time)]) == 1:
              target_order.append(t)
              rob_task_order[r].append(targets[t].label)
        elif prev_loc_val == curr_loc_val:
          for t in tasks:
            if solver.Value(y[(r, t, time)]) == 1:
              target_order.append(t)
              rob_task_order[r].append(targets[t].label)
    ordered_targs = np.full(len(targets), -1)
    end_val = len(targets) - 1
    for i in range(len(target_order) - 1, -1, -1):
      if ordered_targs[target_order[i]] == -1:
        ordered_targs[target_order[i]] = end_val
        end_val -= 1
      if end_val == -1:
        break
    return 0, ordered_targs.astype(int), rob_task_order, status
  else:
    print('No feasible solution found.')
    return num_time_steps + 1, None, None, status

