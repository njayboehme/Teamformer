from __future__ import annotations
import os
from datetime import datetime
import copy
import yaml
import argparse

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor

import torch
torch.backends.cuda.enable_flash_sdp(False) # Doesn't work by itself for whatever reason

# Mem Efficient
torch.backends.cuda.enable_mem_efficient_sdp(False) # didn't work for size 7

# Pytorch C++
torch.backends.cuda.enable_math_sdp(True) # this worked for 7, 15, etc
import numpy as np
import matplotlib.pyplot as plt


from Environment.Target import Target
from Environment.TF_environment import Environment as Test_Env

from tqdm import tqdm
import time

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def reset_prep(env1, env2, num_caps_env_2, same_ta, do_dynamic):
    '''
    Change env2 to have the same environment as env1
    '''
    env2.venv.venv.vec_envs[0].par_env.pop = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.pop)
    env2.venv.venv.vec_envs[0].par_env.pop.max_num_caps = num_caps_env_2
    for i in range(len(env2.venv.venv.vec_envs[0].par_env.pop.population)):
        env2.venv.venv.vec_envs[0].par_env.pop.population[i].total_caps = env2.venv.venv.vec_envs[0].par_env.pop.population[i].total_caps[:num_caps_env_2]
        env2.venv.venv.vec_envs[0].par_env.pop.prev_population[i].total_caps = env2.venv.venv.vec_envs[0].par_env.pop.prev_population[i].total_caps[:num_caps_env_2]
    env2.venv.venv.vec_envs[0].par_env.pop.all_team_info = torch.cat((env2.venv.venv.vec_envs[0].par_env.pop.all_team_info[:, :2], env2.venv.venv.vec_envs[0].par_env.pop.all_team_info[:, 2:num_caps_env_2+2]), dim=-1)
    
    env2.venv.venv.vec_envs[0].par_env.cur_locs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.cur_locs)
    env2.venv.venv.vec_envs[0].par_env.is_split_env = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.is_split_env)
    env2.venv.venv.vec_envs[0].par_env.targets = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.targets)
    env2.venv.venv.vec_envs[0].par_env.target_obs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.target_obs)
    env2.venv.venv.vec_envs[0].par_env.num_targs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.num_targs)
    if do_dynamic:
        env2.venv.venv.vec_envs[0].par_env.target_add_step = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.target_add_step)
        env2.venv.venv.vec_envs[0].par_env.extra_targs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.extra_targs)
        for i in range(len(env2.venv.venv.vec_envs[0].par_env.extra_targs)):
            env2.venv.venv.vec_envs[0].par_env.extra_targs[i].reqs = env2.venv.venv.vec_envs[0].par_env.extra_targs[i].reqs[:num_caps_env_2]
        env2.venv.venv.vec_envs[0].par_env.extra_targs_obs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.extra_targs_obs[:, :2 + num_caps_env_2])
        env2.venv.venv.vec_envs[0].par_env.spawn_point = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.spawn_point)

    # If the same TA method is being used, make sure the target order is the same
    if same_ta:
        env2.venv.venv.vec_envs[0].par_env.task_allocator = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.task_allocator)
        env2.venv.venv.vec_envs[0].par_env.ordered_targets_obs = copy.deepcopy(env1.venv.venv.vec_envs[0].par_env.ordered_targets_obs[:, :2 + num_caps_env_2])

    # Make sure the target reqs are the correct size
    for i in range(len(env2.venv.venv.vec_envs[0].par_env.targets)):
        env2.venv.venv.vec_envs[0].par_env.targets[i].reqs = env2.venv.venv.vec_envs[0].par_env.targets[i].reqs[:num_caps_env_2]
    env2.venv.venv.vec_envs[0].par_env.target_obs = env2.venv.venv.vec_envs[0].par_env.target_obs[:, :2 + num_caps_env_2]


def to_tensor(features):
    t = []
    for f in features:
        t.append(torch.tensor(f).to(dev))
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script for heterogeneous team formation.")
    
    # Add an optional argument (flag)
    parser.add_argument("--config_path", default="test_config.yaml", help="Path to the testing configuration YAML file")
    
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print(config)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")

    rows = config['rows']
    cols = config['cols']

    num_episodes = config['num_env_steps']

    num_agents = config['num_agents']
    tf_num_caps = config['teamformer_num_caps']
    ds_num_caps = config['deep_set_num_caps']
    gnn_num_caps = config['gnn_num_caps']
    max_num_targets = config['max_num_targets']
    min_num_targets = config['min_num_targets']

    # this is the number of reqs per group when do_group_targets is True
    num_reqs_per_target = config['num_reqs_per_target'] # None for random target requirements with do_group False
    max_team_spawn_size = config['max_team_spawn_size']
    Target.do_group_targets = False # False

    num_test_runs = config['num_test_runs']
    do_random_targets = config['do_random_targets']
    same_env = config['same_env']
    strict = False
    testing = True
    split_env_percent = 0.0
    do_ds = config['do_deep_set']
    do_gnn = config['do_gnn']
    do_optimal = config['do_optimal']
    do_tsp = config['do_tsp']
    do_normal_state = config['do_normal_state']
    ta_method_tf = 'mod_tsp'
    ta_method_ds = 'mod_tsp'
    ta_method_gnn = 'mod_tsp'
    ta_method_optimal = 'optimal'

    do_dynamic_targets = config['do_dynamic_targets'] # False
    base_num_targs = config['base_num_targets'] # 20 # None
    dynamic_TA_method = config['dynamic_TA_method'] # TSP or NN

    max_targ_obs_size = config['max_targ_obs_size']
    max_team_obs_size = config['max_team_obs_size']

    tf_num_actions = tf_num_caps + 1
    tf_env_kwargs = {'population_info':[num_agents, None, tf_num_caps, None, split_env_percent],
                    'max_num_targets':max_num_targets,
                    'do_random_targets':do_random_targets,
                    'rows':rows,
                    'cols':cols,
                    'episodes':num_episodes, 
                    'num_actions':tf_num_actions,
                    'min_join_dist':0.0,
                    'use_strict_reward': strict,
                    'testing':testing,
                    'other_num_caps':ds_num_caps if do_ds else tf_num_caps,
                    'test_min_num_targs': min_num_targets,
                    'ta_method': ta_method_tf,
                    'num_reqs_per_target': num_reqs_per_target,
                    'max_team_spawn_size': max_team_spawn_size,
                    'max_targ_obs_size': max_targ_obs_size,
                    'max_team_obs_size': max_team_obs_size,
                    'do_dynamic_targets': do_dynamic_targets,
                    'base_num_targs': base_num_targs,
                    'dynamic_TA_method': dynamic_TA_method,
                    'do_normal_state' : do_normal_state,
                    'sensing_radius': config['sensing_radius'],
                    'spawn_radius': config['spawn_radius'],
                    'join_radius': config['join_radius'],
                    'target_complete_reward': config['target_complete_reward'],
                    'alignment_scale': config['alignment_scale'],
                    'capability_scale': config['capability_scale'],
                    'timestep_penalty': config['timestep_penalty'],
                }
    
    if do_ds:
        ds_num_actions = 2**(ds_num_caps + 1)
        ds_env_kwargs = {'population_info':[num_agents, None, ds_num_caps, None, split_env_percent],
                    'targets':None,
                    'max_num_targets':max_num_targets,
                    'do_random_targets':do_random_targets,
                    'rows':rows,
                    'cols':cols,
                    'episodes':num_episodes, 
                    'num_actions':ds_num_actions,
                    'min_join_dist':0.0,
                    'use_strict_reward': strict,
                    'do_same_env':same_env,
                    'ta_method': ta_method_ds,
                    'max_targ_obs_size': max_targ_obs_size,
                    'max_team_obs_size': max_team_obs_size,
                    'do_dynamic_targets': do_dynamic_targets,
                    'base_num_targs': base_num_targs,
                    'dynamic_TA_method': dynamic_TA_method,
                    'sensing_radius': config['sensing_radius'],
                    'spawn_radius': config['spawn_radius'],
                    'join_radius': config['join_radius'],
                    'target_complete_reward': config['target_complete_reward'],
                    'alignment_scale': config['alignment_scale'],
                    'capability_scale': config['capability_scale'],
                    'timestep_penalty': config['timestep_penalty'],
                    }
    
    if do_gnn:
        gnn_num_actions = 2**(ds_num_caps + 1)
        gnn_env_kwargs = {'population_info':[num_agents, None, gnn_num_caps, None, 0.0],
                        'targets':None,
                        'max_num_targets':max_num_targets,
                        'do_random_targets':do_random_targets,
                        'rows':rows,
                        'cols':cols,
                        'episodes':num_episodes, 
                        'num_actions':gnn_num_actions,
                        'min_join_dist':0.0,
                        'eval':True,
                        'use_strict_reward': False,
                        'do_same_env': same_env,
                        'ta_method': ta_method_gnn,
                        'max_targ_obs_size': max_targ_obs_size,
                        'max_team_obs_size': max_team_obs_size,
                        'use_normal_action_space': True,
                        'do_dynamic_targets': do_dynamic_targets,
                        'base_num_targs': base_num_targs,
                        'dynamic_TA_method': dynamic_TA_method,
                        'sensing_radius': config['sensing_radius'],
                    'spawn_radius': config['spawn_radius'],
                    'join_radius': config['join_radius'],
                    'target_complete_reward': config['target_complete_reward'],
                    'alignment_scale': config['alignment_scale'],
                    'capability_scale': config['capability_scale'],
                    'timestep_penalty': config['timestep_penalty'],
                    }

    # autoregressive
    if do_random_targets:
        tf_env = Test_Env(**tf_env_kwargs)
    tf_env = ss.pettingzoo_env_to_vec_env_v1(tf_env, black_death=True)
    tf_env = ss.concat_vec_envs_v1(tf_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    tf_env = VecMonitor(tf_env)
    tf_env.reset()

    ## TSP environment
    if do_tsp:
        if do_random_targets:
            tf_env_kwargs.update({'do_tsp_policy': True})
            tf_env_kwargs.update({'do_same_env': True})
            tsp_env = Test_Env(**tf_env_kwargs)
        tsp_env = ss.pettingzoo_env_to_vec_env_v1(tsp_env, black_death=True)
        tsp_env = ss.concat_vec_envs_v1(tsp_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
        tsp_env = VecMonitor(tsp_env)
        reset_prep(tf_env, tsp_env, tf_num_caps, True, do_dynamic_targets)
        tsp_env.reset()
    ##

    ## Optimal ordering
    if do_random_targets and do_optimal:
        optimal_env_kwargs = {'population_info':[num_agents, None, tf_num_caps, None, split_env_percent],
                    'targets':None,
                    'max_num_targets':max_num_targets,
                    'do_random_targets':do_random_targets,
                    'rows':rows,
                    'cols':cols,
                    'episodes':num_episodes, 
                    'num_actions':tf_num_actions,
                    'min_join_dist':0.0,
                    'use_strict_reward': strict,
                    'testing':testing,
                    'other_num_caps':ds_num_caps if do_ds else tf_num_caps,
                    'test_min_num_targs': min_num_targets,
                    'ta_method': ta_method_optimal,
                    'num_reqs_per_target': num_reqs_per_target,
                    'max_team_spawn_size': max_team_spawn_size,
                    'max_targ_obs_size': max_targ_obs_size,
                    'max_team_obs_size': max_team_obs_size,
                    'do_optimal': do_optimal,
                    'do_tsp_policy': True,
                    'do_same_env': True,
                    'do_dynamic_targets': do_dynamic_targets,
                    'base_num_targs': base_num_targs,
                    'dynamic_TA_method': dynamic_TA_method,
                    'sensing_radius': config['sensing_radius'],
                    'spawn_radius': config['spawn_radius'],
                    'join_radius': config['join_radius'],
                    'target_complete_reward': config['target_complete_reward'],
                    'alignment_scale': config['alignment_scale'],
                    'capability_scale': config['capability_scale'],
                    'timestep_penalty': config['timestep_penalty'],
                }
        optimal_env = Test_Env(**optimal_env_kwargs)
        optimal_env = ss.pettingzoo_env_to_vec_env_v1(optimal_env, black_death=True)
        optimal_env = ss.concat_vec_envs_v1(optimal_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
        optimal_env = VecMonitor(optimal_env)
    ##


    # Comparison env
    if do_ds:
        if do_random_targets:
            ds_env_kwargs['use_normal_action_space'] = True
            ds_env = Test_Env(**ds_env_kwargs)
        ds_env = ss.pettingzoo_env_to_vec_env_v1(ds_env, black_death=True)
        ds_env = ss.concat_vec_envs_v1(ds_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
        ds_env = VecMonitor(ds_env)
        reset_prep(tf_env, ds_env, ds_num_caps, ta_method_tf == ta_method_ds, do_dynamic_targets)
        ds_env.reset()
    
    if do_gnn:
        if do_random_targets:
            gnn_env = Test_Env(**gnn_env_kwargs)
        gnn_env = ss.pettingzoo_env_to_vec_env_v1(gnn_env, black_death=True)
        gnn_env = ss.concat_vec_envs_v1(gnn_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
        gnn_env = VecMonitor(gnn_env)
        reset_prep(tf_env, gnn_env, gnn_num_caps, ta_method_tf == ta_method_gnn, do_dynamic_targets)
        gnn_env.reset()

    tf_model = PPO.load(config['teamformer_path'])
    my_net = tf_model.policy.mlp_extractor
    my_net.reinit(rows, cols, num_agents, 32)
    my_net.eval()
    my_extractor = tf_model.policy.features_extractor


    if do_ds:
        ds_model = PPO.load(config['deep_set_path'])
        ds_net = ds_model.policy.mlp_extractor
        ds_net.eval()
        ds_extractor = ds_model.policy.features_extractor

    if do_gnn:
        gnn_model = PPO.load(config['gnn_path'])
        gnn_net = gnn_model.policy.mlp_extractor
        gnn_net.eval()
        gnn_extractor = gnn_model.policy.features_extractor

    # Do the experiment runs
    tf_eps_len = np.array([])
    tf_mission_fails = 0
    tf_trial_time = np.array([])

    ds_eps_len = np.array([])
    ds_mission_fails = 0
    ds_trial_time = np.array([])

    tsp_eps_len = np.array([])
    tsp_mission_fails = 0
    tsp_trial_time = np.array([])

    gnn_eps_len = np.array([])
    gnn_mission_fails = 0
    gnn_trial_time = np.array([])

    optimal_eps_len = np.array([])
    optimal_mission_fails = 0
    optimal_trial_time = np.array([])

    tsp_steps = np.array([])
    tf_feat = tf_env.reset()
    for i in tqdm(range(num_test_runs)):
        teamformer_run_time = 0.0
        # Need to make sure that both envs have the same targets and agents
        tsp_steps = np.append(tsp_steps, tf_env.venv.venv.vec_envs[0].par_env.get_tsp_steps())
        if do_tsp:
            reset_prep(tf_env, tsp_env, tf_num_caps, True, do_dynamic_targets)
            tsp_env.reset()
            tsp_run_time = 0.0
        if do_ds:
            reset_prep(tf_env, ds_env, ds_num_caps, ta_method_tf == ta_method_ds, do_dynamic_targets)
            comp_feat = ds_env.reset()
            comp_run_time = 0.0
        if do_gnn:
            reset_prep(tf_env, gnn_env, gnn_num_caps, ta_method_tf == ta_method_gnn, do_dynamic_targets)
            gnn_feat = gnn_env.reset()
            gnn_run_time = 0.0
        if do_optimal:
            reset_prep(tf_env, optimal_env, tf_num_caps, False, do_dynamic_targets)
            _ = optimal_env.reset()
            optimal_run_time = 0.0
        auto_dones = [False]
        comp_dones = [False]
        tsp_dones = [False]
        gnn_dones = [False]
        optimal_dones = [False]
        for j in range(num_episodes + 1):
            # Autoregressive first
            if not auto_dones[0]:
                with torch.no_grad():
                    s_t = time.time()
                    auto_actions = my_net.forward_actor(to_tensor(my_extractor(tf_feat)), testing=False)[0][0]
                    teamformer_run_time += time.time() - s_t
                tf_feat, auto_rewards, auto_dones, auto_infos = tf_env.step(auto_actions.cpu())
            # Then DeepSet
            if not comp_dones[0] and do_ds:
                with torch.no_grad():
                    s_t = time.time()
                    comp_latent = ds_net.forward_actor(to_tensor(ds_extractor(comp_feat)))
                    comp_run_time += time.time() - s_t
                comp_actions = ds_model.policy._get_action_dist_from_latent(comp_latent).get_actions(True)
                comp_feat, comp_rewards, comp_dones, comp_infos = ds_env.step(comp_actions.cpu())
            # Then gnn environment
            if not gnn_dones[0] and do_gnn:
                with torch.no_grad():
                    s_t = time.time()
                    gnn_latent = gnn_net.forward_actor(to_tensor(gnn_extractor(gnn_feat)))
                    gnn_run_time += time.time() - s_t
                gnn_actions = gnn_model.policy._get_action_dist_from_latent(gnn_latent).get_actions(True)
                gnn_feat, gnn_rewards, gnn_dones, gnn_infos = gnn_env.step(gnn_actions.cpu())
            
            # Then TSP
            if not tsp_dones[0] and do_tsp:
                s_t = time.time()
                tsp_feat, tsp_rewards, tsp_dones, tsp_infos = tsp_env.step(auto_actions.cpu())
                tsp_run_time += time.time() - s_t
            
            # Optimal if needed
            if do_optimal and not optimal_dones[0]:
                s_t = time.time()
                optimal_feat, optimal_rewards, optimal_dones, optimal_infos = optimal_env.step(auto_actions.cpu())
                optimal_run_time += time.time() - s_t

            if auto_dones[0] and (comp_dones[0] or not do_ds) and tsp_dones[0] and (gnn_dones[0] or not do_gnn) and (optimal_dones[0] or not do_optimal):
                break
                
            


        if auto_infos[0]['episode']['l'] < (num_episodes + 1):
            tf_eps_len = np.append(tf_eps_len, auto_infos[0]['episode']['l'])
            tf_trial_time = np.append(tf_trial_time, teamformer_run_time)
        else:
            tf_eps_len = np.append(tf_eps_len, np.nan)
            tf_mission_fails += 1
            tf_trial_time = np.append(tf_trial_time, np.nan)
        
        if tsp_infos[0]['episode']['l'] < (num_episodes + 1):
            tsp_eps_len = np.append(tsp_eps_len, tsp_infos[0]['episode']['l'])
            tsp_trial_time = np.append(tsp_trial_time, tsp_run_time)
        else:
            tsp_eps_len = np.append(tsp_eps_len, np.nan)
            tsp_mission_fails += 1
            tsp_trial_time = np.append(tsp_trial_time, np.nan)
            
        if do_ds:
            if comp_infos[0]['episode']['l'] < (num_episodes + 1):
                ds_eps_len = np.append(ds_eps_len, comp_infos[0]['episode']['l'])
                ds_trial_time = np.append(ds_trial_time, comp_run_time)
            else:
                ds_eps_len = np.append(ds_eps_len, np.nan)
                ds_mission_fails += 1
                ds_trial_time = np.append(ds_trial_time, np.nan)
        
        if do_gnn:
            if gnn_infos[0]['episode']['l'] < (num_episodes + 1):
                gnn_eps_len = np.append(gnn_eps_len, gnn_infos[0]['episode']['l'])
                gnn_trial_time = np.append(gnn_trial_time, gnn_run_time)
            else:
                gnn_eps_len = np.append(gnn_eps_len, np.nan)
                gnn_mission_fails += 1
                gnn_trial_time = np.append(gnn_trial_time, np.nan)
        
        if do_optimal:
            if optimal_infos[0]['episode']['l'] < (num_episodes + 1):
                optimal_eps_len = np.append(optimal_eps_len, optimal_infos[0]['episode']['l'])
                optimal_trial_time = np.append(optimal_trial_time, optimal_run_time)
            else:
                optimal_eps_len = np.append(optimal_eps_len, np.nan)
                optimal_mission_fails += 1
                optimal_trial_time = np.append(optimal_trial_time, np.nan)
    

    m_d = datetime.now().strftime("%m_%d")
    h_m = datetime.now().strftime("%H:%M")
    script_dir = os.path.dirname(__file__)
    if do_ds:
        results_dir = os.path.join(script_dir, f'Results/{m_d}/Caps:{tf_num_caps},{ds_num_caps}/ags:{num_agents}, targs:{max_num_targets}/reqs_per_target:{num_reqs_per_target}, max_team_size:{max_team_spawn_size}/rows:{rows}, cols:{cols}/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        np.save(results_dir + f'raw_auto_eps_len_{h_m}', tf_eps_len)
        np.save(results_dir + f'raw_auto_fails_{tf_mission_fails}_{h_m}', tf_mission_fails)
        np.save(results_dir + f'raw_auto_run_time_{h_m}', tf_trial_time)

        np.save(results_dir + f'raw_comp_eps_len_{h_m}', ds_eps_len)
        np.save(results_dir + f'raw_comp_fails_{ds_mission_fails}_{h_m}', ds_mission_fails)
        np.save(results_dir + f'raw_comp_run_time_{h_m}', ds_trial_time)
        np.save(results_dir + f'raw_gnn_eps_len_{h_m}', gnn_eps_len)
        np.save(results_dir + f'raw_gnn_fails_{gnn_mission_fails}_{h_m}', gnn_mission_fails)
        np.save(results_dir + f'raw_gnn_run_time_{h_m}', gnn_trial_time)
        np.save(results_dir + f'raw_tsp_policy_eps_len_{h_m}', tsp_eps_len)
        np.save(results_dir + f'raw_tsp_policy_fails_{tsp_mission_fails}_{h_m}', tsp_mission_fails)
        np.save(results_dir + f'raw_tsp_policy_run_time_{h_m}', tsp_trial_time)

    else:
        results_dir = os.path.join(script_dir, f'Results/{m_d}/Caps:{tf_num_caps}/ags:{num_agents}, targs:{max_num_targets}/reqs_per_target:{num_reqs_per_target}, max_team_size:{max_team_spawn_size}/rows:{rows}, cols:{cols}/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        np.save(results_dir + f'raw_auto_eps_len_{h_m}', tf_eps_len)
        np.save(results_dir + f'raw_auto_fails_{tf_mission_fails}_{h_m}', tf_mission_fails)
        np.save(results_dir + f'raw_auto_run_time_{h_m}', tf_trial_time)
        np.save(results_dir + f'raw_tsp_policy_eps_len_{h_m}', tsp_eps_len)
        np.save(results_dir + f'raw_tsp_policy_fails_{tsp_mission_fails}_{h_m}', tsp_mission_fails)
        np.save(results_dir + f'raw_tsp_policy_run_time_{h_m}', tsp_trial_time)
    
    if do_optimal:
        np.save(results_dir + f'raw_optimal_eps_len_{h_m}', optimal_eps_len)
        np.save(results_dir + f'raw_optimal_run_time_{h_m}', optimal_trial_time)
    

    if not do_ds and not do_gnn and not do_optimal:
        fig_save_path = results_dir + f'{h_m}, raw, runs:{num_test_runs}, TA:{ta_method_tf}, grouped_targs:{Target.do_group_targets}.pdf'
        categories = [f'Teamformer:{tf_num_caps}', 'Heuristic']
        colors = [[0.12156863, 0.47058824, 0.70588235],
                    [1.,         0.49803922, 0.        ],]

        tf_mean = np.nanmean(tf_eps_len)
        
        tf_std = np.nanstd(tf_eps_len) / np.sqrt(num_test_runs - tf_mission_fails)

        tsp_mean = np.nanmean(tsp_eps_len)
        tsp_std = np.nanstd(tsp_eps_len) / np.sqrt(num_test_runs - tsp_mission_fails)
        
        plt.errorbar(categories, [tf_mean, tsp_mean], yerr=[tf_std, tsp_std], fmt='none', capsize=12, color='black')
        plt.bar(categories, [tf_mean, tsp_mean], yerr=[tf_std, tsp_std], color=colors)
        print('Teamformer Fails: ', tf_mission_fails)
        print('TSP Fails: ', tsp_mission_fails)
    elif do_optimal and not do_ds:
        fig_save_path = results_dir + f'{h_m}, raw, runs:{num_test_runs}, TA:{ta_method_tf}, grouped_targs:{Target.do_group_targets}.pdf'
        categories = [f'Teamformer:{tf_num_caps}', 'Heuristic', 'Optimal']
        colors = [[0.12156863, 0.47058824, 0.70588235],
                    [1.,         0.49803922, 0.        ],
                    [0.41568627, 0.23921569, 0.60392157]]

        tf_mean = np.nanmean(tf_eps_len)
        tf_std = np.nanstd(tf_eps_len) / np.sqrt(num_test_runs - tf_mission_fails)

        tsp_mean = np.nanmean(tsp_eps_len)
        tsp_std = np.nanstd(tsp_eps_len) / np.sqrt(num_test_runs - tsp_mission_fails)

        optimal_mean = np.nanmean(optimal_eps_len)
        optimal_std = np.nanstd(optimal_eps_len) / np.sqrt(num_test_runs - optimal_mission_fails)

        plt.errorbar(categories, [tf_mean, tsp_mean, optimal_mean], yerr=[tf_std, tsp_std, optimal_std], fmt='none', capsize=12, color='black')
        plt.bar(categories, [tf_mean, tsp_mean, optimal_mean], yerr=[tf_std, tsp_std, optimal_std], color=colors)
        print('Teamformer Fails: ', tf_mission_fails)
        print('TSP Fails: ', tsp_mission_fails)
        print('Optimal Fails: ', optimal_mission_fails)
    elif do_optimal and do_ds and do_gnn:
        fig_save_path = results_dir + f'{h_m}, raw, runs:{num_test_runs}, TA:{ta_method_tf, ta_method_ds}, grouped_targs:{Target.do_group_targets}.pdf'
        categories = [f'Teamformer:{tf_num_caps}', f'Deep Set:{ds_num_caps}', f'GNN:{gnn_num_caps}', 'Heuristic', 'Optimal']
        colors = [[0.12156863, 0.47058824, 0.70588235],
                    [0.89019608, 0.10196078, 0.10980392],
                    [0.2,        0.62745098, 0.17254902],
                    [1.,         0.49803922, 0.        ],
                    [0.41568627, 0.23921569, 0.60392157]]

        tf_mean = np.nanmean(tf_eps_len)
        tf_std = np.nanstd(tf_eps_len) / np.sqrt(num_test_runs - tf_mission_fails)

        ds_mean = np.nanmean(ds_eps_len)
        ds_std = np.nanstd(ds_eps_len) / np.sqrt(num_test_runs - ds_mission_fails)

        gnn_mean = np.nanmean(gnn_eps_len)
        gnn_std = np.nanstd(gnn_eps_len) / np.sqrt(num_test_runs - gnn_mission_fails)

        tsp_mean = np.nanmean(tsp_eps_len)
        tsp_std = np.nanstd(tsp_eps_len) / np.sqrt(num_test_runs - tsp_mission_fails)

        optimal_mean = np.nanmean(optimal_eps_len)
        optimal_std = np.nanstd(optimal_eps_len) / np.sqrt(num_test_runs - optimal_mission_fails)

        plt.errorbar(categories, [tf_mean, ds_mean, gnn_mean, tsp_mean, optimal_mean], yerr=[tf_std, ds_std, gnn_std, tsp_std, optimal_std], fmt='none', capsize=12, color='black')
        plt.bar(categories, [tf_mean, ds_mean, gnn_mean, tsp_mean, optimal_mean], yerr=[tf_std, ds_std, gnn_std, tsp_std, optimal_std], color=colors)
        print('Teamformer Fails: ', tf_mission_fails)
        print('Teamformer Avg Run time: ', np.nanmean(tf_trial_time))

        print('Deep Set Fails: ', ds_mission_fails)
        print('Deep Set Avg Run time: ', np.nanmean(ds_trial_time))

        print('GNN Fails: ', gnn_mission_fails)
        print('GNN Avg Run time: ', np.nanmean(gnn_trial_time))

        print('TSP Fails: ', tsp_mission_fails)
        print('TSP Avg Run time: ', np.nanmean(tsp_trial_time))

        print('Optimal Fails: ', optimal_mission_fails)
        print('Optimal Avg Run time: ', np.nanmean(optimal_trial_time))
    else:
        fig_save_path = results_dir + f'{h_m}, raw, runs:{num_test_runs}, TA:{ta_method_tf, ta_method_ds}, grouped_targs:{Target.do_group_targets}.pdf'
        categories = [f'Teamformer:{tf_num_caps}', f'Deep Set:{ds_num_caps}', f'GNN:{gnn_num_caps}', 'Heuristic']
        colors = [[0.12156863, 0.47058824, 0.70588235],
                    [0.89019608, 0.10196078, 0.10980392],
                    [0.2,        0.62745098, 0.17254902],
                    [1.,         0.49803922, 0.        ],]
        
        tf_mean = np.nanmean(tf_eps_len)
        tf_std = np.nanstd(tf_eps_len) / np.sqrt(num_test_runs - tf_mission_fails)

        ds_mean = np.nanmean(ds_eps_len)
        ds_std = np.nanstd(ds_eps_len) / np.sqrt(num_test_runs - ds_mission_fails)

        gnn_mean = np.nanmean(gnn_eps_len)
        gnn_std = np.nanstd(gnn_eps_len) / np.sqrt(num_test_runs - gnn_mission_fails)

        tsp_mean = np.nanmean(tsp_eps_len)
        tsp_std = np.nanstd(tsp_eps_len) / np.sqrt(num_test_runs - tsp_mission_fails)

        plt.errorbar(categories, [tf_mean, ds_mean, gnn_mean, tsp_mean], yerr=[tf_std, ds_std, gnn_std, tsp_std], fmt='none', capsize=12, color='black')
        plt.bar(categories, [tf_mean, ds_mean, gnn_mean, tsp_mean], yerr=[tf_std, ds_std, gnn_std, tsp_std], color=colors)
        print('Teamformer Fails: ', tf_mission_fails)
        print('Deep Set Fails: ', ds_mission_fails)
        print('GNN Fails: ', gnn_mission_fails)
        print('TSP Fails: ', tsp_mission_fails)
    
    print("OG TSP avg steps: ", np.average(tsp_steps))
    plt.ylabel('# Environment Steps', fontsize=16)
    plt.title(f'Robots: {num_agents}, Tasks: {max_num_targets}', fontsize=16)
    plt.savefig(fig_save_path, format='pdf')
    plt.show()
    z = ''
