import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from datetime import datetime
import logging
logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.ERROR)
import numpy as np
import supersuit as ss
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

import torch
torch.cuda.empty_cache()
torch.set_anomaly_enabled(True)
# FlashAttn
torch.backends.cuda.enable_flash_sdp(False) # Doesn't work by itself for whatever reason
# Mem Efficient
torch.backends.cuda.enable_mem_efficient_sdp(False) # didn't work for size 7
# Pytorch C++
torch.backends.cuda.enable_math_sdp(True) # this worked for 7, 15, etc

from Stable_Baseline_Utils.policy import CustomPolicy
from Networks.Teamformer.teamformer import Teamformer
from Networks.Comparisons.DeepSet.deep_set_policy import DeepSetPolicy
from Networks.Comparisons.GNN.GNN_policy import GNNPolicy
from Environment.TF_environment import Environment
from Environment.Target import Target  

class MyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
    def forward(self, observations):
        return [observations['observation'], 
                observations['action_mask'], 
                observations['team_obs'], 
                observations['target_obs'], 
                observations['team_mask'],
                observations['target_mask'],
                ]

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for heterogeneous team formation.")
    parser.add_argument("--config_path", default="train_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")

    torch.cuda.empty_cache()
    env_fn = Environment
    num_envs = config['algorithm']['num_envs']
    num_cpus = config['algorithm']['num_cpus']

    batch_size = config['algorithm']['batch_size']
    n_steps = config['algorithm']['num_steps_per_env']
    gamma = config['algorithm']['gamma']
    lr = config['algorithm']['learning_rate']
    ent_coef = config['algorithm']['entropy_coefficient']
    clip_range = config['algorithm']['clip_range']
    n_epochs= config['algorithm']['num_epochs']
    gae_lambda= config['algorithm']['gae_lambda']
    max_grad_norm= config['algorithm']['max_grad_norm']
    vf_coef = config['algorithm']['value_function_coefficient']

    trans_dim = config['policy']['d_model']
    num_enc_heads = config['policy']['num_enc_heads']
    num_enc_layers = config['policy']['num_enc_layers']
    num_dec_heads = config['policy']['num_dec_heads']
    num_dec_layers = config['policy']['num_dec_layers']
    mlp_hidden_dim = config['policy']['mlp_hidden_dim']
    mlp_num_layers = config['policy']['mlp_layers']
    policy_type = config['policy']['type']
    use_normal_action_space = False if policy_type == 'Teamformer' else True
    split_env_percent = config['environment']['split_env_percent']

    max_num_caps = config['environment']['num_capabilities']
    ta_method = config['environment']['task_allocation_method']
    # This dictates whether Teamformer will use each team's entire state space or not (it should)
    do_informed = config['policy']['do_informed']

    date = datetime.now().strftime("%m_%d")
    folder = f'{policy_type}/final_{max_num_caps}/{date}'

    rows = config['environment']['rows']
    cols = config['environment']['cols']
    
    num_episodes = config['environment']['episode_length']
    do_join_mask = False
    extra_caps_pen = -1
    
    num_agents = config['environment']['num_agents']
    max_num_targets = config['environment']['max_num_targets']

    agent_caps = [0, 1, 2, 3]
    agent_start_locations = None
    targets = [Target(4, 4, 0, reqs=np.array([1, 1]))] # If reward = 0, then gamma=1
    do_random_targets = config['environment']['do_random_targets']
    max_targ_obs_size = config['environment']['max_target_obs_size']
    max_team_obs_size = config['environment']['max_team_obs_size']
    env_kwargs = {'population_info':[num_agents, agent_caps, max_num_caps, agent_start_locations, split_env_percent],
                'targets':targets,
                'max_num_targets':max_num_targets,
                'do_random_targets':do_random_targets,
                'rows':rows,
                'cols':cols,
                'episodes':num_episodes, 
                'min_join_dist':0.0,
                'use_normal_action_space': use_normal_action_space,
                'ta_method': ta_method,
                'max_targ_obs_size': max_targ_obs_size,
                'max_team_obs_size': max_team_obs_size,
                'do_normal_state' : True,
                'sensing_radius': config['environment']['sensing_radius'],
                'spawn_radius': config['environment']['spawn_radius'],
                'join_radius': config['environment']['join_radius'],
                'target_complete_reward': config['environment']['target_complete_reward'],
                'alignment_scale': config['environment']['alignment_scale'],
                'capability_scale': config['environment']['capability_scale'],
                'timestep_penalty': config['environment']['timestep_penalty'],
                }
    
    # Training Environment
    env = env_fn(**env_kwargs)

    # Go from parallel env to a bunch of environments for a single agent
    # Black death needs to be true to allow variable number of active agents
    env = ss.pettingzoo_env_to_vec_env_v1(env, black_death=True)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    env.reset()

    # Evaluation Environment
    # Example evaluation agent information
    eval_agent_info = {
                        # delayed join
                        0:{
                            'num_agents':2,
                            'team_sizes':[1, 1],
                            'team_locs':[[0,0], [1, 0]],
                            'team_caps':[1, 2]
                        },
                        # no joining or splitting
                        1:{
                            'num_agents':3,
                            'team_sizes':[2, 1],
                            'team_locs':[[0,0], [1, 0]],
                            'team_caps':[0, 3, 0]
                        },
                    }
    # Example evaluation target information
    eval_target_info = {
                        0:{
                            'targets':[
                                        Target(x=0,y=5,label=0,reqs=np.array([0, 1, 0, 0]), num_reqs=1),
                                        Target(x=7,y=2,label=1,reqs=np.array([0, 0, 1, 0]), num_reqs=1),
                                        Target(x=8,y=9,label=2,reqs=np.array([0, 1, 1, 0]), num_reqs=2)                                       
                                    ],
                            'target_obs':np.array([[0, 5, 0, 1, 0, 0],
                                                    [7, 2, 0, 0, 1, 0],
                                                    [8, 9, 0, 1, 1, 0],
                                                    [0, 0, 0, 0, 0, 0]])
                        },
                        1:{
                            'targets':[
                                        Target(x=6, y=8, label=0, reqs=np.array([1, 0, 0, 1]), num_reqs=2),
                                        Target(x=8, y=4, label=1, reqs=np.array([1, 0, 0, 0]), num_reqs=1)
                                    ],
                            'target_obs':np.array([[6, 8, 1, 0, 0, 1],
                                                    [8, 4, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]])
                        },
                    }
    
    for eval_targ in eval_target_info:
        eval_target_info[eval_targ]['target_obs'] = np.pad(eval_target_info[eval_targ]['target_obs'], ((0,0), (0, max_num_caps - eval_target_info[eval_targ]['target_obs'].shape[1] + 2)))
        for t in eval_target_info[eval_targ]['targets']:
            t.reqs = np.pad(t.reqs, (0, max_num_caps - len(t.reqs)))
    eval_num_episodes = 30
    num_eval_agents = 4
    max_team_obs_size = 4
    eval_env_kwargs = {'population_info':[num_eval_agents, agent_caps, max_num_caps, agent_start_locations, 0.0],
                    'targets':targets,
                    'max_num_targets':max_num_targets,
                    'do_random_targets':False,
                    'rows':10,
                    'cols':10,
                    'episodes':eval_num_episodes, 
                    'use_normal_action_space': use_normal_action_space,
                    'min_join_dist':0.0,
                    'eval':True,
                    'eval_agent_info':eval_agent_info,
                    'eval_target_info':eval_target_info,
                    'use_strict_reward': False,
                    'ta_method': ta_method,
                    'max_targ_obs_size': max_targ_obs_size,
                    'max_team_obs_size': max_team_obs_size,
                    'do_normal_state': True,
                    'sensing_radius': config['environment']['sensing_radius'],
                    'spawn_radius': config['environment']['spawn_radius'],
                    'join_radius': config['environment']['join_radius'],
                    'target_complete_reward': config['environment']['target_complete_reward'],
                    'alignment_scale': config['environment']['alignment_scale'],
                    'capability_scale': config['environment']['capability_scale'],
                    'timestep_penalty': config['environment']['timestep_penalty'],
                    }

    num_eval_envs = len(eval_agent_info)
    num_evals = num_agents * num_eval_envs # should be a multiple of num_agents and should split evenly among all the agents
    eval_env = env_fn(**eval_env_kwargs)
    eval_env.reset()
    # Black death needs to be true to allow variable number of active agents
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env, black_death=True)
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    eval_env.reset()
    eval_env = VecMonitor(eval_env)
    eval_env.reset()
    # This will be called every 500 times env.step (i.e. the train env) has been called
    current_time = datetime.now().strftime("%H:%M:%S")
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=f'{folder}/{current_time}', 
                                 log_path=f'{folder}/{current_time}',
                                 n_eval_episodes=num_evals, 
                                 eval_freq=n_steps + 1, 
                                 deterministic=True, 
                                 render=False, 
                                 verbose=1)

    # Create Checkpoint Callback
    save_freq = config['algorithm']['save_frequency'] / num_agents
    checkpoint_callback = CheckpointCallback(save_freq, f'{folder}/checkpoint')

    # Model Params
    general_info = [rows, cols, max_num_caps, num_agents]
    if policy_type == 'Teamformer':
        policy_class = Teamformer
        enc_dim = trans_dim
        enc_heads = num_enc_heads
        enc_layers = num_enc_layers
        enc_info = [enc_dim, enc_heads, enc_layers]

        dec_dim = trans_dim
        dec_heads = num_dec_heads
        dec_layers = num_dec_layers
        dec_info = [dec_dim, dec_heads, dec_layers]

        mlp_hid_dim=mlp_hidden_dim
        mlp_layers= mlp_num_layers
        mlp_info = [mlp_hid_dim, mlp_layers]

        net_arch=dict(general_info=general_info,
                    enc_info=enc_info,
                    dec_info=dec_info,
                    mlp_info=mlp_info,
                    do_informed=do_informed
                    )
    elif policy_type == 'DeepSet':
        policy_class = DeepSetPolicy
        inp_dim = 2 + max_num_caps + 4 # add four for the indicators

        dec_dim = trans_dim
        dec_heads = num_dec_heads
        dec_layers = num_dec_layers
        dec_info = [dec_dim, dec_heads, dec_layers]

        mlp_hid_dim=mlp_hidden_dim
        mlp_layers= mlp_num_layers
        mlp_info = [mlp_hid_dim, mlp_layers]

        net_arch=dict(general_info=general_info,
                    inp_dim=inp_dim,
                    out_dim=2**(max_num_caps + 1),
                    mlp_info=mlp_info,
                    hidden_dim=trans_dim,
                    num_heads=num_enc_heads
                    )
    elif policy_type == 'GNN':
        policy_class = GNNPolicy
        inp_dim = 2 + max_num_caps + 4 # add four for the indicators
        hidden_dim = trans_dim # this is d_model
        num_heads = num_enc_heads
        critic_out_dim = 1

        net_arch={
            'actor': dict(inp_dim=inp_dim,
                    feature_dim=hidden_dim,
                    num_heads=num_heads,
                    out_dim=2**(max_num_caps + 1) - 1,
                    num_agents=num_agents,
                    max_num_caps=max_num_caps,
                    max_team_obs_size=max_team_obs_size,
                    max_targ_obs_size=max_targ_obs_size,
                    ),

            'critic': dict(inp_dim=inp_dim,
                    feature_dim=hidden_dim,
                    num_heads=num_heads,
                    out_dim=critic_out_dim,
                    num_agents=num_agents,
                    max_num_caps=max_num_caps,
                    max_team_obs_size=max_team_obs_size,
                    max_targ_obs_size=max_targ_obs_size,
                    ),
        }
    optimizer_class = torch.optim.Adam
    policy_kwargs = dict(features_extractor_class=MyFeatureExtractor,
                        features_extractor_kwargs=dict(features_dim=4),
                        net_arch=net_arch,
                        policy_func=policy_class,
                        optimizer_class=optimizer_class
                        )
        
    policy = CustomPolicy
    num_timesteps = config['algorithm']['total_env_steps']
    
    model = PPO(policy, 
            env, 
            verbose=2, 
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=lr, 
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f'{folder}/',
            )

    trained = model.learn(num_timesteps, callback=eval_callback, progress_bar=True)
    trained.save(f'{folder}/final')
