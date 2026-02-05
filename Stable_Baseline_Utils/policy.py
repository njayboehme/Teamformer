from typing import Callable

from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer

from Stable_Baseline_Utils.custom_distribution import CustomDistribution
from Networks.Teamformer.teamformer import Teamformer

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution

# # Let's define a custom actor-critic policy to be used in PPO
class CustomPolicy(ActorCriticPolicy):

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Callable[[float], float], *args, **kwargs):
        self.policy_func = kwargs.pop('policy_func')
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        dist_kwargs = {}
        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dist = CustomDistribution(list(action_space.nvec), True, **dist_kwargs)
        elif isinstance(action_space, spaces.Discrete):
            self.action_dist = CustomDistribution([1], False, **dist_kwargs)

    def _build_mlp_extractor(self) -> None:
        # This passes the output of the feature extractor to the custom network architecute's forward method
        # the net_arch is going to be passed into policy and can pass the necessary arguments into it
        self.mlp_extractor = self.policy_func(net_arch=self.net_arch)        
    
    def _get_action_dist_from_latent(self, latent_pi):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        if isinstance(self.action_dist, CategoricalDistribution):
            # latent_pi is the masked logits 
            action_logits, agent_team_head = latent_pi
            return self.action_dist.proba_distribution(action_logits=action_logits)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=latent_pi)
        elif isinstance(self.action_dist, CustomDistribution):
            if self.policy_func is Teamformer:
                if self.action_dist.is_multi_categorical:
                    actions, log_prob, entropy = latent_pi[0]
                    agent_team_head = latent_pi[1]
                    if agent_team_head is not None:
                        self.action_dist.actions = actions[agent_team_head.squeeze(0)] # This will force all members in a team to have the same actions during training
                        self.action_dist.actions_log_prob = log_prob[agent_team_head.squeeze(0)]
                    else:
                        self.action_dist.actions = None
                        self.action_dist.actions_log_prob = log_prob
                        self.action_dist.actions_entropy = entropy
                    return self.action_dist
                else:
                    action_logits, agent_team_head = latent_pi
                    self.action_dist.agent_team_head = agent_team_head
                    return self.action_dist.proba_distribution(action_logits=action_logits)
            else:
                action_logits, agent_team_head = latent_pi
                self.action_dist.agent_team_head = agent_team_head
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")
        