from typing import List, Tuple
from stable_baselines3.common.distributions import Distribution

import torch as th
import torch.nn as nn
from torch.distributions import Categorical

'''
This Distribution is just an adapted MultiCategorical distribution from SB3
'''

class CustomDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int], is_multi_categorical: bool):
        super().__init__()
        self.action_dims = action_dims
        self.is_multi_categorical = is_multi_categorical
        self.agent_team_head = None
        self.actions = None
        self.actions_log_prob = None
        self.actions_entropy = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor):
        if not self.is_multi_categorical:
            self.distribution = [Categorical(logits=action_logits)]
        else:
            self.distribution = [Categorical(logits=split) for split in th.split(action_logits, list(self.action_dims), dim=1)]
        
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if not self.is_multi_categorical:
            return self.distribution[0].log_prob(actions)
        else:
            if self.actions_log_prob is None:
                # Extract each discrete action and compute log prob for their respective distributions
                return th.stack(
                [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
            ).sum(dim=1)
            else:
                p = self.actions_log_prob
                self.actions_log_prob = None
                return p.sum(dim=1)

    def entropy(self) -> th.Tensor:
        if not self.is_multi_categorical:
            return self.distribution[0].entropy()
        else:
            if self.actions_entropy is None:
                return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)
            else:
                p = self.actions_entropy
                self.actions_entropy = None
                return p.sum(dim=1)

    def sample(self) -> th.Tensor:
        t = th.stack([dist.sample() for dist in self.distribution], dim=1)
        num_envs, num_agents = self.agent_team_head.shape
        acts_reshaped = t.reshape(num_envs, num_agents, -1)
        real_acts = acts_reshaped[th.arange(num_envs).unsqueeze(1), self.agent_team_head].reshape(num_envs * num_agents, -1)
        if not self.is_multi_categorical:
            return real_acts.squeeze(-1)
        return real_acts

    def mode(self) -> th.Tensor:
        if not self.is_multi_categorical:
            return th.argmax(self.distribution[0].probs, dim=1)
        else:
            return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    
    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if not self.is_multi_categorical or self.actions is None:
            if deterministic:
                return self.mode()
            return self.sample()
        else:  
            a = self.actions
            self.actions = None      
            return a