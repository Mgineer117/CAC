import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from policy.layers.building_blocks import MLP


class PPO_Actor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        activation: nn.Module = nn.Tanh(),
        is_discrete: bool = False,
    ):
        super(PPO_Actor, self).__init__()

        # Activation function to be used in the network
        self.act = activation

        # Action dimension (a_dim) and flag to indicate if the action space is discrete
        self.action_dim = a_dim
        self.is_discrete = is_discrete

        # Initialize the model: MLP that outputs actions
        self.model = MLP(
            input_dim, hidden_dim, a_dim, activation=self.act, initialization="actor"
        )
        self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        deterministic: bool = False,
    ):
        # Concatenate the state with reference states and actions
        state = torch.cat((x, xref, uref), dim=-1)

        logits = self.model(state)

        ### Shape the output as desired
        mu = logits
        logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
        std = torch.exp(logstd.expand_as(mu))
        dist = Normal(loc=mu, scale=std)

        if deterministic:
            # For deterministic actions, return the mean of the distribution
            a = mu
        else:
            a = dist.rsample()

        logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy().sum(1)

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Computes log probability of given actions under the distribution.

        Args:
            dist (torch.distributions): The distribution of actions.
            actions (torch.Tensor): The actions for which to compute the log probability.

        Returns:
            logprobs (torch.Tensor): The log probability of the actions.
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        logprobs = dist.log_prob(actions).unsqueeze(-1).sum(1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency, computes entropy of the distribution.

        Args:
            dist (torch.distributions): The distribution to compute entropy for.

        Returns:
            entropy (torch.Tensor): The entropy of the distribution.
        """
        return dist.entropy().unsqueeze(-1).sum(1)


class PPO_Critic(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        # Activation function to be used in the network
        self.act = activation

        # Initialize the model: MLP that outputs the value function (1 output)
        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        # Pass the state through the model to get the value
        value = self.model(x)
        return value
