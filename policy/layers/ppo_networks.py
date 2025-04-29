import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

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

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        x_trim: torch.Tensor,
        xref_trim: torch.Tensor,
        deterministic: bool = False,
    ):
        # Concatenate the state with reference states and actions
        state = torch.cat((x, xref, uref), dim=-1)
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits  # Mean of the action distribution
        logstd = torch.zeros_like(mu)  # Log standard deviation (no uncertainty)
        std = torch.exp(logstd)  # Standard deviation

        # If deterministic, take the mean as the action
        if deterministic:
            a = mu  # Deterministic action
            dist = None
            logprobs = torch.zeros_like(
                mu[:, 0:1]
            )  # Log probabilities for deterministic actions
            probs = torch.ones_like(
                logprobs
            )  # Probabilities are 1 for deterministic actions
            entropy = torch.zeros_like(
                logprobs
            )  # Entropy is zero for deterministic actions

        else:
            # If stochastic, use a multivariate normal distribution
            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            # Sample an action from the distribution
            a = dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)  # Log probabilities of the action
            probs = torch.exp(logprobs)  # Probabilities from the log probabilities

            entropy = dist.entropy()  # Entropy of the distribution

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

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency, computes entropy of the distribution.

        Args:
            dist (torch.distributions): The distribution to compute entropy for.

        Returns:
            entropy (torch.Tensor): The entropy of the distribution.
        """
        return dist.entropy().unsqueeze(-1)


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
