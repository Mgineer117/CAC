import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

from policy.layers.building_blocks import MLP


class C3M_W_Gaussian(nn.Module):
    """
    C3M_W_Gaussian generates a state-dependent matrix W(x) modeled using a Gaussian distribution.
    It leverages a neural network (MLP) to learn parameters of the Gaussian distribution
    (mean and covariance) and constructs a positive semi-definite matrix W(x) by sampling
    from the distribution and ensuring symmetry.

    Attributes:
        x_dim (int): Dimension of the state space.
        state_dim (int): Dimension of the state input.
        device (str): Computational device ('cpu' or 'cuda').
        w_lb (float): Lower bound added to the diagonal of W to ensure positive definiteness.
        model (MLP): Multi-layer perceptron (MLP) that generates parameters for the Gaussian distribution.
        mu (torch.nn.Linear): Linear layer generating the mean of the distribution.
        logstd (torch.nn.Linear): Linear layer generating the log of the standard deviation of the distribution.
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        hidden_dim: list,
        w_lb: float,
        activation: nn.Module = nn.Tanh(),
        device: str = "cpu",
    ):
        super(C3M_W_Gaussian, self).__init__()

        # Initializing model parameters
        self.x_dim = x_dim
        self.state_dim = state_dim
        self.device = device
        self.w_lb = w_lb

        # Define the MLP model with given input dimension and hidden layers
        self.model = MLP(input_dim=x_dim, hidden_dims=hidden_dim, activation=activation)

        # Linear layers for the mean (mu) and log-std (logstd) of the Gaussian distribution
        self.mu = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)
        self.logstd = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass to compute the matrix W(x) based on Gaussian distribution parameters.

        The output matrix W(x) is sampled from a Gaussian distribution with mean (mu)
        and variance (given by the exp(logstd)). If `deterministic` is True, the output
        matrix is set to the mean (mu). Otherwise, a random sample is drawn.

        Args:
            states (torch.Tensor): Input tensor representing the current state(s).
            deterministic (bool): If True, the output is the mean; otherwise, it is a sample.

        Returns:
            W (torch.Tensor): Computed matrix W(x) of shape (n, x_dim, x_dim).
            dict (dict): A dictionary containing distribution information (log probabilities, entropy, etc.)
        """
        n = x.shape[0]

        # Generate logits from the input states via the MLP
        logits = self.model(x)
        # Calculate mean (mu) and log standard deviation (logstd)
        mu = self.mu(logits)

        # If deterministic, use the mean for W(x) and calculate corresponding log probabilities
        if deterministic:
            W = mu  # Use mean (mu) as W(x) in deterministic case
            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        else:
            logstd = self.logstd(logits)

            # Clamping logstd for numerical stability and to prevent extreme values
            logstd = torch.clamp(logstd, min=-5, max=2)
            # Calculate variance as exp(logstd)^2
            std = torch.exp(logstd)

            # change it to multivariate Gaussian
            dist = Normal(loc=mu, scale=std)

            # Sample W(x) from the distribution
            W = dist.rsample()  # Sample from the distribution

            # Calculate log-probability, probability, and entropy
            logprobs = dist.log_prob(W).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().unsqueeze(-1).sum(1)

        # Reshape W(x) to the desired shape (n, x_dim, x_dim) and ensure symmetry
        W = W.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)  # W = WᵀW ensures symmetry

        # Add a lower bound to the diagonal to ensure positive definiteness of W(x)
        W += self.w_lb * torch.eye(self.x_dim).to(self.device).view(
            1, self.x_dim, self.x_dim
        )

        return W, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, W: torch.Tensor):
        """
        Computes the log-probability of the action/sample W(x) under the given distribution.

        Args:
            dist (torch.distributions): The probability distribution (e.g., MultivariateNormal).
            W (torch.Tensor): The sampled matrix.

        Returns:
            logprobs (torch.Tensor): The log-probabilities of the samples.
        """
        W = W.view(W.shape[0], -1)  # Flatten W for compatibility with distribution
        logprobs = dist.log_prob(W).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        Computes the entropy of the given distribution, representing uncertainty.

        Args:
            dist (torch.distributions): The probability distribution (e.g., MultivariateNormal).

        Returns:
            entropy (torch.Tensor): The entropy of the distribution.
        """
        return dist.entropy().unsqueeze(-1)


class C3M_W(nn.Module):
    """
    C3M_W constructs a state-dependent positive semi-definite matrix W(x) used to
    define a quadratic cost metric in control or imitation learning.

    It dynamically builds W(x) through learned models and enforces symmetry and
    positive definiteness via a structured transformation and a lower-bound shift.

    Attributes:
        x_dim (int): Total state dimension.
        state_dim (int): Dimension of the full state input.
        action_dim (int): Dimension of action/control vector.
        w_lb (float): Lower bound value added to ensure positive definiteness of W.
        task (str): Identifier for the task (e.g., car, quadrotor).
        device (str): Computational device (CPU or CUDA).
        model_W (nn.Module): Neural network generating base W matrix.
        model_Wbot (nn.Module): Neural network generating task-specific lower block.
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        action_dim: int,
        w_lb: float,
        task: str,
        hidden_dim: list,
        activation: nn.Module = nn.Tanh(),
        device: str = "cpu",
    ):
        super(C3M_W, self).__init__()

        self.x_dim = x_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.w_lb = w_lb
        self.task = task

        # Instantiate models for generating the W matrix and optional lower blocks
        self.model_W = MLP(
            input_dim=x_dim,
            hidden_dims=hidden_dim,
            output_dim=x_dim * x_dim,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Computes the task-specific matrix W(x) for a batch of inputs.

        W(x) is constructed as W(x) = WᵀW + w_lb * I to ensure symmetry and
        positive definiteness. For some tasks, parts of W are substituted with a
        learned block Wbot(x) that captures invariant or low-dimensional structure.

        Returns:
            W (torch.Tensor): Positive semi-definite matrix of shape (n, x_dim, x_dim)
        """
        n = x.shape[0]

        # Ensure W is symmetric and PSD by computing WᵀW
        W = self.model_W(x).view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)

        # Add lower-bound scaled identity to guarantee positive definiteness
        W += self.w_lb * torch.eye(self.x_dim).to(self.device).view(
            1, self.x_dim, self.x_dim
        )

        return W, {
            "dist": None,
            "probs": torch.ones(self.x_dim * self.x_dim).to(self.device),
            "logprobs": torch.zeros(self.x_dim * self.x_dim).to(self.device),
            "entropy": torch.zeros(self.x_dim * self.x_dim).to(self.device),
        }


def get_u_model(task, x_dim: int, action_dim: int):
    """
    Constructs two neural networks (w1 and w2) that generate dynamic weight matrices
    based on the trimmed current and reference states. These networks are task-agnostic
    and used to compute control-relevant transformations in the C3M_U controller.

    Args:
        task (str): Identifier for the task (currently unused, reserved for future extensions).
        x_dim (int): Full state dimension.
        action_dim (int): Dimension of the action space.

    Returns:
        w1 (nn.Sequential): Network mapping input to a flattened tensor of shape (c * x_dim),
                            later reshaped to (c, x_dim) for transforming the error vector.
        w2 (nn.Sequential): Network mapping input to a flattened tensor of shape (c * action_dim),
                            later reshaped to (action_dim, c) to map the transformed error to control.
    """
    input_dim = 2 * x_dim  # Concatenated trimmed x and x_ref
    c = 3 * x_dim  # Intermediate dimension multiplier

    # First weight generator (for projecting error vector to latent space)
    w1 = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c * x_dim, bias=True),
    )

    # Second weight generator (for projecting latent to action space)
    w2 = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c * action_dim, bias=True),
    )

    return w1, w2


import torch
import torch.nn as nn
import torch.nn.functional as F


class C3M_U_Gaussian(nn.Module):
    """
    C3M_U: Control model to predict control input 'u' based on state, reference state,
    and learned task-specific parameters using neural networks.

    The model generates weight matrices from the trimmed states via neural networks,
    and applies them to the error between current and reference states to compute the control action.
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        action_dim: int,
        task: str,
    ):
        """
        Initialize the control model.

        Args:
            x_dim (int): Dimension of the state vector x.
            state_dim (int): Total dimension of the combined state vector.
            action_dim (int): Dimension of the control/action vector u.
            task (str): Identifier for the task used to get task-specific model parameters.
        """
        super(C3M_U_Gaussian, self).__init__()

        self.x_dim = x_dim  # Dimension of state x
        self.state_dim = state_dim  # Total dimension of input state
        self.action_dim = action_dim  # Dimension of action u

        self.task = task

        # Obtain task-specific neural networks that generate weight matrices
        self.w1, self.w2 = get_u_model(self.task, x_dim, self.action_dim)
        self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass to compute control input u.

        Args:
            x (torch.Tensor): Current state x, shape (batch_size, x_dim)
            xref (torch.Tensor): Reference state x_ref, shape (batch_size, x_dim)
            uref (torch.Tensor): Reference control input u_ref, unused here
            deterministic (bool): Placeholder for compatibility; unused

        Returns:
            u (torch.Tensor): Computed control input, shape (batch_size, action_dim)
            dict: Empty dictionary (placeholder for potential future use)
        """
        n = x.shape[0]  # Batch size

        # Concatenate trimmed current and reference state
        x_xref = torch.cat((x, xref), axis=-1)

        # Compute the error between x and x_ref
        e = (x - xref).unsqueeze(-1)  # Shape: (batch_size, x_dim, 1)

        # Generate weight matrices from the neural networks
        w1 = self.w1(x_xref).reshape(
            n, -1, self.x_dim
        )  # Shape: (batch_size, hidden_dim, x_dim)
        w2 = self.w2(x_xref).reshape(
            n, self.action_dim, -1
        )  # Shape: (batch_size, action_dim, hidden_dim)

        # Compute intermediate representation
        l1 = F.tanh(torch.matmul(w1, e))  # Shape: (batch_size, hidden_dim, 1)

        # Final control output
        mu = torch.matmul(w2, l1).squeeze(-1)  # Shape: (batch_size, action_dim)

        if deterministic:
            # For deterministic actions, return the mean of the distribution
            u = mu

            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        else:
            logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
            std = torch.exp(logstd.expand_as(mu))
            dist = Normal(loc=mu, scale=std)

            u = dist.rsample()

            logprobs = dist.log_prob(u).unsqueeze(-1).sum(1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy().sum(1)

        return u, {
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


class C3M_U(nn.Module):
    """
    C3M_U: Control model to predict control input 'u' based on state, reference state,
    and learned task-specific parameters using neural networks.

    The model generates weight matrices from the trimmed states via neural networks,
    and applies them to the error between current and reference states to compute the control action.
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        action_dim: int,
        task: str,
    ):
        """
        Initialize the control model.

        Args:
            x_dim (int): Dimension of the state vector x.
            state_dim (int): Total dimension of the combined state vector.
            action_dim (int): Dimension of the control/action vector u.
            task (str): Identifier for the task used to get task-specific model parameters.
        """
        super(C3M_U, self).__init__()

        self.x_dim = x_dim  # Dimension of state x
        self.state_dim = state_dim  # Total dimension of input state
        self.action_dim = action_dim  # Dimension of action u

        self.task = task

        # Obtain task-specific neural networks that generate weight matrices
        self.w1, self.w2 = get_u_model(self.task, x_dim, self.action_dim)
        # self.model = MLP(
        #     input_dim=self.state_dim, hidden_dims=[128, 128], output_dim=action_dim
        # )

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass to compute control input u.

        Args:
            x (torch.Tensor): Current state x, shape (batch_size, x_dim)
            xref (torch.Tensor): Reference state x_ref, shape (batch_size, x_dim)
            uref (torch.Tensor): Reference control input u_ref, unused here
            deterministic (bool): Placeholder for compatibility; unused

        Returns:
            u (torch.Tensor): Computed control input, shape (batch_size, action_dim)
            dict: Empty dictionary (placeholder for potential future use)
        """
        n = x.shape[0]  # Batch size

        # Concatenate trimmed current and reference state
        x_xref = torch.cat((x, xref), axis=-1)

        # Compute the error between x and x_ref
        e = (x - xref).unsqueeze(-1)  # Shape: (batch_size, x_dim, 1)

        # Generate weight matrices from the neural networks
        w1 = self.w1(x_xref).reshape(
            n, -1, self.x_dim
        )  # Shape: (batch_size, hidden_dim, x_dim)
        w2 = self.w2(x_xref).reshape(
            n, self.action_dim, -1
        )  # Shape: (batch_size, action_dim, hidden_dim)

        # Compute intermediate representation
        l1 = F.tanh(torch.matmul(w1, e))  # Shape: (batch_size, hidden_dim, 1)

        # Final control output
        u = torch.matmul(w2, l1).squeeze(-1)  # Shape: (batch_size, action_dim)

        # u = u + uref  # Add reference control input to the computed action

        return u, {}
