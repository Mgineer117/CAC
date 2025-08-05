import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
        self.model = MLP(
            input_dim=state_dim, hidden_dims=hidden_dim, activation=activation
        )

        # Linear layers for the mean (mu) and log-std (logstd) of the Gaussian distribution
        self.mu = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)
        self.logstd = torch.nn.Linear(hidden_dim[-1], x_dim * x_dim)

        # self.model = MLP(
        #     input_dim=state_dim,
        #     hidden_dims=hidden_dim,
        #     output_dim=x_dim * x_dim,
        #     activation=activation,
        # )
        # self.logstd = nn.Parameter(torch.zeros(1, self.x_dim * self.x_dim))

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
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
        states = torch.cat((x, xref, uref), dim=-1)
        n = states.shape[0]

        # Generate logits from the input states via the MLP
        # logits = self.model(states)
        # mu = logits
        # logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
        # std = torch.exp(logstd.expand_as(mu))

        # # For the stochastic case, use a multivariate Gaussian to sample W(x)
        # dist = Normal(loc=mu, scale=std)

        logits = self.model(states)
        # Calculate mean (mu) and log standard deviation (logstd)
        mu = self.mu(logits)
        logstd = self.logstd(logits)

        # Clamping logstd for numerical stability and to prevent extreme values
        logstd = torch.clamp(logstd, min=-5, max=2)
        # logstd = torch.clamp(logstd, min=-2, max=3)
        # Calculate variance as exp(logstd)^2
        std = torch.exp(logstd)

        dist = Normal(loc=mu, scale=std)

        # If deterministic, use the mean for W(x) and calculate corresponding log probabilities
        if deterministic:
            W = mu  # Use mean (mu) as W(x) in deterministic case
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        else:
            # Sample W(x) from the distribution
            W = dist.rsample()  # Sample from the distribution

            # Calculate log-probability, probability, and entropy
            logprobs = dist.log_prob(W).unsqueeze(-1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy()

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
        effective_indices (list): Indices of state components used for trimmed representation.
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
        effective_indices: list,
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
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim
        self.device = device
        self.w_lb = w_lb
        self.task = task

        # Instantiate models for generating the W matrix and optional lower blocks
        self.model_W = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dim,
            output_dim=x_dim * x_dim,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
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

        # if self.task == "car":
        #     # Generate base matrix W and substitute top-left block with Wbot
        #     W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
        #     Wbot = self.model_Wbot(torch.ones(n, 1).to(self.device)).view(
        #         n, self.x_dim - self.action_dim, self.x_dim - self.action_dim
        #     )
        #     W[:, : self.x_dim - self.action_dim, : self.x_dim - self.action_dim] = Wbot
        #     W[:, self.x_dim - self.action_dim :, : self.x_dim - self.action_dim] = 0

        # elif self.task == "neurallander":
        #     # Wbot depends on z-dimension (x[:, 2])
        #     W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
        #     Wbot = self.model_Wbot(x[:, 2:3]).view(
        #         n, self.x_dim - self.action_dim, self.x_dim - self.action_dim
        #     )
        #     W[:, : self.x_dim - self.action_dim, : self.x_dim - self.action_dim] = Wbot
        #     W[:, self.x_dim - self.action_dim :, : self.x_dim - self.action_dim] = 0

        # elif self.task in ("pvtol", "turtlebot"):
        #     # Fully learned W without substitution
        #     W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)

        # elif self.task == "quadrotor":
        #     # Wbot depends on selected effective input dimensions (excluding action dims)
        #     W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
        #     input_Wbot = x[:, self.effective_indices[: -self.action_dim]]
        #     Wbot = self.model_Wbot(input_Wbot).view(
        #         n, self.x_dim - self.action_dim, self.x_dim - self.action_dim
        #     )
        #     W[:, : self.x_dim - self.action_dim, : self.x_dim - self.action_dim] = Wbot
        #     W[:, self.x_dim - self.action_dim :, : self.x_dim - self.action_dim] = 0

        # else:
        #     raise NotImplementedError(f"Task '{self.task}' not supported.")

        # Ensure W is symmetric and PSD by computing WᵀW
        states = torch.cat(
            (x, xref, uref), dim=-1
        )  # Concatenate current and reference states
        W = self.model_W(states).view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)

        # Add lower-bound scaled identity to guarantee positive definiteness
        W += self.w_lb * torch.eye(self.x_dim).to(self.device).view(
            1, self.x_dim, self.x_dim
        )

        return W


def get_u_model(task, x_dim: int, effective_x_dim: int, action_dim: int):
    """
    Constructs two neural networks (w1 and w2) that generate dynamic weight matrices
    based on the trimmed current and reference states. These networks are task-agnostic
    and used to compute control-relevant transformations in the C3M_U controller.

    Args:
        task (str): Identifier for the task (currently unused, reserved for future extensions).
        x_dim (int): Full state dimension.
        effective_x_dim (int): Dimension of trimmed (selected) state used as input.
        action_dim (int): Dimension of the action space.

    Returns:
        w1 (nn.Sequential): Network mapping input to a flattened tensor of shape (c * x_dim),
                            later reshaped to (c, x_dim) for transforming the error vector.
        w2 (nn.Sequential): Network mapping input to a flattened tensor of shape (c * action_dim),
                            later reshaped to (action_dim, c) to map the transformed error to control.
    """
    input_dim = 2 * effective_x_dim  # Concatenated trimmed x and x_ref
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
        effective_indices: list,
        action_dim: int,
        task: str,
    ):
        """
        Initialize the control model.

        Args:
            x_dim (int): Dimension of the state vector x.
            state_dim (int): Total dimension of the combined state vector.
            effective_indices (list): Indices of x to be used for control (feature selection).
            action_dim (int): Dimension of the control/action vector u.
            task (str): Identifier for the task used to get task-specific model parameters.
        """
        super(C3M_U, self).__init__()

        self.x_dim = x_dim  # Dimension of state x
        self.state_dim = state_dim  # Total dimension of input state
        self.effective_x_dim = len(effective_indices)  # Dimension of trimmed state x
        self.action_dim = action_dim  # Dimension of action u
        self.effective_indices = effective_indices  # Selected indices for trimmed state

        self.task = task

        # Obtain task-specific neural networks that generate weight matrices
        self.w1, self.w2 = get_u_model(self.task, x_dim, x_dim, self.action_dim)

    def trim_state(self, state: torch.Tensor):
        """
        Split and trim the input state into x, x_ref, and u_ref.

        Args:
            state (torch.Tensor): Tensor of shape (batch_size, state_dim)

        Returns:
            x (torch.Tensor): Current state x
            xref (torch.Tensor): Reference state x_ref
            uref (torch.Tensor): Reference action u_ref
            x_trim (torch.Tensor): Trimmed current state (selected indices)
            xref_trim (torch.Tensor): Trimmed reference state (selected indices)
        """
        # Split state into components
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        # Select only effective indices from x and xref
        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

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
            x_trim (torch.Tensor): Trimmed x based on effective indices
            xref_trim (torch.Tensor): Trimmed x_ref based on effective indices
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

        return u, {}
