from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.env_base import BaseEnv

# NEURAL-LANDER PARAMETERS
rho = 1.225
drone_height = 0.09
g = 9.81
mass = 1.47

# X bounds
X_MIN = np.array([-10.0, -10.0, 0.0, -1.0, -1.0, -1.0]).reshape(-1, 1)
X_MAX = np.array([10.0, 10.0, 3.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

# Initial reference state bounds
XREF_INIT_MIN = np.array([-3.0, -3.0, 0.5, 1.0, 0.0, 0.0])
XREF_INIT_MAX = np.array([3.0, 3.0, 1.0, 1.0, 0.0, 0.0])

# Initial reference state perturbation bounds
XE_INIT_MIN = np.array([-1, -1, -0.4, -1.0, -1.0, 0.0])
XE_INIT_MAX = np.array([1, 1.0, 1.0, 1.0, 1.0, 0.0])

# reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1, 1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1, 1)

# reference control bounds
UREF_MIN = np.array([-1.0, -1.0, -3.0]).reshape(-1, 1)
UREF_MAX = np.array([1.0, 1.0, 9.0]).reshape(-1, 1)


# NEURAL-LANDER FUNCTIONS
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device("cpu"))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


env_config = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 6,
    "num_dim_control": 3,
    "pos_dimension": 3,
    "dt": 0.03,
    "time_bound": 6.0,
    "use_learned_dynamics": False,
    "q": 1.0,  # state cost weight
    "r": 0.1,  # control cost weight
}


class NeuralLanderEnv(BaseEnv):
    def __init__(self, sample_mode: str = "uniform"):
        """
        State: tracking error between current and reference trajectory
        Reward: 1 / (The 2-norm of tracking error + 1)
        """

        # env specific parameters
        self.task = "neurallander"
        self.Fa_model = read_weight("model/Fa_net_12_3_full_Lip16.pth")

        # initialize the base environment
        env_config["sample_mode"] = sample_mode
        env_config["Bbot_func"] = None

        super(NeuralLanderEnv, self).__init__(env_config)

    def Fa_func(self, x: torch.Tensor | np.ndarray):
        """Calculates the aerodynamic force using the neural network model."""
        # copy and use x
        x = deepcopy(x)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if len(x.shape) == 1:
            x = x.view(1, -1)

        z = x[:, 2]
        vx = x[:, 3]
        vy = x[:, 4]
        vz = x[:, 5]

        # use prediction from NN as ground truth
        n = z.shape[0]
        state = torch.zeros((n, 12))
        state[:, 0] = z + drone_height
        state[:, 1] = vx  # velocity
        state[:, 2] = vy  # velocity
        state[:, 3] = vz  # velocity
        state[:, 7] = 1.0
        state[:, 8:12] = 6508.0 / 8000

        with torch.no_grad():
            if next(self.Fa_model.parameters()).device != z.device:
                self.Fa_model.to(z.device)
            Fa = self.Fa_model(state) * torch.tensor([30.0, 15.0, 10.0])

        return Fa

    def _f_logic(self, x, lib):
        """Calculates the f(x) vector using the provided library."""
        n = x.shape[0]
        Fa = self.Fa_func(x)

        if lib == np:
            Fa = Fa.numpy()

        x, y, z, vx, vy, vz = [x[:, i] for i in range(self.num_dim_x)]

        f = lib.zeros((n, self.num_dim_x))
        f[:, 0] = vx
        f[:, 1] = vy
        f[:, 2] = vz
        f[:, 3] = Fa[:, 0] / mass
        f[:, 4] = Fa[:, 1] / mass
        f[:, 5] = Fa[:, 2] / mass - g
        return f

    def _B_logic(self, x, lib):
        """Calculates the B(x) matrix using the provided library."""
        n = x.shape[0]
        B = lib.zeros((n, self.num_dim_x, self.num_dim_control))

        B[:, 3, 0] = 1
        B[:, 4, 1] = 1
        B[:, 5, 2] = 1
        return B

    def _B_null_logic(self, n, lib):
        """Builds the B_null matrix batch using the provided library."""

        # Calculate the dimensions for the component matrices
        eye_dims = self.num_dim_x - self.num_dim_control
        zero_dims = (self.num_dim_control, eye_dims)

        if lib == torch:
            # 1. Create the base 2D matrix
            Bbot = torch.cat(
                (torch.eye(eye_dims), torch.zeros(zero_dims)),
                dim=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            return Bbot.repeat(n, 1, 1)
        else:  # lib == np
            # 1. Create the base 2D matrix
            Bbot = np.concatenate(
                (np.eye(eye_dims), np.zeros(zero_dims)),
                axis=0,
            )
            # 2. Repeat it 'n' times to create a 3D batch
            #    (np.newaxis adds the first dimension for repeating)
            return np.repeat(Bbot[np.newaxis, :, :], n, axis=0)

    def define_initial_state(self):
        """Define the initial state of the environment."""
        xref_0 = self.XREF_INIT_MIN + np.random.rand(len(self.XREF_INIT_MIN)) * (
            self.XREF_INIT_MAX - self.XREF_INIT_MIN
        )
        xe_0 = self.XE_INIT_MIN + np.random.rand(len(self.XE_INIT_MIN)) * (
            self.XE_INIT_MAX - self.XE_INIT_MIN
        )
        x_0 = xref_0 + xe_0

        # this is newly added for this env
        self.Fa = self.Fa_func(xref_0.reshape(1, -1)).flatten().numpy()

        return xref_0, xe_0, x_0

    def sample_reference_controls(self, freqs, weights, _t, add_noise=False):
        uref = np.array([0, 0, g]) - self.Fa / mass  # ref
        for freq, weight in zip(freqs, weights):
            uref += np.array(
                [
                    weight[0] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[1] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                    weight[2] * np.sin(freq * _t / self.time_bound * 2 * np.pi),
                ]
            )
        if add_noise:
            # add gaussian noise
            uref += np.random.normal(0, np.abs(0.1 * uref), size=uref.shape)

        uref = np.clip(uref, UREF_MIN.flatten(), UREF_MAX.flatten())
        return uref

    def render(self, mode="human"):
        pass
