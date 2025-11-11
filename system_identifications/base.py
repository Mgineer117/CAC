import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import lstsq  # <-- Import least squares solver
from scipy import stats

# === Parameters === #
STATE_DIM = 10
ACTION_DIM = 4

LABELS = [
    r"x ($m$)",
    r"y ($m$)",
    r"z ($m$)",
    r"vx ($m/s$)",
    r"vy ($m/s$)",
    r"vz ($m/s$)",
    r"thrust ($1/s^2$)",
    r"roll ($rad$)",
    r"pitch ($rad$)",
    r"yaw ($rad$)",
]

BASE_DIR = "system_identifications/"
MIN_THRUST, MAX_THRUST = 10001, 60000
TIME_INTERVAL = 0.05  # seconds
N = 600  # number of time steps

g = 9.81


# === Functions === #
def f_nominal(x):
    """Calculates the nominal f(x) vector from the image"""
    f_nom = np.zeros(STATE_DIM)
    f_nom[0] = x[3]  # x_dot = vx
    f_nom[1] = x[4]  # y_dot = vy
    f_nom[2] = x[5]  # z_dot = vz
    # --- FIX THE PHYSICS TO BE Z-UP ---

    # The X and Y acceleration equations need to swap
    # to match the regression's findings (and a standard frame).

    # vx_dot (state 3) depends on ROLL (x[7])
    f_nom[3] = x[6] * np.cos(x[8]) * np.sin(x[7])

    # vy_dot (state 4) depends on PITCH (x[8])
    f_nom[4] = -x[6] * np.sin(x[8])

    # vz_dot (state 5) is (Thrust) - (Gravity)
    # f_nom[5] = x[6] * np.cos(x[8]) * np.cos(x[7]) - g
    f_nom[5] = x[6] * np.cos(x[8]) * np.cos(x[7])  # REMOVE '- g'
    # --- END FIX ---

    # f_nom[3] = -x[6] * np.sin(x[8])
    # f_nom[4] = x[6] * np.cos(x[8]) * np.sin(x[7])
    # f_nom[5] = g - x[6] * np.cos(x[8]) * np.cos(x[7])
    return f_nom


def B_nominal():
    """Builds the 10x4 nominal B matrix from the image"""
    B_nom = np.zeros((STATE_DIM, ACTION_DIM))
    B_nom[6, 0] = 1.0
    B_nom[7, 1] = 1.0
    B_nom[8, 2] = 1.0
    B_nom[9, 3] = 1.0
    return B_nom


def call_mdp_data(data_type: str = "train"):
    # === Call mdp_data.json data === #
    with open(f"{BASE_DIR}/data/mdp_data/{data_type}_data.json", "r") as infile:
        mdp_data = json.load(infile)

    states = mdp_data["states"]
    actions = mdp_data["actions"]
    next_states = mdp_data["next_states"]
    terminals = mdp_data["terminals"]
    dynamics = mdp_data["dynamics"]

    for i in range(len(states)):
        # print if does not match
        if not (
            len(states[i])
            == len(actions[i])
            == len(next_states[i])
            == len(terminals[i])
        ):
            assert False, f"Data length mismatch in trajectory {i}:"

    # check if the number of trajectory match
    assert (
        len(states) == len(actions) == len(next_states) == len(terminals)
    ), "Number of trajectories do not match."

    return states, actions, next_states, terminals, dynamics


def call_policy(algo_name: str) -> list:
    # === Call policy.json data === #
    if algo_name in ["c3m", "c3m-approx"]:
        from policy.layers.c3m_networks import C3M_U

        policy_class = C3M_U
    elif algo_name in ["ppo", "cac", "cac-approx"]:
        from policy.layers.c3m_networks import C3M_U_Gaussian

        policy_class = C3M_U_Gaussian
    else:
        raise NotImplementedError(f"Policy {algo_name} is not implemented.")

    policy = policy_class(x_dim=10, state_dim=24, action_dim=4, task="flapper")

    # Define the path
    model_path = f"system_identifications/policies/{algo_name}.pth"

    # 1. Load the state dictionary from the file (using map_location is safer)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # 2. Load the dictionary into the policy
    policy.load_state_dict(state_dict)

    return policy


def smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def FD_fourth_order(points: list, method: str = "central") -> list:
    if method == "forward":
        return (
            -(25 / 12) * points[0]
            + (4) * points[1]
            - 3 * points[2]
            + (4 / 3) * points[3]
            - (1 / 4) * points[4]
        ) / TIME_INTERVAL
    elif method == "backward":
        return (
            (25 / 12) * points[-1]
            - 4 * points[-2]
            + 3 * points[-3]
            - (4 / 3) * points[-4]
            + (1 / 4) * points[-5]
        ) / TIME_INTERVAL
    elif method == "central":
        return (
            (1 / 12) * points[0]
            - (2 / 3) * points[1]
            + (2 / 3) * points[3]
            - (1 / 12) * points[4]
        ) / TIME_INTERVAL
    else:
        raise ValueError("Invalid method for finite difference.")


def FD_second_order(points: list, method: str = "central") -> list:
    if method == "forward":
        return (-3 * points[0] + 4 * points[1] - 1 * points[2]) / (2 * TIME_INTERVAL)
    elif method == "backward":
        return (3 * points[-1] - 4 * points[-2] + 1 * points[-3]) / (2 * TIME_INTERVAL)
    elif method == "central":
        return (points[2] - points[0]) / (2 * TIME_INTERVAL)
    else:
        raise ValueError("Invalid method for finite difference.")


def compute_dynamics(states: list) -> list:
    # === Generate x_dot using finite difference === #
    # x_dots = []
    # for i in range(len(states)):
    x_dot_trajectory = []

    try:
        # The full trajectory is the list of states + the final "next_state"
        full_trajectory = [np.array(s) for s in states]
    except IndexError:
        # Handle cases where a trajectory might be empty
        raise ValueError(f"Trajectory is empty or malformed.")

    # 2. Validate the assumption that N=600 + 1
    if len(full_trajectory) != N + 1:
        raise ValueError(
            f"Trajectory length is {len(full_trajectory)}, but we assumed N={N}."
        )

    # First point (t=0): Use second-order FORWARD difference
    x_dot_0 = FD_second_order(
        [full_trajectory[0], full_trajectory[1], full_trajectory[2]],
        method="forward",
    )
    x_dot_trajectory.append(x_dot_0.tolist())

    # Middle points (t=1 to N-2, i.e., 1 to 297): Use second-order CENTRAL difference
    for t in range(1, N):
        x_dot_t = (full_trajectory[t + 1] - full_trajectory[t - 1]) / (
            2 * TIME_INTERVAL
        )
        x_dot_trajectory.append(x_dot_t.tolist())
    return x_dot_trajectory
    #     print(len(x_dot_trajectory))
    #     x_dots.append(x_dot_trajectory)

    # return x_dots


# === Compute x_hat_dot using finite difference === #
def compute_nominal_dynamics(states: list, actions: list) -> list:
    N_trajectory = len(states)

    # === Generate x_hat_dot using weighted average of finite difference and nominal model === #
    train_indices = list(range(N_trajectory))

    x_hat_dots = []
    for j in train_indices:  # Loop over training trajectories
        x_hat_dot = []

        # Reconstruct the full trajectory
        full_trajectory = [np.array(s) for s in states[j]]

        for t in range(N):  # Loop from t=0 to N-2
            x_t = full_trajectory[t]
            u_t = np.array(actions[j][t])
            # Calculate the nominal regressor vector h(x, u)
            x_hat_dot.append(f_nominal(x_t) + B_nominal() @ u_t)

        x_hat_dots.append(x_hat_dot)

    return x_hat_dots


def thrust_normalization(thrust: float) -> float:
    """Normalize thrust value between 0 and 1."""
    return (thrust - MIN_THRUST) / (MAX_THRUST - MIN_THRUST)


def thrust_denormalization(norm_thrust: float) -> float:
    """Denormalize thrust value from 0-1 back to actual thrust."""
    return norm_thrust * (MAX_THRUST - MIN_THRUST) + MIN_THRUST


def compose_state(flight_data: dict, index: int) -> list:
    """Compose a state vector from flight data at a given index."""
    # Velocity from using Kalman filter is too noisy so we resort to FD
    pose = flight_data["mocap_pose"][index]
    stabilizer = flight_data["stabilizer"][index]
    cmd = flight_data["controller_cmd"][index]

    if index == 0:
        current_pos = np.array([pose[0], pose[1], pose[2]])
        next_pos_1 = np.array(
            [
                flight_data["mocap_pose"][index + 1][0],
                flight_data["mocap_pose"][index + 1][1],
                flight_data["mocap_pose"][index + 1][2],
            ]
        )
        next_pos_2 = np.array(
            [
                flight_data["mocap_pose"][index + 2][0],
                flight_data["mocap_pose"][index + 2][1],
                flight_data["mocap_pose"][index + 2][2],
            ]
        )

        vx = FD_second_order(
            [current_pos[0], next_pos_1[0], next_pos_2[0]], method="forward"
        )
        vy = FD_second_order(
            [current_pos[1], next_pos_1[1], next_pos_2[1]], method="forward"
        )
        vz = FD_second_order(
            [current_pos[2], next_pos_1[2], next_pos_2[2]], method="forward"
        )
    else:
        before_pos_1 = np.array(
            [
                flight_data["mocap_pose"][index - 1][0],
                flight_data["mocap_pose"][index - 1][1],
                flight_data["mocap_pose"][index - 1][2],
            ]
        )
        next_pos_1 = np.array(
            [
                flight_data["mocap_pose"][index + 1][0],
                flight_data["mocap_pose"][index + 1][1],
                flight_data["mocap_pose"][index + 1][2],
            ]
        )

        vx = FD_second_order([before_pos_1[0], None, next_pos_1[0]], method="central")
        vy = FD_second_order([before_pos_1[1], None, next_pos_1[1]], method="central")
        vz = FD_second_order([before_pos_1[2], None, next_pos_1[2]], method="central")

    state = [
        pose[0],  # x position
        pose[1],  # y position
        pose[2],  # z position
        vx,  # x velocity
        vy,  # y velocity
        vz,  # z velocity
        thrust_normalization(stabilizer["stabilizer.thrust"]),  # normalized thrust
        pose[3],  # roll
        pose[4],  # pitch
        pose[5],  # yaw
    ]
    return state


def compose_action(flight_data: dict, index: int) -> list:
    """Compose an action vector from flight data at a given index."""
    # THIS IS FOR PID DESIRED RATE CONTROLLER which is too drastic
    # attitude_list = flight_data["controller_attitude_rate"][index]
    # attitude_list["controller.rollRate"],  # roll command
    # attitude_list["controller.pitchRate"],  # pitch command
    # attitude_list["controller.yawRate"],  # yaw command

    # === action = [thrust_rate, roll_rate, pitch_rate, yaw_rate] === #
    thrust_list = flight_data["controller_cmd"]
    attitude_list = flight_data["mocap_pose"]

    # thrust dynamics using second-order finite difference
    if index == 0:
        current_thrust = thrust_normalization(
            thrust_list[index]["controller.cmd_thrust"]
        )
        next_thrust_1 = thrust_normalization(
            thrust_list[index + 1]["controller.cmd_thrust"]
        )
        next_thrust_2 = thrust_normalization(
            thrust_list[index + 2]["controller.cmd_thrust"]
        )

        current_angle = np.array(attitude_list[index][3:6])
        next_angle_1 = np.array(attitude_list[index + 1][3:6])
        next_angle_2 = np.array(attitude_list[index + 2][3:6])

        thrust_rate = FD_second_order(
            [current_thrust, next_thrust_1, next_thrust_2], method="forward"
        )
        roll_rate = FD_second_order(
            [current_angle[0], next_angle_1[0], next_angle_2[0]], method="forward"
        )
        pitch_rate = FD_second_order(
            [current_angle[1], next_angle_1[1], next_angle_2[1]], method="forward"
        )
        yaw_rate = FD_second_order(
            [current_angle[2], next_angle_1[2], next_angle_2[2]], method="forward"
        )
    else:
        before_thrust_1 = thrust_normalization(
            thrust_list[index - 1]["controller.cmd_thrust"]
        )
        next_thrust_1 = thrust_normalization(
            thrust_list[index + 1]["controller.cmd_thrust"]
        )

        before_angle_1 = np.array(attitude_list[index - 1][3:6])
        next_angle_1 = np.array(attitude_list[index + 1][3:6])

        thrust_rate = FD_second_order(
            [before_thrust_1, None, next_thrust_1], method="central"
        )
        roll_rate = FD_second_order(
            [before_angle_1[0], None, next_angle_1[0]], method="central"
        )
        pitch_rate = FD_second_order(
            [before_angle_1[1], None, next_angle_1[1]], method="central"
        )
        yaw_rate = FD_second_order(
            [before_angle_1[2], None, next_angle_1[2]], method="central"
        )

    action = [
        thrust_rate,  # thrust command
        roll_rate,  # roll command
        pitch_rate,  # pitch command
        yaw_rate,  # yaw command
    ]
    return action


def lstq_regression(
    x_dots: list,
    x_hat_dots: list,
    outlier_removal: bool = False,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits the model x_dot = (v * h(x,u)) + c element-wise, where v and c are
    10D vectors. 'x_dots' is the target (Y) and 'x_hat_dots' is the
    regressor h(x,u) (the 'X' in y=mx+b).

    Args:
        x_dots: A list of trajectories, where each trajectory is a list of
                x_dot vectors (e.g., [[x_dot_t0, x_dot_t1, ...], ...]).
        x_hat_dots: A list of trajectories, where each trajectory is a list of
                    h(x,u) vectors (e.g., [[h_t0, h_t1, ...], ...]).

    Returns:
        A tuple (v_hat, c_hat):
        - v_hat: The 10D scaling vector (the 'm' in y=mx+b for each dim).
        - c_hat: The 10D intercept vector (the 'b' in y=mx+b for each dim).
    """
    # Ensure the input lists are not empty
    if not x_dots or not x_hat_dots:
        raise ValueError("Input lists are empty.")

    # Stack all data points from all trajectories into two giant matrices
    # Y_matrix is all x_dot targets, shape (Total_Samples, 10)
    Y_matrix = np.vstack([np.array(traj) for traj in x_dots])
    # A_matrix is all h(x,u) regressors, shape (Total_Samples, 10)
    A_matrix = np.vstack([np.array(traj) for traj in x_hat_dots])

    if outlier_removal:
        # === Outlier Removal Implemented (Checking X and Y) === #

        # 1. Define the indices of the columns to check
        noisy_indices = [5]  # We only care about state 5 (vz_dot)

        # 2. Calculate Z-scores for the Y_matrix (the "true" derivatives)
        z_scores_Y = np.abs(stats.zscore(Y_matrix, axis=0))
        z_scores_to_check_Y = z_scores_Y[:, noisy_indices]
        is_not_outlier_Y = np.all(z_scores_to_check_Y < threshold, axis=1)

        # 3. Calculate Z-scores for the A_matrix (the "model" regressors)
        z_scores_A = np.abs(stats.zscore(A_matrix, axis=0))
        z_scores_to_check_A = z_scores_A[:, noisy_indices]
        is_not_outlier_A = np.all(z_scores_to_check_A < threshold, axis=1)

        # 4. Create a final mask. A point is kept ONLY if it is
        #    NOT an outlier in Y AND NOT an outlier in A.
        is_not_outlier = is_not_outlier_Y & is_not_outlier_A

        # 5. Filter both your data and regressor matrices using this combined mask
        Y_matrix_clean = Y_matrix[is_not_outlier]
        A_matrix_clean = A_matrix[is_not_outlier]

        print(f"[INFO] Outlier Removal: Original data had {len(Y_matrix)} points.")
        print(
            f"    Filtered data has {len(Y_matrix_clean)} points ({len(Y_matrix) - len(Y_matrix_clean)} removed)."
        )
        Y_matrix = Y_matrix_clean
        A_matrix = A_matrix_clean

    if Y_matrix.shape != A_matrix.shape:
        raise ValueError(
            f"Shape mismatch: x_dots matrix is {Y_matrix.shape} but "
            f"x_hat_dots matrix is {A_matrix.shape}."
        )

    N_samples, state_dim = Y_matrix.shape
    v_hat = np.zeros(state_dim)
    c_hat = np.zeros(state_dim)
    std_dev_hat = np.zeros(state_dim)  # <-- NEW: To store std dev of residuals
    total_residuals = 0

    # Solve 10 independent 1D linear regressions (y = m*x + b)
    plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
    )

    for i in range(state_dim):
        # Y_i = [A_i, 1] @ [v_i, c_i]^T

        # Y is the target vector for this dimension (shape N_samples,)
        Y_i = Y_matrix[:, i]

        # A is the regressor vector (our 'x' in y=mx+b)
        A_i_regressor = A_matrix[:, i]

        # Build the regressor matrix with an intercept column
        # This is our [x, 1] matrix, shape (N_samples, 2)
        A_with_intercept = np.vstack([A_i_regressor, np.ones(N_samples)]).T

        # Solve for the parameters [v_i, c_i]
        try:
            (v_i, c_i), res, _, _ = lstsq(A_with_intercept, Y_i, rcond=None)
            v_hat[i] = v_i
            c_hat[i] = c_i

            if res.size > 0:
                total_residuals += res.sum()
                # --- NEW CODE: Calculate std dev of residuals ---
                # res[0] is the sum of squared residuals: sum((Y_true - Y_pred)^2)
                # std_dev = sqrt( sum_of_squared_residuals / N_samples )
                std_dev_hat[i] = np.sqrt(res[0] / N_samples)
                # --- END NEW CODE ---
            else:
                std_dev_hat[i] = np.nan  # if res is empty

        except np.linalg.LinAlgError as e:
            print(f"Error solving for dimension {i}: {e}")
            v_hat[i] = np.nan
            c_hat[i] = np.nan
            std_dev_hat[i] = np.nan  # <-- NEW

        # plot the least squares plot for each dimension
        plt.subplot(5, 2, i + 1)
        # do not plot if there are too many points
        if len(A_i_regressor) > 10000000:  # Increased limit
            plt.scatter(
                A_i_regressor[::10],
                Y_i[::10],
                alpha=0.3,
                label="Data Points (subsampled)",
                s=10,
                color="blue",
            )
        else:
            plt.scatter(
                A_i_regressor,
                Y_i,
                alpha=0.3,
                label="Data Points",
                s=10,
                color="blue",
            )

        # Plot the fitted line
        x_vals = np.array([A_i_regressor.min(), A_i_regressor.max()])
        y_vals = v_hat[i] * x_vals + c_hat[i]
        plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Fitted Line")

        # what is x being plotted?

        plt.ylabel(LABELS[i], fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.7, linewidth=1.5)
    plt.xlabel("x_hat_dot", fontsize=12)
    plt.suptitle("Least Squares Regression for Each State Dimension", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/figs/least_squares_fit.svg")
    plt.close()

    print(f"[INFO] Solved for VECTOR v and INTERCEPT c")
    print(f"   v = {np.array_str(v_hat, precision=4, suppress_small=True)}")
    print(f"   c = {np.array_str(c_hat, precision=4, suppress_small=True)}")
    # --- NEW PRINT STATEMENT ---
    print(f" std = {np.array_str(std_dev_hat, precision=4, suppress_small=True)}")
    # --- END NEW PRINT ---
    print(f"[INFO] Total Residual: {total_residuals:.4f}")

    return v_hat, c_hat, std_dev_hat  # <-- MODIFIED RETURN


def predict_next_state(
    x: np.ndarray, u: np.ndarray, v_hat: np.ndarray, c_hat: np.ndarray
):
    """Predicts x_{t+1} using Euler integration and learned dynamics"""
    f_x = f_nominal(x)
    B_u = B_nominal() @ u

    # Calculate nominal dynamics
    h = f_x + B_u

    # Apply the learned scaling vector element-wise
    x_hat_dot = v_hat * h + c_hat

    x_next = x + x_hat_dot * TIME_INTERVAL
    return x_next


def plot_prediction(true_matrices: list, pred_matrices: list):
    # === Plot the prediction over time for each state variable === #
    # smooth the true and predicted matrices
    for i in range(len(true_matrices)):
        for j in range(STATE_DIM):
            true_matrices[i][:, j] = smooth(true_matrices[i][:, j], weight=0.9)
            pred_matrices[i][:, j] = smooth(pred_matrices[i][:, j], weight=0.9)

    for i in range(len(true_matrices)):
        # Plot error over time for each state variable
        plt.subplots(
            nrows=5,
            ncols=2,
            figsize=(16, 8),
            sharex=True,  # constrained_layout=True
        )
        for j in range(STATE_DIM):
            time_steps = np.arange(true_matrices[i].shape[0]) * TIME_INTERVAL

            plt.subplot(5, 2, j + 1)
            plt.plot(
                time_steps,
                true_matrices[i][:, j],
                linewidth=2.5,
                alpha=0.9,
                label="True",
            )
            plt.plot(
                time_steps,
                pred_matrices[i][:, j],
                linewidth=2.5,
                alpha=0.9,
                label="Predicted",
            )
            plt.ylabel(LABELS[j], fontsize=14)
            plt.grid(True, linestyle=":", alpha=0.7, linewidth=1.5)

        plt.xlabel("Time (s)")
        plt.suptitle("Prediction Error with 95% Confidence Interval", fontsize=18)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{BASE_DIR}/figs/prediction_{i}.svg")
        plt.close()


def plot_prediction_error(error_matrices: list):
    # === Plot the prediction error over time for each state variable === #
    # smooth the error matrices
    for i in range(len(error_matrices)):
        for j in range(STATE_DIM):
            error_matrices[i][:, j] = smooth(error_matrices[i][:, j], weight=0.9)

    # if len(error_matrices) > 1, average them and plot 95% confidence interval.
    if len(error_matrices) > 1:
        avg_error_matrix = np.mean(error_matrices, axis=0)
        std_error_matrix = np.std(error_matrices, axis=0)
    else:
        avg_error_matrix = error_matrices[0]
        std_error_matrix = np.zeros_like(avg_error_matrix)

    # Plot error over time for each state variable
    plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
        sharex=True,  # constrained_layout=True
    )
    for i in range(STATE_DIM):
        time_steps = np.arange(avg_error_matrix.shape[0]) * TIME_INTERVAL

        plt.subplot(5, 2, i + 1)
        plt.plot(time_steps, avg_error_matrix[:, i], linewidth=2.5, alpha=0.9)
        plt.fill_between(
            time_steps,
            avg_error_matrix[:, i] - 1.96 * std_error_matrix[:, i],
            avg_error_matrix[:, i] + 1.96 * std_error_matrix[:, i],
            color="b",
            alpha=0.2,
        )
        plt.ylabel(LABELS[i], fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.7, linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.suptitle("Prediction Error with 95% Confidence Interval", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/figs/prediction_error.svg")
    plt.close()


def simulate(
    test_states: list,
    test_actions: list,
    v: nn.Module | np.ndarray,
    c: nn.Module | np.ndarray,
):
    # === Run simulation on the test trajectory === #
    print(f"[INFO] Simulating and Plotting Error for Test Trajectory.")
    true_matrices = []
    pred_matrices = []
    error_matrices = []
    N_trajectories = len(test_states)
    for idx in range(N_trajectories):
        # reconstruct the full trajectory
        test_traj_states = [np.array(s) for s in test_states[idx]]
        test_traj_actions = [np.array(u) for u in test_actions[idx]]

        # Initialize the predicted trajectory with the true initial state
        x_pred_trajectory = [test_traj_states[0]]

        # Loop for N-1 steps to generate N states
        for t in range(N - 1):
            x_t = x_pred_trajectory[-1]  # Get the last predicted state
            u_t = test_traj_actions[t]  # Get the true action

            # Predict the next state using our learned model
            if isinstance(v, nn.Module) and isinstance(c, nn.Module):
                with torch.no_grad():
                    v_hat = v(torch.tensor(x_t, dtype=torch.float32)).detach().numpy()
                    c_hat = c(torch.tensor(x_t, dtype=torch.float32)).detach().numpy()
            else:
                v_hat = v
                c_hat = c
            # print(v_hat, c_hat)
            # print(x_t)
            x_t_plus_1 = predict_next_state(x_t, u_t, v_hat, c_hat)
            x_pred_trajectory.append(x_t_plus_1)

        # Convert lists to numpy arrays for easy subtraction
        true_matrix = np.array(test_traj_states)
        pred_matrix = np.array(x_pred_trajectory)

        #
        true_matrices.append(true_matrix)
        pred_matrices.append(pred_matrix)

        # 2. Calculate the error matrix
        error_matrix = np.abs(true_matrix - pred_matrix)
        error_matrices.append(error_matrix)
    return true_matrices, pred_matrices, error_matrices


# def fit_reference_controls(actions: list, test_indices: list):
#     test_indices = [0]
#     actions = [actions[i] for i in test_indices]
#     # === find freq and weights that best match the reference controls in data === #
#     # freq = 0.1
#     # weights = np.array([0.25, 0.25, 0.25, 0.25])  # equal weights for all controls

#     freqs = [0.1 * i for i in range(1, 11)]  # flapper is vulnerable to high freq
#     weights = np.random.randn(len(freqs), 4)
#     weights = (weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
#     # print(weights)

#     urefs = []
#     for _t in range(N):
#         uref = np.array([0.0, 0.0, 0.0, 0.0])
#         for freq, weight in zip(freqs, weights):
#             uref += np.array(
#                 [
#                     weight[0] * np.sin(freq * _t / N * 2 * np.pi),
#                     weight[1] * np.sin(freq * _t / N * 2 * np.pi),
#                     weight[2] * np.sin(freq * _t / N * 2 * np.pi),
#                     weight[3] * np.sin(freq * _t / N * 2 * np.pi),
#                 ]
#             )
#         urefs.append(uref)

#     # Find the best matching frequency and weights
#     # plot actions by dimension and urefs
#     plt.subplots(
#         nrows=2,
#         ncols=2,
#         figsize=(12, 8),
#     )
#     for i in range(ACTION_DIM):
#         plt.subplot(2, 2, i + 1)
#         for j in range(len(actions)):
#             action_dim_i = [actions[j][t][i] for t in range(N)]
#             plt.plot(action_dim_i, label="Actual Actions", alpha=0.4)

#         plt.plot(
#             [urefs[t][i] for t in range(N)],
#             label="Reference",
#             color="black",
#             linestyle="--",
#             linewidth=2,
#             alpha=0.8,
#         )
#         plt.xlabel("Time Step")
#         plt.ylabel(f"{LABELS[i + 6]}")

#     plt.savefig(f"{BASE_DIR}/figs/reference_controls.svg")
