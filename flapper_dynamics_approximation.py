# === Approximate the empirical coefficients === #
# Dynamics is as follows:
# x_dot = v * (f_nominal(x) + B_nominal * u), where v is a 10D vector
# We will estimate the 10 components of 'v' using least squares.

import json
from math import ceil, floor

import matplotlib.pyplot as plt  # <-- Import plotting
import numpy as np  # <-- Import numpy for vector operations
from numpy.linalg import lstsq  # <-- Import least squares solver
from scipy import stats

from read import MAX_THRUST, MIN_THRUST, TIME_INTERVAL

# === Parameters === #
N = 299
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

m = 0.102  # mass of the flapper in kg (102g)
g = 9.81


# === Functions === #
def call_data():
    # === Call mdp_flight_data.json data === #
    with open("mdp_flight_data.json", "r") as infile:
        mdp_data = json.load(infile)

    states = mdp_data["states"]
    actions = mdp_data["actions"]
    next_states = mdp_data["next_states"]
    terminals = mdp_data["terminals"]

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

    return states, actions, next_states, terminals


# === ADD THE DYNAMICS APPROXIMATION CODE === #
def f_nominal(x):
    """Calculates the nominal f(x) vector from the image"""
    f_nom = np.zeros(STATE_DIM)
    f_nom[0] = x[3]  # x_dot = vx
    f_nom[1] = x[4]  # y_dot = vy
    f_nom[2] = x[5]  # z_dot = vz
    f_nom[3] = -x[6] * np.sin(x[8])
    f_nom[4] = x[6] * np.cos(x[8]) * np.sin(x[7])
    f_nom[5] = g - x[6] * np.cos(x[8]) * np.cos(x[7])
    return f_nom


def B_nominal():
    """Builds the 10x4 nominal B matrix from the image"""
    B_nom = np.zeros((STATE_DIM, ACTION_DIM))
    B_nom[6, 0] = 1.0
    B_nom[7, 1] = 1.0
    B_nom[8, 2] = 1.0
    B_nom[9, 3] = 1.0
    return B_nom


# === Compute x_dot using finite difference === #
def compute_x_dot(states: list, next_states: list) -> list:
    # === Generate x_dot using finite difference === #
    x_dots = []
    for i in range(len(states)):
        x_dot_trajectory = []

        try:
            # The full trajectory is the list of states + the final "next_state"
            full_trajectory = [np.array(s) for s in states[i]] + [
                np.array(next_states[i][-1])
            ]
        except IndexError:
            # Handle cases where a trajectory might be empty
            raise ValueError(f"Trajectory {i} is empty or malformed.")

        # 2. Validate the assumption that N=299
        if len(full_trajectory) != N:
            raise ValueError(
                f"Trajectory {i} length is {len(full_trajectory)}, but we assumed N={N}."
            )

        # First point (t=0): Use second-order FORWARD difference
        x_dot_0 = (
            -(1 / 2) * full_trajectory[2]
            + 2 * full_trajectory[1]
            - (3 / 2) * full_trajectory[0]
        ) / TIME_INTERVAL
        x_dot_trajectory.append(x_dot_0.tolist())

        # Middle points (t=1 to N-2, i.e., 1 to 297): Use second-order CENTRAL difference
        for t in range(1, N - 1):
            x_dot_t = (full_trajectory[t + 1] - full_trajectory[t - 1]) / (
                2 * TIME_INTERVAL
            )
            x_dot_trajectory.append(x_dot_t.tolist())

        # # Last point (t=N-1, i.e., 298): Use second-order BACKWARD difference
        # x_dot_N_minus_1 = (
        #     (3 / 2) * full_trajectory[N - 1]
        #     - 2 * full_trajectory[N - 2]
        #     + (1 / 2) * full_trajectory[N - 3]
        # ) / TIME_INTERVAL
        # x_dot_trajectory.append(x_dot_N_minus_1.tolist())

        x_dots.append(x_dot_trajectory)

    return x_dots


# === Compute x_hat_dot using finite difference === #
def compute_x_hat_dot(states: list, actions: list, ratios: list = [0.8, 0.2]) -> list:
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"

    N_trajectory = len(states)
    N_train, N_test = floor(N_trajectory * ratios[0]), ceil(N_trajectory * ratios[1])

    # === Generate x_hat_dot using weighted average of finite difference and nominal model === #
    train_indices = list(range(N_train))
    test_indices = list(range(N_train, N_train + N_test))
    print(
        f"Training on {len(train_indices)} trajectories, testing on {len(test_indices)} trajectories."
    )

    x_hat_dots = []
    for j in train_indices:  # Loop over training trajectories
        x_hat_dot = []

        # Reconstruct the full trajectory
        full_trajectory = [np.array(s) for s in states[j]] + [
            np.array(next_states[j][-1])
        ]

        for t in range(N - 1):  # Loop from t=0 to N-2
            x_t = full_trajectory[t]
            u_t = np.array(actions[j][t])
            # Calculate the nominal regressor vector h(x, u)
            x_hat_dot.append(f_nominal(x_t) + B_nominal() @ u_t)

        x_hat_dots.append(x_hat_dot)

    return x_hat_dots, train_indices, test_indices


def least_square_regression(
    x_dots: list,
    x_hat_dots: list,
    outlier_removal: bool = False,
    threshold: float = 2.5,
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
        # === Outlier Removal Implemented === #

        # 1. Define the indices of the columns we know are noisy
        #    x_dots[3,4,5] (velocity derivatives) and x_dots[6] (thrust derivative)
        noisy_indices = [3, 4, 5, 6]

        # 2. Calculate Z-scores for the ENTIRE Y_matrix
        z_scores = np.abs(stats.zscore(Y_matrix, axis=0))

        # 3. Select ONLY the Z-scores for our noisy columns
        z_scores_to_check = z_scores[:, noisy_indices]

        # 4. Create a boolean mask for all rows that are NOT outliers.
        #    A row is kept (True) only if ALL of its "noisy" dimensions
        #    have a Z-score less than 3. Outliers in "clean" mocap
        #    columns (0,1,2,7,8,9) will be ignored.
        is_not_outlier = np.all(z_scores_to_check < threshold, axis=1)

        # 5. Filter both your data and regressor matrices using this mask
        Y_matrix_clean = Y_matrix[is_not_outlier]
        A_matrix_clean = A_matrix[is_not_outlier]

        print(f"[INFO] Outlier Removal: Original data had {len(Y_matrix)} points.")
        print(
            f"  Filtered data has {len(Y_matrix_clean)} points ({len(Y_matrix) - len(Y_matrix_clean)} removed)."
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
        except np.linalg.LinAlgError as e:
            print(f"Error solving for dimension {i}: {e}")
            v_hat[i] = np.nan
            c_hat[i] = np.nan

        # plot the least squares plot for each dimension
        plt.subplot(5, 2, i + 1)
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
        # plt.xlabel("x_hat_dot", fontsize=12)
        plt.ylabel(LABELS[i], fontsize=12)
        # plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7, linewidth=1.5)
    plt.suptitle("Least Squares Regression for Each State Dimension", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"least_squares_fit.svg")
    plt.close()

    print(f"[INFO] Solved for VECTOR v and INTERCEPT c")
    print(f"  v (scaling) = {np.array_str(v_hat, precision=4, suppress_small=True)}")
    print(f"  c (intercept) = {np.array_str(c_hat, precision=4, suppress_small=True)}")
    print(f"[INFO] Total Residual: {total_residuals:.4f}")

    return v_hat, c_hat


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


def simulate(test_indices: list, v_hat: np.ndarray, c_hat: np.ndarray):
    # === Run simulation on the test trajectory === #
    print(
        f"[INFO] Simulating and Plotting Error for Test Trajectory {test_indices} ---"
    )
    error_matrices = []
    for idx in test_indices:
        # reconstruct the full trajectory
        test_traj_states = [np.array(s) for s in states[idx]] + [
            np.array(next_states[idx][-1])
        ]
        test_traj_actions = [np.array(u) for u in actions[idx]]

        # Initialize the predicted trajectory with the true initial state
        x_pred_trajectory = [test_traj_states[0]]

        # Loop for N-1 steps to generate N states
        for t in range(N - 1):
            x_t = x_pred_trajectory[-1]  # Get the last predicted state
            u_t = test_traj_actions[t]  # Get the true action

            # Predict the next state using our learned model
            x_t_plus_1 = predict_next_state(x_t, u_t, v_hat, c_hat)
            x_pred_trajectory.append(x_t_plus_1)

        # Convert lists to numpy arrays for easy subtraction
        true_matrix = np.array(test_traj_states)
        pred_matrix = np.array(x_pred_trajectory)

        # 2. Calculate the error matrix
        error_matrix = np.abs(true_matrix - pred_matrix)
        error_matrices.append(error_matrix)
    return error_matrices


def smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


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
    plt.savefig("prediction_error.svg")
    plt.close()


if __name__ == "__main__":
    states, actions, next_states, terminals = call_data()
    x_dots = compute_x_dot(states, next_states)
    x_hat_dots, train_indices, test_indices = compute_x_hat_dot(
        states, actions, ratios=[0.9, 0.1]
    )
    v_hat, c_hat = least_square_regression(
        [x_dots[i] for i in train_indices], x_hat_dots, outlier_removal=True
    )
    error_matrices = simulate(test_indices, v_hat, c_hat)
    plot_prediction_error(error_matrices)
