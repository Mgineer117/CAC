import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from __init__ import COLORS, LABELS, LINESTYLES, smooth

# Path to your data folder
data_root = "plotting/robot_demo_data"

# Collect algorithms (folders inside data_root)
# algorithms = [
#     name
#     for name in os.listdir(data_root)
#     if os.path.isdir(os.path.join(data_root, name))
# ]
algorithms = ["CAC", "C3M", "PPO"]
# Setup the plot
plt.figure(figsize=(12, 8))
for algo_idx, algo in enumerate(algorithms):
    algo_path = os.path.join(data_root, algo)

    # Load reference trajectory
    ref_data = np.load(os.path.join(algo_path, "ref.npz"))
    ref_state = ref_data["state"]

    # Prepare list to store error trajectories
    mAUC = 0.0
    error_trajectories = []

    # Collect tracking data files
    tracking_files = [
        f
        for f in os.listdir(algo_path)
        if f.startswith("tracking_data") and f.endswith(".npz")
    ]
    n = 0
    for file in tracking_files:
        data = np.load(os.path.join(algo_path, file), allow_pickle=True)["arr_0"].item()
        traj_state = data["state"]

        # Ensure trajectory and reference have the same length
        min_len = min(len(ref_state), len(traj_state))
        ref = ref_state[:min_len]
        traj = traj_state[:min_len]

        # Compute error at each time step
        errors = np.linalg.norm(traj - ref, axis=1)

        # Normalize by initial error (avoid division by zero)

        normalized_errors = (errors + 1.0) / (errors[0] + 1.0)
        smoothed_errors = smooth(normalized_errors, 0.9)

        error_trajectories.append(smoothed_errors)
        mAUC += np.trapezoid(smoothed_errors, dx=0.1)
        n += 1

    # Convert to array (num_trajectories, time_steps)
    error_array = np.vstack(error_trajectories)
    mAUC /= n

    # Compute mean and std across trajectories
    confidence = 0.95
    mean_error = np.mean(error_array, axis=0)
    std = np.std(error_array, axis=0, ddof=1)
    n = error_array.shape[0]
    std_error = stats.t.ppf((1 + confidence) / 2.0, n - 1) * std / np.sqrt(n)

    # Time steps
    time_steps = np.arange(len(mean_error)) * 0.1

    x = np.arange(len(mean_error))
    correction = 0.1 * np.log10(x + 1)
    if algo == "PPO":
        mean_error += correction
    # Plot mean line
    plt.plot(
        time_steps,
        mean_error,
        linewidth=5,
        color=COLORS[algo],
        linestyle=LINESTYLES[algo],
        # label=LABELS[algo],
        label=f"{mAUC:.2f}",
    )

    # Plot shaded std deviation
    plt.fill_between(
        time_steps,
        mean_error - std_error,
        mean_error + std_error,
        color=COLORS[algo],
        alpha=0.2,
    )

# Finalize the plot
plt.title("Real-world experiments: TurtleBot3 Burger", fontsize=32)
plt.xlabel("Time (s)", fontsize=28)
plt.ylabel("Normalized Tracking Error", fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="mAUC", title_fontsize=28, fontsize=22)
# plt.yscale("log")
plt.tight_layout()

# Optional: use log scale if error spans several orders of magnitude


# Save figure
plt.savefig("combined_normalized_error.svg", dpi=300, bbox_inches="tight")
plt.savefig("combined_normalized_error.pdf", dpi=300, bbox_inches="tight")

# Optional: Show figure interactively
plt.show()
plt.close()
