import os

import matplotlib.pyplot as plt
import numpy as np

# Path to your data folder
data_root = "plotting/robot_demo_data"

# Collect algorithms (folders inside data_root)
algorithms = [
    name
    for name in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, name))
]

for algo in algorithms:
    algo_path = os.path.join(data_root, algo)

    # Load reference trajectory
    ref_data = np.load(os.path.join(algo_path, "ref.npz"))
    ref_x = ref_data["state"][:, 0]
    ref_y = ref_data["state"][:, 1]

    # Start plotting
    plt.figure(figsize=(8, 6))
    plt.plot(
        ref_x, ref_y, "k--", linewidth=3, label="Reference (target)"
    )  # black dotted line

    # Mark start and end of reference
    plt.scatter(
        ref_x[0], ref_y[0], color="black", marker="o", s=80, label="Reference Start"
    )
    plt.scatter(
        ref_x[-1], ref_y[-1], color="black", marker="x", s=80, label="Reference End"
    )

    # Assign unique colors for tracking data
    tracking_files = [
        f
        for f in os.listdir(algo_path)
        if f.startswith("tracking_data") and f.endswith(".npz")
    ]
    colors = plt.cm.get_cmap(
        "tab20b", len(tracking_files)
    )  # use tab20 colormap for better variety

    for idx, file in enumerate(tracking_files):
        data = np.load(os.path.join(algo_path, file), allow_pickle=True)["arr_0"].item()
        pos_x = data["state"][:, 0]
        pos_y = data["state"][:, 1]

        label = (
            "Trajectories" if idx == 0 else None
        )  # ✅ only first line gets legend label

        plt.plot(pos_x, pos_y, color=colors(idx), linewidth=3, label=label)
        # plt.plot(pos_x, pos_y, color="steelblue", linewidth=2, label=label)

        # Start and end points (optional, keep or remove from legend)
        plt.scatter(pos_x[0], pos_y[0], color="black", marker="o", s=60)
        plt.scatter(pos_x[-1], pos_y[-1], color="black", marker="x", s=60)

    # Title and labels with larger font
    if algo == "CAC":
        plt.title(f"Trajectories for {algo} (ours)", fontsize=32)
    else:
        plt.title(f"Trajectories for {algo}", fontsize=32)
    plt.xlabel("Position X", fontsize=28)
    plt.ylabel("Position Y", fontsize=28)

    # Reduce number of ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=5)

    plt.grid(True, linestyle="--", alpha=0.6)

    # Place legend outside the plot
    # plt.legend(fontsize=34, loc='lower center', bbox_to_anchor=(0.5, 1.02),
    #        ncol=5, frameon=True, fancybox=True)

    plt.tight_layout()

    # Optional: Save figure as high-quality image
    plt.savefig(f"{algo}_trajectory.svg", dpi=300)
    # plt.show()
