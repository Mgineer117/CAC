import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Optional: Set a style for cleaner academic plots
sns.set_style("white")


def generate_distinct_centered_data(mode, n_samples=1000):
    """
    Generates data (x, u) centered in the plot using specific sampling strategies.
    """
    # Plot center coordinates
    c_x, c_u = 0.5, 0.5

    if mode == "real_world":
        # --- 1. Real-World: TINY, Centered Blob (Gaussian Mixture) ---
        cov1 = [[0.008, 0.004], [0.004, 0.008]]
        data1 = np.random.multivariate_normal(
            [c_x - 0.02, c_u - 0.02], cov1, int(n_samples / 2)
        )

        cov2 = [[0.002, -0.001], [-0.001, 0.002]]
        data2 = np.random.multivariate_normal(
            [c_x + 0.02, c_u + 0.02], cov2, int(n_samples / 2)
        )

        data = np.vstack((data1, data2))
        x = data[:, 0]
        u = data[:, 1]
        return x, u

    elif mode == "control_focused":
        # --- 2. Control-Focused: Wide coverage ---
        width = 0.9
        height = 0.9
        q = 3  # n_control_per_x

        x_min = c_x - (width / 2)
        u_min = c_u - (height / 2)

        batch_size = int(n_samples / q)

        x_base = np.random.uniform(x_min, x_min + width, batch_size)
        u_base = np.random.uniform(u_min, u_min + height, batch_size)

        x = np.concatenate([x_base] * q, axis=0)
        u = np.concatenate(
            [u_base[np.random.permutation(len(u_base))] for _ in range(q)],
            axis=0,
        )
        return x, u

    elif mode == "state_focused":
        # --- 3. State-Focused: Baseline ---
        width = 0.9
        height = 0.9
        q = 1

        x_min = c_x - (width / 2)
        u_min = c_u - (height / 2)

        batch_size = int(0.3 * n_samples / q)

        x_base = np.random.uniform(x_min, x_min + width, batch_size)
        u_base = np.random.uniform(u_min, u_min + height, batch_size)

        x = np.concatenate([x_base] * q, axis=0)
        u = np.concatenate(
            [u_base[np.random.permutation(len(u_base))] for _ in range(q)],
            axis=0,
        )
        return x, u


def draw_final_plot(ax, x_data, u_data, title):
    """
    Draws the plot.
    heatmap (KDE) is active.
    Scatter plot is provided as a comment for alternative visualization.
    """
    PLOT_MAX = 1.0

    # ---------------------------------------------------------
    # OPTION 1: Scatter Plot (Uncomment to use instead of Heatmap)
    # ---------------------------------------------------------
    # sns.scatterplot(
    #     x=x_data, y=u_data, ax=ax,
    #     s=10, color="#1f77b4", alpha=0.3, edgecolor=None
    # )

    # ---------------------------------------------------------
    # OPTION 2: Density Heatmap (Normalized visually)
    # ---------------------------------------------------------
    # bw_adjust=0.5: Sharper edges to fit the uniform shape
    # thresh=0.05: Cuts off the very low density tails for a cleaner look
    sns.kdeplot(
        x=x_data,
        y=u_data,
        ax=ax,
        fill=True,
        cmap="Blues",
        alpha=0.9,
        levels=10,  # Increased levels for smoother gradient
        thresh=0.05,
        bw_adjust=0.5,
        zorder=2,
    )

    # --- Draw Contour Outline (Optional) ---
    sns.kdeplot(
        x=x_data,
        y=u_data,
        ax=ax,
        fill=False,
        color="gray",
        alpha=0.3,
        linewidths=1.5,
        levels=[0.05],  # Match the thresh of the fill
        bw_adjust=0.5,
        zorder=3,
    )

    # # --- Add Colorbar (Normalized) ---
    # # Only adding to the last plot, but treating it as "Relative Density"
    # if title == "Control-focused":
    #     # Create colorbar based on the contour set
    #     mappable = ax.collections[0]
    #     cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    #     # NORMALIZATION FIX:
    #     # Instead of showing raw PDF values (which differ wildly between plots),
    #     # we set the ticks to simply show 0 (Min) and 1 (Max/Peak).
    #     cbar.set_ticks([mappable.get_array().min(), mappable.get_array().max()])
    #     cbar.set_ticklabels(["Low", "High"])
    #     # 3. ENLARGE THE LABELS HERE:
    #     cbar.ax.tick_params(labelsize=24)
    #     cbar.set_label("Relative Density", fontsize=28, weight="bold")

    # --- Styling ---
    ax.set_xlim(0, PLOT_MAX)
    ax.set_ylim(0, PLOT_MAX)

    # Custom thick axes arrows
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Arrowheads
    ax.plot(
        1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=10
    )
    ax.plot(
        0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=10
    )

    # Labels
    ax.set_xlabel(r"$|\mathcal{X}|$", fontsize=30, loc="right", weight="bold")
    ax.set_ylabel(r"$|\mathcal{U}|$", fontsize=30, loc="top", rotation=0, weight="bold")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=24, pad=20, weight="bold")


# --- Main Execution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 0. Baseline (State-Focused)
x3, u3 = generate_distinct_centered_data("state_focused")
draw_final_plot(axes[0], x3, u3, "Baseline")

# 1. Real-World
x1, u1 = generate_distinct_centered_data("real_world")
draw_final_plot(axes[1], x1, u1, "Real-world-focused")

# 2. Control-Focused
x2, u2 = generate_distinct_centered_data("control_focused")
draw_final_plot(axes[2], x2, u2, "Control-focused")

plt.tight_layout()
plt.savefig("synthetic_data_final.svg", dpi=300)
