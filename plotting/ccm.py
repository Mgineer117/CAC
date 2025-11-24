import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_inverse_surface():
    # --- Configuration ---
    range_limit = 3.0  # Expanded slightly to show the asymptotic tail
    num_points = 100

    # Visual Configuration
    # Same custom colormap: Dark Blue -> Grey/Blue -> Gold/Yellow
    colors = ["#003366", "#4d79ff", "#ffe066", "#ffcc00"]
    custom_cmap = LinearSegmentedColormap.from_list("BlueGold", colors)

    # ---------------------

    # 1. Generate Grid
    x = np.linspace(-range_limit, range_limit, num_points)
    y = np.linspace(-range_limit, range_limit, num_points)
    X, Y = np.meshgrid(x, y)

    # 2. Calculate Z (The Inverse Function)
    # Formula: Z = 1 / (1 + f(x,y)) where f(x,y) = x^2 + y^2
    Z = 1.0 / (1.0 + X**2 + Y**2)

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=custom_cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.9,
    )

    # 4. Clean up the look
    ax.set_axis_off()

    # Adjust view angle to show the peak clearly
    ax.view_init(elev=30, azim=45)

    # 5. Save as SVG
    output_filename = "inverse_surface.svg"
    plt.savefig(
        output_filename,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    print(f"3D Plot saved successfully as '{output_filename}'")


def plot_quadratic_surface():
    # --- Configuration ---
    range_limit = 2.0
    num_points = 100  # Higher = smoother surface

    # Visual Configuration
    # Creating a custom colormap: Dark Blue -> Grey/Blue -> Gold/Yellow
    colors = ["#003366", "#4d79ff", "#ffe066", "#ffcc00"]
    custom_cmap = LinearSegmentedColormap.from_list("BlueGold", colors)

    # ---------------------

    # 1. Generate Grid
    x = np.linspace(-range_limit, range_limit, num_points)
    y = np.linspace(-range_limit, range_limit, num_points)
    X, Y = np.meshgrid(x, y)

    # 2. Calculate Z (Quadratic / Paraboloid function)
    # Z = X^2 + Y^2
    Z = X**2 + Y**2

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    # rstride/cstride controls the "grid" density appearance
    # linewidth=0 removes the mesh lines for a smooth look (or add small width for grid)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=custom_cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.9,
    )

    # 4. Clean up the look (Remove axes, ticks, panes)
    ax.set_axis_off()

    # Optional: Adjust view angle to match the reference image
    ax.view_init(elev=30, azim=45)

    # 5. Save as SVG
    output_filename = "quadratic_surface.svg"
    # transparent=True ensures the background doesn't have a white box
    plt.savefig(
        output_filename,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    print(f"3D Plot saved successfully as '{output_filename}'")

    # plt.show()


if __name__ == "__main__":
    plot_quadratic_surface()
    plot_inverse_surface()
