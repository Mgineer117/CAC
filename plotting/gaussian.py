import matplotlib.pyplot as plt
import numpy as np


def plot_gaussian():
    # --- Configuration: User Choices ---
    # Define your X-axis range here
    x_min = -10.0
    x_max = 10.0
    num_points = 1000  # Higher number = smoother curve

    # Define the Gaussian parameters
    mu = 0.0  # Mean (center of the peak)
    sigma = 2.0  # Standard deviation (width/spread)

    # Label for the X-axis
    x_axis_label = "Control Input"

    # # Visual Configuration
    # fill_color = "cornflowerblue"  # Change this to any color (e.g., 'red', '#FF5733')
    # line_color = "royalblue"  # Change the outline color
    # fill_alpha = 0.3  # Transparency (0.0 is invisible, 1.0 is solid)
    # -----------------------------------
    # Visual Configuration
    fill_color = "orange"  # Change this to any color (e.g., 'red', '#FF5733')
    line_color = "darkorange"  # Change the outline color
    fill_alpha = 0.3  # Transparency (0.0 is invisible, 1.0 is solid)

    # 1. Generate X data points based on your choice
    x = np.linspace(x_min, x_max, num_points)

    # 2. Calculate Y data (Probability Density Function)
    # Formula: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    term1 = 1 / (sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y = term1 * term2

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Plot the line
    plt.plot(
        x,
        y,
        color=line_color,
        linewidth=2.5,
        label=f"Gaussian ($\mu={mu}, \sigma={sigma}$)",
    )

    # Fill the area under the curve (optional, makes it look nice)
    plt.fill_between(x, y, color=fill_color, alpha=fill_alpha)

    # 4. Customization
    # plt.title("Gaussian Distribution", fontsize=16)
    plt.xlabel(x_axis_label, fontsize=22)
    plt.ylabel("Probability Density", fontsize=22)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(fontsize=12)

    # Remove tick labels (the numbers on the axes)
    plt.xticks([])
    plt.yticks([])

    # Set x-axis limits explicitly to user choice
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0)

    # 5. Save as SVG
    output_filename = "gaussian_distribution.svg"
    plt.savefig(output_filename, format="svg")
    print(f"Plot saved successfully as '{output_filename}'")

    # Show the plot (optional if running locally)
    # plt.show()


if __name__ == "__main__":
    plot_gaussian()
