import matplotlib.pyplot as plt
import numpy as np
import torch

from envs.flapper import FlapperEnv
from system_identifications.base import TIME_INTERVAL, call_mdp_data, call_policy

if __name__ == "__main__":
    # fix torch seed
    torch.manual_seed(42)

    # call test data
    test_states, test_actions, test_next_states, test_terminals, test_x_dots = (
        call_mdp_data(data_type="test")
    )

    # call policy
    policy = call_policy(algo_name="cac")

    # call env
    env = FlapperEnv()

    # Get reference trajectory
    traj_index = 0  # you can change this to test different trajectories
    xref = torch.tensor(test_states[traj_index], dtype=torch.float32)
    uref = torch.tensor(test_actions[traj_index], dtype=torch.float32)

    # --- Define Noise Parameters ---
    gamma = 0.998
    noise_scale = torch.tensor(
        [
            0.5,
            0.5,
            0.3,
            0.2,
            0.2,
            0.2,
            0.1,
            0.5,
            0.5,
            0.5,
        ],
        dtype=torch.float32,
    )

    # --- MULTI-SIMULATION SETUP ---
    num_simulations = 5  # How many different starting points to test
    all_simulated_trajectories = []

    print(f"Running {num_simulations} simulations from different initial states...")

    for i in range(num_simulations):
        print(f"  Simulation {i+1}/{num_simulations}...")

        # 1. Generate a *new* noisy initial state for this simulation
        x_initial = xref[0] + torch.abs(torch.randn_like(xref[0]) * noise_scale)

        # --- SIMULATION LOOP ---
        x_current = x_initial.unsqueeze(0)  # Shape: (1, state_dim)

        # Create a list to store this single simulation's trajectory
        simulated_trajectory_i = [x_current]

        # Loop through all time steps
        for t in range(xref.shape[0] - 1):
            x_t = simulated_trajectory_i[-1]

            # Get the target state and reference action for this time step
            xref_t = xref[t, :].unsqueeze(0)
            uref_t = uref[t, :].unsqueeze(0)

            # Get policy action
            with torch.no_grad():
                actions_t = policy(x_t, xref_t, uref_t, deterministic=True)[0]
                u_t = actions_t + uref_t
                u_t.clamp_(-1.0, 1.0)

            # Get next state dynamics
            f_x, B_x, _ = env.get_f_and_B(x_t)
            x_hat_dot = f_x + torch.matmul(B_x, u_t.unsqueeze(-1)).squeeze(-1)

            # Use device-safe tensor creation to prevent CPU/GPU errors
            v = torch.as_tensor(env.v, device=x_hat_dot.device, dtype=x_hat_dot.dtype)
            c = torch.as_tensor(env.c, device=x_hat_dot.device, dtype=x_hat_dot.dtype)

            # Use deterministic dynamics for a clean simulation
            x_dot = v * x_hat_dot + c

            # Euler integration
            x_next = x_t + x_dot * TIME_INTERVAL

            # Store the result
            simulated_trajectory_i.append(x_next)

        # Convert list of tensors to a single tensor for plotting
        x_sim_tensor_i = torch.cat(simulated_trajectory_i, dim=0)

        # Add this full trajectory to our list of all trajectories
        all_simulated_trajectories.append(x_sim_tensor_i)

    # --- END SIMULATION LOOPS ---

    # --- PLOTTING ---
    time_steps_np = (torch.arange(0, xref.shape[0]) * TIME_INTERVAL).numpy()
    plt.figure(figsize=(12, 8))

    xref_np = xref.detach().numpy()
    plt.title("Simulated Trajectory Tracking (XYZ) from Multiple Initial States")

    # --- Plot X ---
    plt.subplot(3, 1, 1)
    plt.plot(
        time_steps_np,
        xref_np[:, 0],
        label="Target X (xref)",
        color="blue",
        linewidth=2.5,
    )

    # Plot all simulated X trajectories
    for i, x_sim_tensor in enumerate(all_simulated_trajectories):
        x_sim_np = x_sim_tensor.detach().numpy()
        label = "Simulated X" if i == 0 else None  # Only add one label
        plt.plot(
            time_steps_np,
            x_sim_np[:, 0],
            label=label,
            color="orange",
            linestyle="--",
            alpha=0.7,
        )

    plt.ylabel("X Position (m)")
    plt.legend()
    plt.grid(True)

    # --- Plot Y ---
    plt.subplot(3, 1, 2)
    plt.plot(
        time_steps_np,
        xref_np[:, 1],
        label="Target Y (xref)",
        color="blue",
        linewidth=2.5,
    )

    # Plot all simulated Y trajectories
    for i, x_sim_tensor in enumerate(all_simulated_trajectories):
        x_sim_np = x_sim_tensor.detach().numpy()
        label = "Simulated Y" if i == 0 else None
        plt.plot(
            time_steps_np,
            x_sim_np[:, 1],
            label=label,
            color="orange",
            linestyle="--",
            alpha=0.7,
        )

    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)

    # --- Plot Z ---
    plt.subplot(3, 1, 3)
    plt.plot(
        time_steps_np,
        xref_np[:, 2],
        label="Target Z (xref)",
        color="blue",
        linewidth=2.5,
    )

    # Plot all simulated Z trajectories
    for i, x_sim_tensor in enumerate(all_simulated_trajectories):
        x_sim_np = x_sim_tensor.detach().numpy()
        label = "Simulated Z" if i == 0 else None
        plt.plot(
            time_steps_np,
            x_sim_np[:, 2],
            label=label,
            color="orange",
            linestyle="--",
            alpha=0.7,
        )

    plt.ylabel("Z Position (m)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    # --- Finalize ---
    plt.tight_layout()
    # Save the figure *before* showing it
    plt.savefig("Position_PID_MultiSim.svg")
    plt.show()
