import matplotlib.pyplot as plt
import torch

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

    noise_scale = torch.tensor(
        [
            0.5,
            0.5,
            0.5,
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

    # perturb the test data with noise
    traj_index = 0  # you can change this to test different trajectories
    xref = torch.tensor(test_states[traj_index], dtype=torch.float32)
    next_xref = torch.tensor(test_next_states[traj_index], dtype=torch.float32)

    # expand it to the same size as test data
    gamma = 0.995
    discounts = torch.pow(gamma, torch.arange(next_xref.shape[0], dtype=torch.float32))
    noise_scale = noise_scale.expand_as(xref) * discounts.unsqueeze(1)
    print("Noise scale per state dimension:", noise_scale.numpy())

    x = xref + torch.randn_like(xref) * noise_scale
    uref = torch.tensor(test_actions[traj_index], dtype=torch.float32)

    with torch.no_grad():
        actions = policy(x, xref, uref, deterministic=True)[0]
        u = actions + uref
        # u = uref
        # u = torch.zeros_like(actions)

        u.clamp_(-1.0, 1.0)
    # --- ADDED UREF STATS ---
    # Convert uref to numpy for mean/std calculation
    print("--- Reference Action (uref) Statistics ---")
    print(
        f"Thrust Rate: mean={torch.mean(uref[:, 0]):.4f}, std={torch.std(uref[:, 0]):.4f}"
    )
    print(
        f"Roll Rate:   mean={torch.mean(uref[:, 1]):.4f}, std={torch.std(uref[:, 1]):.4f}"
    )
    print(
        f"Pitch Rate:  mean={torch.mean(uref[:, 2]):.4f}, std={torch.std(uref[:, 2]):.4f}"
    )
    print(
        f"Yaw Rate:    mean={torch.mean(uref[:, 3]):.4f}, std={torch.std(uref[:, 3]):.4f}"
    )
    print("------------------------------------------")
    # --- ADDED UREF STATS ---
    # Convert uref to numpy for mean/std calculation
    print("--- Reference Action (actions) Statistics ---")
    print(
        f"Thrust Rate: mean={torch.mean(actions[:, 0]):.4f}, std={torch.std(actions[:, 0]):.4f}"
    )
    print(
        f"Roll Rate:   mean={torch.mean(actions[:, 1]):.4f}, std={torch.std(actions[:, 1]):.4f}"
    )
    print(
        f"Pitch Rate:  mean={torch.mean(actions[:, 2]):.4f}, std={torch.std(actions[:, 2]):.4f}"
    )
    print(
        f"Yaw Rate:    mean={torch.mean(actions[:, 3]):.4f}, std={torch.std(actions[:, 3]):.4f}"
    )
    print("------------------------------------------")

    # clip actions to be within action limits
    # actions = torch.clamp(actions, min=MIN_THRUST, max=MAX_THRUST)

    # compute the next state
    thrust = x[:, 6] + u[:, 0] * TIME_INTERVAL
    roll = x[:, 7] + u[:, 1] * TIME_INTERVAL
    pitch = x[:, 8] + u[:, 2] * TIME_INTERVAL
    yaw = x[:, 9] + u[:, 3] * TIME_INTERVAL

    # plot the results
    time_steps = torch.arange(0, x.shape[0]) * TIME_INTERVAL
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.title(f"RMSE {torch.abs(next_xref[:, 6] - thrust).mean().item():.4f}")
    plt.plot(time_steps, thrust.numpy(), label="Thrust")
    plt.plot(time_steps, next_xref[:, 6].numpy(), label="Next Thrust", linestyle="--")
    plt.ylabel(f"Thrust (N)")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.title(f"RMSE {torch.abs(next_xref[:, 7] - roll).mean().item():.4f}")
    plt.plot(time_steps, roll.numpy(), label="Roll")
    plt.plot(time_steps, next_xref[:, 7].numpy(), label="Next Roll", linestyle="--")
    plt.ylabel(f"Roll (rad)")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.title(f"RMSE {torch.abs(next_xref[:, 8] - pitch).mean().item():.4f}")
    plt.plot(time_steps, pitch.numpy(), label="Pitch")
    plt.plot(time_steps, next_xref[:, 8].numpy(), label="Next Pitch", linestyle="--")
    plt.ylabel(f"Pitch (rad)")
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.title(f"RMSE {torch.abs(next_xref[:, 9] - yaw).mean().item():.4f}")
    plt.plot(time_steps, yaw.numpy(), label="Yaw")
    plt.plot(time_steps, next_xref[:, 9].numpy(), label="Next Yaw", linestyle="--")
    plt.ylabel(f"Yaw (rad)")
    plt.xlabel("Time (s)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("simulator_control.svg")
