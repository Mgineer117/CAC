import json
import os

import matplotlib.pyplot as plt
import numpy as np
from base import (
    BASE_DIR,
    TIME_INTERVAL,
    N,
    compose_action,
    compose_state,
    compute_dynamics,
)


def read_flight_data(file_path):
    """Read and return flight data from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)

    # check their length is over 600
    if len(data["pose"]) < N:
        raise ValueError(
            f"Flight data in {file_path} is too short in length {len(data['pose'])} < {N}."
        )
    return data


if __name__ == "__main__":
    # make it as a usable MDP dataset
    mdp_data = {
        "states": [],
        "actions": [],
        "next_states": [],
        "dynamics": [],
        "terminals": [],
    }

    # given data_dir, find all names of json files in a list

    data_dir = BASE_DIR + "data/raw_data/train/"

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    for file_name in json_files:
        flight_data = read_flight_data(os.path.join(data_dir, file_name))
        states, actions, next_states, dynamics, terminals = [], [], [], [], []

        for i in range(N):
            state = compose_state(flight_data, i)
            next_state = compose_state(flight_data, i + 1)
            action = compose_action(flight_data, i)
            terminal = False if i < N - 1 else True

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            terminals.append(terminal)

        dynamics = compute_dynamics(states + [next_states[-1]])

        mdp_data["states"].append(states)
        mdp_data["actions"].append(actions)
        mdp_data["next_states"].append(next_states)
        mdp_data["dynamics"].append(dynamics)
        mdp_data["terminals"].append(terminals)

    # check they are in same length
    for i in range(len(mdp_data["states"])):
        assert (
            len(mdp_data["states"][i])
            == len(mdp_data["actions"][i])
            == len(mdp_data["next_states"][i])
            == len(mdp_data["terminals"][i])
            == len(mdp_data["dynamics"][i])
            == N
        )

    # Save the MDP dataset to a JSON file
    with open(f"{BASE_DIR}/data/mdp_flight_data.json", "w") as outfile:
        json.dump(mdp_data, outfile, indent=4)

    # Plot all trajectories of state variables
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
        sharex=True,  # constrained_layout=True
    )
    labels = [
        r"x ($m$)",
        r"y ($m$)",
        r"z ($m$)",
        r"vx ($m/s$)",
        r"vy ($m/s$)",
        r"vz ($m/s$)",
        r"thrust ($m/s^2$)",
        r"roll ($rad$)",
        r"pitch ($rad$)",
        r"yaw ($rad$)",
    ]
    for i in range(len(mdp_data["states"])):
        states = np.array(mdp_data["states"][i])
        time_steps = np.arange(states.shape[0]) * TIME_INTERVAL

        for j in range(states.shape[1]):
            plt.subplot(5, 2, j + 1)
            plt.plot(time_steps, states[:, j], linewidth=2.5, alpha=0.9)
            plt.ylabel(labels[j], fontsize=14)
            plt.grid(True, linestyle=":", alpha=0.7, linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/figs/state_variables_plot.svg")
    plt.close()
