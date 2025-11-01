import json
import os

import matplotlib.pyplot as plt
import numpy as np

MIN_THRUST, MAX_THRUST = 10001, 60000
TIME_INTERVAL = 0.1  # seconds


def compose_state(flight_data: dict, index: int) -> list:
    """Compose a state vector from flight data at a given index."""
    pose = flight_data["pose"][index]
    vel = flight_data["vel"][index]
    cmd = flight_data["controller_cmd"][index]

    state = [
        pose[0],  # x position
        pose[1],  # y position
        pose[2],  # z position
        vel["stateEstimate.vx"],  # x velocity
        vel["stateEstimate.vy"],  # y velocity
        vel["stateEstimate.vz"],  # z velocity
        cmd["controller.cmd_thrust"]
        / (MAX_THRUST - MIN_THRUST)
        * 100,  # normalized thrust
        pose[3],  # roll
        pose[4],  # pitch
        pose[5],  # yaw
    ]
    return state


def compose_action(
    flight_data: dict, index: int, states: list, next_states: list
) -> list:
    """Compose an action vector from flight data at a given index."""
    # action = [roll_rate, pitch_rate, yaw_rate, thrust_rate]
    attitude = flight_data["controller_attitude_rate"][index]

    action = [
        (next_states[-1] - states[-1]) / TIME_INTERVAL,  # thrust command
        attitude["controller.rollRate"],  # roll command
        attitude["controller.pitchRate"],  # pitch command
        attitude["controller.yawRate"],  # yaw command
    ]

    # when applying this action to the environment,
    # action[-1] -> (action[-1] * TIME_INTERVAL + states[-1])
    # implement bound checker [0, 100].
    return action


def read_flight_data(file_path):
    """Read and return flight data from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)

    # check their length is over 300
    if len(data["pose"]) < 299:
        raise ValueError(
            f"Flight data in {file_path} is too short in length {len(data['pose'])} < 299."
        )
    return data


if __name__ == "__main__":
    # make it as a usable MDP dataset
    mdp_data = {
        "states": [],
        "actions": [],
        "next_states": [],
        "terminals": [],
    }

    # given data_dir, find all names of json files in a list
    data_dir = "data"
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    for file_name in json_files:
        flight_data = read_flight_data(os.path.join(data_dir, file_name))
        states, actions, next_states, terminals = [], [], [], []

        for i in range(len(flight_data["pose"]) - 1):
            state = compose_state(flight_data, i)
            next_state = compose_state(flight_data, i + 1)
            action = compose_action(flight_data, i, state, next_state)
            terminal = False if i < len(flight_data["pose"]) - 2 else True

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            terminals.append(terminal)

        mdp_data["states"].append(states)
        mdp_data["actions"].append(actions)
        mdp_data["next_states"].append(next_states)
        mdp_data["terminals"].append(terminals)

    # Save the MDP dataset to a JSON file
    with open("mdp_flight_data.json", "w") as outfile:
        json.dump(mdp_data, outfile, indent=4)

    # Plot all trajectories of state variables
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
        sharex=True,  # constrained_layout=True
    )
    labels = [
        "x (m)",
        "y (m)",
        "z (m)",
        "vx (m/s)",
        "vy (m/s)",
        "vz (m/s)",
        "thrust (%)",
        "roll (rad)",
        "pitch (rad)",
        "yaw (rad)",
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
    plt.savefig("state_variables_plot.svg")
    plt.close()
