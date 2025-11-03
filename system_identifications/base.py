import numpy as np

BASE_DIR = "system_identifications/"
MIN_THRUST, MAX_THRUST = 10001, 60000
TIME_INTERVAL = 0.05  # seconds
N = 600  # number of time steps


def FD_second_order(points: list, method: str = "central") -> list:
    if method == "forward":
        return (-3 * points[0] + 4 * points[1] - 1 * points[2]) / 2 * TIME_INTERVAL
    elif method == "backward":
        return (3 * points[-1] - 4 * points[-2] + 1 * points[-3]) / 2 * TIME_INTERVAL
    elif method == "central":
        return (points[2] - points[0]) / (2 * TIME_INTERVAL)
    else:
        raise ValueError("Invalid method for finite difference.")


def compute_dynamics(states: list) -> list:
    # === Generate x_dot using finite difference === #
    # x_dots = []
    # for i in range(len(states)):
    x_dot_trajectory = []

    try:
        # The full trajectory is the list of states + the final "next_state"
        full_trajectory = [np.array(s) for s in states]
    except IndexError:
        # Handle cases where a trajectory might be empty
        raise ValueError(f"Trajectory is empty or malformed.")

    # 2. Validate the assumption that N=600 + 1
    if len(full_trajectory) != N + 1:
        raise ValueError(
            f"Trajectory length is {len(full_trajectory)}, but we assumed N={N}."
        )

    # First point (t=0): Use second-order FORWARD difference
    x_dot_0 = FD_second_order(
        [full_trajectory[0], full_trajectory[1], full_trajectory[2]],
        method="forward",
    )
    x_dot_trajectory.append(x_dot_0.tolist())

    # Middle points (t=1 to N-2, i.e., 1 to 297): Use second-order CENTRAL difference
    for t in range(1, N):
        x_dot_t = (full_trajectory[t + 1] - full_trajectory[t - 1]) / (
            2 * TIME_INTERVAL
        )
        x_dot_trajectory.append(x_dot_t.tolist())
    return x_dot_trajectory
    #     print(len(x_dot_trajectory))
    #     x_dots.append(x_dot_trajectory)

    # return x_dots


def compose_state(flight_data: dict, index: int) -> list:
    """Compose a state vector from flight data at a given index."""
    # Velocity from using Kalman filter is too noisy so we resort to FD
    # vel = flight_data["vel"][index]
    # vel["stateEstimate.vx"],  # x velocity
    # vel["stateEstimate.vy"],  # y velocity
    # vel["stateEstimate.vz"],  # z velocity

    pose = flight_data["pose"][index]
    cmd = flight_data["controller_cmd"][index]

    if index == 0:
        current_pos = np.array([pose[0], pose[1], pose[2]])
        next_pos_1 = np.array(
            [
                flight_data["pose"][index + 1][0],
                flight_data["pose"][index + 1][1],
                flight_data["pose"][index + 1][2],
            ]
        )
        next_pos_2 = np.array(
            [
                flight_data["pose"][index + 2][0],
                flight_data["pose"][index + 2][1],
                flight_data["pose"][index + 2][2],
            ]
        )

        vx = FD_second_order(
            [current_pos[0], next_pos_1[0], next_pos_2[0]], method="forward"
        )
        vy = FD_second_order(
            [current_pos[1], next_pos_1[1], next_pos_2[1]], method="forward"
        )
        vz = FD_second_order(
            [current_pos[2], next_pos_1[2], next_pos_2[2]], method="forward"
        )
    else:
        before_pos_1 = np.array(
            [
                flight_data["pose"][index - 1][0],
                flight_data["pose"][index - 1][1],
                flight_data["pose"][index - 1][2],
            ]
        )
        next_pos_1 = np.array(
            [
                flight_data["pose"][index + 1][0],
                flight_data["pose"][index + 1][1],
                flight_data["pose"][index + 1][2],
            ]
        )

        vx = FD_second_order([before_pos_1[0], None, next_pos_1[0]], method="central")
        vy = FD_second_order([before_pos_1[1], None, next_pos_1[1]], method="central")
        vz = FD_second_order([before_pos_1[2], None, next_pos_1[2]], method="central")

    state = [
        pose[0],  # x position
        pose[1],  # y position
        pose[2],  # z position
        vx,  # x velocity
        vy,  # y velocity
        vz,  # z velocity
        cmd["controller.cmd_thrust"] / (MAX_THRUST - MIN_THRUST),  # normalized thrust
        pose[3],  # roll
        pose[4],  # pitch
        pose[5],  # yaw
    ]
    return state


def compose_action(flight_data: dict, index: int) -> list:
    """Compose an action vector from flight data at a given index."""
    # THIS IS FOR PID DESIRED RATE CONTROLLER which is too drastic
    # attitude_list = flight_data["controller_attitude_rate"][index]
    # attitude_list["controller.rollRate"],  # roll command
    # attitude_list["controller.pitchRate"],  # pitch command
    # attitude_list["controller.yawRate"],  # yaw command

    # === action = [thrust_rate, roll_rate, pitch_rate, yaw_rate] === #
    thrust_list = flight_data["controller_cmd"]
    attitude_list = flight_data["pose"]

    # thrust dynamics using second-order finite difference
    if index == 0:
        current_thrust = thrust_list[index]["controller.cmd_thrust"] / (
            MAX_THRUST - MIN_THRUST
        )
        next_thrust_1 = thrust_list[index + 1]["controller.cmd_thrust"] / (
            MAX_THRUST - MIN_THRUST
        )
        next_thrust_2 = thrust_list[index + 2]["controller.cmd_thrust"] / (
            MAX_THRUST - MIN_THRUST
        )

        current_angle = np.array(attitude_list[index][3:6])
        next_angle_1 = np.array(attitude_list[index + 1][3:6])
        next_angle_2 = np.array(attitude_list[index + 2][3:6])

        thrust_rate = FD_second_order(
            [current_thrust, next_thrust_1, next_thrust_2], method="forward"
        )
        roll_rate = FD_second_order(
            [current_angle[0], next_angle_1[0], next_angle_2[0]], method="forward"
        )
        pitch_rate = FD_second_order(
            [current_angle[1], next_angle_1[1], next_angle_2[1]], method="forward"
        )
        yaw_rate = FD_second_order(
            [current_angle[2], next_angle_1[2], next_angle_2[2]], method="forward"
        )
    else:
        before_thrust_1 = thrust_list[index - 1]["controller.cmd_thrust"] / (
            MAX_THRUST - MIN_THRUST
        )
        next_thrust_1 = thrust_list[index + 1]["controller.cmd_thrust"] / (
            MAX_THRUST - MIN_THRUST
        )

        before_angle_1 = np.array(attitude_list[index - 1][3:6])
        next_angle_1 = np.array(attitude_list[index + 1][3:6])

        thrust_rate = FD_second_order(
            [before_thrust_1, None, next_thrust_1], method="central"
        )
        roll_rate = FD_second_order(
            [before_angle_1[0], None, next_angle_1[0]], method="central"
        )
        pitch_rate = FD_second_order(
            [before_angle_1[1], None, next_angle_1[1]], method="central"
        )
        yaw_rate = FD_second_order(
            [before_angle_1[2], None, next_angle_1[2]], method="central"
        )

    action = [
        thrust_rate,  # thrust command
        roll_rate,  # roll command
        pitch_rate,  # pitch command
        yaw_rate,  # yaw command
    ]
    return action
