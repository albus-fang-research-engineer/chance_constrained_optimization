import matplotlib.pyplot as plt
import numpy as np
ROBOT_RADIUS = 0.105
def draw_world(obstacles):
    for (xmin, xmax, ymin, ymax) in obstacles:
        plt.fill(
            [xmin, xmax, xmax, xmin],
            [ymin, ymin, ymax, ymax],
            alpha=0.3
        )
def draw_robot(p, color):
    circle = plt.Circle(p, ROBOT_RADIUS, fill=False, color=color)
    plt.gca().add_patch(circle)


def plot_traj(traj, color, label):
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], "-o", color=color, label=label)

    for p in traj:
        draw_robot(p, color)

# def plot_traj(traj):
#     traj = np.array(traj)
#     plt.plot(traj[:, 0], traj[:, 1], "-o")