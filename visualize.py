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

def plot_mu_sigma_per_waypoint(traj, mus, sigmas):

    n = len(mus)

    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]

        p = traj[i]

        circle = plt.Circle(p, ROBOT_RADIUS, fill=False)
        ax.add_patch(circle)

        ax.set_title(f"k={i}\nμ={mus[i]:.3f}, σ={sigmas[i]:.3f}")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("mu_sigma_per_waypoint.png")
    plt.show()
    # for p in traj:
    #     draw_robot(p, color)

# def plot_traj(traj):
#     traj = np.array(traj)
#     plt.plot(traj[:, 0], traj[:, 1], "-o")