import numpy as np
import matplotlib.pyplot as plt

from world import make_3_box_obstacles
from optimizer import solve_step
from rollout import rollout_nominal, rollout_optimized
from visualize import draw_world, plot_traj

obstacles = make_3_box_obstacles()

start = np.array([-1.8, -1.8])

path = [
    np.array([-1.0, -1.0]),
    np.array([0.0, -0.8]),
    np.array([0.8, 0.0]),
    np.array([1.5, 1.5]),
]

nominal = rollout_nominal(start, path)
optimized = rollout_optimized(start, path, obstacles, solve_step)

plt.figure(figsize=(7, 7))

draw_world(obstacles)

plot_traj(nominal, "red", "Nominal path")
plot_traj(optimized, "blue", "Chance-constrained path")

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.gca().set_aspect("equal")
plt.legend()
plt.title("Nominal vs Chance-Constrained Trajectory")

plt.show()