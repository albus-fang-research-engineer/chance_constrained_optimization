import numpy as np
import matplotlib.pyplot as plt

from world import make_3_box_obstacles
from optimizer import solve_step
from rollout import rollout_optimized
from visualize import draw_world, plot_traj
from astar import astar_path, simplify_path

def main():
    obstacles = make_3_box_obstacles()

    start = np.array([-1.8, -0.0], dtype=float)
    goal  = np.array([ 1.5,  -0.5], dtype=float)

    # A* nominal (collision-free)
    dense_nominal = astar_path(
        obstacles=obstacles,
        start=start,
        goal=goal,
        bounds=(-2, 2, -2, 2),
        res=0.05,              # grid resolution (smaller = better, slower)
        safety_margin=0.00,    # require mu >= 0 (already accounts for robot radius)
        allow_diagonal=True
    )

    # Convert dense A* path to fewer waypoints for the optimizer
    nominal = simplify_path(dense_nominal, min_dist=0.10)

    # Optimized: track A* waypoints with chance constraint
    optimized = rollout_optimized(start, nominal[1:], obstacles, solve_step)

    # Plot
    plt.figure(figsize=(7, 7))
    draw_world(obstacles)
    plot_traj(nominal, "red", "Nominal (A*)")
    plot_traj(optimized, "blue", "Optimized (chance-constrained)")

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("A* nominal vs Chance-Constrained Optimization")
    plt.savefig("demo.png")
    plt.show()

if __name__ == "__main__":
    main()