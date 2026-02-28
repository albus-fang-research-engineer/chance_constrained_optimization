import numpy as np
import matplotlib.pyplot as plt

from world import make_3_box_obstacles, sample_obstacle_points
from optimizer import solve_step
from rollout import rollout_optimized_plot
from visualize import draw_world, plot_traj, plot_mu_sigma_per_waypoint
from astar import astar_path, simplify_path
from load_njsdf.inference import load_sdf_2d_model
def main():
    obstacles = make_3_box_obstacles()
    obstacle_points = sample_obstacle_points(obstacles, num_points=2000)
    model, device = load_sdf_2d_model()
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
    # optimized = rollout_optimized(start, nominal[1:], obstacle_points, solve_step, model, device)
    optimized, mus, sigmas = rollout_optimized_plot(start, nominal[1:], obstacle_points, solve_step, model, device)
    
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
    plot_mu_sigma_per_waypoint(optimized, mus, sigmas)
    # plt.show()

if __name__ == "__main__":
    main()