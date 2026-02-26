import numpy as np


def rollout_nominal(start, path):
    traj = [start.copy()]
    for wp in path:
        traj.append(wp.copy())
    return traj


def rollout_optimized(start, path, obstacles, solver):
    p = start.copy()
    traj = [p.copy()]

    for wp in path:
        p = solver(p, wp, obstacles)
        traj.append(p.copy())

    return traj