import heapq
import numpy as np
from world import mu_and_grad

def world_to_grid(p, origin, res):
    # origin is (xmin, ymin)
    gx = int(round((p[0] - origin[0]) / res))
    gy = int(round((p[1] - origin[1]) / res))
    return gx, gy

def grid_to_world(gx, gy, origin, res):
    return np.array([origin[0] + gx * res, origin[1] + gy * res], dtype=float)

def build_occupancy(obstacles, bounds, res, safety_margin=0.0):
    """
    occupancy[gx,gy] = True means blocked (in collision)
    blocked if mu(p) < safety_margin
    """
    xmin, xmax, ymin, ymax = bounds
    nx = int(np.floor((xmax - xmin) / res)) + 1
    ny = int(np.floor((ymax - ymin) / res)) + 1
    origin = (xmin, ymin)

    occ = np.zeros((nx, ny), dtype=bool)

    for gx in range(nx):
        for gy in range(ny):
            p = grid_to_world(gx, gy, origin, res)
            mu, _ = mu_and_grad(p, obstacles)
            if mu < safety_margin:
                occ[gx, gy] = True

    return occ, origin, nx, ny

def astar_path(obstacles, start, goal, bounds=(-2, 2, -2, 2), res=0.05, safety_margin=0.0, allow_diagonal=True):
    """
    Returns a list of world points [p0, p1, ..., pN] from start to goal.
    """
    occ, origin, nx, ny = build_occupancy(obstacles, bounds, res, safety_margin=safety_margin)

    s = world_to_grid(start, origin, res)
    g = world_to_grid(goal, origin, res)

    def in_bounds(n):
        return 0 <= n[0] < nx and 0 <= n[1] < ny

    if not in_bounds(s) or not in_bounds(g):
        raise ValueError("Start or goal out of bounds.")

    if occ[s[0], s[1]]:
        raise ValueError("Start is in collision (blocked).")
    if occ[g[0], g[1]]:
        raise ValueError("Goal is in collision (blocked).")

    # 4- or 8-connected neighbors
    if allow_diagonal:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        nbrs = [(-1,0),(1,0),(0,-1),(0,1)]

    def heuristic(a, b):
        # Euclidean in grid units
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.hypot(dx, dy)

    # A*
    open_heap = []
    heapq.heappush(open_heap, (0.0, s))

    came_from = {}
    gscore = {s: 0.0}

    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)

        if cur == g:
            # reconstruct
            path_grid = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path_grid.append(cur)
            path_grid.reverse()
            # to world
            path_world = [grid_to_world(x, y, origin, res) for (x, y) in path_grid]
            return path_world

        for dx, dy in nbrs:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not in_bounds(nxt):
                continue
            if occ[nxt[0], nxt[1]]:
                continue

            step_cost = np.hypot(dx, dy)  # 1 or sqrt(2)
            tentative = gscore[cur] + step_cost

            if nxt not in gscore or tentative < gscore[nxt]:
                came_from[nxt] = cur
                gscore[nxt] = tentative
                f = tentative + heuristic(nxt, g)
                heapq.heappush(open_heap, (f, nxt))

    raise RuntimeError("A* failed to find a path.")

def simplify_path(path, min_dist=0.15):
    """
    Downsample a dense A* path into waypoints spaced ~min_dist apart.
    Keeps first and last.
    """
    if len(path) <= 2:
        return path

    out = [path[0]]
    acc = 0.0
    for i in range(1, len(path)):
        seg = np.linalg.norm(path[i] - path[i-1])
        acc += seg
        if acc >= min_dist:
            out.append(path[i])
            acc = 0.0
    if np.linalg.norm(out[-1] - path[-1]) > 1e-9:
        out.append(path[-1])
    return out