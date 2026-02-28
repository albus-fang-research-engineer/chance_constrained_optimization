import numpy as np

ROBOT_RADIUS = 0.105
# SIGMA = 0.02

def make_3_box_obstacles():
    return [
        (-1.4, -0.6, -0.2,  0.6),
        ( 0.3,  1.2, -1.2, -0.4),
        (-0.2,  0.6,  0.5,  1.4),
    ]


def sample_obstacle_points(boxes, num_points=2000):
    pts = []

    for xmin, xmax, ymin, ymax in boxes:
        n = num_points // len(boxes)

        xs = np.random.uniform(xmin, xmax, n)
        ys = np.random.uniform(ymin, ymax, n)

        pts.append(np.stack([xs, ys], axis=1))

    return np.concatenate(pts, axis=0)

def sdf_point_to_box(p, box):
    xmin, xmax, ymin, ymax = box

    dx = max(xmin - p[0], 0, p[0] - xmax)
    dy = max(ymin - p[1], 0, p[1] - ymax)

    outside_dist = np.hypot(dx, dy)

    if dx == 0 and dy == 0:
        # inside → negative distance
        return -min(
            p[0] - xmin,
            xmax - p[0],
            p[1] - ymin,
            ymax - p[1],
        )

    return outside_dist


def grad_sdf_point_to_box(p, box):
    xmin, xmax, ymin, ymax = box

    px = np.clip(p[0], xmin, xmax)
    py = np.clip(p[1], ymin, ymax)

    closest = np.array([px, py])
    v = p - closest
    norm = np.linalg.norm(v)

    if norm < 1e-8:
        return np.zeros(2)

    return v / norm


def mu_and_grad(p, obstacles):
    dists = []
    grads = []

    for box in obstacles:
        d = sdf_point_to_box(p, box)
        g = grad_sdf_point_to_box(p, box)
        dists.append(d)
        grads.append(g)

    i = np.argmin(dists)

    mu = dists[i] - ROBOT_RADIUS
    grad = grads[i]

    return mu, grad