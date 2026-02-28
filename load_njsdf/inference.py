import numpy as np
import torch
RADIUS = 0.105
def predict_mu_var(model, x):
    pred = model(x)
    mu, logvar = torch.chunk(pred, 2, dim=-1)
    logvar = torch.clamp(logvar, -20.0, 10.0)
    return mu.squeeze(-1), torch.exp(logvar).squeeze(-1)

def mu_sigma_grad_nn(robot_xy, obstacle_points, model, device):
    """
    robot_xy: (2,)
    obstacle_points: (N,2)

    returns:
        mu      -> scalar (closest mean distance - radius)
        sigma   -> scalar
        grad    -> (2,) gradient wrt robot position
    """

    N = obstacle_points.shape[0]

    robot_rep = np.repeat(robot_xy[None, :], N, axis=0)
    x_input = np.concatenate([robot_rep, obstacle_points], axis=1)

    x = torch.tensor(x_input, dtype=torch.float32, device=device, requires_grad=True)

    mu, var = predict_mu_var(model, x)

    # subtract robot radius (same as analytic version)
    mu = mu - RADIUS

    # find most critical point
    idx = torch.argmin(mu)

    mu_min = mu[idx]
    sigma_min = torch.sqrt(var[idx])

    # gradient wrt robot position
    grad_full = torch.autograd.grad(mu_min, x, retain_graph=False)[0]

    grad_robot = grad_full[idx, 0:2]   # only d/d(robot_x, robot_y)

    return (
        mu_min.item(),
        sigma_min.item(),
        grad_robot.detach().cpu().numpy()
    )