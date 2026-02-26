import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from world import mu_and_grad, SIGMA

DELTA = 0.05
BETA = norm.ppf(1 - DELTA)


def solve_step(p0, p_goal, obstacles):

    def objective(x):
        dp = x[:2]
        slack = x[2:]
        return dp @ dp + 10.0 * (slack @ slack)

    def chance_constraint(x):
        dp = x[:2]
        mu, grad = mu_and_grad(p0, obstacles)
        return mu + grad @ dp - BETA * SIGMA

    def tracking_constraint(x):
        dp = x[:2]
        slack = x[2:]
        return p0 + dp - p_goal - slack

    cons = [
        {"type": "ineq", "fun": chance_constraint},
        {"type": "eq", "fun": tracking_constraint},
    ]

    res = minimize(objective, np.zeros(4), constraints=cons, method="SLSQP")
    return p0 + res.x[:2]#, res