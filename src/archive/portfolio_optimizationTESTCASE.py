"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Codebase for Control Barrier Functions applied to problems in Finance and Economics.

---------------------------------------------------------------------------

[describe problem and give textbook]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import quadprog
from scipy.optimize import minimize
import time
import sdeint


class StochasticCBF:
    def obj_fun(self, u, u_des):
        """
        Objective function to minimize norm between actual and desired control.

        """
        return np.linalg.norm(u - u_des)
        # return u

    def h_x(self, x, x_min):
        return (x**2) - x_min**2

    def barrier_constraint(self, u, x, x_min, mu, r, sigma):
        return (
            2 * x * (r * x + (mu - r) * u)
            + (sigma * u) ** 2
            + self.alpha(self.h_x(x, x_min))
        )

    def alpha(self, x):
        return 10 * x

    def asif_NLP(self, u, x_curr, u_max, x_min, mu, r, sigma):
        # Check if control is safe
        constraint = [
            {
                "type": "ineq",
                "fun": self.barrier_constraint,
                "args": (x_curr, x_min, mu, r, sigma),
            }
        ]
        constraints = tuple(constraint)
        bnds = ((0.0, u_max),)

        u_0 = u / 3
        # u_0 = 0
        tic = time.perf_counter()
        result = minimize(
            self.obj_fun,
            u_0,
            constraints=constraints,
            method="SLSQP",
            bounds=bnds,
            args=u,
            tol=1e-5,
        )
        toc = time.perf_counter()
        solver_dt = toc - tic

        u_act = result.x[0]
        if not result.success:
            print("Fail, x_curr:", x_curr)
            # u_act = u_act / 3

        return u_act, solver_dt


class PortfolioOptimization(StochasticCBF):
    def primaryControl(self, l, g, sigma, r, T, t):
        """
        Optimal control solution.

        """
        u = l / (g * sigma) * np.exp(-r * (T - t))
        return u

    def eulerMaruyamaInt(self, x_tkm, del_t, u, mu, r, sigma):
        # Discrete integration via Euler–Maruyama method
        x_tk = (
            x_tkm
            + self.f_fun(x_tkm, 0, u, mu, r) * del_t
            + self.g_fun(0, 0, u, sigma)
            * (self.rng.standard_normal(size=(1)) * np.sqrt(del_t))
        )
        return x_tk

    def f_fun(self, x, t, u, mu, r):
        return u * (mu - r) + r * x

    def g_fun(self, x, t, u, sigma):
        return u * sigma

    def runSimulation(self):
        # Constants
        r = 0.02
        mu = 0.07
        sigma = 0.15
        g = 0.01
        l = (mu - r) / sigma

        self.seed = np.random.randint(0, 99999)
        self.seed = 7267  # Reproducibility
        self.rng = np.random.default_rng(self.seed)

        # State constraint
        x_min = 480

        # Data statistics
        numPts = 2
        timestep = 0.1
        tspan = np.arange(0, numPts * timestep, timestep)

        # Initial conditions
        x_0 = np.array([481])  # x
        x_curr = x_0

        def close(func, *args):
            def newfunc(x, t):
                return func(x, t, *args)

            return newfunc

        for i in range(1, numPts):
            # Generate control desired
            t = tspan[i - 1]
            u = self.primaryControl(l, g, sigma, r, max(tspan), t)
            # u = 0.6 * x_curr

            # Apply Stochastic CBF
            u_max = x_curr
            u_act = self.asif_NLP(u, x_curr, u_max, x_min, mu, r, sigma)
            u = u_act

            args1 = (u, mu, r)
            args2 = (u, sigma)

            # Apply control and propagate state using the stochastic diff eq.
            x_curr = sdeint.itoint(
                close(self.f_fun, *args1),
                close(self.g_fun, *args2),
                x_curr[0],
                [tspan[i - 1], tspan[i]],
                generator=self.rng,
            )[-1]

            print(x_curr)

        print("Seed:", self.seed)

        return tspan, numPts


if __name__ == "__main__":
    env = PortfolioOptimization()
    (
        tspan,
        numPts,
    ) = env.runSimulation()
