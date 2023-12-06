"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Codebase for Control Barrier Functions applied to problems in Finance and Economics.

---------------------------------------------------------------------------

Problem solution to Optimal Advertising problem introduced in 
"Optimal Control Theory with Applications in Economics" (Weber 2014)
http://econspace.net/teaching/MGT-626/MGT-626-Notes-2014.pdf

Exercise 3.8
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

    def h_x(self, x, x_max):
        return (x**2) - x_max**2

    def barrier_constraint(self, u, x, x_max, mu, r, sigma):
        return (
            2 * x * (r * x + (mu - r) * u)
            + (sigma**2) * (u**2)
            + self.alpha(self.h_x(x, x_max))
        )

    def alpha(self, x):
        return 10 * x

    def asif(self, u, x_curr, x_max, mu, r, sigma):
        # Check if control is safe
        constraint = [
            {
                "type": "ineq",
                "fun": self.barrier_constraint,
                "args": (u, x_curr, x_max, mu, r, sigma),
            }
        ]
        constraints = tuple(constraint)
        u_max = x_curr
        bnds = (0, u_max)


class PortfolioOptimization(StochasticCBF):
    def primaryControl(self, l, g, sigma, r, T, t):
        """
        Optimal control solution.

        """
        u = l / (g * sigma) * np.exp(-r * (T - t))
        return u

    # def propFun(self, x_full, t, u, r, mu, sigma):  # TODO DELETE THIS LATER
    #     """
    #     Propagate dynamics.

    #     """
    #     x = x_full
    #     dx = np.zeros_like(x_full)
    #     dx[0] = (u * (mu - r) + r * x) + u * sigma * (
    #         self.rng.standard_normal(size=(1))
    #     )
    #     return dx

    def f_fun(self, x, t, u, mu, r):
        return u * (mu - r) + r * x

    def g_fun(self, x, t, u, sigma):
        return u * sigma

    def runSimulation(self):
        # Constants
        r = 0.024
        mu = 0.1
        sigma = 0.14
        g = 0.1
        l = (mu - r) / sigma
        # u_max = 12

        self.seed = np.random.randint(0, 99999)
        # self.seed = 97072  # Reproducibility
        self.rng = np.random.default_rng(self.seed)

        # State constraint
        x_max = 190

        # Data statistics
        numPts = 500
        timestep = 0.01
        tspan = np.arange(0, numPts * timestep, timestep)

        # Tracking variables
        x_store = np.zeros((2, numPts))
        u_store = np.zeros(numPts)
        u_des_store = np.zeros(numPts)

        # Initial conditions
        x_0 = np.array([350])  # x
        x_curr = x_0

        # Storing values
        x_store[:, 0] = x_0
        u_store[0] = 0
        u_des_store[0] = 0
        solver_times = []
        intervened = [False] * numPts

        def close(func, *args):
            def newfunc(x, t):
                return func(x, t, *args)

            return newfunc

        for i in range(1, numPts):
            # Generate control desired
            t = tspan[i - 1]
            u = self.primaryControl(l, g, sigma, r, max(tspan), t)
            # u = 1

            u_act, sovler_dt = self.asif_QP(x_curr[0], x_max, u, b, u_max)
            solver_times.append(sovler_dt)
            if abs(u_act - u) > 0.001:
                intervened[i] = True
                # print("Intervened")
            u_des_store[i] = u
            u = u_act

            args1 = (u, mu, r)
            args2 = (u, sigma)

            # Apply control and propagate state
            x_curr = sdeint.itoint(
                close(self.f_fun, *args1),
                close(self.g_fun, *args2),
                x_curr[0],
                [tspan[i - 1], tspan[i]],
                generator=self.rng,
            )[-1]

            # # Apply control and propagate state
            # x_curr = odeint(
            #     self.propFun, x_curr, [tspan[i - 1], tspan[i]], args=(u, r, mu, sigma)
            # )[-1]

            # # Store data
            x_store[:, i] = x_curr
            u_store[i] = u

        print("Seed:", self.seed)
        # print(f"Average solver time: {1000*np.average(solver_times):0.4f} ms")
        # print(f"Maximum single solver time: {1000*np.max(solver_times):0.4f} ms")

        return tspan, x_store, numPts, u_store


class Plotter:
    def individualPlot(self, tspan, x_store, numPts, u_store):
        # Plotting
        fontsz = 24
        legend_sz = 24
        ticks_sz = 20
        x_line_opts = {"linewidth": 3, "color": "b"}
        u_line_opts = {"linewidth": 3, "color": "g", "label": "$\mathbf{u}_{\\rm act}$"}
        udes_line_opts = {"linewidth": 3, "color": "r", "label": "$\mathbf{u}^{*}$"}

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
            }
        )

        # State plot
        axf = plt.figure(figsize=(10, 7), dpi=100)
        ax = axf.add_subplot(111)
        ax.grid(True)
        plt.plot(tspan, x_store[0], **x_line_opts)
        # plt.axhline(x_max, color="k", linestyle="--")
        # plt.ylabel("$\mathbf{x}$", fontsize=fontsz + 4)
        # plt.ylabel("Installed Customer Base", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1]])

        plt.tight_layout()

        # Control plot
        axf = plt.figure(figsize=(10, 7), dpi=100)
        ax = axf.add_subplot(111)
        ax.grid(True)
        plt.plot(tspan, u_store, **u_line_opts)
        # plt.ylabel("$\mathbf{u}$", fontsize=fontsz + 4)
        # plt.ylabel("Advertising Activity", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1]])
        # ax.set_ylim([0, 1.1 * max(max(u_store), max(u_des_store))])

        ax.legend(fontsize=legend_sz, loc="upper left")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = PortfolioOptimization()
    (
        tspan,
        x_store,
        numPts,
        u_store,
    ) = env.runSimulation()
    plotter_env = Plotter()
    plotter_env.individualPlot(tspan, x_store, numPts, u_store)
    # plotter_env.subPlots(tspan, x_store, numPts, x_max, u_des_store, u_store)
