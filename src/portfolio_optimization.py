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
        return x / 10

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
        # self.seed = 7267  # Reproducibility
        self.rng = np.random.default_rng(self.seed)

        # State constraint
        x_min = 480

        # Data statistics
        numPts = 400
        timestep = 0.1
        tspan = np.arange(0, numPts * timestep, timestep)

        # Tracking variables
        x_store = np.zeros((2, numPts))
        x_EM_store = np.zeros((1, numPts))
        u_store = np.zeros(numPts)
        u_des_store = np.zeros(numPts)

        # Initial conditions
        x_0 = np.array([500])  # x
        x_curr = x_0

        # Storing values
        x_store[:, 0] = x_0
        x_EM_store[:, 0] = x_0
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
            # u = 0.6 * x_curr

            if i == 50:
                x_curr[0] = x_min + 3
            # x_min = x_curr * 0.94
            # Apply Stochastic CBF
            u_max = x_curr
            u_act, sovler_dt = self.asif_NLP(u, x_curr, u_max, x_min, mu, r, sigma)
            solver_times.append(sovler_dt)
            if abs(u_act - u) > 0.001:
                intervened[i] = True
            u_des_store[i] = u
            u = u_act

            args1 = (u, mu, r)
            args2 = (u, sigma)

            x_EM_store[:, i] = self.eulerMaruyamaInt(x_curr, timestep, u, mu, r, sigma)

            # Apply control and propagate state using the stochastic diff eq.
            x_curr = sdeint.itoint(
                close(self.f_fun, *args1),
                close(self.g_fun, *args2),
                x_curr[0],
                [tspan[i - 1], tspan[i]],
                generator=self.rng,
            )[-1]

            # Store data
            x_store[:, i] = x_curr
            u_store[i] = u

        print("Seed:", self.seed)
        print(f"Average solver time: {1000*np.average(solver_times):0.4f} ms")
        print(f"Maximum single solver time: {1000*np.max(solver_times):0.4f} ms")

        return tspan, x_store, x_EM_store, numPts, u_des_store, u_store


class Plotter:
    def individualPlot(self, tspan, x_store, x_EM_store, numPts, u_des_store, u_store):
        # Plotting
        fontsz = 24
        legend_sz = 24
        ticks_sz = 20
        x_line_opts = {"linewidth": 3, "color": "b"}
        x_line_opts2 = {"linewidth": 3, "color": "pink"}
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
        # plt.plot(tspan, x_EM_store[0], **x_line_opts2)
        # plt.axhline(x_min, color="k", linestyle="--")
        plt.ylabel("$\mathbf{x}$", fontsize=fontsz + 4)
        # plt.ylabel("Installed Customer Base", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.004])

        plt.tight_layout()

        # Control plot
        axf = plt.figure(figsize=(10, 7), dpi=100)
        ax = axf.add_subplot(111)
        ax.grid(True)
        plt.plot(tspan, u_des_store, **udes_line_opts)
        plt.plot(tspan, u_store, **u_line_opts)
        plt.ylabel("$\mathbf{u}$", fontsize=fontsz + 4)
        # plt.ylabel("Advertising Activity", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.004])
        ax.set_ylim([0, 1.1 * max(max(u_store), max(u_des_store))])

        ax.legend(fontsize=legend_sz, loc="upper right")
        plt.tight_layout()
        plt.show()

    def subPlots(self, tspan, x_store, numPts, u_des_store, u_store):
        # Plotting
        fontsz = 24
        legend_sz = 24
        x_line_opts = {"linewidth": 3, "color": "b"}
        u_line_opts = {"linewidth": 3, "color": "g", "label": "$\mathbf{u}_{\\rm act}$"}
        udes_line_opts = {"linewidth": 3, "color": "r", "label": "$\mathbf{u}^{*}$"}

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
            }
        )

        plt.figure(figsize=(10, 8))

        # State plot
        ax = plt.subplot(2, 1, 1)
        ax.grid(True)
        plt.plot(tspan, x_store[0], **x_line_opts)
        # plt.axhline(x_max, color="k", linestyle="--")
        plt.ylabel("$\mathbf{x}$", fontsize=fontsz + 3)
        plt.xlabel("Time", fontsize=fontsz)
        ax.set_xlim([0, tspan[-1]])
        ax.set_ylim([0, 1])
        # y1_fill = np.ones(numPts) * 0
        # y2_fill = np.ones(numPts) * x_max
        # ax.fill_between(
        #     tspan,
        #     y1_fill,
        #     y2_fill,
        #     color=(244 / 255, 249 / 255, 241 / 255),  # Green, safe set
        # )
        # ax.fill_between(
        #     tspan,
        #     y2_fill,
        #     np.ones(numPts),
        #     color=(255 / 255, 239 / 255, 239 / 255),  # Red, unsafe set
        # )

        # Control plot
        ax = plt.subplot(2, 1, 2)
        ax.grid(True)
        plt.plot(tspan, u_des_store, **udes_line_opts)
        plt.plot(tspan, u_store, **u_line_opts)
        plt.ylabel("$\mathbf{u}$", fontsize=fontsz + 3)
        plt.xlabel("Time", fontsize=fontsz)
        ax.set_xlim([0, tspan[-1]])

        ax.legend(fontsize=legend_sz, loc="upper left")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = PortfolioOptimization()
    (
        tspan,
        x_store,
        x_EM_store,
        numPts,
        u_des_store,
        u_store,
    ) = env.runSimulation()
    plotter_env = Plotter()
    plotter_env.individualPlot(tspan, x_store, x_EM_store, numPts, u_des_store, u_store)
    # plotter_env.subPlots(tspan, x_store, numPts, u_des_store, u_store)
