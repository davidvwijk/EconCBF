"""
--------------------------------------------------------------------------

van Wijk, David
Texas A&M University
Aerospace Engineering

Codebase for Control Barrier Functions applied to problems in Finance and Economics.

---------------------------------------------------------------------------

[describe problem and give textbook] TODO
"""

import time
from alive_progress import alive_bar
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import quadprog
import sdeint


class StochasticCBF:
    def obj_fun(self, u, u_des):
        """
        Objective function to minimize norm between actual and desired control.

        """
        return np.linalg.norm(u - u_des)

    def h_x(self, x, x_max):
        return -(x**2) + (x_max) ** 2

    def barrier_constraint(self, u, x, x_max, delta, r, sigma):
        dhdx = -2 * x
        dhdx_sqred = -2
        return (
            dhdx * (r * u * np.sqrt(1 - x) - delta * x)
            + 0.5 * (dhdx_sqred * (sigma * x) ** 2)
            + self.alpha(self.h_x(x, x_max))
        )

    def h_x2(self, x, x_max):
        return (x_max - x) ** 2

    def barrier_constraint2(self, u, x, x_max, delta, r, sigma):
        dhdx = 2 * (x_max - x)
        dhdx_sqred = -2
        return (
            dhdx * (r * u * np.sqrt(1 - x) - delta * x)
            + 0.5 * (dhdx_sqred * (sigma * x) ** 2)
            + self.alpha(self.h_x2(x, x_max))
        )

    def alpha(self, x):
        return x

    def asif_NLP(self, u, x_curr, u_max, x_max, delta, r, sigma):
        # Check if control is safe
        constraint = [
            {
                "type": "ineq",
                "fun": self.barrier_constraint,
                "args": (x_curr, x_max, delta, r, sigma),
            }
        ]
        constraints = tuple(constraint)
        bnds = ((0.0, u_max),)

        u_0 = u
        tic = time.perf_counter()
        result = minimize(
            self.obj_fun,
            u_0,
            constraints=constraints,
            method="SLSQP",
            bounds=bnds,
            args=u,
            # tol=1e-5,
        )
        toc = time.perf_counter()
        solver_dt = toc - tic

        u_act = result.x[0]
        if not result.success and self.verbose:
            print("Fail, x_curr:", x_curr)
            u_act = 0

        return u_act, solver_dt

    def asif_QP(self, u, x_curr, u_max, x_max, delta, r, sigma):
        M = np.eye(2)
        q = np.array([u, 0])  # Need to append the control with 0 to get 2 dimensions

        # Actuation constraints (0,u_max)
        G = np.vstack((np.eye(2), -np.eye(2)))
        h = np.array([0, 0, -u_max, 0])

        dhdx = -2 * x_curr
        dhdx_sqred = -2

        g_constraint = np.array([dhdx * (r * u * np.sqrt(1 - x_curr)), 0])
        g_constraint = np.vstack([g_constraint, np.zeros(2)])
        h_constraint = (
            (dhdx * delta * x_curr)
            - (0.5 * (dhdx_sqred * (sigma * x_curr) ** 2))
            - self.alpha(self.h_x(x_curr, x_max))
        )

        G = np.vstack([G, g_constraint])
        h = np.vstack(
            [h.reshape((-1, 1)), np.array([h_constraint, 0]).reshape((-1, 1))]
        )
        d = h.reshape((len(h),))

        tic = time.perf_counter()
        try:
            u_act = quadprog.solve_qp(M, q, G.T, d, 0)[0]
        except:
            u_act = [0]
        toc = time.perf_counter()
        solver_dt = toc - tic

        return u_act[0], solver_dt


class Advertising(StochasticCBF):
    def primaryControl(self, x, rho, delta, pi, r):
        """
        Optimal control solution.

        """
        l = (np.sqrt((rho + delta) ** 2 + pi * r**2) - (rho + delta)) / ((r**2) / 2)
        u = (r * l * np.sqrt(1 - x)) / 2
        return u

    def eulerMaruyamaInt(self, x_tkm, del_t, u, delta, r, sigma):
        # Discrete integration via Euler–Maruyama method
        x_tk = (
            x_tkm
            + self.f_fun(x_tkm, 0, u, delta, r) * del_t
            + self.g_fun(0, 0, u, sigma)
            * (self.rng.standard_normal(size=(1)) * np.sqrt(del_t))
        )
        return x_tk

    def f_fun(self, x, t, u, delta, r):
        return r * u * np.sqrt(1 - x) - delta * x

    def g_fun(self, x, t, u, sigma):
        return x * sigma

    def runSimulation(self, verbose=True, SCBF_flag=True):
        self.verbose = verbose
        self.SCBF_flag = SCBF_flag

        # Constants
        r = 0.02
        delta = 0.07
        sigma = 0.03
        rho = 0.1
        pi = 50
        u_max = 10

        self.seed = np.random.randint(0, 99999)
        # self.seed = 7267  # Reproducibility
        self.rng = np.random.default_rng(self.seed)

        # State constraint
        x_max = 0.4

        # Data statistics
        numPts = 500
        timestep = 0.1
        tspan = np.arange(0, numPts * timestep, timestep)

        # Tracking variables
        x_store = np.zeros(numPts)
        x_EM_store = np.zeros(numPts)
        u_store = np.zeros(numPts)
        u_des_store = np.zeros(numPts)

        # Initial conditions
        x_0 = np.array([0.1])  # x
        x_curr = x_0

        # Storing values
        x_store[0] = x_0[0]
        x_EM_store[0] = x_0[0]
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
            u = self.primaryControl(x_curr, rho, delta, pi, r)

            # Apply Stochastic CBF
            if self.SCBF_flag:
                # NLP and QP Implementation for verification

                # u_act, sovler_dt = self.asif_NLP(
                #     u, x_curr, u_max, x_max, delta, r, sigma
                # )
                u_act, sovler_dt = self.asif_QP(
                    u[0], x_curr[0], u_max, x_max, delta, r, sigma
                )
                solver_times.append(sovler_dt)
                if abs(u_act - u) > 0.001:
                    intervened[i] = True
                u_des_store[i] = u[0]
                u = u_act

            args1 = (u, delta, r)
            args2 = (u, sigma)

            x_EM_store[i] = self.eulerMaruyamaInt(x_curr, timestep, u, delta, r, sigma)[
                0
            ]

            # Apply control and propagate state using the stochastic diff eq.
            x_curr = sdeint.itoint(
                close(self.f_fun, *args1),
                close(self.g_fun, *args2),
                x_curr[0],
                [tspan[i - 1], tspan[i]],
                generator=self.rng,
            )[-1]

            # Store data
            x_store[i] = x_curr[0]
            u_store[i] = u

        if self.verbose:
            print("Seed:", self.seed)
            # print(f"Average solver time: {1000*np.average(solver_times):0.4f} ms")
            # print(f"Maximum single solver time: {1000*np.max(solver_times):0.4f} ms")

        return tspan, x_store, x_EM_store, numPts, u_des_store, u_store, x_max

    def runMC(self, numMCPts, SCBF_flag):
        MC_store = []
        with alive_bar(numMCPts) as bar:
            for _ in range(numMCPts):
                (
                    tspan,
                    x_store,
                    _,
                    _,
                    _,
                    _,
                    x_max,
                ) = self.runSimulation(verbose=False, SCBF_flag=SCBF_flag)
                MC_store.append(x_store)
                bar()

        return MC_store, tspan, x_max


class Plotter:
    def individualPlot(
        self, tspan, x_store, x_EM_store, numPts, u_des_store, u_store, x_max
    ):
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
        plt.plot(tspan, x_store, **x_line_opts)
        # plt.plot(tspan, x_EM_store[0], **x_line_opts2)
        plt.axhline(x_max, color="k", linestyle="--")
        plt.ylabel("Installed Customer Base", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.003])

        plt.tight_layout()

        # Control plot
        axf = plt.figure(figsize=(10, 7), dpi=100)
        ax = axf.add_subplot(111)
        ax.grid(True)
        plt.plot(tspan, u_des_store, **udes_line_opts)
        plt.plot(tspan, u_store, **u_line_opts)
        plt.ylabel("Advertising Activity", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.003])
        ax.set_ylim([0, 1.1 * max(max(u_store), max(u_des_store))])

        ax.legend(fontsize=legend_sz, loc="upper right")
        plt.tight_layout()
        plt.show()

    def MCplot(self, MC_store, tspan, x_max):
        fontsz = 24
        ticks_sz = 20
        x_line_opts = {"linewidth": 1.5}

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
        for i in range(len(MC_store)):
            plt.plot(tspan, MC_store[i], **x_line_opts)

        plt.axhline(x_max, color="k", linestyle="--")
        plt.ylabel("Installed Customer Base", fontsize=fontsz)
        plt.xlabel("Time", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.003])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = Advertising()
    plotter_env = Plotter()

    individual_run = False
    MC_run = True

    if individual_run:
        (
            tspan,
            x_store,
            x_EM_store,
            numPts,
            u_des_store,
            u_store,
            x_max,
        ) = env.runSimulation(verbose=True, SCBF_flag=True)
        plotter_env.individualPlot(
            tspan, x_store, x_EM_store, numPts, u_des_store, u_store, x_max
        )
    if MC_run:
        MC_store, tspan, x_max = env.runMC(100, SCBF_flag=True)
        plotter_env.MCplot(MC_store, tspan, x_max)
