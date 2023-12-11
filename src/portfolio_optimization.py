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
from alive_progress import alive_bar
from scipy.optimize import minimize
import time
import sdeint


class StochasticCBF:
    def obj_fun(self, u, u_des):
        """
        Objective function to minimize norm between actual and desired control.

        """
        return np.linalg.norm(u - u_des)

    def h_x(self, x, x_min):
        return (x) ** 2 - (x_min) ** 2

    def barrier_constraint(self, u, x, x_min, mu, r, sigma):
        return (
            2 * x * (r * x + (mu - r) * u)
            + (sigma * u) ** 2
            + self.alpha(self.h_x(x, x_min))
        )

    def barrier_constraint_new(self, u, x, x_min, mu, r, sigma):
        dh_dx = 2 * x
        dh_dx_sqrd = 2
        mu_tilde = dh_dx * (r * x + (mu - r) * u) + 0.5 * dh_dx_sqrd * (sigma * u) ** 2
        sigma_tilde = dh_dx * u * sigma
        SBC = (
            mu_tilde
            - (sigma_tilde**2 / self.h_x(x, x_min))
            + self.alpha(self.h_x(x, x_min)) * (self.h_x(x, x_min)) ** 2
        )
        return SBC

    def alpha(self, x):
        return x / 1e12

    def asif_NLP(self, u, x_curr, u_max, x_min, mu, r, sigma):
        # Check if control is safe
        constraint = [
            {
                "type": "ineq",
                "fun": self.barrier_constraint_new,
                "args": (x_curr, x_min, mu, r, sigma),
            }
        ]
        constraints = tuple(constraint)
        bnds = ((0.0, u_max),)

        # u_0 = u / 3
        # u_0 = 0
        u_0 = u
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
        if not result.success and self.verbose:
            print("Fail, x_curr:", x_curr)
            u_act = 0

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

    def runSimulation(self, verbose=True):
        """
        Full simulation run.

        """
        self.verbose = verbose
        # Constants
        r = 0.02
        mu = 0.11
        sigma = 0.15
        g = 0.01
        l = (mu - r) / sigma

        self.seed = np.random.randint(0, 99999)
        # self.seed = 2222  # Reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Data statistics
        numPts = 400
        timestep = 0.1
        tspan = np.arange(0, numPts * timestep, timestep)

        # State constraint
        x_min = 500
        withdrawal_idx = 150

        # Tracking variables
        x_store = np.zeros(numPts)
        x_EM_store = np.zeros(numPts)
        u_store = np.zeros(numPts)
        u_des_store = np.zeros(numPts)

        # Initial conditions
        x_0 = np.array([520])  # x
        x_curr = x_0

        # Storing values
        x_store[0] = x_0[0]
        x_EM_store[0] = x_0[0]
        u_store[0] = 0
        u_des_store[0] = 0
        solver_times = []
        intervened = [False] * numPts
        violation_count = 0

        def close(func, *args):
            def newfunc(x, t):
                return func(x, t, *args)

            return newfunc

        for i in range(1, numPts):
            # Keep track of any constraint violations
            if x_curr < x_min:
                violation_count += 1

            # Generate control desired
            t = tspan[i - 1]
            u = self.primaryControl(l, g, sigma, r, max(tspan), t)

            if i == withdrawal_idx:
                x_curr[0] = x_min * 1.05

            # Apply Stochastic CBF using NLP
            u_max = x_curr
            u_act, sovler_dt = self.asif_NLP(u, x_curr, u_max, x_min, mu, r, sigma)
            solver_times.append(sovler_dt)
            if abs(u_act - u) > 0.001:
                intervened[i] = True
            u_des_store[i] = u
            u = u_act

            args1 = (u, mu, r)
            args2 = (u, sigma)

            # x_EM_store[:, i] = self.eulerMaruyamaInt(x_curr, timestep, u, mu, r, sigma)

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

        avg_solver_t = 1000 * np.average(solver_times)  # in ms
        max_solver_t = 1000 * np.max(solver_times)  # in ms
        if self.verbose:
            print("Seed:", self.seed)
            print(f"Average solver time: {avg_solver_t:0.4f} ms")
            print(f"Maximum single solver time: {max_solver_t:0.4f} ms")
            print(
                f"Single trial Pr(success): {(1-(violation_count/numPts)) * 100:0.2f}% \n"
            )

        return (
            tspan,
            x_store,
            x_EM_store,
            numPts,
            u_des_store,
            u_store,
            x_min,
            [avg_solver_t, max_solver_t],
            violation_count,
        )

    def runMC(self, numMCPts):
        """
        Run a monte carlo simulation with the number of samples: numMCPts

        """
        MC_store = []
        MC_violations_count = 0
        with alive_bar(numMCPts) as bar:
            for _ in range(numMCPts):
                (
                    tspan,
                    x_store,
                    _,
                    numPts,
                    _,
                    _,
                    x_min,
                    _,
                    violations,
                ) = self.runSimulation(verbose=False)
                MC_store.append(x_store)
                MC_violations_count += violations
                bar()

        MC_prob_success = (1 - (MC_violations_count / (numPts * numMCpts))) * 100

        return MC_store, tspan, x_min, MC_prob_success


class Plotter:
    def individualPlot(
        self, tspan, x_store, x_EM_store, numPts, u_des_store, u_store, x_min
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
        # plt.plot(tspan, x_EM_store, **x_line_opts2)
        plt.axhline(x_min, color="k", linestyle="--")
        plt.ylabel("Total Wealth (K USD)", fontsize=fontsz)
        plt.xlabel("Time (arbitrary)", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.004])
        y1_fill = np.ones(len(tspan)) * 0
        y2_fill = np.ones(len(tspan)) * x_min
        ax.fill_between(
            tspan,
            y1_fill,
            y2_fill,
            color=(255 / 255, 239 / 255, 239 / 255),  # Red, unsafe set
        )
        ax.fill_between(
            tspan,
            y2_fill,
            2 * max(x_store) * np.ones(len(tspan)),
            color=(244 / 255, 249 / 255, 241 / 255),  # Green, safe set
        )
        ax.set_ylim([0, 1.05 * max(x_store)])

        plt.tight_layout()

        # Control plot
        axf = plt.figure(figsize=(10, 7), dpi=100)
        ax = axf.add_subplot(111)
        ax.grid(True)
        plt.plot(tspan, u_des_store, **udes_line_opts)
        plt.plot(tspan, u_store, **u_line_opts)
        plt.ylabel("Wealth in Risky Asset (K USD)", fontsize=fontsz)
        plt.xlabel("Time (arbitrary)", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.004])
        ax.set_ylim([0, 1.1 * max(max(u_store), max(u_des_store))])

        ax.legend(fontsize=legend_sz, loc="upper left")
        plt.tight_layout()
        plt.show()

    def MCplot(self, MC_store, tspan, x_min):
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
        global_max = 0
        for i in range(len(MC_store)):
            plt.plot(tspan, MC_store[i], **x_line_opts)
            if max(MC_store[i]) > global_max:
                global_max = max(MC_store[i])

        y1_fill = np.ones(len(tspan)) * 0
        y2_fill = np.ones(len(tspan)) * x_min
        ax.fill_between(
            tspan,
            y1_fill,
            y2_fill,
            color=(255 / 255, 239 / 255, 239 / 255),  # Red, unsafe set
        )
        ax.fill_between(
            tspan,
            y2_fill,
            2 * global_max * np.ones(len(tspan)),
            color=(244 / 255, 249 / 255, 241 / 255),  # Green, safe set
        )
        ax.set_ylim([0, 1.05 * global_max])
        plt.axhline(x_min, color="k", linestyle="--")
        plt.ylabel("Total Wealth (K USD)", fontsize=fontsz)
        plt.xlabel("Time (arbitrary)", fontsize=fontsz)
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax.set_xlim([0, tspan[-1] * 1.003])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = PortfolioOptimization()
    plotter_env = Plotter()

    individual_run = True
    MC_run, numMCpts = True, 1000
    if individual_run:
        print("Running single trial for stochastic portfolio optimization")
        (
            tspan,
            x_store,
            x_EM_store,
            numPts,
            u_des_store,
            u_store,
            x_min,
            _,
            _,
        ) = env.runSimulation()
        plotter_env.individualPlot(
            tspan, x_store, x_EM_store, numPts, u_des_store, u_store, x_min
        )
    if MC_run:
        print(
            f"Running Monte Carlo simulation for stochastic portfolio optimization with {numMCpts} samples"
        )
        MC_store, tspan, x_min, MC_prob_success = env.runMC(numMCpts)
        plotter_env.MCplot(MC_store, tspan, x_min)
        print(f"Monte Carlo Pr(success) = {MC_prob_success:0.3f}%")
