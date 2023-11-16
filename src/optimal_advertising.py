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
import seaborn as sns

# Constants
k = 0.5
r = 0.001
b = 0.1
c = 10
g = (r + b) / c
u_max = 12

# State constraint (0,1)
x_max = 0.6

# Data statistics
numPts = 250
timestep = 0.01
tspan = np.arange(0, numPts * timestep, timestep)

# Tracking variables
x_store = np.zeros((2, numPts))
u_store = np.zeros(numPts)
u_des_store = np.zeros(numPts)

# Initial conditions
x_0 = np.array([0.02, 1])  # x, v
x_curr = x_0
x_store[:, 0] = x_0
u_store[0] = 0
u_des_store[0] = 0


def barrier_constraint(x, x_max, u, b):
    return 2 * b * x**2 - 2 * x * (1 - x) * u + alpha(h_x(x, x_max))


def asif_QP(x, x_max, u_des, b, u_max):
    M = np.eye(2)
    q = np.array([u_des, 0])  # Need to append the control with 0 to get 2 dimensions

    # Actuation constraints (0,u_max)
    G = np.vstack((np.eye(2), -np.eye(2)))
    h = np.array([0, 0, -u_max, 0])

    grad_h = -2 * x
    g_x = 1 - x
    f_x = -b * x

    g_constraint = np.array([grad_h * g_x, 0])
    g_constraint = np.vstack([g_constraint, np.zeros(2)])
    h_constraint = -grad_h * f_x - alpha(h_x(x, x_max))

    G = np.vstack([G, g_constraint])
    h = np.vstack([h.reshape((-1, 1)), np.array([h_constraint, 0]).reshape((-1, 1))])
    d = h.reshape((len(h),))

    u_act = quadprog.solve_qp(M, q, G.T, d, 0)[0]
    return u_act[0]


def alpha(x):
    return 10 * x


def h_x(x, x_max):
    return -(x**2) + x_max**2


def primaryControl(x, v, t, k, r, u_max):
    # Optimal control solution
    u = ((k * v * (1 - x)) / (np.exp(-r * t))) ** (k / (1 - k))
    return min(u, u_max)


def propFun(x_full, t, u, r, g, b):
    # Propagate the dynamics and the costate
    x, v = x_full
    dx = np.zeros_like(x_full)
    dx[0] = (1 - x) * u - b * x
    dx[1] = -g * (np.exp(-r * t)) + v * (u + b)
    return dx


intervened = [False] * numPts
for i in range(1, numPts):
    # Generate control desired
    t = tspan[i - 1]
    u = primaryControl(x_curr[0], x_curr[1], t, k, r, u_max)

    # Check if control is safe
    u_act = asif_QP(x_curr[0], x_max, u, b, u_max)
    if abs(u_act - u) > 0.001:
        intervened[i] = True
        # print("Intervened")
    u_des_store[i] = u
    u = u_act

    # Apply control and propagate state
    x_curr = odeint(propFun, x_curr, [tspan[i - 1], tspan[i]], args=(u, r, g, b))[-1]

    # Store data
    x_store[:, i] = x_curr
    u_store[i] = u

# Plotting
fontsz = 20
legend_sz = 15
x_line_opts = {"linewidth": 2, "color": "b"}
u_line_opts = {"linewidth": 2, "color": "g", "label": "$\mathbf{u}_{act}$"}
udes_line_opts = {"linewidth": 2, "color": "r", "label": "$\mathbf{u}^{*}$"}

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
plt.axhline(x_max, color="k", linestyle="--")
plt.ylabel("$\mathbf{x}$", fontsize=fontsz)
plt.xlabel("Time", fontsize=fontsz)
ax.set_xlim([0, tspan[-1]])
ax.set_ylim([0, 1])
y1_fill = np.ones(numPts) * 0
y2_fill = np.ones(numPts) * x_max
ax.fill_between(
    tspan,
    y1_fill,
    y2_fill,
    color=(244 / 255, 249 / 255, 241 / 255),  # Green, safe set
)
ax.fill_between(
    tspan,
    y2_fill,
    np.ones(numPts),
    color=(255 / 255, 239 / 255, 239 / 255),  # Red, unsafe set
)

# Control plot
ax = plt.subplot(2, 1, 2)
ax.grid(True)
plt.plot(tspan, u_des_store, **udes_line_opts)
plt.plot(tspan, u_store, **u_line_opts)
plt.ylabel("$\mathbf{u}$", fontsize=fontsz)
plt.xlabel("Time", fontsize=fontsz)
ax.set_xlim([0, tspan[-1]])

ax.legend(fontsize=legend_sz, loc="upper left")
plt.tight_layout()
plt.show()
