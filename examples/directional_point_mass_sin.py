"""Trajectory tracking problem.

Manufactured example to demonstrate a simple path following
problem. The problem involves finding the control to manoeuvre
a point mass in the plane such that it follows a sinusoidal path.
It is strongly inspired by a similar example from pycollo.

The point mass is controllable by a single point force whose
direction can be directly controlled.

The current method is mainly a proof of concept and it would be
preferable to actually implement a more decent approach. One of
the features, which would also be nice to add, is to make have
periodic constraints, e.g. same state at the beginning and end
of the trajectory.

x is the position of the point horizontally from the origin (x-axis)
y is the position of the point vertically from the origin (y-axis)
F is the magnitude of the force applied to the point
theta is the angle of the force applied to the point
m is the mass of the point

"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from opty.direct_collocation import Problem
from opty.utils import parse_free

# State variables
t = me.dynamicsymbols._t
x = me.dynamicsymbols("x")
y = me.dynamicsymbols("y")
dx = me.dynamicsymbols("dx")
dy = me.dynamicsymbols("dy")

# Control variables
F = me.dynamicsymbols("F")
theta = me.dynamicsymbols("theta")

# Static parameter variable
m = sm.Symbol("m")

ddx = me.dynamicsymbols("dx", 1)
ddy = me.dynamicsymbols("dy", 1)
eoms = sm.Matrix([
    dx - x,
    dy - y,
    ddx - (F * sm.cos(theta)) / m,
    ddy - (F * sm.sin(theta)) / m,
])
constants = {
    m: 1.0
}
bounds = {
    x: (0, 2 * sm.pi),
    y: (-1, 1),
    dx: (-50, 50),
    dy: (-50, 50),
    theta: (-sm.pi, sm.pi),
    F: (0, 200),
}
initial_state_constraints = {
    x: 0,
    dx: 0,
    y: 0,
    dy: 0,
}
final_state_constraints = {
    x: 2 * sm.pi,
    y: 0,
}
end_point_constraints = []  # (x, y, dx, dy, theta, F)
instance_constraints = []

state_symbols = (x, dx, y, dy)

objective = theta ** 2 + F ** 2  # Should implement this kind of syntax

# Settings
T = 1.0  # duration
N = 100  # number of collocation nodes
n = len(state_symbols)  # number of state trajectories
interval = T / (N - 1)

instance_constraints.extend(
    [
        xi.xreplace({t: 0}) - xi_val for xi, xi_val in initial_state_constraints.items()
    ] + [
        xi.xreplace({t: T}) - xi_val for xi, xi_val in final_state_constraints.items()
    ] + [
        xi.xreplace({t: 0}) - xi.xreplace({t: T}) for xi in end_point_constraints
    ])

path = sm.sin(x) - y
max_average_tracking_error = 0.1
int_path = me.dynamicsymbols("int_path")

eoms = eoms.col_join(sm.Matrix([path ** 2 - int_path]))
state_symbols += (int_path,)
bounds[int_path] = (0, max_average_tracking_error ** 2 * T)


def obj(free):
    return interval * np.sum((free[n * N:]) ** 2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[n * N:] = 2 * interval * free[n * N:]
    return grad


# Problem instantiation
prob = Problem(
    obj=obj,
    obj_grad=obj_grad,
    equations_of_motion=eoms,
    state_symbols=state_symbols,
    num_collocation_nodes=N,
    node_time_interval=interval,
    known_parameter_map=constants,
    bounds=bounds,
    instance_constraints=tuple(instance_constraints),
)

initial_guess = np.zeros(prob.num_free)
initial_guess[:N] = np.linspace(0, 2 * np.pi, N)

# Find the optimal solution.
solution, info = prob.solve(initial_guess)

state_traj, input_traj, constants = parse_free(
    solution, prob.collocator.num_states,
    prob.collocator.num_unknown_input_trajectories,
    prob.collocator.num_collocation_nodes)
time = np.linspace(
    0, prob.collocator.num_collocation_nodes * prob.collocator.node_time_interval,
    num=prob.collocator.num_collocation_nodes)

x_sol, y_sol = state_traj[0, :], state_traj[2, :]

# Make some plots
prob.plot_trajectories(solution)
prob.plot_constraint_violations(solution)
prob.plot_objective_value()

plt.figure()
plt.plot(x_sol, np.sin(x_sol), "k")
plt.plot(x_sol, y_sol, "r")

plt.show()
