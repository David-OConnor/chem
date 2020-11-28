# todo: Sort through this

"""Time-evolution approaches:
- Break into basis of energy eigenstates (n = 1, 3 etc); evolve using (exp(-i*(t - t0) * E / ħ))
- Just solve the PDE of x and time (Challenge: Solving PDEs)
- Solve time as ODE after initially solving the spatial ode??
"""
from dataclasses import dataclass
from functools import partial
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import simps

from .main import Nucleus, nuc_pot, solve

ATOM_ARR_LEN = 5


@dataclass
class Electron:
    # ψ: List[float]
    ψ: Callable[[float], float]
    spin: bool  # True for up


def atoms_to_array(atoms: List[Nucleus]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.sx, atom.vx])
    return result


def h_static_pure(E: float) -> Tuple[np.ndarray, np.ndarray]:
    """No massasging past the singularity"""
    ψ0 = 0.2
    ψ_p0 = 0

    x_span = (-40, 0.0000001)

    V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])

    soln = solve(E, V_elec, ψ0, ψ_p0, x_span)

    x, soln = soln.t, soln.y[0]

    norm = simps(np.conj(soln) * soln, x=x)
    return x, soln / norm ** 0.5


def electron_potential(soln, n_electrons, sx: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0] ** 2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


def evolve_basis(state: np.ndarray, dt: float, E: float) -> np.ndarray:
    """e^(-itH/ħ), using an energy eigenbasis as the state."""
    return state * exp(-i * dt * E / ħ)


def td_schrod_t(d2𝚿_dx2: complex, V: float, t: float, 𝚿: complex):
    """
    d𝚿/dt = -ħ²/2miħ * d²𝚿/dx² + V(x)𝚿
    """
    return -(ħ ** 2) / (2 * m_e * i * ħ) * (d2𝚿_dx2 + V) * 𝚿


def evolve_de(
    x: np.ndarray, ψ0: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    In this approach, we solve the full PDE of x and t: iħ*dψ/dt = Hψ.
    We use the Method of Lines to discretize x, then solve as an ODE over t.

    iħ * d𝚿/dt = -1/2 * d^2𝚿/dx^2 + V𝚿
    d^2𝚿/dx^2 = (𝚿(xi+1, t) - 2𝚿(xi, t) + 𝚿(xi-1, t)) / Δx^2
    iħ * d𝚿/dt = -1/2 (𝚿_i+1(t) - 2𝚿_i(t) + 𝚿_i-1(t)) / Δx^2 + V𝚿

    """
    t_span = (0, 10)
    N = 100

    result = np.empty([N, x.size])
    result[0] = ψ0

    t = np.arange(0, x.size)  # todo
    ψ = ψ0
    x_span = (-40, 0.0000001)  # todo sync with other one
    V = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])

    # Iterate over each x value, to find its corresponding one one time-step later.
    # todo: For now, this is euler-esque, stepping over t.
    for t in range(N):
        # Calculate d𝚿/dt for each value of x.
        d2ψ_dx2 = np.diff(np.diff(ψ))  # todo: Check the offset imposed by d2ing
        d2ψ_dx2 = np.append(d2ψ_dx2, [0, 0])

        d𝚿_dt = np.empty(x.size, dtype=np.csingle)
        # d𝚿_dt = np.empty(x.size, dtype=np.csingle)
        for j in range(x.size):
            d𝚿_dt[j] = td_schrod_t(d2ψ_dx2[j], V(x[j]), 0, ψ[j])

        # print(d2ψ_dx2)
        # print(d𝚿_dt)
        ψ = ψ + d𝚿_dt
        result[t] = ψ

        # rhs = partial(td_schrod_x, d𝚿_dt, V)

        # ψ = solve_ivp(rhs, x_span, (0, .2),  t_eval=np.linspace(x_span[0], x_span[1], 10000)).y[0]

        # result[j] = ψ
    # todo: Can we assume t is invariant across the integration?
    return x, t, result


def plot_h_evolve_de():
    dt = 1
    n = 1
    E1 = -2 / (n + 1) ** 2
    E2 = -2 / (3 + 1) ** 2

    _, state1 = h_static_pure(E1)
    x, state2 = h_static_pure(E2)

    ψ_0 = (
        sqrt(2) / 2 * state1 + sqrt(2) / 2 * state2
    )  # A wall boundary condition, across all x, for t=0

    ψ_0 = state1

    _, t, soln = evolve_de(x, ψ_0, dt)

    fig, ax = plt.subplots()

    # for t in soln:
    #     ax.plot()

    ax.plot(x, soln[0])
    ax.plot(x, soln[2])
    ax.plot(x, soln[4])
    ax.plot(x, soln[6])
    ax.plot(x, soln[8])
    ax.plot(x, soln[10])

    plt.ylim(-1, 1)

    ax.grid(True)
    plt.show()


def plot_h_evolve():
    dt = 0.5
    ev = lambda E: exp(-i * dt * E / ħ)

    n = 1
    E1 = -2 / (n + 1) ** 2
    E2 = -2 / (3 + 1) ** 2

    # Eigenfunctions as basis
    x, ψ1 = h_static(E1)
    _, ψ2 = h_static(E2)

    state = [(sqrt(2) / 2, E1), (sqrt(2) / 2, E2)]

    fig, ax = plt.subplots()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for j in range(10):
        # Evolved here are the coefficients
        evolved1 = state[0][0] * ev(state[0][1])
        evolved2 = state[1][0] * ev(state[1][1])
        # evolved3 = state[2][0] * ev(state[2][1])

        # state = [(evolved1, state[0][1]), (evolved2, state[1][1]), (evolved3, state[2][1])]
        state = [(evolved1, state[0][1]), (evolved2, state[1][1])]

        # ψ = state[0][0] * ψ1 + state[1][0] * ψ2 + state[2][0] * ψ3
        ψ = state[0][0] * ψ1 + state[1][0] * ψ2
        # print(state)
        # print(np.abs(state[0][0]), np.abs(state[1][0]))
        # ax.plot(x, ψ)
        ax.plot(x, np.conj(ψ) * ψ)

    # (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))

    # Axes3D.plot_surface(x, y, Z)

    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


# def calc_h_static_eig():
#     D = op.diff_op()
#     D2 = D @ D
#
#     E = -1/2
#     N = 50
#     x = np.linspace(N) - 25
#
#     V = nuc_potential([Nucleus(1, 0, 0, 0)], x)
#
#     return 1/(2*E*V) * (D2 @ x)


def rhs_nbody(atoms: Iterable[Nucleus], t: float, y: np.array):
    # Un-flatten, back into a cleaner form.
    y_flat = y.reshape(-1, ATOM_ARR_LEN)

    # Inefficient, but makes code cleaner
    # Don't have number of neutrons here
    atoms2 = [Nucleus(a[1], 0, a[3], a[4]) for a in y_flat]

    # Chicken-egg scenario with calculating electric potential, so use the previous
    # iteration's field.
    soln_elec = solve(E, V_prev)

    # todo wrong! y[t] is y at diff posit, not time.
    E = i * ħ * soln_elec.y[t] - soln_elec.y[t - 1]

    # is this right???
    total_n_elec = sum(atom.n_elec for atom in y_flat)

    # potential = k * charge / r
    # force = k * charge1 * charge2 /r**2
    # force = potential * charge (other)

    result = []

    for j in range(y_flat.shape[0]):
        mass, n_prot, n_elec, sx, dx_dt = y_flat[j]

        V_nuc = nuc_pot(atoms2, sx)
        V_elec = electron_potential(soln_elec, total_n_elec, sx)
        force = (V_nuc + V_elec) * (n_prot * e)

        force_x = force

        # Calculate the acceleration
        ddx_ddt = force_x / mass

        # First three values don't change (mass, number of prots/elecs)
        result.extend([0, 0, 0, dx_dt, ddx_ddt])

    return result


def nbody():
    # Solve the nbody problem using classical nuclei

    atoms = [
        Nucleus(1, 0, 0, 0),  # Stationary hydrogen atom at origin
        # Atom(1, 0, -1, 1),
        # Atom(1, 0, 1, -1),
    ]

    atoms_array = atoms_to_array(atoms)

    # Convert atoms into a 1d array, which scipy can integrate.
    atoms_flat = atoms_array.flatten()

    rhs = partial(rhs_nbody, atoms)

    t_span = (0, 100)

    # for Ψ0 in np.linspace()

    soln = solve_ivp(
        rhs, t_span, atoms_flat, t_eval=np.linspace(t_span[0], t_span[1], 1000)
    )

    plt.plot(soln.y[0])
    plt.show()
