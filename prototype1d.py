from dataclasses import dataclass


from functools import partial
from typing import List, Iterable, Callable, Tuple

import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, simps

from consts import *
import op

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0

i = complex(0, 1)



# Lookin into matrix mechanics, and Feynman path integral approaches too

# orbitals are characterized (in simple cases) by quantum numbers n, l, and m, corresponding to
# energy, angular momentum, and magnetic (ang momentum vec component)

# spin?

# Do electrons (electrically) interact with themselves?

# Breaking up a numerical problem into a number of solveable analytic ones??? Eg set up
# an arbitrary V as a series of step Vs which have anal solns


# Free variables: 2? Energy, and ψ_p_0(ψ). Eg we can set ψ to what we wish, find the ψ_p that
# works with it (and the E set), then normalize.

@dataclass
class Nucleus:
    n_prot: float
    n_neut: float
    sx: float
    vx: float

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


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


def nuc_potential(nuclei: Iterable[Nucleus], sx: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e / abs(nuclei.sx - sx)

    return result


def ti_schrod(E: float, V: Callable, x: float, y: Tuple[complex, complex]) -> Tuple[complex, complex]:
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def nuc_elec(E: float, V: Callable, ψ0: float, ψ_p0: float, x_span: Tuple[float, float]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    rhs = partial(ti_schrod, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def h_static(E: float) -> Tuple[np.ndarray, np.ndarray]:
    ψ0 = .2
    ψ_p0 = 0

    # ψ0 = 1
    # ψ_p0 = 0

    x_span = (-40, .0000001)
    # x_span = (-40, 10)

    # V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0), Nucleus(1, 0, 3, 0)])
    V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln_orig = nuc_elec(E, V_elec, ψ0, ψ_p0, x_span)
    soln_left = soln_orig.y[0]
    soln_right = np.flip(soln_left)
    soln = np.concatenate([soln_left, -soln_right])
    x = np.concatenate([soln_orig.t, np.flip(-soln_orig.t)])

    norm = simps(np.conj(soln) * soln, x=x)
    return x, soln/norm**.5
    # norm = simps(np.conj(soln_orig.y[0]) * soln_orig.y[0], x=soln_orig.t)
    # return soln_orig.t, soln_orig.y[0]/norm**.5


def plot_h_static():
    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.
    # E should be a whittaker energy, ie -1/2, -2/9, -1/8, -.08 etc
    # Only odd states (n = 1, 3, 5 etc) correspond to 3d H atom.
    n = 1
    E = -2/(n+1)**2

    # E = -.44194

    x, ψ = h_static(E)

    fig, ax = plt.subplots()
    ax.plot(x, ψ)
    ax.plot(x, np.conj(ψ) * ψ)
    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


"""Time-evolution approaches:
- Break into basis of energy eigenstates (n = 1, 3 etc); evolve using (exp(-i*(t - t0) * E / ħ))
- Just solve the PDE of x and time (Challenge: Solving PDEs)
- Solve time as ODE after initially solving the spatial ode??
"""


def evolve_basis(state: np.ndarray, dt: float, E: float) -> np.ndarray:
    """e^(-itH/ħ), using an energy eigenbasis as the state."""
    return state * exp(-i * dt * E / ħ)


def td_schrod_t(d2ψ_dx2: complex, V: float, t: float, 𝚿: complex):
    """
    2 * m_e / ħ ** 2 * (V(x) - E) * ψ
    Return d𝚿/dt
    """
    return (-ħ**2/(2*m_e) * d2ψ_dx2 + V * 𝚿) / (i*ħ)


def td_schrod_x(d𝚿_dt: complex, V: Callable, x: float, y: Tuple[complex, complex]) -> Tuple[complex, complex]:
    """
    This is similar to `ti_schrod`, but uses d𝚿_dt instead of E.
    todo: Just use the same fn? Only diff is the i*ħ factor.
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - i*ħ*d𝚿_dt) * ψ

    return ψ_p, φ_p


def evolve_de(x: np.ndarray, ψ0: np.ndarray, dt: float, E: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Iterate over each x value, to find its corresponding one one time-step later.
    for j in range(x.size):
        x_ = x[j]
        ψ = ψ0[j]
        x_span = (-40, .0000001) # todo sync with other one

        # Calculate d𝚿/dt for each value of x.
        d2ψ_dx2 = np.diff(np.diff(ψ))  # todo: Check the offset imposed by d2ing!
        V_x = nuc_potential([Nucleus(1, 0, 0, 0)], x_)

        d𝚿_dt = td_schrod_t(d2ψ_dx2[j], V_x, 0, ψ[j])

        rhs = partial(td_schrod_x, d𝚿_dt)
        ψ, _ = solve_ivp(rhs, x_span, (),  t_eval=np.linspace(x_span[0], x_span[1], 10000)).y

        result[j] = ψ
    # todo: Can we assume t is invariant across the integration?
    return x, t, result


def plot_h_evolve_de():
    dt = 1
    n = 1
    E1= -2 / (n + 1) ** 2
    E2 = -2 / (3 + 1) ** 2

    x, ψ_0 = sqrt(2)/2 * h_static(E1) + sqrt(2)/2 * h_static(E2)  # A wall boundary condition, across all x, for t=0

    t, soln = evolve_de(x, ψ_0, dt, E)
    breakpoint()

    fig, ax = plt.subplots()

    for t in soln:
        ax.plot()


    ax.grid(True)
    plt.show()


def plot_h_evolve():
    dt = 10
    ev = lambda E: exp(-i * dt * E / ħ)

    n = 1
    E1 = -2 / (n + 1) ** 2
    E2 = -2 / (3 + 1) ** 2
    E3 = -2 / (5 + 1) ** 2

    # Eigenfunctions as basis
    x, ψ1 = h_static(E1)
    _, ψ2 = h_static(E2)
    # _, ψ3 = h_static(E3)

    # state = [(sqrt(3)/3, E1), (sqrt(3)/3, E2), (sqrt(3)/3, E3)]
    state = [(sqrt(2)/2, E1), (sqrt(2)/2, E2)]

    fig, ax = plt.subplots()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for j in range(1):
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


def electron_potential(soln, n_electrons, sx: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0]**2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


def rhs_nbody(atoms: Iterable[Nucleus], t: float, y: np.array):
    # Un-flatten, back into a cleaner form.
    y_flat = y.reshape(-1, ATOM_ARR_LEN)

    # Inefficient, but makes code cleaner
    # Don't have number of neutrons here
    atoms2 = [Nucleus(a[1], 0, a[3], a[4]) for a in y_flat]


    # Chicken-egg scenario with calculating electric potential, so use the previous
    # iteration's field.
    soln_elec = nuc_elec(E, V_prev)

    # todo wrong! y[t] is y at diff posit, not time.
    E = i * ħ * soln_elec.y[t] - soln_elec.y[t-1]

    # is this right???
    total_n_elec = sum(atom.n_elec for atom in y_flat)

    # potential = k * charge / r
    # force = k * charge1 * charge2 /r**2
    # force = potential * charge (other)

    result = []

    for j in range(y_flat.shape[0]):
        mass, n_prot, n_elec, sx, dx_dt = y_flat[j]

        V_nuc = nuc_potential(atoms2, sx)
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

    soln = solve_ivp(rhs, t_span, atoms_flat, t_eval=np.linspace(t_span[0], t_span[1], 1000))

    plt.plot(soln.y[0])
    plt.show()


if __name__ == "__main__":
    plot_h_static()
