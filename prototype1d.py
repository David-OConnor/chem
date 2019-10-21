from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable, Tuple

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps

from consts import *

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0

import op


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


def schrod(E: float, V: Callable, x: float, y):
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def elec(E: float, V: Callable, ψ0: float, ψ_p0: float):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    x_span = (-10, 10)

    rhs = partial(schrod, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def nuc_potential(nuclei: Iterable[Nucleus], sx: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e / abs(nuclei.sx - sx)

    return result


def h_static(ψ0, ψ_p0, E: float):

    """A time-independet simulation of the electron cloud surrounding a hydrogen atom"""

    # ground level hydrogen: 13.6eV

    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.

    V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0)])

    return elec(E, V_elec, ψ0, ψ_p0)


def h_static_sph(ψ0: float, ψ_p0: float, E: float):

    """A time-independet simulation of the electron cloud surrounding a hydrogen atom"""

    # ground level hydrogen: 13.6eV, or 1 hartree

    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.

    # todo: are
    V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0)])

    l = 0  # what should this be?
    V_centrip = lambda r: ħ**2 * l*(l+1) / (2*m_e * r**2)
    V = lambda r: V_elec(r) + V_centrip(r)

    # x = np.linspace(.01, 10, 1000)
    # y = V(x)
    # plt.plot(x, y)
    # plt.show()
    # return

    return elec(E, V, ψ0, ψ_p0)


def calc_hydrogen_static():
    ψ0 = 0
    ψ_p0 = -1

    # E should be a whittaker energy, ie -1/2, -2/9, -1/8, -.08 etc
    # Only odd states (n = 1, 3, 5 etc) correspond to 3d H atom.
    n = 1
    E = -2/(n+1)**2

    E = 50

    soln = h_static(ψ0, ψ_p0, E)

    return soln


def calc_h_static_eig():
    D = op.diff_op()
    D2 = D @ D

    E = -1/2
    N = 50
    x = np.linspace(N) - 25

    V = nuc_potential([Nucleus(1, 0, 0, 0)], x)

    return 1/(2*E*V) * (D2 @ x)


def plot_hydrogen_static():
    # soln = calc_hydrogen_static()
    soln = calc_hydrogen_static()

    plt.plot(soln.t, soln.y[0])
    plt.show()


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
    soln_elec = elec(E, V_prev)

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
    plot_hydrogen_static()

# plot_ics()

# find_ψ_p0()
