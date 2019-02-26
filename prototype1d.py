from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from consts import *

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0


# Lookin into matrix mechanics, and Feynman path integral approaches too

# orbitals are characterized (in simple cases) by quantum numbers n, l, and m, corresponding to
# energy, angular momentum, and magnetic (ang momentum vec component)

# spin?

# Do electrons (electrically) interact with themselves?

# Breaking up a numerical problem into a number of solveable analytic ones???


# Free variables: 2? Energy, and ψ_p_0(ψ). Eg we can set ψ to what we wish, find the ψ_p that
# works with it (and the E set), then normalize.

@dataclass
class Atom:
    n_prot: float
    n_neut: float
    n_elec: float
    sx: float
    vx: float

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


def atoms_to_array(atoms: List[Atom]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.n_elec, atom.sx, atom.vx])
    return result


def schrod(E: float, V: Callable, x: float, y: float):
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ
    # print(V(x))

    return ψ_p, φ_p


def elec(E: float, V: Callable):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    x_span = (0.001, 11000)

    # todo: How do I set these ICs/BCs?
    # ψ0 = .3
    # ψ_p0 = 0.00291164261770918944735

    # ψ0 = 1/2
    # ψ_p0 = 0.004847101011705505

    ψ0 = 1/2
    ψ_p0 = 0.004847101011705505

    rhs = partial(schrod, E, V)

    soln = solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 1000))

    return soln


def nuc_potential(nuclei: Iterable[Atom], n_electrons: int, sx: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        # result -= e * n_electrons * nuclei.charge() / abs(body.sx - sx)
        result -= e / abs(nuclei.sx - sx)

    return result


def hydrogen_static():
    """A time-independet simulation of the electron cloud surrounding a hydrogen atom"""

    # ground level hydrogen: 13.6eV
    n = 1
    E = -ħ**2/(2*m_e*1**2 * n**2)
    print(E)

    # Negative E implies bound state; positive scattering.
    E= -0.01

    V = partial(nuc_potential, [Atom(1, 0, 1, 0, 0)], 1)

    soln = elec(E, V)

    # x_ = np.linspace(.001, .01, int(1e3))
    # plt.plot(x_, V(x_))

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


def rhs_nbody(atoms: Iterable[Atom], t: float, y: np.array):
    # Un-flatten, back into a cleaner form.
    y_flat = y.reshape(-1, ATOM_ARR_LEN)

    # Inefficient, but makes code cleaner
    # Don't have number of neutrons here
    atoms2 = [Atom(a[1], 0, a[2], a[3], a[4]) for a in y_flat]


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
        Atom(1, 0, 1, 0, 0),  # Stationary hydrogen atom at origin
        # Atom(1, 0, 1, -1, 1),
        # Atom(1, 0, 1, 1, -1),
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


def run_2d():
    pass


hydrogen_static()
