# Use a radial approach, where we only use a single atom, taking advantage of
# the potential's spherical symmetry.u

from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from consts import *

# E = n**2 * np.pi**2 * ħ**2 / (2 * m_e * L**2)

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0


@dataclass
class Atom:
    n_prot: float
    n_neut: float
    n_elec: float
    sx: float
    sy: float
    sz: float
    vx: float
    vy: float
    vz: float

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
        result[j] = np.array([atom.mass(), atom.n_prot, atom.n_elec,
                              atom.sx, atom.sy, atom.sz,
                              atom.vx, atom.vy, atom.vz])
    return result


def rhs_elec(E: float, V: Callable, x: float, y: float):
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
    x_span = (0.001, 2)

    # todo: How do I set these ICs/BCs?
    ψ0 = .1
    ψ_p0 = .022

    rhs = partial(rhs_elec, E, V)

    soln = solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 1000))

    return soln


def nuc_potential(bodies: Iterable[Atom], sx: float, sy: float, sz: float) -> float:
    result = 0
    for body in bodies:
        # Coulomb potential
        # result += e * body.charge() / np.sqrt((body.sx - sx)**2 + (body.sy - sy**2) + (body.sz - sz)**2)
        result += e**2 * body.charge() / np.sqrt((body.sx - sx)**2 + (body.sy - sy**2) + (body.sz - sz)**2)

    return result


def hydrogen_static():
    """A time-independent simulation of the electron cloud surrounding a hydrogen atom"""
    E = -.007  # todo how do I set this?

    # ground level hydrogen: 13.6eV
    E = 1/2

    V = partial(nuc_potential, [Atom(1, 0, 1, 0, 0, 0, 0, 0, 0)])

    V2 = lambda x: -V(x)  # Due to lower potential of electrons near protons. ?

    soln = elec(E, V2)

    # plt.plot(np.linspace(.001, .01), V2(np.linspace(0.001, .01)))

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


hydrogen_static()
