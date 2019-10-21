from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable, Tuple

import numpy as np
from numpy import sqrt
from math import factorial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps

from consts import *

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0

import op


@dataclass
class Nucleus:
    n_prot: float
    n_neut: float
    sx: float
    vx: float
    sy: float
    vy: float

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


@dataclass
class Electron:
    ψ: Callable[[float], float]
    spin: bool  # True for up


def atoms_to_array(atoms: List[Nucleus]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.sx, atom.vx])
    return result


def schrod(E: float, V: Callable, r: Tuple[float, float], psi):
    ψ, φ = psi
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(r) - E) * ψ

    return ψ_p, φ_p


def elec(E: float, V: Callable, ψ0: float, ψ_p0: float):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    x_span = (-10, 10)

    rhs = partial(schrod, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def nuc_potential(nuclei: Iterable[Nucleus], sx: float, sy: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e / abs(sqrt((nuclei.sx - sx)**2 + (nuclei.sy - sy)**2))

    return result


def h_static(ψ0, ψ_p0, E: float):
    """A time-independent simulation of the electron cloud surrounding a hydrogen atom"""

    V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0)])

    return elec(E, V_elec, ψ0, ψ_p0)


def calc_hydrogen_static():
    # todo: Try this - use the same idea as 1d, but mark all points on a radius.
    ψ0 = 0
    ψ_p0 = 1

    E = -.5

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


def electron_potential(soln, n_electrons, sx: float, xy: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0]**2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


if __name__ == "__main__":
    plot_hydrogen_static()

