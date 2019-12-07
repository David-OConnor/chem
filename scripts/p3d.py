# Use a radial approach, where we only use a single atom, taking advantage of
# the potential's spherical symmetry.u

from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from consts import *

# E = n**2 * np.pi**2 * ħ**2 / (2 * m_e * L**2)

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0


@dataclass
class Coord:
    x: float
    y: float
    z: float


@dataclass
class Nucleus:
    n_prot: float
    n_neut: float
    # n_elec: float
    s: Coord
    v: Coord

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


@dataclass
class Electron:
    s: Coord
    spin: bool  # True for up


def atoms_to_array(atoms: List[Nucleus]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.n_elec,
                              atom.s.x, atom.s.y, atom.s.z,
                              atom.v.x, atom.v.y, atom.v.z])
    return result


def schrod(E: float, V: Callable, s: Coord, y):
    ψ, φx, φy, φz = y

    dψ_dx, dψ_dy, dψ_dz = φx, φy, φz

    c = 2 * m_e / ħ ** 2 * (V(s) - E)

    ddψ_dx2 = c - ddψ_dy2 - ddψ_dz2
    ddψ_dy2 = c - ddψ_dx2 - ddψ_dz2
    ddψ_dz2 = c - ddψ_dx2 - ddψ_dy2

    return dψ_dx, ddψ_dx2, dψ_dy, ddψ_dy2, dψ_dz, ddψ_dz2


def schrod_split(ddψ_da2: float, ddψ_db2: float, E: float, V: Callable, s: Coord, y):
    # a and b are the other two second derivatives: ie if we're using this to find
    # change with respect to x, a is y, and b is z.
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(s) - E) * ψ - ddψ_da2 - ddψ_db2

    return ψ_p, φ_p


# function schrod_split(ddψ_da2, ddψ_db2, u, E, V, s)
#     # a and b are the other two second derivatives: ie if we're using this to find
#     # change with respect to x, a is y, and b is z.
#     ψ, φ = u
#     ψ_p = φ
#     φ_p = 2m_e / ħ ** 2 * (V(s) - E) * ψ - ddψ_da2 - ddψ_db2
#
#     ψ_p, φ_p
# end


def elec(E: float, V: Callable):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    x_span = (0.001, 2)

    ψ0 = .1
    ψ_p0 = .022

    rhs = partial(schrod, E, V)

    soln = solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 1000))

    return soln


def nuc_potential(bodies: Iterable[Nucleus], n_electrons: int, s: Coord) -> float:
    result = 0
    for body in bodies:
        # Coulomb potential
        # result += e * body.charge() / np.sqrt((body.sx - sx)**2 + (body.sy - sy**2) + (body.sz - sz)**2)

        dist = np.sqrt((body.s.x - s.x)**2 + (body.s.y - s.y**2) + (body.s.z - s.z)**2)

        # Coulomb
        result -= e * n_electrons * body.charge() / dist

        l = 1

        # Centrifigal force
        result += ħ**2 * l*(l+1) / (2*m_e * dist**2)

    return result


def hydrogen_static():
    """A time-independent simulation of the electron cloud surrounding a hydrogen atom"""
    E = -.007  # todo how do I set this?

    # ground level hydrogen: 13.6eV
    E = 1/2

    V = partial(nuc_potential, [Nucleus(1, 0, 0, 0, 0, 0, 0, 0)])

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


def wave_func(x: Coord) -> float:
    """Return the value of the wave function at a given position."""
    pass


hydrogen_static()
