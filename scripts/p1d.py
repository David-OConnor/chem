from dataclasses import dataclass
from scipy.integrate import solve_ivp, simps

from consts import *

from functools import partial
from typing import List, Iterable, Callable, Tuple

from numpy import exp, sqrt

import numpy as np


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# import seaborn as sns
# import plotly
# import plotly.graph_objects as go


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





def nuc_potential(nuclei: Iterable[Nucleus], sx: float) -> float:
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e / abs(nuclei.sx - sx)

    return result


def ti_schrod(E: float, V: Callable, x: float, y: Tuple[complex, complex]) -> Tuple[complex, complex]:
    """
    d²ψ/dx² = 2m/ħ² * (V(x) - E)ψ
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ**2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def nuc_elec(E: float, V: Callable, ψ0: float, ψ_p0: float, x_span: Tuple[float, float]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    rhs = partial(ti_schrod, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def h_static(E: float) -> Tuple[np.ndarray, np.ndarray]:
    ψ0 = 0
    ψ_p0 = .2

    x_span = (-100, .0000001)

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

    x, ψ = h_static(E)

    fig, ax = plt.subplots()
    ax.plot(x, ψ)
    # ax.plot(x, np.conj(ψ) * ψ)
    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


if __name__ == "__main__":
    plot_h_static()
