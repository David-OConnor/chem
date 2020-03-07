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


def nuc_pot(nuclei: Iterable[Nucleus], sx: float) -> float:
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e**2 / abs(nuclei.sx - sx)

    return result

###


def ti_schrod_rhs_new(E: float, V: Callable, x: complex, y: Tuple[complex, complex]) -> Tuple[complex, complex]:
    """
    d²ψ/dx² = 2m/ħ² * (V(x) - E)ψ
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ**2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def solve_new(E: float, V: Callable, ψ0: complex, ψ_p0: complex, x_span: Tuple[complex, complex]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """

    # todo: Cage yourself. You're evaluating two related concepts: 1 -Applying complex numbers
    # todo to all phases of the problem (Try making each axis complex), and 2 - Trying to find a more global approach to iterating,
    # todo that solves the problem without iterating along x from a point. If you can produce
    # todo the same result as the traditional approach without encountering a singularity,
    # todo, there's a good chance you've succeeded.
    # todo: Some type of self-consistent procedure?

    rhs = partial(ti_schrod_rhs_new, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def h_static_new(E: float) -> Tuple[np.ndarray, np.ndarray]:

    # todo: Think about what your constraints should be.
    # todo: Perhaps you're looking for any constraint that guarantees a "non-blown-up" soln.

    # todo Shower thought: Is this wf shape we see on a graph a slice of a higher-D landscape?

    # how does E tie in ?
    # How can we find the energy *landscape* as hills/valleys etc, where E values can
    # be chosen?

    ψ_inf = 0 + 0 * i

    x_span = (-100, 100)

    V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln = solve_new(E, V_elec, ψ0, ψ_p0, x_span)

    norm = simps(np.conj(soln) * soln, x=soln.t)
    return soln.t, soln/norm**.5

###


def ti_schrod_rhs(E: float, V: Callable, x: float, y: Tuple[complex, complex]) -> Tuple[complex, complex]:
    """
    d²ψ/dx² = 2m/ħ² * (V(x) - E)ψ
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ**2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def solve(E: float, V: Callable, ψ0: float, ψ_p0: float, x_span: Tuple[float, float]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    rhs = partial(ti_schrod_rhs, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def h_static(E: float) -> Tuple[np.ndarray, np.ndarray]:
    ψ0 = 0
    ψ_p0 = 5

    x_span = (-100, .0000001)

    V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln_orig = solve(E, V_elec, ψ0, ψ_p0, x_span)
    soln_left = soln_orig.y[0]
    soln_right = np.flip(soln_left)
    soln = np.concatenate([soln_left, -soln_right])
    x = np.concatenate([soln_orig.t, np.flip(-soln_orig.t)])

    norm = simps(np.conj(soln) * soln, x=x)
    return x, soln/norm**.5


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
