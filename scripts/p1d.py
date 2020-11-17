from dataclasses import dataclass
from scipy.integrate import solve_ivp, simps
from scipy.fft import fft
from scipy.stats import invgauss#, invgauss_gen

from consts import *

from functools import partial
from typing import List, Iterable, Callable, Tuple

from numpy import exp, ndarray, sqrt

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from series import Fourier, Taylor

τ = 2 * np.pi
i = complex(0, 1)

# 2020-11-15
"""
One of your goals is to figure out if you can use hydrogen (1d?) WFs as basis functions to create
arbitrary solns to the Schrodinger equation, thereby making chemistry simulation and modeling
much more computationally efficient.

It appears that for Hydrogen atoms, you can use linear combinations of 1D WFs as basis functions
in 2 adn 3d by choosing the right coefficients, and the right modifier fn (sin, cos etc) across
θ and φ to apply to 2 and 3d situations.

You need to verify that this is correct adn quantify. A challenge is finding accurate 2D orbitals
to compare your results to, and in visualizing and/or quantifying your 3D results to compare
to real results in 3d.

In parallel to verifying this, assume it's right, and try to model a 2-nucleus system. For
example, a H2 molecule. Attempt, in 1D, to find a combination of H atomic orbitals (perhaps
offset in x) that create the H2 molecular orbitals. These orbitals you're attempting to
match can be taken from real data, or by integrating. (May need to break up integration
into three areas, to avoid singularities at each nucleus).
"""


matplotlib.use("Qt5Agg")
# import seaborn as sns
# import plotly
# import plotly.graph_objects as go

# A global state var
V_prev: Callable = lambda sx: 0

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
        result -= e / abs(nuclei.sx - sx)

    return result


def ti_schrod_rhs(
    E: float, V: Callable, x: float, y: Tuple[complex, complex]
) -> Tuple[complex, complex]:
    """
    d²ψ/dx² = 2m/ħ² * (V(x) - E)ψ
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def solve(E: float, V: Callable, ψ0: float, ψ_p0: float, x_span: Tuple[float, float]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    rhs = partial(ti_schrod_rhs, E, V)
    return solve_ivp(
        rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000)
    )


def h_static(E: float) -> Tuple[ndarray, ndarray]:
    ψ0 = 0
    ψ_p0 = 1
    x_span = (-100, 0.0000001)
    # x_span = (.000001, 1)

    V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])
    # V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0), Nucleus(1, 0, 1, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln_orig = solve(E, V_elec, ψ0, ψ_p0, x_span)
    soln_left = soln_orig.y[0]
    soln_right = np.flip(soln_left)
    soln = np.concatenate([soln_left, -soln_right])
    x = np.concatenate([soln_orig.t, np.flip(-soln_orig.t)])

    norm = simps(np.conj(soln) * soln, x=x)
    return x, soln / norm ** 0.5


def plot_h_static(n: int = 1):
    """This 1d model represents the radial component of the wave function;
    ie all of a 2d shell condensed down 2 dimensions to a point."""
    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.
    # E should be a whittaker energy, ie -1/2, -2/9, -1/8, -.08 etc
    # Only odd states (n = 1, 3, 5 etc) correspond to 3d H atom.
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    fig, ax = plt.subplots()
    ax.plot(x, ψ)

    ax.grid(True)
    plt.xlim(-20, 20)
    plt.show()


def plot_h_static3d(n: int = 1):
    """Like H static, but perhaps this is the right model for 3D."""
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    ψ = sqrt(ψ**2 / x**2)

    # Post-process by flipping between 0s, to make up for info lost
    # during square root.
    ε = 1e-5  # thresh for hit a 0.
    ψ_processed = ψ
    in_inversion = False

    for j in range(ψ.size):
        if ψ[j] <= ε:
            in_inversion = not in_inversion

        if in_inversion:
            ψ_processed[j] = -ψ[j]
        else:
            ψ_processed[j] = ψ[j]

    fig, ax = plt.subplots()
    ax.plot(x, ψ)

    ax.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(-3, 3)
    plt.show()


def reimann():
    n = 1
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    ψ = ψ.astype(np.complex128)

    x = np.linspace(-10, 10, x.size, dtype=np.complex128)
    ψ = np.zeros(len(x), dtype=np.complex128)
    ψ += exp(1 * i * x)  # this should give us 0, [1], [0]

    fig, ax = plt.subplots()
    ax.plot(x, result)
    ax.grid(True)
    # plt.xlim(-10, 10)
    plt.show()

    return result


def check_wf_1d(x: ndarray, ψ: ndarray, E: float) -> ndarray:
    """Given a wave function as a set of discrete points, (Or a fn?) determine how much
    it close it is to the schrodinger equation by analyzing the derivatives.
    The result is a percent diff.

    ψ = -1/2ψ'' / (E-V)
    ψ = -1/2ψ'' / (E-1/abs(r))
    or, reversed:
    ψ'' = -2(E - 1/abs(r))ψ
    """

    # todo: Center it up? This approach lags.
    # ψ_pp = np.diff(np.diff(ψ))
    dx = (x[-1] - x[0]) / x.size

    ψ_pp = np.diff(np.diff(ψ)) / dx
    ψ_pp = np.append(ψ_pp, np.array([0, 0]))  # make the lengths match

    ψ_pp_ideal = -2 * (E - 1/np.abs(x)) * ψ

    # plt.plot(x, ψ)
    # plt.plot(x, ψ_pp)
    # plt.xlim(0, 10)
    # plt.show()

    # For now, assume assume a single protein in the nucleus, at x=0.

    ψ_ideal = -1/2 * ψ_pp / (E - 1/np.abs(x))

    # plt.plot(x, ψ_ideal)
    # plt.plot(x, ψ)
    # plt.xlim(0, 10)
    # plt.show()

    plt.plot(x, ψ)
    # plt.plot(x, ψ_pp_ideal)
    plt.xlim(0, 10)
    plt.show()

    # result = (ψ - ψ_ideal) / ψ_ideal
    result = (ψ_pp - ψ_pp_ideal) / ψ_pp_ideal

    # plt.plot(x, result)
    # plt.xlim(0, 10)
    # plt.show()

    return result


# def check_wf(ψ: Callable[(float, float), ]):
def check_wf_2d(ψ: ndarray):
    """Given a wave function as a set of discrete points, (Or a fn?) determine how much
    it close it is to the schrodinger equation by analyzing the derivatives."""
    pass


def run_check():
    n = 1
    E = -2 / (n + 1) ** 2

    x, ψ = h_static(E)
    print(check_wf_1d(x, ψ, E))


if __name__ == "__main__":
    plot_h_static3d(3)
    # test_fft()
    # run_fft()
    # reimann()
    # test_taylor()
    # test_fourier()
    # inv_gauss()
    # h2()

    # run_check()