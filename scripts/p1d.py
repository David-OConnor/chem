from dataclasses import dataclass
from scipy.integrate import solve_ivp, simps
from scipy.fft import fft
from scipy.stats import invgauss#, invgauss_gen

from consts import *

from functools import partial
from typing import List, Iterable, Callable, Tuple

from numpy import exp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from series import Fourier, Taylor

τ = 2 * np.pi
i = complex(0, 1)


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


###


def ti_schrod_rhs_new(
    E: float, V: Callable, x: complex, y: Tuple[complex, complex]
) -> Tuple[complex, complex]:
    """
    d²ψ/dx² = 2m/ħ² * (V(x) - E)ψ
    """
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def solve_new(
    E: float, V: Callable, ψ0: complex, ψ_p0: complex, x_span: Tuple[complex, complex]
):
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
    return solve_ivp(
        rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000)
    )


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
    return soln.t, soln / norm ** 0.5


###


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


def h_static(E: float) -> Tuple[np.ndarray, np.ndarray]:
    ψ0 = 0
    ψ_p0 = 1
    # x_span = (-100, 0.0000001)
    x_span = (-100, 0.0000001)

    V_elec = partial(nuc_pot, [Nucleus(1, 0, 0, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln_orig = solve(E, V_elec, ψ0, ψ_p0, x_span)
    soln_left = soln_orig.y[0]
    soln_right = np.flip(soln_left)
    soln = np.concatenate([soln_left, -soln_right])
    x = np.concatenate([soln_orig.t, np.flip(-soln_orig.t)])

    norm = simps(np.conj(soln) * soln, x=x)
    return x, soln / norm ** 0.5


def plot_h_static(n: int = 1):
    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.
    # E should be a whittaker energy, ie -1/2, -2/9, -1/8, -.08 etc
    # Only odd states (n = 1, 3, 5 etc) correspond to 3d H atom.
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    fig, ax = plt.subplots()
    ax.plot(x, ψ)
    # ax.plot(x, np.conj(ψ) * ψ)
    ax.grid(True)
    plt.xlim(-20, 20)
    plt.show()


def test_fft():
    f = Fourier(0, [1], [0])
    f.plot((-8, 8))


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


def run_fft():
    n = 1
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    ψ = ψ.astype(np.complex128)

    x = np.linspace(-10, 10, x.size, dtype=np.complex128)

    ψ = np.zeros(len(x), dtype=np.complex128)
    ψ += exp(1 * i * x)  # this should give us 0, [1], [0]

    result = fft(ψ)

    fig, ax = plt.subplots()
    ax.plot(x, result)
    ax.grid(True)
    # plt.xlim(-10, 10)
    plt.show()

    return result


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


def fft2():
    from numpy import ones_like, cos, pi, sin, allclose

    T = 1.5  # any real number

    def f(t):
        """example of periodic function in [0,T]"""
        n1, n2, n3 = 1.0, 4.0, 7.0  # in Hz, or nondimensional for the matter.
        a0, a1, b4, a7 = 4.0, 2.0, -1.0, -3
        return (
            a0 / 2 * ones_like(t)
            + a1 * cos(2 * pi * n1 * t / T)
            + b4 * sin(2 * pi * n2 * t / T)
            + a7 * cos(2 * pi * n3 * t / T)
        )

    N_chosen = 10
    a0, a, b = fourier_series_coeff_numpy(f, T, N_chosen)

    # we have as expected that
    assert allclose(a0, 4)
    assert allclose(a, [2, 0, 0, 0, 0, 0, -3, 0, 0, 0])
    assert allclose(b, [0, 0, 0, -1, 0, 0, 0, 0, 0, 0])


def test_taylor():
    f = Taylor(0, [0, -.5, 0, 1])
    f.plot((-8, 8))


def test_fourier():
    f = Fourier(0, [2 + i, 0, -1 - 2*i, 0, 1, 0, -1, i/2, 1, 0, 0, 0, 1, 0, i, i, i, 0], [])
    f.plot((-8, 8))


def inv_gauss():
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html"""
    λ = 0.01
    mu = 100

    # x = np.linspace(invgauss.ppf(λ, mu), invgauss.ppf(0.95, mu), 100)

    x = np.linspace(0, 10, 1_000)

    plt.plot(x, invgauss.pdf(x, mu), 'r-', lw=1, alpha=0.6, label='invgauss pdf')
    plt.show()


def gen_inv_gaus():
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html#scipy.stats.geninvgauss"""
    pass


def make_hyd_dist(a: float, b: float) -> np.ndarray:
    pass


def hyd_dist():
    """Use linear superpositions of variations of the n=1 (E=-1/2) solution of the 1D Hydrogen
    atom to get solutions for higher n."""

    # We're trying to replicate this:
    n = 3
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    # By superimposing variants of n=1
    n = 1
    E = -2 / (n + 1) ** 2
    x, ψ_working = h_static(E)

    ψ_working *= -1

    fig, ax = plt.subplots()

    ax.plot(x, ψ)
    ax.plot(x, ψ_working)

    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


def h2():
    """Use linear superpositions of variations of the n=1 (E=-1/2) solution of the 1D Hydrogen
    atom to get solutions for higher n."""

    # We're trying to replicate this:
    n = 1
    E = -2 / (n + 1) ** 2
    x, ψ = h_static(E)

    ψ_working = ψ

    fig, ax = plt.subplots()

    ax.plot(x, ψ)
    ax.plot(x, ψ_working)

    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


if __name__ == "__main__":
    plot_h_static(1)
    # test_fft()
    # run_fft()
    # reimann()
    # test_taylor()
    # test_fourier()
    # inv_gauss()
    # h2()