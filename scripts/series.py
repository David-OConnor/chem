from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy import pi as π, exp, sin, cos, exp
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from p1d import h_static

i = complex(0, 1)
τ = 2 * π


@dataclass
class FourierReal:
    """A Fourier series to arbitrary precision"""

    a_0: complex  # todo: Eschew in favor of being the first coef?
    coeffs: List[Tuple[complex, complex]]  # ie [(a1, b1), (a2, b2)]
    P: float  # Period

    def value(self, x: complex) -> complex:
        result = self.a_0 / 2
        for n_, (a, b) in enumerate(self.coeffs, 1):
            result += a * np.cos(2 * π * n_ * x / self.P) + b * np.sin(
                2 * π * n_ * x / self.P
            )
        return result

    def plot(self, range_: Tuple[float, float]) -> None:
        x = np.linspace(range_[0], range_[1], 1000)
        y = []
        for v in x:
            y.append(self.value(v))

        plt.plot(x, y)
        plt.show()


@dataclass
class Fourier:
    """A Fourier series to arbitrary precision."""

    coeff_0: complex  # n = 0
    coeffs_pos: List[complex]  # Positive coefficients: n = 1, 2, 3...
    coeffs_neg: List[complex]  # Negative coefficients: n = -1, -2, -3...
    P: float = τ  # Period

    # todo: Vectorize?
    def value(self, t: float) -> complex:
        """Get a single value."""
        result = self.coeff_0

        for n, c in enumerate(self.coeffs_pos, 1):
            result += c * exp(i * τ * n * t / self.P)

        for n, c in enumerate(self.coeffs_neg, 1):
            result += c * exp(-i * τ * n * t / self.P)

        return result

    def populate(
        self, range_: Tuple[float, float], size: int = 10_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make a series of the data."""
        t = np.linspace(range_[0], range_[1], size)
        y = np.empty(t.size, dtype=np.complex128)

        for j, v in enumerate(t):
            y[j] = self.value(v)

        return t, y

    def plot(self, range_: Tuple[float, float], size: int = 10_000) -> None:
        t, y = self.populate(range_, size)

        plt.plot(t, y.real)
        plt.plot(t, y.imag)
        plt.show()

        # def scatter(self, range_: Tuple[float, float]) -> None:
        #     t = np.linspace(range_[0], range_[1], 1000)
        #     y = []   # todo: Make arrays
        #     x = []
        #     for v in t:
        #         value = self.value(v)
        #         x.append(value.real)
        #         y.append(value.imag)
        #
        #     plt.scatter(x, y)
        #     plt.show()


@dataclass
class Taylor:
    """A Taylor series to arbitrary precision"""

    a: complex  # The center
    coeffs: List[complex]  # ie f(a), f'(a), f''(a)...

    def value(self, x: complex) -> complex:
        result = 0
        for n_, f_a in enumerate(self.coeffs):
            result += f_a * (x - self.a) ** n_ / factorial(n_)
        return result

    def populate(
        self, range_: Tuple[float, float], size: int = 10_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        # todo: DRY with Fourier populate.
        t = np.linspace(range_[0], range_[1], size)
        y = np.empty(t.size, dtype=np.complex128)

        for j, v in enumerate(t):
            y[j] = self.value(v)

        return t, y

    def plot(self, range_: Tuple[float, float], size: int = 10_000) -> None:
        # todo: DRY with Fourier plot.
        t, y = self.populate(range_, size)

        plt.plot(t, y.real)
        plt.plot(t, y.imag)
        plt.show()


# todo: Can we use this 1d WF, then as we rotation through (2d for now) space, vary
# todo the coefficients to produce (2d for now )wfs? Start with just a single H atom.
# ns: Just rotate the [n, 0...] around with no modifications.

@dataclass
class Hydrogen:
    """A series using s=0 hydrogen atom wavefunctions at different energy levels."""
    # todo: Or is it n = 1, 3, 5...
    coeffs: List[complex]  # Positive coefficients: n = 0, 1, 2, 3...

    x: np.ndarray
    components: List[np.ndarray]

    def __init__(self, coeffs: List[complex]):
        self.coeffs = coeffs
        self.components = []

        n = 1   # todo: Only odd 1d coeffs for now.
        for c in self.coeffs:
            E = -2 / (n + 1) ** 2
            x, ψ = h_static(E)

            if n == 1:
                self.x = x
            self.components.append(c * ψ)

            n += 2

    def value(self, x: float) -> complex:
        """Get a single value."""
        result = 0
        for comp in self.components:
            result += np.interp([x], self.x, comp)

        return result[0]

    def plot(self, range_: Tuple[float, float] = (-20, 20), size: int = 10_000) -> None:

        ψ = np.zeros(len(self.x), dtype=np.complex128)
        for ψi in self.components:
            ψ += ψi



        # todo: DRY with other series'
        plt.plot(self.x, np.conj(ψ.real) * ψ.real)
        # plt.plot(self.x, ψ.imag)
        # plt.xlim(range_[0], range_[1])
        plt.xlim(0, range_[1])
        plt.show()

    def plot_2d(self) -> None:
        # The "blending" etc can be a periodic fn, like a sinusoid etc, over radials.

        # all 0 counts include the decay, and center.

        # for 1D:
        # n=0: start high, decay. Peaks: .15
        # n=1: start 0, decay. Peaks: 1
        # n=2: start high, cross once, decay. Peaks: .1, 2.7
        # n=3: Start 0, cross once, decay. Peaks: .76, 5.23
        # n=4: Start high, cross 2x, decay
        # n=5: Start 0, cross 2x, decay. Peaks: .74, 4.18, 13
        # n=6: start high: cross 3x, decay

        # Let n be odd only: (n+1)/2 + 1 = num zeros.

        # good diagrams:
        # https://en.wikipedia.org/wiki/Atomic_orbital

        # for 2D:
        # (n, l, (m_(l/s)). n is number of 0s. l is number of symmetry lines?. m is...?

        # 3D:
        # n is number of 0s. l is number of symmetry lines?. m+1 = num of rotations
        # - m=0 have one rotational sym axis.

        # todo: Start with n=2, and a sin wave of amplitude
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the mesh in polar coordinates and compute corresponding Z.
        # r = np.linspace(0, 1.25, 50)
        r = np.linspace(0, 8, 50)
        p = np.linspace(0, τ, 50)

        R, P = np.meshgrid(r, p)

        # Z = ((R ** 2 - 1) ** 2)
        Z = np.array([self.value(r) for r in R]) #* sin(P)
        # Z = R
        Z = Z**2

        # Express the mesh in the cartesian system.
        X, Y = R * cos(P), R * sin(P)

        # Plot the surface.
        ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

        # Tweak the limits and add latex math labels.
        ax.set_zlim(0, 10)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$\psi$')

        plt.show()


def fourier_ode():
    """
    Assume BVP, where boundaries are the value goes to 0 at +- ∞
    """
    y_0 = (0, 1)
    t = (0, 10)
    dt = 0.1

    rhs = lambda x, dx: dx

    f = FourierReal(0, [(1, 1), (1, 1), 2 * np.pi])
