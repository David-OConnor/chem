from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy import pi as π, exp, sin, cos, exp
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from p1d import h_static, h_static_3d

i = complex(0, 1)
τ = 2 * π

# todo: See if you can get actual distance of h2 atoms.


# Goals (2020-11-17):
# - Model and display full hydrogen wave function range from S-orbital basis functions.
# - Model H2 molecular orbitals.

# Try this next: Calculate energy by numerically integrating the wave functions (Start with 1D).
# Do you get the energies you used to generate them?

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

# @dataclass
# class Hydrogen:
#     """A series using s=0 hydrogen atom wavefunctions at different energy levels."""
#     # todo: Or is it n = 1, 3, 5...
#     coeffs: List[complex]  # Positive coefficients: n = 0, 1, 2, 3...
#
#     x: np.ndarray
#     components: List[np.ndarray]
#
#     def __init__(self, coeffs: List[complex]):
#         self.coeffs = coeffs
#         self.components = []
#
#         # n = 1
#         n = 0
#         for j, c in enumerate(self.coeffs):
#             E = -2 / (n + 1) ** 2
#             x, ψ = h_static_3d(E)
#
#             # if n == 1:
#             if n == 0:
#                 self.x = x
#
#             self.components.append(c * ψ)
#
#             # n += 2
#             n += 1
#
#     def value(self, x: float) -> complex:
#         """Get a single value."""
#         result = 0
#         for comp in self.components:
#             result += np.interp([x], self.x, comp)[0]
#
#         return result
#
#     def value_comp(self, x: np.ndarray, j: int) -> np.ndarray:
#         """Get a single value, from a specific component."""
#
#         # Freeze values close to 0, to prevent asymptote from making vis unusable.
#         # vals = np.copy(self.components[j])
#         # thresh_x = 0.5
#         # thresh_val = np.interp([thresh_x], self.x, vals)[0]
#         #
#         # for k in range(self.x.size):
#         #     if self.x[k] < thresh_x:
#         #         vals[k] = thresh_val
#
#         # return np.interp([x], self.x, vals)[0]
#         return np.interp([x], self.x, self.components[j])[0]
#
#     def plot(self, range_: Tuple[float, float] = (0, 20), shift: float = 0., size: int = 10_000, show: bool = True) -> None:
#         ψ = np.zeros(len(self.x), dtype=np.complex128)
#         for ψi in self.components:
#             ψ += ψi
#
#         fig, ax = plt.subplots()
#
#         # todo: DRY with other series'
#         ax.plot(self.x + shift, ψ.real)
#         ax.grid(True)
#
#         # plt.plot(self.x, ψ.imag)
#         # plt.xlim(range_[0], range_[1])
#         plt.xlim(0, range_[1])
#         plt.ylim(-0.02, 0.02)
#
#         if show:
#             plt.show()
#
#     def plot_2d(self) -> None:
#         # The "blending" etc can be a periodic fn, like a sinusoid etc, over radials.
#
#         # Let n be odd only: (n+1)/2 + 1 = num zeros.
#
#         # good diagrams:
#         # https://en.wikipedia.org/wiki/Atomic_orbital
#
#         # todo: Start with n=2, and a sin wave of amplitude
#         fig = plt.figure(figsize=(10, 10))
#         ax = fig.add_subplot(111, projection='3d')
#
#         N = 150
#
#         # Create the mesh in polar coordinates and compute corresponding Z.
#
#         # Example r ranges for useful visualization:
#         # n = 1: 10
#         # n = 2: 15
#         # n = 3: 30
#
#         r = np.linspace(0, 15, N)
#         p = np.linspace(0, τ, N)
#
#         R, P = np.meshgrid(r, p)
#
#         Z = np.zeros([N, N])
#         # for j in range(len(self.components)):
#         #     Z += np.array([self.value_comp(r, j) for r in R])
#
#         Z += np.array([self.value_comp(r, 1) for r in R]) * sin(P)
#         Z += np.array([self.value_comp(r, 3) for r in R]) * sin(P)
#         Z += np.array([self.value_comp(r, 5) for r in R]) * sin(P)
#
#         # Don't let center asymptote cause a spike.
#         print(Z.shape)
#         thresh_val = 0.1
#         for j in range(Z.shape[0]):
#             for k in range(Z.shape[1]):
#                 if np.abs(Z[j][k]) > thresh_val:
#                     if Z[j][k] >= 0:
#                         Z[j][k] = thresh_val
#                     else:
#                         Z[j][k] = -thresh_val
#
#         Z = Z**2  # Uncomment to square the WF.
#
#         # Express the mesh in the cartesian system.
#         X, Y = R * cos(P), R * sin(P)
#
#         # Plot the surface.
#         ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
#
#         # Tweak the limits and add latex math labels.
#         ax.set_zlim(0, 0.4)
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$y$')
#         ax.set_zlabel(r'$\psi$')
#         ax.grid(False)
#
#         plt.show()
#
#         # Reproducing Hydrogen orbitals. 2D for now.
#         # Since we're plotting in 2D, we leave out some Ms, since they're rotations
#         # in a non-applicable plane.
#
#         # Legend
#         # (n, l, m): n, modifier 1, modifier 2
#
#         # S:
#         # (n, 0, 0): n, None, None
#
#         # P:
#         # (2, 1, 0): n, no modification
#         # (3, 1, 1): n, no modification
#
#         # D:
#         # (3, 2, 0): n, no modification
#         # (4, 2, 1): n, no modification
#         # (5, 2, 2): n, no modification
#
#
# def plot_2d(self) -> None:
#     """An attempt to visualize 3D WFs by scatter density/color plots."""
#
#     # todo: Start with n=2, and a sin wave of amplitude
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     N = 150
#
#     # Create the mesh in polar coordinates and compute corresponding Z.
#
#     # Example r ranges for useful visualization:
#     # n = 1: 10
#     # n = 2: 15
#     # n = 3: 30
#
#     r = np.linspace(0, 15, N)
#     p = np.linspace(0, τ, N)
#
#     R, P = np.meshgrid(r, p)
#
#     Z = np.zeros([N, N])
#     for j in range(len(self.components)):
#         Z += np.array([self.value_comp(r, j) for r in R])
#
#     # Express the mesh in the cartesian system.
#     X, Y = R * cos(P), R * sin(P)
#
#     # Plot the surface.
#     ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
#
#     # Tweak the limits and add latex math labels.
#     ax.set_zlim(0, 0.4)
#     ax.set_xlabel(r'$x$')
#     ax.set_ylabel(r'$y$')
#     ax.set_zlabel(r'$\psi$')
#     ax.grid(False)
#
#     plt.show()
#

@dataclass
class Hydrogen3d:
    """A Hydrogen 3d superposition"""
    # todo: Or is it n = 1, 3, 5...
    # coeffs: List[complex]  # Positive coefficients: n = 0, 1, 2, 3...

    n: int
    l: int
    m: int

    x: np.ndarray
    components: List[np.ndarray]

    def __init__(self, coeffs: List[complex]):
        self.coeffs = coeffs
        self.components = []

        # n = 1   # todo: Only odd 1d coeffs for now.
        n = 0   # todo: Only odd 1d coeffs for now.
        for c in self.coeffs:
            E = -2 / (n + 1) ** 2
            x, ψ = h_static_3d(E)

            # if n == 1:
            if n == 0:
                self.x = x

            self.components.append(c * ψ)

            # n += 2
            n += 1

    def value(self, r: float, θ: float, φ: float) -> complex:
        """Get a single value."""
        result = 0
        for comp in self.components:
            result += np.interp([r], self.x, comp)[0]

        return result

    def value_comp(self, x: float, j: int) -> complex:
        """Get a single value, from a specific component."""
        return np.interp([x], self.x, self.components[j])[0]

    def plot(self, range_: Tuple[float, float] = (-20, 20), shift: float = 0., size: int = 10_000, show: bool = True) -> None:
        ψ = np.zeros(len(self.x), dtype=np.complex128)
        for ψi in self.components:
            ψ += ψi

        # todo: DRY with other series'
        plt.plot(self.x + shift, ψ.real)
        # plt.plot(self.x, ψ.imag)
        # plt.xlim(range_[0], range_[1])
        plt.xlim(0, range_[1])

        if show:
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


def plot_multiple_h(comps: List[Tuple[List[complex], float]], range_: Tuple[float, float] = (-20, 20)) -> None:
    """Comps is a list of Hydrogen series data/shifts"""

    x = Hydrogen3d(comps[0][0]).x  # todo: awk/fallible
    ψ = np.zeros(len(x), dtype=np.complex128)

    for ser, shift in comps:
        h = Hydrogen3d(ser)
        # todo: Dry(plot)

        ψj = np.zeros(len(x), dtype=np.complex128)
        for ψi in h.components:
            ψj += ψi

        # convert our numerical shift to an index-based integer
        dx = (x[-1] - x[0]) / x.size
        shift_val = shift / dx
        print(f"SF: {shift_val} dx: {dx}")
        ψj = np.roll(ψj, int(shift_val))

        ψ += ψj

    plt.plot(x, ψ.real)
    plt.xlim(range_[0], range_[1])

    plt.show()


if __name__ == "__main__":
    # n=         1     2     3
    # Hydrogen([0, 0, 0, 1, 0, 0]).plot_2d()

    plot_multiple_h([
        ([0, 1, 0, 0], 10),
        ([0, 1, 0, 0], -1),
    ])