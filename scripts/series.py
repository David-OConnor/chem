from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy import pi as π
from math import factorial
import matplotlib.pyplot as plt

i = complex(0, 1)
τ = 2 * np.pi


@dataclass
class FourierReal:
    """A Fourier series to arbitrary precision"""

    a_0: complex  # todo: Eschew in favor of being the first coef?
    coeffs: List[Tuple[complex, complex]]  # ie [(a1, b1), (a2, b2)]
    P: float  # Period

    # @classmethod
    # def from_fn(cls, a_0: float, fn: Callable, precision: float):
    #     Fourier(a_0, callable(, 0) for n in range(1, 100))

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
    P: float  # Period

    def __init__(
        self,
        coeff_0: complex,
        coeffs_pos: List[complex],
        coeffs_neg: List[complex],

        P: float = τ,
    ):
        """This constructor allows us to use a default period."""
        self.coeffs_pos = coeffs_pos
        self.coeffs_neg = coeffs_neg
        self.coeff_0 = coeff_0
        self.P = P

    def value(self, t: float) -> complex:
        result = self.coeff_0

        for n, c in enumerate(self.coeffs_pos, 1):
            result += c * np.exp(i * τ * n * t / self.P)

        for n, c in enumerate(self.coeffs_neg, 1):
            result += c * np.exp(-i * τ * n * t / self.P)

        return result

    def plot(self, range_: Tuple[float, float], size: int = 10_000) -> None:
        t = np.linspace(range_[0], range_[1], size)
        y_real = np.empty(t.size)
        y_imag = np.empty(t.size)
        for j, v in enumerate(t):
            value = self.value(v)
            y_real[j] = value.real
            y_imag[j] = value.imag

        plt.plot(t, y_real)
        plt.plot(t, y_imag)
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

    def plot(self, range_: Tuple[float, float]) -> None:
        x = np.linspace(range_[0], range_[1], 1000)
        y = []
        for v in x:
            y.append(self.value(v))

        plt.plot(x, y)
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
