from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Fourier:
    """A Fourier series to arbitrary precision"""
    a_0: complex
    coeffs: List[Tuple[complex, complex]]  # ie [(a1, b1), (a2, b2)]
    P: float  # Period

    # @classmethod
    # def from_fn(cls, a_0: float, fn: Callable, precision: float):
    #     Fourier(a_0, callable(, 0) for n in range(1, 100))

    def value(self, x: complex) -> complex:
        result = self.a_0 / 2
        for n_, (a, b) in enumerate(self.coeffs, 1):
            result += a * np.cos(2*π*n_*x / self.P) + b * np.sin(2*π*n_*x / self.P)
        return result

    def plot(self, range_: Tuple[float, float]) -> None:
        x = np.linspace(range_[0], range_[1], 1000)
        y = []
        for v in x:
            y.append(self.value(v))

        plt.plot(x, y)
        plt.show()


@dataclass
class FourierCplx:
    """A Fourier series to arbitrary precision, in the complex plane"""
    coeffs: List[complex]
    P: float  # Period
    i = complex(0, 1)

    def value(self, t: float) -> complex:
        τ = 2 * np.pi
        result = 0
        for n_, c in enumerate(self.coeffs):
            result += c * np.exp(n_*τ*i*t / self.P)
        return result

    def plot(self, range_: Tuple[float, float]) -> None:
        t = np.linspace(range_[0], range_[1], 1000)
        y = []
        x = []
        for v in t:
            value = self.value(v)
            x.append(value.real)
            y.append(value.imag)

        plt.scatter(x, y)
        plt.show()


@dataclass
class Taylor:
    """A Taylor series to arbitrary precision"""
    a: complex  # The center
    coeffs: List[complex]  # ie f(a), f'(a), f''(a)...

    def value(self, x: complex) -> complex:
        result = 0
        for n_, f_a in enumerate(self.coeffs):
            result += f_a * (x-self.a)**n_ / factorial(n_)
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

    f = Fourier(0, [(1, 1), (1, 1), 2*np.pi])
