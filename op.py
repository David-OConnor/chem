import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


class OpComponent(Enum):
    SCALE = auto
    # DERIV = auto
    ROTATION = auto


@dataclass
class Operator:
    """A linear operator"""
    # eigenvalues are not tied to a basis.
    eigvals: List[complex]
    eigvecs: List[np.ndarray]
    # todo: combine eigvals and vecs as below. Sep now for experimenting
    # eig: List[(complex, np.ndarray)]

    components: List[Tuple[OpComponent, float]]

    @classmethod
    def make_scalar(const: complex, N: float) -> 'Operator':
        return Operator([const] * N)

    def natural_basis(self) -> np.ndarray:
        return np.diagflat(self.eigvals)

    def in_basis(self, basis: np.ndarray) -> np.ndarray:
        return np.linalg.inv(basis) @ self.natural_basis() @ basis

    def det(self) -> complex:
        return reduce(lambda a, b: a * b, self.eigvals)

    def trace(self) -> complex:
        return sum(self.eigvals)

    def to_std_basis(self, basis: np.ndarray) -> np.ndarray:
        return np.linalg.inv(basis) @ self.natural_basis() @ basis

    def on(self, state: np.ndarray) -> np.ndarray:
        # todo: Overload matrix mult or mult?
        return self.natural_basis() @ state


from numpy import pi

# from scipy.constants import epsilon_0, hbar_si, m_e as me_si, elementary_charge as e_si
# Atomic units
ħ = 1
m_e = 1
i = complex(0, 1)
a_0 = 1
from numpy import pi

N = 50
xmin = 0
xmax = 10
x = np.linspace(xmin, xmax, N)
dx = (xmax - xmin) / (N)


def posit_to_ind(posit: float) -> int:
    """Calculate the matrix index position from a position value"""
    port_through = (posit - xmin) / (xmax - xmin)
    # todo: Averaging logic? Return the portion between indexes? These problems vanish
    # with high N.
    return int(round(port_through / N))


def ind_to_posit(ind: int) -> float:
    """Inverse of ind_to_posit"""
    port_through = ind / N
    return (xmax - xmin) * port_through + xmin


# -1
def diff_op(N2: Optional[int]) -> np.ndarray:
    N_ = N2 if N2 else N
    part1 = np.array([.5] * N_)
    return (np.roll(np.diagflat(-part1), -1) + np.roll(np.diagflat(part1), 1)) / dx


def diff_sq_op() -> np.ndarray:
    """A more direct variant that squaring the diff op."""
    return (-2 * np.eye(N) + np.roll(np.eye(N), -1) + np.roll(np.eye(N), 1)) / dx




def hamiltonian(V: np.ndarray) -> np.ndarray:
    """ie RHS of Schrodinger's equation"""
    D = diff_op()
    KE = -ħ ** 2 / (2 * m_e) * (D @ D)

    # print("D: ", D)
    # print("KE: ", KE)

    return KE + V


def hamilt_h() -> np.ndarray:
    V = np.diagflat([1 / ind_to_posit(i) for i in range(N)])
    print("V: ", V)

    return hamiltonian(V)


def flatten2d(A: np.ndarray) -> np.ndarray:
    """Convert to a format that can be differentiated with a 1d solver."""
    return np.concatenate([A.flatten(), A.T.flatten()])


def reassemble2d(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Undo flatten"""
    N_ = int(np.sqrt(A.shape[0]/2))

    pt1, pt2 = np.split(A, 2)

    pt1 = pt1.reshape(N_, N_)
    pt2 = pt2.reshape(N_, N_)

    return pt1, pt2


def main():
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

    N_ = 5

    A = np.array([
        1, 2, 3, 4, 5,
        3, 4, 5, 6, 7,
        5, 6, 7, 8, 9,
        7, 8, 9, 10, 11,
        9, 10, 11, 12, 13,
    ]).reshape(N_, N_)

    D = diff_op(2 * N_**2) * dx  # todo temp compensate dx

    print(reassemble2d(D @ flatten2d(A)))

