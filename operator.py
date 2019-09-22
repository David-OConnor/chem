import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from typing import Tuple, List
import matplotlib.pyplot as plt


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

N = 5
xmin = 0
xmax = 5
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
def diff_op() -> np.ndarray:
    part1 = np.array([.5] * N)
    return (np.roll(np.diagflat(-part1), -1) + np.roll(np.diagflat(part1), 1)) / dx


def diff_op_2d() -> np.ndarray:
    a0 = 1 / 2 * np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    a1 = 1 / 2 * np.array([[0, -1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    a2 = 1 / 2 * np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 1], [0, 0, 1, 0]])
    a3 = 1 / 2 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])

    return np.stack([a0, a1, a2, a3], 0)


def diff_op_k() -> np.ndarray:
    a0 = diff_op()
    a1 = diff_op()

    return np.kron(a0, a1)


def diff_by_ax2d(D: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff_0 = np.empty([N, N])
    for i in range(N):
        diff_0[i] = D @ state[i]

    transp = state.T
    diff_1 = np.empty([N, N])
    for i in range(N):
        diff_1[i] = D @ transp[i]
    return (diff_0, diff_1)


def diff_by_ax3d(D: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff_0 = np.empty([N, N, N])
    for i in range(N):
        diff_0[i] = D @ state[i]

    transp = np.transpose(state, [1, 0, 2])
    diff_1 = np.empty([N, N, N])
    for i in range(N):
        for j in range(N):
            diff_1[i, j] = D @ transp[i, j]

    transp2 = np.transpose(state, [1, 2, 0])
    diff_2 = np.empty([N, N, N])
    for i in range(N):
        for j in range(N):
            diff_2[i, j] = D @ transp2[i, j]

    return (diff_0, diff_1, diff_2)


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


def hyd_ode():
    pass


def main():
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

    # Derivative operator
    # D = diff_op()
    # print("DX: ", dx)

    # H = hamilt_h()

    # eigs = np.linalg.eigvals(H)
    # print("H: ", H)
    # print("E: ", eigs)
    # print("min: ", min(abs(eigs)))

    # n = 1
    # h = lambda r: 1/np.sqrt(pi) * (1/a_0)**(3/2) * np.exp(-r/a_0)

    # plt.plot(eigs)
    # plt.show()

    # D2 = diff_op_2d()
    D = diff_op()
    D2k = diff_op_k()
    # print("D2d: ", D2d)

    ax1 = np.array([
        0, 1, 2, 3, 4,
        2, 3, 4, 5, 6,
        4, 5, 6, 7, 8,
        6, 7, 8, 9, 10,
        8, 9, 10, 11, 12,
    ]).reshape(N, N)

    print(ax1.flatten("F"))
    x = D2k @ ax1.flatten("F")
    print("X: ", x)

    ax2 = ax1 + 3
    ax3 = ax1 + 6
    ax4 = ax1 + 9
    ax5 = ax2 + 12

    state = np.stack([ax1, ax2, ax3, ax4, ax5], 0)
    # print("state", state)

    # (diff_0, diff_1, diff_2) = diff_by_ax3d(D, state)

    # print("D2 on state: ", (D2d @ state).reshape(5, 5))
    # print("axis 0: ", diff_0)
    # print("axis 1: ", diff_1)
    # print("axis 2: ", diff_2)

    # KE = -ħ**2 / (2*m_e) * (D @ D)
    # V = np.diagflat([1 / ind_to_posit(i) for i in range(N)])

    # plt.plot(x, KE @ x)
    # plt.show()


main()
