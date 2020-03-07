"""Testing new methods of integration that may be appropriate to use on the
Schrodinger equation"""

from numpy import pi as π
import numpy as np
from numpy import array, linspace
import matplotlib.pyplot as plt

from powerseries import Taylor

τ = 2*π


def rhs(x: np.ndarray) -> np.ndarray:
    """Returns the 2nd deriv"""
    return -x


def interp_with_sec_deriv(x: np.ndarray, x_result: np.ndarray, ψ: np.ndarray, ψ_pp: np.ndarray) -> np.ndarray:
    pass


def method1(x: np.ndarray, ψ: np.ndarray) -> np.ndarray:
    """Interpolation, alternating between midpoints"""
    N = 1  # Number of times to iterate

    # right shift; midpoints.
    x_shifted = x + (x[1] - x[0])
    # ψ_shifted = np.empty(ψ.size - 1)

    plt.grid()
    # plt.axes()

    curve_x = linspace(x[0], x[-1], 1000)

    for i in range(N):  # iterations
        ψ_pp = rhs(ψ)

        for j in range(ψ.size):  # nodes
            ts = Taylor(x[j], [0, 0, ψ_pp[j]])

            # for k in range(x - curve_buffer, x + curve_buffer, 100):
            curve = np.empty(curve_x.size)
            for k in range(curve_x.size):  # values of x to calc the 2nd deriv
                curve[k] = ts.value(curve_x[k])

            curve += ψ[j]

            plt.plot(curve_x, curve)

        # ψ_shifted = interpolate.spline(ψ, ψ_pp)

        # node x, x we're solving for, ψ at nodes, ψ'' at nodes
        # ψ_shifted = interpolate(x, x_shifted, ψ, ψ_pp)

        # plt.plot(x, ψ_pp)
        plt.show()

    # result = np.empty(ψ.size + ψ_shifted.size)
    # result[0::2] = ψ
    # result[1::2] = ψ_shifted
    # return result

    return ψ


def run_method1() -> None:
    x_min = 0
    x_max = 2*τ

    initial_x = linspace(x_min, x_max, 9)

    # initialize to arbitrary values. This may affect which soln we converge to, if multiple exist.
    initial_ψ = array([1, -1, 1, -1, 1, -1, 1, -1, 1])
    assert(initial_x.size == initial_ψ.size)

    method1(initial_x, initial_ψ)


run_method1()
