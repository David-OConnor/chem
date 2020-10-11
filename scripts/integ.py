"""Testing new methods of integration that may be appropriate to use on the
Schrodinger equation"""

from numpy import pi as π
import numpy as np
from numpy import array, linspace
import matplotlib.pyplot as plt

from series import Taylor

from math import e

τ = 2 * π


def rhs(x: np.ndarray) -> np.ndarray:
    """Returns the 2nd deriv"""
    return -x


def interp_with_sec_deriv(
    x: np.ndarray, x_result: np.ndarray, ψ: np.ndarray, ψ_pp: np.ndarray
) -> np.ndarray:
    pass


def method1(x: np.ndarray, ψ: np.ndarray) -> np.ndarray:
    """Interpolation, alternating between midpoints"""
    N = 1  # Number of times to iterate

    # right shift; midpoints.
    # x_shifted = x + (x[1] - x[0])
    # ψ_shifted = np.empty(ψ.size - 1)

    plt.grid()
    # plt.axes()

    curve_x = linspace(x[0], x[-1], 1000)

    for i in range(N):  # iterations
        ψ_pp = rhs(ψ)

        taylors = []

        for j in range(x.size):  # nodes
            # For ψ', use the slope between the prev and following nodes.
            # if 1 < j < x.size - 1:
            #     ψ_p = (ψ[j+1] - ψ[j-1]) / (x[j+1] - x[j-1])
            # else:
            #     ψ_p = 0

            # ψ_p = 1

            for θ in linspace(0, τ / 2, 5):
                ψ_p = np.tan(θ)
                ts = Taylor(x[j], [ψ[j], ψ_p, ψ_pp[j]])
                # taylors.append(ts)

                curve = np.empty(curve_x.size)
                for k in range(curve_x.size):  # values of x to calc the 2nd deriv
                    curve[k] = ts.value(curve_x[k])

                plt.plot(curve_x, curve)

        # Now massage, attempting to get tangency.
        for j in range(1, x.size - 1):
            pass
            # l_curve_val = taylors[j-1].value(x[j])
            # r_curve_val = taylors[j+1].value(x[j])

        # ψ_shifted = interpolate.spline(ψ, ψ_pp)

        # node x, x we're solving for, ψ at nodes, ψ'' at nodes
        # ψ_shifted = interpolate(x, x_shifted, ψ, ψ_pp)

        # plt.plot(x, ψ_pp)
        plt.plot(x, ψ, linestyle="", marker="o")
        plt.ylim([-5, 5])
        plt.show()

    # result = np.empty(ψ.size + ψ_shifted.size)
    # result[0::2] = ψ
    # result[1::2] = ψ_shifted
    # return result

    return ψ


def run_method1() -> None:
    x_min = 0
    x_max = 2 * τ

    SIZE = 9
    initial_x = linspace(x_min, x_max, SIZE)
    print(initial_x)
    # initialize to arbitrary values. This may affect which soln we converge to, if multiple exist.
    # initial_ψ = array([0, 3, 0, -1, 0, 1, 0, -1, 0])
    # initial_ψ = array([1, 4.81047, 23.1406926, 111.317778, 535.491656, 12575.9705, 0, -1, 0])
    initial_ψ = e ** (initial_x)

    # initial_ψ[3] = 60
    # initial_ψ[2] = 10
    # initial_ψ[1] = 20

    initial_ψ = np.sin(initial_x)
    initial_ψ[2] = 1
    assert initial_x.size == initial_ψ.size

    method1(initial_x, initial_ψ)


run_method1()
