# Testing numerical eigen solving
import numpy as np
import matplotlib.pyplot as plt

import op

#target v: e^x

N = 30
min_ = 0
max_ = 10

x = np.linspace(min_, max_, N)
dx = (max_ - min_) / N

D = op.diff_op(N)

def main():


    # d/dx f(x) = λ f(x),  ie Dv = λv
    return solve(D, x)


def test():
    print(dx)
    v = np.exp(x)
    result = D @ v * dx

    # We show that the derivative of v is the same as  v.
    plt.plot(x, v)
    plt.plot(x, result)


    print(v)
    print(result)

    plt.show()


# main()
test()
