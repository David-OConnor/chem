from functools import partial
import numpy as np

from scripts import main
import op

import matplotlib
# matplotlib.use('qt5')  # For WSL

N = 10


def main():
    x = np.linspace(N)

    D2 = op.diff_op_sq(N)

    V_fn = partial(main.nuc_pot, [main.Nucleus(1, 0, 0, 0)])

    V_vec = np.empty(N)
    for j, x_ in enumerate(x):
        V_vec[j] = V_fn(x_)
    # Convert from a vector to a diag matrix
    V = np.diagflat(V_vec)

    ψ = -1/2 * D2 + V


if __name__ == '__main__':
    main()
