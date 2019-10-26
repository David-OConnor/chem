from dataclasses import dataclass


from functools import partial
from typing import List, Iterable, Callable, Tuple

import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps

from consts import *
import op

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0

i = complex(0, 1)



# Lookin into matrix mechanics, and Feynman path integral approaches too

# orbitals are characterized (in simple cases) by quantum numbers n, l, and m, corresponding to
# energy, angular momentum, and magnetic (ang momentum vec component)

# spin?

# Do electrons (electrically) interact with themselves?

# Breaking up a numerical problem into a number of solveable analytic ones??? Eg set up
# an arbitrary V as a series of step Vs which have anal solns


# Free variables: 2? Energy, and ψ_p_0(ψ). Eg we can set ψ to what we wish, find the ψ_p that
# works with it (and the E set), then normalize.

@dataclass
class Nucleus:
    n_prot: float
    n_neut: float
    sx: float
    vx: float

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


@dataclass
class Electron:
    # ψ: List[float]
    ψ: Callable[[float], float]
    spin: bool  # True for up


def atoms_to_array(atoms: List[Nucleus]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.sx, atom.vx])
    return result


def nuc_potential(nuclei: Iterable[Nucleus], sx: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        result -= e / abs(nuclei.sx - sx)

    return result


def ti_schrod(E: float, V: Callable, x: float, y):
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def elec(E: float, V: Callable, ψ0: float, ψ_p0: float, x_span: Tuple[float, float]):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """


    rhs = partial(ti_schrod, E, V)
    return solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 10000))


def h_static(E: float) -> Tuple[np.ndarray, np.ndarray]:
    ψ0 = 0
    ψ_p0 = -.0001
    x_span = (-40, 0.0001)

    V_elec = partial(nuc_potential, [Nucleus(1, 0, 0, 0)])

    # Left and right of the x=0 coulomb singularity. Assume odd solution around x=0.
    soln_orig = elec(E, V_elec, ψ0, ψ_p0, x_span)
    soln_left = soln_orig.y[0]
    soln_right = np.flip(soln_left)
    soln = np.concatenate([soln_left, -soln_right])
    t = np.concatenate([soln_orig.t, np.flip(-soln_orig.t)])

    norm = simps(np.conj(soln) * soln, x=t)
    return t, soln/norm**.5


def evolve(state: np.ndarray, t0: float, t: float, E: float) -> np.ndarray:
    """e^(-itH/ħ)"""
    # todo eig st basis?
    # e^(i*E*t/ħ)
    # Assume H doesn't depend on time?
    # H w/power expansion instead of E ??
    # print(exp(-i*(t - t0) * E / ħ))
    return state * exp(-i*(t - t0) * E / ħ)


def td_schrod(E: float, V: Callable, x: float, ψ: complex):
    return E * ψ / (i*ħ)


def evolve2(state: np.ndarray, t0: float, t: float, E: float):
    """iħ*dψ/dt = Hψ"""
    t_span = (-10, 10)
    ψ0, ψ_p0 = 0, 1

    rhs = partial(ti_schrod, E)
    return solve_ivp(rhs, t_span, (ψ0, ψ_p0), t_eval=np.linspace(t_span[0], t_span[1], 10000))


def plot_h_static():
    # Negative E implies bound state; positive scattering.
    # ψ_p0 should be 0 for continuity across the origin.
    # E should be a whittaker energy, ie -1/2, -2/9, -1/8, -.08 etc
    # Only odd states (n = 1, 3, 5 etc) correspond to 3d H atom.
    n = 1
    E = -2/(n+1)**2

    t, ψ = h_static(E)

    fig, ax = plt.subplots()
    ax.plot(t, ψ)
    ax.plot(t, np.conj(ψ) * ψ)

    integ = simps(np.conj(ψ) * ψ, x=t)
    # print("Norm sq: ", integ)

    ax.grid(True)
    plt.xlim(-10, 10)
    plt.show()


def plot_h_static_evolve():
    dt = 0.01

    n = 1
    E = -2 / (n + 1) ** 2 # must match in calc_hydrogen static

    soln = h_static(E)
    for i in range(1):
        evolved = evolve(soln.y[0]**2, 0, dt, E)
        plt.plot(soln.t, evolved)

    plt.show()

    # def calc_h_static_eig():
    #     D = op.diff_op()
    #     D2 = D @ D
    #
    #     E = -1/2
    #     N = 50
    #     x = np.linspace(N) - 25
    #
    #     V = nuc_potential([Nucleus(1, 0, 0, 0)], x)
    #
    #     return 1/(2*E*V) * (D2 @ x)


def electron_potential(soln, n_electrons, sx: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0]**2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


def rhs_nbody(atoms: Iterable[Nucleus], t: float, y: np.array):
    # Un-flatten, back into a cleaner form.
    y_flat = y.reshape(-1, ATOM_ARR_LEN)

    # Inefficient, but makes code cleaner
    # Don't have number of neutrons here
    atoms2 = [Nucleus(a[1], 0, a[3], a[4]) for a in y_flat]


    # Chicken-egg scenario with calculating electric potential, so use the previous
    # iteration's field.
    soln_elec = elec(E, V_prev)

    # todo wrong! y[t] is y at diff posit, not time.
    E = i * ħ * soln_elec.y[t] - soln_elec.y[t-1]

    # is this right???
    total_n_elec = sum(atom.n_elec for atom in y_flat)

    # potential = k * charge / r
    # force = k * charge1 * charge2 /r**2
    # force = potential * charge (other)

    result = []

    for j in range(y_flat.shape[0]):
        mass, n_prot, n_elec, sx, dx_dt = y_flat[j]

        V_nuc = nuc_potential(atoms2, sx)
        V_elec = electron_potential(soln_elec, total_n_elec, sx)
        force = (V_nuc + V_elec) * (n_prot * e)

        force_x = force

        # Calculate the acceleration
        ddx_ddt = force_x / mass

        # First three values don't change (mass, number of prots/elecs)
        result.extend([0, 0, 0, dx_dt, ddx_ddt])

    return result


def nbody():
    # Solve the nbody problem using classical nuclei

    atoms = [
        Nucleus(1, 0, 0, 0),  # Stationary hydrogen atom at origin
        # Atom(1, 0, -1, 1),
        # Atom(1, 0, 1, -1),
    ]

    atoms_array = atoms_to_array(atoms)

    # Convert atoms into a 1d array, which scipy can integrate.
    atoms_flat = atoms_array.flatten()

    rhs = partial(rhs_nbody, atoms)

    t_span = (0, 100)

    # for Ψ0 in np.linspace()

    soln = solve_ivp(rhs, t_span, atoms_flat, t_eval=np.linspace(t_span[0], t_span[1], 1000))

    plt.plot(soln.y[0])
    plt.show()


if __name__ == "__main__":
    plot_hydrogen_static()

# plot_ics()

# find_ψ_p0()
