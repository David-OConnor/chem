from dataclasses import dataclass

from functools import partial
from typing import List, Iterable, Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps

from consts import *

ATOM_ARR_LEN = 5

# A global state var
V_prev: Callable = lambda sx: 0


# Lookin into matrix mechanics, and Feynman path integral approaches too

# orbitals are characterized (in simple cases) by quantum numbers n, l, and m, corresponding to
# energy, angular momentum, and magnetic (ang momentum vec component)

# spin?

# Do electrons (electrically) interact with themselves?

# Breaking up a numerical problem into a number of solveable analytic ones???


# Free variables: 2? Energy, and ψ_p_0(ψ). Eg we can set ψ to what we wish, find the ψ_p that
# works with it (and the E set), then normalize.

@dataclass
class Atom:
    n_prot: float
    n_neut: float
    n_elec: float
    sx: float
    vx: float

    def mass(self):
        return self.n_prot * m_p + self.n_neut * m_n

    def charge(self):
        # Charge from protons only.
        return self.n_prot * e


def atoms_to_array(atoms: List[Atom]) -> np.array:
    # Convert atoms to an array we'll intergrate.
    # Each row is mass, n_protons, n_electrons, position, and velocity.
    result = np.empty(len(atoms), ATOM_ARR_LEN)

    for j, atom in enumerate(atoms):
        result[j] = np.array([atom.mass(), atom.n_prot, atom.n_elec, atom.sx, atom.vx])
    return result


def schrod(E: float, V: Callable, x: float, y: float):
    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ
    # print(V(x))

    return ψ_p, φ_p


def elec(E: float, V: Callable, ψ0: float, ψ_p0: float):
    """
    Calculate the wave function for electrons in an arbitrary potential, at a single snapshot
    in time.
    """
    x_span = (1e-5, 10000)

    rhs = partial(schrod, E, V)
    soln = solve_ivp(rhs, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 1000))

    return soln


def nuc_potential(nuclei: Iterable[Atom], n_electrons: int, sx: float) -> float:
    # In 1d, we have no angular momentum/centripetal potential: Only coulomb potential.
    result = 0
    for nuclei in nuclei:
        # Coulomb potential
        # result -= e * n_electrons * nuclei.charge() / abs(body.sx - sx)
        result -= e / abs(nuclei.sx - sx)

    return result


def find_ψ_p0(ψ0: float=1/2, E: float=-.01) -> float:
    """There should be only one bound state corresponding to a combination of ψ0 and E.
    I suspect that it's really only 1 bound state for a given E, where all ψ0 and ψ'0 combinations
    that normalize are equivalent."""

    # Use the shooting method to find our solution. Narrow down the range of possible values
    # iteratively.

    START = -10
    END = 10
    check_x = 6000  # Value to confirm soln is close to 0. Higher is more precise.
    ϵ = 1e-4  # Set smaller for higher precision, but isn't as sensitive as check_x.

    MAX_ATTEMPTS = 10000
    MAX_PRECISION = 1e-17  # Just return the result if upper and lower are very close

    lower = START
    upper = END
    guess = (END - START) / 2

    for i in range(MAX_ATTEMPTS):
        soln = hydrogen_static(ψ0, guess, E)

        # ψ should be near 0 for a bound state when far from the origin, represented
        # by check_val
        # check_val = soln.y[int(len(soln.y) / 2)][0]  # todo nope. Find closest to t!
        check_val = soln.y[0][800]  # todo nope. Find closest to t!

        # Debugging code
        # print(f"upper:{upper}, lower:{lower}, guess:{guess}, check: {check_val}")
        # plt.plot(soln.y[0])
        # plt.show()

        if abs(check_val) <= ϵ or abs(upper - lower) <= MAX_PRECISION:
            return guess

        # Assume negative check_val means our guess is too low.
        if check_val < 0:
            # We know the soln is bounded below by lower.
            lower = guess
            guess += (upper - lower) / 2
        else:
            upper = guess
            guess -= (upper - lower) / 2
    return guess


def find_ψ0(E: float=-.01) -> float:
    """Find the ψ0 value leading to a normalized bound solution. There should be only 1.
    The integration result should be 1/2, since we're only integrating half the wave func."""
    # todo DRY between here and find_ψ_p0!

    TARGET = 1/2

    START = 0
    END = 2
    ϵ = 1e-4  # Set smaller for higher precision, but isn't as sensitive as check_x.

    lower = START
    upper = END
    guess = (END - START) / 2

    while True:
        soln = hydrogen_static(guess, find_ψ_p0(guess, E), E)

        norm = simps(soln.y[0][:200])

        # Debugging
        print(f"upper:{upper}, lower:{lower}, guess: {guess}, norm: {norm}")

        if abs(norm - TARGET) <= ϵ:
            return guess

        # Assume norm less than target means our guess is too low.
        if norm < TARGET:
            # We know the soln is bounded below by lower.
            lower = guess
            guess += (upper - lower) / 2
        else:
            upper = guess
            guess -= (upper - lower) / 2


def hydrogen_static(ψ0, ψ_p0, E: float):
    """A time-independet simulation of the electron cloud surrounding a hydrogen atom"""

    # ground level hydrogen: 13.6eV

    # Negative E implies bound state; positive scattering.

    V = partial(nuc_potential, [Atom(1, 0, 1, 0, 0)], 1)
    return elec(E, V, ψ0, ψ_p0)


def find_ics(E: float) -> Tuple[float, float]:
    """Return (ψ0, ψ'0) for the normalized bound state at the given energy."""
    ψ0 = 1  # Arbitrary starting value.
    # ψ0 = find_ψ0(E)
    ψ_p0 = find_ψ_p0(ψ0, E)

    soln = hydrogen_static(ψ0, ψ_p0, E)
    norm = simps(soln.y[0][:500]) * 2

    # Now that we know how to normalize, modify ψ0 appropriately:

    return ψ0 / norm, find_ψ_p0(ψ0/norm, E)


def plot_ics():
    """Plot the initial conditions over bound state."""
    SIZE = 20
    E = np.linspace(0, -.3, SIZE)
    ψ0 = np.empty(SIZE)
    ψ_p0 = np.empty(SIZE)

    for i in range(SIZE):
        ψ0[i], ψ_p0[i] = find_ics(E[i])

    print(ψ0)
    print(ψ_p0)

    plt.plot(E, ψ0)
    plt.show()

    plt.plot(E, ψ_p0)
    plt.show()


def calc_hydrogen_static():
    E = -.006
    ψ0, ψ_p0 = find_ics(E)

    soln = hydrogen_static(ψ0, ψ_p0, E)
    norm = simps(soln.y[0][:500]) * 2

    print("ψ0", ψ0, "ψ'0", ψ_p0, "NORM: ", norm)

    return soln


def electron_potential(soln, n_electrons, sx: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0]**2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


def rhs_nbody(atoms: Iterable[Atom], t: float, y: np.array):
    # Un-flatten, back into a cleaner form.
    y_flat = y.reshape(-1, ATOM_ARR_LEN)

    # Inefficient, but makes code cleaner
    # Don't have number of neutrons here
    atoms2 = [Atom(a[1], 0, a[2], a[3], a[4]) for a in y_flat]


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
        Atom(1, 0, 1, 0, 0),  # Stationary hydrogen atom at origin
        # Atom(1, 0, 1, -1, 1),
        # Atom(1, 0, 1, 1, -1),
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


def run_2d():
    pass


soln = calc_hydrogen_static()
plt.plot(soln.t, soln.y[0])
plt.show()

# plot_ics()

# find_ψ_p0()
