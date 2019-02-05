from dataclasses import dataclass

from functools import partial
from typing import List, Iterable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ħ = 1
m_e = 0.00054858
m_p = 1
m_n = 1
e = 1  # Does it?? elementary charge

n = 1
L = 10
E = n**2 * np.pi**2 * ħ**2 / (2 * m_e * L**2)



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


def V(x):
    if x < 0 or x > L:
        return 2
    return 0


def rhs_elec(x, y):

    ψ, φ = y
    ψ_p = φ
    φ_p = 2 * m_e / ħ ** 2 * (V(x) - E) * ψ

    return ψ_p, φ_p


def elec(E: float):
    # Things to control:
    # E: Normalization; can take multiple, quantized values
    # s0: ?
    # v0: ?

    x_span = (-15, 15)

    # for ψ0 in np.linspace(-.2, .2, 5):
    #     for ψ_p0 in np.linspace(-.2, .2, 5):

    ψ_0 = .1
    ψ_p0 = .1
    soln = solve_ivp(rhs_elec, x_span, (ψ0, ψ_p0), t_eval=np.linspace(x_span[0], x_span[1], 1000))

            # plt.plot(soln.t, soln.y[0])
    # plt.show()
    return soln


def nuc_potential_1d(bodies: Iterable[Atom], sx: float) -> float:
    result = 0
    for body in bodies:
        result += body.charge() / abs(body.sx - sx)

    return result


def electron_potential_1d(soln, n_electrons, sx: float) -> float:
    # Approximate the electric potential by sampling points from the solution to the TISE.
    prob = soln.y[0]**2

    # Normalize the probability.
    total_prob = sum(prob)

    result = 0
    for i in soln.len():
        result += n_electrons * -e * (prob[i] / total_prob) / abs(soln.t[i] - sx)

    return result


def rhs_nbody(atoms: Iterable[Atom], t, y, ):
    sx, v = y

    soln_elec = elec(E)

    # is this right???
    n_electrons = sum(atom.n_elec for atom in atoms)

    V = nuc_potential_1d(atoms, sx) + electron_potential_1d(soln_elec, n_electrons, sx)

    


def nbody():
    # Solve the nbody problem using classical nuclei

    atoms = [
        Atom(1, 0, 1, -1, 1),
        Atom(1, 0, 1, 1, -1),
    ]


    t_span = (0, 100)

    for Ψ0 in np.linspace()

    soln = solve_ivp(rhs_nbody, t_span, (ψ0, ψ_p0), t_eval=np.linspace(t_span[0], t_span[1], 1000))




def run_2d():
    pass


nbody()