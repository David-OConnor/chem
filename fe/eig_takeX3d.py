#Feras Aldahlawi
#Schrodinger equation solver using fenics

from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # For WSL

#define mesh and function space
mesh = UnitCubeMesh(20, 20, 20)
V = FunctionSpace(mesh, 'Lagrange', 3)

#build essential boundary conditions
def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,Constant(0.0) , u0_boundary)

#define functions
u = TrialFunction(V)
v = TestFunction(V)


Pot = Expression('0.0', degree=1)

#define problem
a = (inner(grad(u), grad(v)) + Pot*u*v)*dx
m = u*v*dx

A = PETScMatrix()
M = PETScMatrix()
_ = PETScVector()
L = Constant(0.)*v*dx

assemble_system(a, L, bc, A_tensor=A, b_tensor=_)
#assemble_system(m, L,bc, A_tensor=M, b_tensor=_)
assemble_system(m, L, A_tensor=M, b_tensor=_)

#create eigensolver
eigensolver = SLEPcEigenSolver(A,M)
eigensolver.parameters['spectrum'] = 'smallest magnitude'
eigensolver.parameters['tolerance'] = 1.e-15

#solve for eigenvalues
eigensolver.solve(5)

u = Function(V)
for i in range(0, 5):
    #extract next eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    print('eigenvalue:', r)

    #assign eigenvector to function
    u.vector()[:] = rx

    breakpoint()

    plot(u, interactive=True)
    plt.show()

# ?:
# _ (2 1 0) (2 1 1) (3 2 1) _ (4 3 2)?
