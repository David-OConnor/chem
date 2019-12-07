from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # For WSL

eps = 1e-14


# Create mesh and define function space
domain = Circle(Point(0, 0), 64)
mesh = generate_mesh(domain, 64)
# V = FunctionSpace(mesh, 'P', 1)

# mesh = CircleMesh(Point(0,0),5,0.7)
V = FunctionSpace(mesh, 'Lagrange', 3)

# Define boundary condition
ψ_D = Expression('0', degree=1)


# Circular boundary condition. Adapt to 0-1 domain, with origin at x/y = 1/2
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, ψ_D, boundary)

# Define functions
ψ = TrialFunction(V)
v = TestFunction(V)

Pot = Expression('0.0', degree=1)

# Some Schrodinger-specific details
E = -1/2
# Pot = Expression('-1 / (sqrt(pow(x[0], 2) + pow(x[1], 2)))', degree=2)

#define problem
# from https://fenicsproject.org/qa/3215/solving-2d-schrodinger-equation/
a = (inner(grad(ψ), grad(v) + Pot*ψ*v))*dx
m = ψ*v*dx

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

ψ = Function(V)


# a = dot(grad(ψ), grad(v))*dx
# L = f*v*dx
#

#
for i in range(0,5):
    #extract next eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    print('eigenvalue:', r)

    #assign eigenvector to function
    ψ.vector()[:] = rx

    plot(ψ, interactive=True)

# Compute solution
# ψ = Function(V)
# solve(a == L, ψ, bc)
#
# # Plot solution and mesh
# plot(ψ)
# plt.legend()
# plot(mesh)

# # Save solution to file in VTK format
# vtkfile = File('poisson/solution.pvd')
# vtkfile << ψ
#
# # Compute error in L2 norm
# error_L2 = errornorm(ψ_D, ψ, 'L2')
#
# # Compute maximum error at vertices
# vertex_values_ψ_D = ψ_D.compute_vertex_values(mesh)
# vertex_values_ψ = ψ.compute_vertex_values(mesh)
#
# error_max = np.max(np.abs(vertex_values_ψ_D - vertex_values_ψ))
#
# # Print errors
# print('error_L2  =', error_L2)
# print('error_max =', error_max)
#
# # Hold plot

plt.show()
