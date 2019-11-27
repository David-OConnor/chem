from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

eps = 1e-14

# Define boundary condition
# 0 for x on the boundary; our starting and ending TI states for t on the boundary.
# todo: Cauchy or Neumann BC for t? Probably Cauchy using the energy??
ψ_D = Expression('abs(x[0]) < eps || abs(x[0]) - 1 < eps ? 0 : ', degree=1, eps=eps)


def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, ψ_D, boundary)

# Define variational problem
E = -1/2
# todo x[1] is time; formulate it that way.
V = Expression('-1. / pow(x[0], 2)', degree=2)
f = Expression('(V - E)*[ψ]', degree=2, V=V, E=E)

ψ = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(ψ), grad(v))*dx
L = f*v*dx

# Compute solution
ψ = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << ψ

# Compute error in L2 norm
error_L2 = errornorm(ψ_D, ψ, 'L2')

# Compute maximum error at vertices
vertex_values_ψ_D = ψ_D.compute_vertex_values(mesh)
vertex_values_ψ = ψ.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_ψ_D - vertex_values_ψ))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot

plt.show()
