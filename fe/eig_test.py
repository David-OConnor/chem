from fenics import *
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')  # For WSL


# Define mesh, function space
mesh = Mesh("box_with_dent.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
# a = dot(grad(u), grad(v))*dx

a = grad(u)*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

# Extract largest (first) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(0)

print("Largest eigenvalue: ", r)

# Initialize function and assign eigenvector
u = Function(V)
u.vector()[:] = rx

# Plot eigenfunction
plt.plot(u)
plt.show()