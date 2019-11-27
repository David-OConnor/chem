from fenics import *
from mshr import *
import matplotlib.pyplot as plt


# Create mesh and define function space
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 64)

V = FunctionSpace(mesh, 'P', 1)

eps = 1e-14

#build essential boundary conditions
def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,Constant(0.0) , u0_boundary)

#define functions
u = TrialFunction(V)
v = TestFunction(V)

# Pot = Expression('0.0')
E = -1/2
Pot = Expression('-1 / (sqrt(pow(x[0], 2) + pow(x[1], 2)))', degree=2)

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
for i in range(0,5):
    #extract next eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    print('eigenvalue:', r)

    #assign eigenvector to function
    u.vector()[:] = rx

    plot(u, interactive=True)