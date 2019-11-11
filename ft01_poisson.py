from fenics import *
import matplotlib.pyplot as plt
# from IPython import embed; embed()

def boundary(x, on_boundary):
    return on_boundary


def solver(f: Expression, uD: Expression, Nx: float, Ny: float, degree=1) -> Function:
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    bc = DirichletBC(V, uD, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u


def plot_soln(u: Function) -> None:
    # Plot solution and mesh
    plot(u)
    # plot(mesh)

    # Save solution to file in VTK format
    vtkfile = File('poisson/solution.pvd')
    vtkfile << u

    plt.show()


def calc_errors(u_D: Expression, u: Function):
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    import numpy as np

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

    # Print errors
    print('error_L2  =', error_L2)
    print('error_max =', error_max)


def run_solver():
    f = Constant(-6.0)
    # Define boundary condition
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    u_ = solver(f, u_D, 8, 8)

    plot_soln(u_)
    # calc_errors(u_D, u_)


if __name__ == '__main__':
    run_solver()
