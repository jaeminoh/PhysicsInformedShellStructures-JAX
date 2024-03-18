import scipy.sparse as spp
import jax.experimental.sparse as sp
from dolfin import *

meshlevel = 100
degree = 1
dim = 2
mesh = UnitDiscMesh.create(MPI.comm_world, meshlevel, degree, dim)
V = FunctionSpace(mesh, "Lagrange", 1)


def boundary(x):
    boundary = near(x[0] ** 2 + x[1] ** 2, 1, 1e-2)
    return boundary


u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("1 - pow(x[0], 2) - pow(x[1], 2)", degree=2)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx
# accounting for essential boundary condition
A, b = assemble_system(a, L, bcs=bc)
# conversion to sparse matrix and vector
x = mesh.coordinates()
size = x[:,0].size
indptr, indices, data = as_backend_type(A).mat().getValuesCSR()
A_spp = spp.csr_matrix((data, indices, indptr))
A_sp = sp.BCSR.from_scipy_sparse(A_spp)

b_vec = as_backend_type(b).vec().getArray()
uhat = sp.linalg.spsolve(data, indices, indptr, b_vec)
# due to the sparsity pattern, reordering is necessary.
b_recon = sp.csr_matvec(A_sp, uhat)
print(abs(b_recon - b_vec).sum() / abs(b_vec).sum())
