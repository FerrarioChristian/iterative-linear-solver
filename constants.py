from linear_solver.solvers import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)

TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
MATRICES = [
    "matrices/spa1.mtx",
    "matrices/spa2.mtx",
    "matrices/vem1.mtx",
    "matrices/vem2.mtx",
]
