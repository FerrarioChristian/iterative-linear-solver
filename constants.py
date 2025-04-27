from linear_solver.solvers import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)

TOLERANCES = [10e-4, 10e-6, 10e-8, 10e-10]
SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
MATRICES = [
    "matrices/spa1.mtx",
    "matrices/spa2.mtx",
    "matrices/vem1.mtx",
    "matrices/vem2.mtx",
]
