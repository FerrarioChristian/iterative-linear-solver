from linear_solver.solvers import (
    ConjugateGradientSolver,
    GaussSeidelSolver,
    GradientSolver,
    JacobiSolver,
)
import glob

TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
MATRICES = sorted(glob.glob("matrices/*.mtx"))
