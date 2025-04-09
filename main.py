import numpy as np
from scipy.io import mmread

from linear_solver.analysis.benchmark import benchmark
from linear_solver.solvers import *


def main():
    # Caricamento della matrice A e del vettore b
    A = mmread("./matrici/spa1.mtx").toarray()
    n = A.shape[0]

    x = np.array([1] * n)
    b = A @ x

    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]

    print("Insert max iterations: ")
    maxIter = int(input())

    solvers = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]

    # Esecuzione dei solutori con diverse tolleranze
    for tol in tolerances:
        print(f"\n### Tolleranza: {tol:.0e} ###")
        for solver_class in solvers:
            res = benchmark(solver_class, A, b, tol=tol, max_iter=maxIter)
            print_result(res, x)
        for solver_class in solvers:
            res = benchmark(solver_class, A, b, tol=tol, max_iter=maxIter)
            print_result(res, x)


def print_result(result: dict, x_true: np.ndarray):
    x = result["solution"][0]
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"\n--- {result['solver_class']} ---")
    print(f"Errore relativo: {err:.6e}")
    print(f"Iterazioni:     {result['iterations']}")
    print(f"Tempo (s):      {result['execution_time']:.6f}")


if __name__ == "__main__":
    main()
