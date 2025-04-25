import numpy as np
import pandas as pd
from scipy.io import mmread

from linear_solver.analysis.benchmark import benchmark
from linear_solver.matrix_analysis.structure import analyze_matrix
from linear_solver.solvers import *


def main():
    all_results = []

    maxIter = int(input("Insert the maximum number of iterations: "))

    TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
    SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
    MATRICES = [
        "matrices/spa1.mtx",
        "matrices/spa2.mtx",
        "matrices/vem1.mtx",
        "matrices/vem2.mtx",
    ]

    for matrix in MATRICES:
        A = mmread(matrix)
        if not isinstance(A, np.ndarray):
            A = A.toarray()

        n = A.shape[0]
        x = np.array([1] * n)
        b = A @ x

        print_matrix_properties(matrix, A)

        for tol in TOLERANCES:
            print(f"\n### Tolleranza: {tol:.0e} ###")
            for solver_class in SOLVERS:
                res = benchmark(solver_class, A, b, tol=tol, max_iter=maxIter)
                res["matrix"] = matrix.split("/")[-1]
                all_results.append(res)
                print_result(res, x, maxIter)

    df = pd.DataFrame(all_results)
    df.to_csv("results.csv", index=False)
    df.to_json("results.json", orient="records", lines=True)

    print("\n--- Risultati ---")
    print(df)


def print_result(result: dict, x_true: np.ndarray, maxIter):
    x = result["solution"]
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"\n--- {result['solver_class']} ---")
    print(f"Errore relativo: {err:.2e}")
    print(f"Iterazioni:      {result['iterations']}")
    print(f"Tempo (s):       {result['execution_time']:.6f}")

    if result["iterations"] == maxIter:
        print("Errore: non convergente")


def print_matrix_properties(path, A):
    mp = analyze_matrix(A)
    print()
    print("======================================================")
    print(f"Matrice: {path} | Condizionamento: {mp.condition_number:.2e}")
    print(f"Simmetria: {mp.is_symmetric} | ", end="")
    print(f"Definita positiva: {mp.is_positive_definite} | ", end="")
    print(f"Dominanza diagonale: {mp.is_diagonally_dominant}")
    print("======================================================")


if __name__ == "__main__":
    main()
