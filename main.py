import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

from constants import MATRICES, SOLVERS, TOLERANCES
from linear_solver.analysis.benchmark import BenchmarkResult, benchmark_solver
from linear_solver.analysis.compare_plot import plot_execution_time
from linear_solver.matrix_analysis.structure import analyze_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-iter", type=int, default=20000, help="Maximum number of iterations"
)
parser.add_argument(
    "--skip-check", action="store_true", help="Skip matrix properties check"
)
args = parser.parse_args()

max_iterations = args.max_iter
skip_check = args.skip_check


def main():
    all_results = []

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
                res = benchmark_solver(
                    solver_class, A, b, tol=tol, max_iter=max_iterations
                )
                matrix_path = Path(matrix)
                res.matrix = matrix_path.name
                all_results.append(res)
                print_result(res, x, max_iterations)

    df = pd.DataFrame(all_results)
    df.to_csv("results.csv", index=False)
    df.to_json("results.json", orient="records", lines=True)

    print("\n--- Risultati ---")
    print(df)

    for matrix in df["matrix"].unique():
        for tol in TOLERANCES:
            subset = df[(df["matrix"] == matrix) & (df["tolerance"] == tol)]
            if not subset.empty:
                plot_execution_time(subset)


def print_result(
    result: BenchmarkResult, x_true: np.ndarray, max_iterations: int
) -> None:
    x = result.solution
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"\n--- {result.solver_class} ---")
    print(f"Errore relativo: {err:.2e}")
    print(f"Iterazioni:      {result.iterations}")
    print(f"Tempo (s):       {result.execution_time:.6f}")

    if result.iterations == max_iterations:
        print("Errore: non convergente")


def print_matrix_properties(path, A):
    print()
    print("======================================================")
    print(f"Matrice: {path} ", end="")

    if not skip_check:
        mp = analyze_matrix(A)
        print(f"| Condizionamento: {mp.condition_number:.2e}")
        print(f"Simmetria: {mp.is_symmetric} | ", end="")
        print(f"Definita positiva: {mp.is_positive_definite} | ", end="")
        print(f"Dominanza diagonale: {mp.is_diagonally_dominant}", end="")

    print("\n======================================================")


if __name__ == "__main__":
    main()
