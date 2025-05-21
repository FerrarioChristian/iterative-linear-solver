from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

from demo.cli import bcolors, parse_arguments
from demo.constants import MATRICES, SOLVERS, TOLERANCES
from demo.logger import (
    print_intermediate_result,
    print_matrix_properties,
    print_results,
    save_results,
    visualize_results,
)
from linear_solver.analysis.benchmark import benchmark_solver
from linear_solver.analysis.plot import spy_matrices

args = parse_arguments()
max_iterations = args.max_iter
skip_check = args.skip_check
spy = args.spy
TOLERANCES = args.tolerances
output_dir = args.output_dir


def main():

    if spy:
        to_plot = [load_matrix(matrix) for matrix in MATRICES]
        spy_matrices(to_plot, save_path=output_dir)
        return

    df = run_benchmark(
        MATRICES,
        SOLVERS,
        TOLERANCES,
        max_iterations,
    )

    save_results(df, output_dir)
    print_results(df)
    visualize_results(df, output_dir)


def load_matrix(matrix_path):
    """Carica una matrice da file e la converte in un array denso."""
    A = mmread(matrix_path).tocsr()
    return A


def run_benchmark(matrices, solvers, tolerances, max_iterations):
    """Esegue il benchmark per i solver e le matrici specificate."""
    all_results = []
    for matrix in matrices:
        A = load_matrix(matrix)
        n = A.shape[0]
        x = np.array([1] * n)
        b = A @ x

        print_matrix_properties(matrix, A, skip_check)

        for tol in tolerances:
            print(bcolors.OKBLUE + f"\n### Tolleranza: {tol:.0e} ###" + bcolors.ENDC)
            for solver_class in solvers:
                res = benchmark_solver(
                    solver_class, A, b, tol=tol, max_iter=max_iterations
                )
                matrix_path = Path(matrix)
                res.matrix = matrix_path.name
                all_results.append(res)
                print_intermediate_result(res, x, max_iterations)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    main()
