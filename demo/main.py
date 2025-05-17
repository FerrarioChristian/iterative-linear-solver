import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

from demo.cli import bcolors, parse_arguments
from demo.constants import MATRICES, SOLVERS, TOLERANCES
from linear_solver.analysis.benchmark import BenchmarkResult, benchmark_solver
from linear_solver.analysis.compare_plot import plot_execution_time, plot_relative_error
from linear_solver.analysis.plot import spy_matrices
from linear_solver.matrix_analysis.structure import analyze_matrix

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
    visualize_results(df)


def load_matrix(matrix_path):
    """Carica una matrice da file e la converte in un array denso."""
    A = mmread(matrix_path)
    if not isinstance(A, np.ndarray):
        A = A.toarray()
    return A


def run_benchmark(matrices, solvers, tolerances, max_iterations):
    """Esegue il benchmark per i solver e le matrici specificate."""
    all_results = []
    for matrix in matrices:
        A = load_matrix(matrix)
        n = A.shape[0]
        x = np.array([1] * n)
        b = A @ x

        print_matrix_properties(matrix, A)

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


def save_results(df, save_path=None):
    """Salva i risultati in formato CSV"""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "results.csv"), index=False)


def print_results(df):
    """Stampa i risultati in modo leggibile."""
    print(
        bcolors.HEADER
        + "\n-----------------------------------------------------------------------------"
    )
    print(
        "----------------------------- Risultati Finali ------------------------------"
    )
    print(
        "-----------------------------------------------------------------------------"
        + bcolors.ENDC
    )
    for matrix in df["matrix"].unique():
        print(f"\nMatrice: {bcolors.BOLD}{bcolors.WARNING}{matrix}{bcolors.ENDC}")
        # Filtra i risultati per la matrice corrente
        subset = df[df["matrix"] == matrix].copy()
        subset.drop(columns=["matrix"], inplace=True)
        subset.sort_values(by=["tolerance", "solver_class"], inplace=True)
        subset = subset.drop(columns=["solution"])
        subset["tolerance"] = subset["tolerance"].apply(lambda x: f"{float(x):.0e}")
        columns_order = [
            "solver_class",
            "tolerance",
            "relative_error",
            "execution_time",
            "iterations",
        ]
        subset = subset[columns_order]
        print(subset.to_string(index=False))


def visualize_results(df):
    plot_execution_time(df, output_dir)
    plot_relative_error(df, output_dir)


def print_intermediate_result(
    result: BenchmarkResult, x_true: np.ndarray, max_iterations: int
) -> None:
    x = result.solution
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    result.relative_error = float(err)

    print(f"\n--- {result.solver_class} ---")
    print(f"Errore relativo: {err:.2e}")
    print(f"Iterazioni:      {result.iterations}")
    print(f"Tempo (s):       {result.execution_time:.6f}")

    if result.iterations == max_iterations:
        print(bcolors.FAIL + "Errore: non convergente" + bcolors.ENDC)


def print_matrix_properties(path, A):
    """Stampa le proprietà della matrice in modo leggibile."""
    print(bcolors.HEADER + "\n======================================================")
    print(f"Matrice: {path}")

    if skip_check:
        print("Analisi delle proprietà della matrice saltata.")
    else:
        mp = analyze_matrix(A)
        properties = [
            f"Dimensione: {A.shape[0]} x {A.shape[1]}",
            f"Condizionamento: {mp.condition_number:.2e}",
            f"Simmetria: {mp.is_symmetric}",
            f"Definita positiva: {mp.is_positive_definite}",
            f"Dominanza diagonale: {mp.is_diagonally_dominant}",
        ]
        print("\n".join(properties))

    print("======================================================" + bcolors.ENDC)


if __name__ == "__main__":
    main()
