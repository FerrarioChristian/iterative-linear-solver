import os

import numpy as np

from demo.cli import bcolors
from linear_solver.analysis.benchmark import BenchmarkResult
from linear_solver.analysis.compare_plot import plot_execution_time, plot_relative_error
from linear_solver.matrix_analysis.structure import analyze_matrix


def print_results(df):
    """
    Print the final results of the benchmark in a readable table format.
    Args:
        df (pd.DataFrame): DataFrame containing the benchmark results.
    """
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


def print_intermediate_result(
    result: BenchmarkResult, x_true: np.ndarray, max_iterations: int
) -> None:
    """
    Print the intermediate solve results of the benchmark.
    Args:
        result (BenchmarkResult): The result of the benchmark.
        x_true (np.ndarray): The true solution vector.
        max_iterations (int): Maximum number of iterations for the solver.
    """
    x = result.solution
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    result.relative_error = float(err)

    print(f"\n--- {result.solver_class} ---")
    print(f"Errore relativo: {err:.2e}")
    print(f"Iterazioni:      {result.iterations}")
    print(f"Tempo (s):       {result.execution_time:.6f}")

    if result.iterations == max_iterations:
        print(bcolors.FAIL + "Errore: non convergente" + bcolors.ENDC)


def print_matrix_properties(path, A, skip_check):
    """
    Prints the properties of the matrix.
    Args:
        path (str): Path to the matrix file.
        A (np.ndarray): The matrix to analyze.
        skip_check (bool): Whether to skip the matrix properties check.
    """
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


def visualize_results(df, output_dir):
    """
    Visualizes the benchmark results by plotting execution time and relative error.
    Args:
        df (pd.DataFrame): DataFrame containing the benchmark results.
        output_dir (str): Directory to save the plots.
    """
    plot_execution_time(df, output_dir)
    plot_relative_error(df, output_dir)


def save_results(df, save_path=None):
    """
    Save the benchmark results to a CSV file.
    Args:
        df (pd.DataFrame): DataFrame containing the benchmark results.
        save_path (str): Directory to save the results. If None, no file is saved.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "results.csv"), index=False)
