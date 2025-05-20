import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_execution_time(df, save_path: Optional[str] = None):
    """
    Plot the execution time for each solver and tolerance.
    Args:
        df (pd.DataFrame): DataFrame containing the benchmark results.
        save_path (str, optional): Path to save the plots. If None, the plots will be shown.
    """
    for matrix in df["matrix"].unique():
        plt.figure(figsize=(10, 6))
        tolerances = df["tolerance"].unique()

        for tol in tolerances:
            subset = df[(df["matrix"] == matrix) & (df["tolerance"] == tol)]
            if not subset.empty:
                plt.plot(
                    subset["solver_class"],
                    subset["execution_time"],
                    marker="o",
                    linestyle="-",
                    label=f"Tol={tol:.0e}",
                )

        plt.title(f"Execution Time Comparison for {matrix}")
        plt.xlabel("Solver")
        plt.ylabel("Execution Time (seconds)")
        plt.grid()
        plt.legend(title="Tolerances")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            save_path = os.path.join(save_path, "plots")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/execution_time_{matrix}.png")
        else:
            plt.show()


def plot_relative_error(df, save_path: Optional[str] = None):
    """
    Plot the relative error for each solver and tolerance.
    Args:
        df (pd.DataFrame): DataFrame containing the benchmark results.
        save_path (str, optional): Path to save the plots. If None, the plots will be shown.
    """
    for matrix in df["matrix"].unique():
        plt.figure(figsize=(10, 6))
        tolerances = df["tolerance"].unique()

        for tol in tolerances:
            subset = df[(df["matrix"] == matrix) & (df["tolerance"] == tol)]
            if not subset.empty:
                plt.plot(
                    subset["solver_class"],
                    subset["relative_error"],
                    marker="o",
                    linestyle="-",
                    label=f"Tol={tol:.0e}",
                )

        plt.title(f"Relative Error Comparison for {matrix}")
        plt.xlabel("Solver")
        plt.ylabel("Relative Error")
        plt.grid()
        plt.legend(title="Tolerances")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            save_path = os.path.join(save_path, "plots")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/relative_error_{matrix}.png")
        else:
            plt.show()


def plot_sparsity(
    A: np.ndarray, matrix_name: str, save_path: Optional[str] = None
) -> None:
    """
    Plot the sparsity pattern of a matrix.
    Args:
        A (np.ndarray): The matrix to visualize.
        matrix_name (str): Name of the matrix for the title.
        save_path (str, optional): Path to save the plot. If None, the plot will be shown.
    """
    plt.figure(figsize=(8, 8))
    plt.spy(A, markersize=1)
    plt.title(f"Sparsity Pattern - {matrix_name}")
    plt.xlabel("Colonne")
    plt.ylabel("Righe")
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
