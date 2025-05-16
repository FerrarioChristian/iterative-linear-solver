import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_execution_time(df, save_path: Optional[str] = None):
    """Genera un grafico per ogni matrice, confrontando tutte le tolleranze."""
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
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/execution_time_{matrix}.png")
        else:
            plt.show()


def plot_relative_error(df, save_path: Optional[str] = None):
    """Genera un grafico per ogni matrice, confrontando tutte le tolleranze."""
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
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/relative_error_{matrix}.png")
        else:
            plt.show()


def plot_sparsity(
    A: np.ndarray, matrix_name: str, save_path: Optional[str] = None
) -> None:
    """
    Plot the sparsity pattern of a matrix.

    :param A: La matrice da visualizzare.
    :param matrix_name: Nome della matrice per il titolo del grafico.
    :param save_path: (Opzionale) Percorso per salvare l'immagine.
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
