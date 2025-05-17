import os
from collections.abc import Iterable
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def spy_matrices(*matrices, save_path: Optional[str] = None):
    """
    Visualize the matrices in sparse format.
    Accepts one or more matrix objects as arguments or a single iterable of matrices.
    """
    if len(matrices) == 1 and isinstance(matrices[0], Iterable):
        matrices_to_plot = list(matrices[0])
    else:
        matrices_to_plot = list(matrices)

    num_matrices = len(matrices_to_plot)
    if num_matrices == 0:
        print("nessuna matrice fornita.")
        return

    if num_matrices == 1:
        fig, ax = plt.subplots(1, 1)
        axs = [ax]  # Crea una lista contenente il singolo oggetto Axes
    else:
        cols = (num_matrices + 1) // 2
        rows = (num_matrices + cols - 1) // cols
        _, axs = plt.subplots(rows, cols, figsize=(14, 8))
        axs = axs.flatten()

    for i, matrix in enumerate(matrices_to_plot):
        try:
            nnz = np.count_nonzero(matrix)
            total_elements = matrix.size
            sparsity_percentage = (nnz / total_elements) * 100

            axs[i].spy(matrix)
            axs[i].set_title(f"Matrice {i+1}")
            axs[i].set_xlabel(f"Densità: {sparsity_percentage:.2f}%")

        except Exception as e:
            axs[i].text(
                0.5,
                0.5,
                f"Errore con la matrice {i+1}:\n{e}",
                ha="center",
                va="center",
            )
            axs[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    if save_path:
        save_path = os.path.join(save_path, "plots")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/spy.png")
    else:
        plt.show()


def plot_convergence(
    residuals: List[float],
    iterations: Optional[int] = None,
    title: str = "Convergence of the Iterative Solver",
    xlabel: str = "Iterations",
    ylabel: str = "Residual Norm",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the convergence of the iterative solver.

    :param residuals: List of residual norms at each iteration.
    :param iterations: Number of iterations performed.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the plot. If None, the plot will be shown.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, marker="o", linestyle="-", color="b")

    if iterations is not None:
        plt.axhline(
            y=residuals[iterations], color="r", linestyle="--", label="Final Residual"
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_execution_time(
    execution_times: List[float],
    title: str = "Execution Time of the Iterative Solver",
    xlabel: str = "Iterations",
    ylabel: str = "Execution Time (seconds)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the execution time of the iterative solver.

    :param execution_times: List of execution times at each iteration.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the plot. If None, the plot will be shown.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(execution_times, marker="o", linestyle="-", color="g")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_relative_error(
    relative_errors: List[float],
    title: str = "Relative Error of the Iterative Solver",
    xlabel: str = "Iterations",
    ylabel: str = "Relative Error",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the relative error of the iterative solver.

    :param relative_errors: List of relative errors at each iteration.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the plot. If None, the plot will be shown.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(relative_errors, marker="o", linestyle="-", color="m")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
