from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


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
