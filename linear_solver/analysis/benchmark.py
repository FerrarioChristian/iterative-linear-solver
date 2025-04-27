import time
from dataclasses import dataclass
from typing import Any, Type

import numpy as np

from linear_solver.solvers.base_solver import BaseIterativeSolver


@dataclass
class BenchmarkResult:
    """
    Class to store the results of a benchmark.
    Attributes:
        solver_class (str): Name of the solver class.
        solution (np.ndarray): Solution vector.
        execution_time (float): Time taken to solve the system.
        iterations (int): Number of iterations taken to converge.
        tolerance (float): Tolerance used for convergence.
    """

    solver_class: str
    solution: np.ndarray
    execution_time: float
    iterations: int
    tolerance: float
    matrix: str = "Unknown"


def benchmark_solver(
    solver_class: Type[BaseIterativeSolver], A: np.ndarray, b: np.ndarray, **kwargs
) -> BenchmarkResult:
    """
    Esegue il benchmark di un solver: tempo di esecuzione, iterazioni, tolleranza.

    Args:
        solver_class: Classe che estende LinearSolver.
        A: Matrice dei coefficienti.
        b: Vettore dei termini noti.
        **kwargs: Argomenti passati al metodo `solve()` del solver (es. tol, max_iter).

    Returns:
        dict con:
            - solver_class (str)
            - solution (np.ndarray)
            - execution_time (float)
            - iterations (int)
            - tolerance (float)
    """
    solver = solver_class(A, b)
    start = time.perf_counter()
    solution = solver.solve(**kwargs)
    end = time.perf_counter()

    return BenchmarkResult(
        solver_class=solver_class.__name__,
        solution=solution,
        execution_time=end - start,
        iterations=solver.get_iterations(),
        tolerance=solver.get_tolerance(),
    )
