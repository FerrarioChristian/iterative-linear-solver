import time
from typing import Any, Type

import numpy as np

from linear_solver.solvers.base_solver import (
    BaseIterativeSolver,
)  # o il path relativo corretto


def benchmark(
    solver_class: Type[BaseIterativeSolver], A: np.ndarray, b: np.ndarray, **kwargs
) -> dict[str, Any]:
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

    return {
        "solver_class": solver_class.__name__,
        "solution": solution,
        "execution_time": end - start,
        "iterations": solver._iterations,
        "tolerance": kwargs.get("tol", 1e-10),
    }
