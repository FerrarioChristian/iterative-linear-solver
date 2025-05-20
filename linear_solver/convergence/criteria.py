from typing import Callable

import numpy as np

StoppingCriterion = Callable[[np.ndarray, float, float, int, int], bool]


def default_stopping_criterion(
    r: np.ndarray, b: float, tol: float, iteration: int, max_iterations: int
) -> bool:
    """
    Default stopping criterion for iterative solvers.
    """
    return not (np.linalg.norm(r) / b < tol or iteration >= max_iterations)
