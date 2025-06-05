from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)


class BaseIterativeSolver(ABC):
    """
    Abstract base class for linear solvers.
    This class provides a common interface for all iterative solvers.
    It defines the basic structure and methods that all solvers must implement.
    :param A: Coefficient matrix.
    :param b: Right-hand side vector.
    :param tol: Tolerance for convergence.
    :param max_iter: Maximum number of iterations.
    """

    def __init__(
        self, A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int = 20000
    ):
        self.A = A
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self._iterations = 0
        self._residuals = []
        self._solution = None

    @abstractmethod
    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        :param tol: Tolerance for convergence.
        :param max_iter: Maximum number of iterations.
        :param stopping_criterion: Stopping criterion function.
        :return: Solution vector x.
        """
        NotImplementedError("You must implement the solve method.")

    def get_iterations(self) -> int:
        return self._iterations

    def get_residuals(self) -> list[float]:
        return self._residuals

    def get_tolerance(self) -> float:
        return self.tol

    def get_max_iterations(self) -> int:
        return self.max_iter

    def get_solution(self) -> np.ndarray:
        if self._solution is None:
            raise ValueError("Solution not computed yet.")
        return self._solution
