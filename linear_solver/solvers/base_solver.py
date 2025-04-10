from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseIterativeSolver(ABC):
    """
    Abstract base class for linear solvers.
    """

    def __init__(
        self, A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int = 1000
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
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        :param tol: Tolerance for convergence.
        :param max_iter: Maximum number of iterations.
        :return: Solution vector x.
        """
        pass

    def get_iterations(self) -> int:
        return self._iterations

    def get_residuals(self) -> list[float]:
        return self._residuals

    def get_solution(self) -> np.ndarray:
        if self._solution is None:
            raise ValueError("Solution not computed yet.")
        return self._solution
