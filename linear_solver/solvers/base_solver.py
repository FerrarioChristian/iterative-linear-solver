from abc import ABC, abstractmethod

import numpy as np


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b
        self.iterations = 0
        self.solution = None

    @abstractmethod
    def solve(self, tol=1e-10, max_iter=1000) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        :param tol: Tolerance for convergence.
        :param max_iter: Maximum number of iterations.
        :return: Solution vector x.
        """
        pass
