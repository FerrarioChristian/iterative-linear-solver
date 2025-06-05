from typing import Optional

import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)

from .base_solver import BaseIterativeSolver


class GradientSolver(BaseIterativeSolver):
    """
    Gradient Descent Method for solving linear systems Ax = b,
    where A is a symmetric positive-definite matrix.
    """

    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b using the Gradient Descent method.
            :param tol: Tolerance for convergence.
            :param max_iter: Maximum number of iterations.
            :param stopping_criterion: Function to determine when to stop the iterations.
            :return: The solution vector x.
        """

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        bi = np.linalg.norm(self.b)
        r = np.array([1] * n)
        while stopping_criterion(
            r, float(bi), self.tol, self._iterations, self.max_iter
        ):
            xold = xnew
            r = self.b - self.A @ xold
            tr = np.transpose(r)
            d = (tr @ r) / (tr @ (self.A @ r))
            xnew = xold + d * r
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew
