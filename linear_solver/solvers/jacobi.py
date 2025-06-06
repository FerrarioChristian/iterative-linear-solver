from typing import Optional

import numpy as np
from scipy.sparse import diags

from linear_solver.convergence.criteria import (StoppingCriterion,
                                                default_stopping_criterion)

from .base_solver import BaseIterativeSolver

# A = P-N, se A e P simmetriche e def positive, se 2P-A definita positiva e simmetrica allora si converge


class JacobiSolver(BaseIterativeSolver):
    """
    Jacobi iterative solver for linear systems of equations.
    This class implements the Jacobi method for solving the system of equations Ax = b,
    where A is a square matrix and b is a vector.
    """

    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b using the Jacobi method.
        Args:
            tol (float, optional): Tolerance for convergence. Default is 1e-10.
            max_iter (int, optional): Maximum number of iterations. Default is 1000.
            stopping_criterion (function, optional): Stopping criterion function.
                Default is default_stopping_criterion.
        Returns:
            np.ndarray: Solution vector x.
        """

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        D = 1.0 / self.A.diagonal()

        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)
        while stopping_criterion(
            r, float(bi), self.tol, self._iterations, self.max_iter
        ):
            xold = xnew
            r = self.b - self.A @ xnew
            xnew = xold + D * r
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew
