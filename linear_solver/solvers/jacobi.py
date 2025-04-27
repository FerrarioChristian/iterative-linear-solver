from typing import Optional

import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)

from .base_solver import BaseIterativeSolver

# A = P-N, se A e P simmetriche e def positive, se 2P-A definita positiva e simmetrica allora si converge


class JacobiSolver(BaseIterativeSolver):
    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        D = np.diag(np.diag(self.A))
        D = np.linalg.inv(D)
        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)
        while stopping_criterion(r, float(bi), self.tol, self._iterations, self.max_iter):
            xold = xnew
            r = self.b - self.A @ xnew
            xnew = xold + D @ r
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew
