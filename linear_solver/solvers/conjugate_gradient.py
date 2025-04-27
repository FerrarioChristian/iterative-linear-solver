from typing import Optional

import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)
from linear_solver.solvers.base_solver import BaseIterativeSolver


class ConjugateGradientSolver(BaseIterativeSolver):
    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:

        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        rold = self.b - self.A @ xold
        dold = rold
        bi = np.linalg.norm(self.b)
        while stopping_criterion(rold, float(bi), tol, self._iterations, max_iter):
            h = self.A @ dold
            alpha = (dold @ rold) / (dold @ h)
            xnew = xold + alpha * dold
            rnew = rold - alpha * h
            beta = ((h) @ rnew) / ((h) @ dold)
            dnew = rnew - beta * dold

            xold = xnew
            rold = rnew
            dold = dnew

            self._iterations += 1
            self._residuals.append(rold)

        self._solution = xnew
        return xnew
