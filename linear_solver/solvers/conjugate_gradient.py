from typing import Optional

import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import BaseIterativeSolver


class ConjugateGradientSolver(BaseIterativeSolver):
    def solve(
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:

        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        rold = self.b - self.A @ xold
        dold = rold

        while criterioDiArresto(self.A, xnew, self.b, tol, self._iterations, max_iter):
            alpha = (dold @ rold) / (dold @ self.A @ dold)
            xnew = xold + alpha * dold
            rnew = rold - alpha * self.A @ dold
            beta = ((self.A @ dold) @ rnew) / ((self.A @ dold) @ dold)
            dnew = rnew - beta * dold

            xold = xnew
            rold = rnew
            dold = dnew

            self._iterations += 1

        return xnew
