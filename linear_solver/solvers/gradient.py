from typing import Optional

import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import BaseIterativeSolver


class GradientSolver(BaseIterativeSolver):
    def solve(
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:

        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        bi = np.linalg.norm(self.b)
        r = np.array([1] * n)
        while criterioDiArresto(r, bi, tol, self._iterations, max_iter):
            xold = xnew
            r = self.b - self.A @ xold
            d = (np.transpose(r) @ r) / (np.transpose(r) @ (self.A @ r))
            xnew = xold + d * r
            self._iterations += 1

        return xnew
