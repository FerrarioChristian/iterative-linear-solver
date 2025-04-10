from typing import Optional

import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import BaseIterativeSolver


class JacobiSolver(BaseIterativeSolver):
    def solve(
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:

        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]

        D = np.diag(np.diag(self.A))

        B = D - self.A

        xnew = np.array([0] * n)
        xold = xnew + 1

        while criterioDiArresto(self.A, xnew, self.b, tol, self._iterations, max_iter):
            xold = xnew
            xnew = np.linalg.inv(D) @ (B @ xold + self.b)
            self._iterations += 1

        return xnew
