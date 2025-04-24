from typing import Optional

import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import BaseIterativeSolver

# A= P-N, se A e P simmetriche e def positive, se 2P-A definita positiva e simmetrica allora si converge
class JacobiSolver(BaseIterativeSolver):
    def solve(
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:

        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        
        D = np.diag(np.diag(self.A))
        D = np.linalg.inv(D)
        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)
        while criterioDiArresto(r, bi, tol, self._iterations, max_iter):
            xold = xnew
            r = self.b - self.A@xnew
            xnew = xold + D@r
            self._iterations += 1
            
        return xnew
