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

        B = D - self.A
        #print("Superamento criterio convergenza Jacobi: ", np.allclose(D, D.T) and np.all(np.linalg.eigvals(D) > 0) and np.all(np.linalg.eigvals((2*D)-self.A) > 0) and np.allclose((2*D)-self.A, ((2*D)-self.A).T))

        xnew = np.array([0] * n)
        xold = xnew + 0.5

        while criterioDiArresto(self.A, xnew, self.b, tol, self._iterations, max_iter):
            xold = xnew
            xnew = np.linalg.inv(D) @ (B @ xold + self.b)
            self._iterations += 1

        return xnew
