from typing import Optional
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve_triangular



import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)

from .base_solver import BaseIterativeSolver


class GaussSeidelSolver(BaseIterativeSolver):
    """
    Gauss-Seidel method for solving linear systems.
    """

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
        L = tril(self.A, format='csc')
        B = self.A - L
        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)

        while stopping_criterion(r, float(bi), self.tol, self._iterations, self.max_iter):
            r = self.b - self.A @ xnew
            xnew = xnew + spsolve_triangular(L, r, lower=True)
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew


def main():
    return False


if __name__ == "__main__":
    main()
