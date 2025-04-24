from typing import Optional

import numpy as np

from linear_solver.utils import criterioDiArresto

from . import lower_triangular
from .base_solver import BaseIterativeSolver


class GaussSeidelSolver(BaseIterativeSolver):
    """
    Gauss-Seidel method for solving linear systems.
    """

    def solve(
        self, tol: Optional[float] = None, max_iter: Optional[int] = None
    ) -> np.ndarray:
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        L = np.tril(self.A)
        B = self.A - L
        xnew = np.array([0] * n)
        xold = xnew + 0.5

        while criterioDiArresto(self.A, xnew, self.b, tol, self._iterations, max_iter):
            xold = xnew
            xnew = lower_triangular.solve(L, (self.b - B @ xold))
            self._iterations += 1

        return xnew


def main():
    return False


if __name__ == "__main__":
    main()
