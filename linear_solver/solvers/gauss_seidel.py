from typing import Optional
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve_triangular



import numpy as np

from linear_solver.utils import criterioDiArresto

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
        L = tril(self.A, format='csc')
        B = self.A - L
        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)

        while criterioDiArresto(r, bi, tol, self._iterations, max_iter):
            r = self.b - self.A @ xnew
            xnew = xnew + spsolve_triangular(L, r, lower=True)
            self._iterations += 1

        return xnew


def main():
    return False


if __name__ == "__main__":
    main()
