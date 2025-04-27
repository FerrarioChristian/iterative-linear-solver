from typing import Optional

import numpy as np

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)

from . import lower_triangular
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
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        L = np.tril(self.A)
        B = self.A - L
        xnew = np.array([0] * n)
        xold = xnew + 1
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)

        while stopping_criterion(r, float(bi), tol, self._iterations, max_iter):
            r = self.b - self.A @ xnew
            xnew = xnew + lower_triangular.solve(L, r)
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew


def main():
    return False


if __name__ == "__main__":
    main()
