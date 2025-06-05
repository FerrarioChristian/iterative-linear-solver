from typing import Optional

import numpy as np
from scipy.sparse import tril

from linear_solver.convergence.criteria import (
    StoppingCriterion,
    default_stopping_criterion,
)
from linear_solver.solvers.lower_triangular import lower_triangular_solve

from .base_solver import BaseIterativeSolver


class GaussSeidelSolver(BaseIterativeSolver):
    """
    Gauss-Seidel method for solving linear systems.
    This method iteratively refines the solution to the system of equations Ax = b,
    where A is a square matrix, x is the solution vector, and b is the right-hand side vector.
    """

    def solve(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        stopping_criterion: StoppingCriterion = default_stopping_criterion,
    ) -> np.ndarray:
        """
        Solve the linear system Ax = b using the Gauss-Seidel method.
            :param tol: Tolerance for convergence.
            :param max_iter: Maximum number of iterations.
            :param stopping_criterion: Function to determine when to stop the iterations.
            :return: The solution vector x.
        """

        self.tol = tol if tol is not None else self.tol
        self.max_iter = max_iter if max_iter is not None else self.max_iter
        self._iterations = 0

        n = self.A.shape[0]
        L = tril(self.A, format="csr")
        xnew = np.array([0] * n)
        r = np.array([1] * n)
        bi = np.linalg.norm(self.b)

        while stopping_criterion(
            r, float(bi), self.tol, self._iterations, self.max_iter
        ):
            r = self.b - self.A @ xnew
            xnew = xnew + lower_triangular_solve(L, r)
            self._iterations += 1
            self._residuals.append(r)

        self._solution = xnew
        return xnew


def main():
    return False


if __name__ == "__main__":
    main()
