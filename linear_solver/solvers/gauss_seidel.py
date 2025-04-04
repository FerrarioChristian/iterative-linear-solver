import numpy as np

from linear_solver.utils import criterioDiArresto

from . import lower_triangular
from .base_solver import LinearSolver


class GaussSeidelSolver(LinearSolver):
    """
    Gauss-Seidel method for solving linear systems.
    """

    def solve(self, tol=1e-8, max_iter=1000):
        n = self.A.shape[0]

        L = np.tril(self.A)
        B = self.A - L

        xnew = np.array([0] * n)
        xold = xnew + 1
        k = 0

        while criterioDiArresto(self.A, xnew, self.b, tol, k, max_iter):
            xold = xnew
            xnew = lower_triangular.solve(L, (self.b - B @ xold))
            k += 1

        return xnew, k


def main():
    return False


if __name__ == "__main__":
    main()
