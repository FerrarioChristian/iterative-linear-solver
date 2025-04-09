import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import LinearSolver


class JacobiSolver(LinearSolver):
    def solve(self, tol=1e-8, max_iter=1000):
        n = self.A.shape[0]

        D = np.diag(np.diag(self.A))

        B = D - self.A

        xnew = np.array([0] * n)
        xold = xnew + 1

        while criterioDiArresto(self.A, xnew, self.b, tol, self.iterations, max_iter):
            xold = xnew
            xnew = np.linalg.inv(D) @ (B @ xold + self.b)
            self.iterations += 1

        return xnew
