import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import LinearSolver


class GradientSolver(LinearSolver):
    def solve(self, tol=1e-8, max_iter=1000):
        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        k = 0

        while criterioDiArresto(self.A, xnew, self.b, tol, k, max_iter):
            xold = xnew
            r = self.b - self.A @ xold
            d = (np.transpose(r) @ r) / (np.transpose(r) @ (self.A @ r))
            xnew = xold + d * r
            k += 1

        return xnew, k
