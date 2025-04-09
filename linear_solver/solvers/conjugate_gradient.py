import numpy as np

from linear_solver.utils import criterioDiArresto

from .base_solver import LinearSolver


class ConjugateGradientSolver(LinearSolver):
    def solve(self, tol=1e-8, max_iter=1000):
        n = self.A.shape[0]

        xnew = np.array([0] * n)
        xold = xnew
        rold = self.b - self.A @ xold
        dold = rold

        while criterioDiArresto(self.A, xnew, self.b, tol, self.iterations, max_iter):
            alpha = (dold @ rold) / (dold @ self.A @ dold)
            xnew = xold + alpha * dold
            rnew = rold - alpha * self.A @ dold
            beta = ((self.A @ dold) @ rnew) / ((self.A @ dold) @ dold)
            dnew = rnew - beta * dold

            xold = xnew
            rold = rnew
            dold = dnew

            self.iterations += 1

        return xnew
