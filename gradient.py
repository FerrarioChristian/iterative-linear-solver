import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    print("\n\n", A, "\n\n")

    xnew = np.random.rand(n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(xnew, xold, tol, k, maxIter):
        xold = xnew
        r = b - A @ xold
        d = (np.transpose(r) @ r) / (np.transpose(r) @ (A @ r))
        xnew = xold + d * r
        k += 1

    error = np.linalg.norm(xnew - xold, np.inf)
    return xnew, error
