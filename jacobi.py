import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    D = np.diag(np.diag(A))

    B = D - A

    xnew = np.random.rand(n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(xnew, xold, tol, k, maxIter):
        xold = xnew
        xnew = np.linalg.inv(D) @ (B @ xold + b)
        k += 1

    error = np.linalg.norm(xnew - xold, np.inf)
    return xnew, error
