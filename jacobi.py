import time

import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    time_jacobi = time.perf_counter()

    D = np.diag(np.diag(A))

    B = D - A

    xnew = np.array([0] * n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(A, xnew, b, tol, k, maxIter):
        xold = xnew
        xnew = np.linalg.inv(D) @ (B @ xold + b)
        k += 1

    time_jacobi = time.perf_counter() - time_jacobi
    # error = np.linalg.norm(xnew - xold, np.inf)
    return xnew, time_jacobi, k
