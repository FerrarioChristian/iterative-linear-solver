import time

import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    time_grad = time.perf_counter()

    xnew = np.array([0] * n)
    xold = xnew
    k = 0

    while criterioDiArresto(A, xnew, b, tol, k, maxIter):
        xold = xnew
        r = b - A @ xold
        d = (np.transpose(r) @ r) / (np.transpose(r) @ (A @ r))
        xnew = xold + d * r
        k += 1

    time_grad = time.perf_counter() - time_grad

    return xnew, time_grad, k
