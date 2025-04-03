import time

import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    time_conj = time.perf_counter()

    xnew = np.array([0] * n)
    xold = xnew
    rold = b - A @ xold
    dold = rold

    k = 0

    while criterioDiArresto(A, xnew, b, tol, k, maxIter):
        alpha = (dold @ rold) / (dold @ A @ dold)
        xnew = xold + alpha * dold
        rnew = rold - alpha * A @ dold
        beta = ((A @ dold) @ rnew) / ((A @ dold) @ dold)
        dnew = rnew - beta * dold

        xold = xnew
        rold = rnew
        dold = dnew

        k += 1

    time_conj = time.perf_counter() - time_conj
    return xnew, time_conj, k
