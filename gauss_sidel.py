import time

import numpy as np

import lower_triangular
from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    time_gauss = time.perf_counter()

    L = np.tril(A)
    B = A - L

    xnew = np.array([0] * n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(A, xnew, b, tol, k, maxIter):
        xold = xnew
        xnew = lower_triangular.solve(L, (b - B @ xold))
        k += 1

    time_gauss = time.perf_counter() - time_gauss

    return xnew, time_gauss, k


def main():
    return False


if __name__ == "__main__":
    main()
