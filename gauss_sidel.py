import numpy as np

import lower_triangular
from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    L = np.tril(A)
    B = A - L

    xnew = np.random.rand(n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(xnew, xold, tol, k, maxIter):
        xold = xnew
        xnew = lower_triangular.solve(L, (b - B @ xold))
        k += 1

    error = np.linalg.norm(xnew - xold, np.inf)
    return xnew, error


def main():
    return False


if __name__ == "__main__":
    main()
