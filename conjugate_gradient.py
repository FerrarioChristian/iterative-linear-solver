import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]

    xnew = np.random.rand(n)
    xold = xnew + 1
    rold = b - A @ xold
    dold = rold
    error = 1
    xcheck = xold

    k = 0

    while criterioDiArresto(xnew, xcheck, tol, k, maxIter):
        alpha = (dold @ rold) / (dold @ A @ dold)
        xnew = xold + alpha * dold
        rnew = rold - alpha * A @ dold
        beta = ((A @ dold) @ rnew) / ((A @ dold) @ dold)
        dnew = rnew - beta * dold

        error = np.linalg.norm(xnew - xold, np.inf)
        xcheck = xold
        xold = xnew
        rold = rnew
        dold = dnew

        k += 1

    return xnew, error
