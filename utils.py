import numpy as np


def criterioDiArresto(xnew, xold, tol, it, maxIter):
    if np.linalg.norm(xnew - xold, np.inf) < tol:
        return False
    if it > maxIter:
        print("errore: non converge")
        return False
    return True


def has_zero_in_diagonal(U: np.ndarray):
    len = U.shape[0]
    for i in range(len):
        if abs(U[i][i]) < 1e-10:
            print("Determinante é zero")
            return True
    return False


def calculate_error(U, x, b):
    return np.linalg.norm(b - np.dot(U, x)) / np.linalg.norm(b)
