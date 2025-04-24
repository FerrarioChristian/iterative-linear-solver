import numpy as np


def criterioDiArresto(A, xnew, b, tol, it, maxIter):
    if np.linalg.norm((A @ xnew) - b, np.inf) / np.linalg.norm(b) < tol:
        return False
    if it >= maxIter:
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

def check_dominance(A):
        for i in range(A.shape[0]):
            diag = abs(A[i, i])  # Diagonal element
            off_diag_sum = np.sum(np.abs(A[i, :])) - diag  # Sum of off-diagonal elements
            if diag < off_diag_sum:
                return False  # Not diagonally dominant
        return True

def calculate_error(U, x, b):
    return np.linalg.norm(b - np.dot(U, x)) / np.linalg.norm(b)
