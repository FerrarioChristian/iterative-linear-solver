import numpy as np


def criterioDiArresto(r, b,tol, it, maxIter):
    if np.linalg.norm(r)/ b < tol:
        return False
    if it >= maxIter:
        
        return False
    return True


def has_zero_in_diagonal(U: np.ndarray):
    len = U.shape[0]
    for i in range(len):
        if abs(U[i][i]) < 1e-10:
            print("Determinante é zero")
            return True
    return False

def info_matrice(A):
        dominanza = True
        for i in range(A.shape[0]):
            diag = abs(A[i, i])  # Diagonal element
            off_diag_sum = np.sum(np.abs(A[i, :])) - diag  # Sum of off-diagonal elements
            if diag < off_diag_sum:
                dominanza = False
            return {
                "condizionamento": np.round(np.linalg.cond(A)),
                "simmetria": np.allclose(A, A.T),
                "positività": np.all(np.linalg.eigvals(A) > 0),
                "dominanza": dominanza,
            }

def calculate_error(U, x, b):
    return np.linalg.norm(b - np.dot(U, x)) / np.linalg.norm(b)
