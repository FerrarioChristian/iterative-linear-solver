from dataclasses import dataclass

import numpy as np


@dataclass
class MatrixStructureInfo:
    condition_number: float
    is_symmetric: bool
    is_positive_definite: bool
    is_diagonally_dominant: bool


def is_symmetric(A: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix A is symmetric."""
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A: np.ndarray) -> bool:
    """Check if matrix A is positive definite (all eigenvalues > 0)."""
    eigvals = np.linalg.eigvals(A)
    return bool(np.all(eigvals > 0))


def condition_number(A: np.ndarray) -> float:
    """Compute the condition number of matrix A."""
    return np.linalg.cond(A)


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """Check if matrix A is diagonally dominant."""
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        if diag < off_diag_sum:
            return False
    return True


def has_zero_in_diagonal(U: np.ndarray):
    """Check if matrix U has zero in diagonal."""
    len = U.shape[0]
    for i in range(len):
        if abs(U[i][i]) < 1e-10:
            return True
    return False


def analyze_matrix(A: np.ndarray) -> MatrixStructureInfo:
    """Return a dictionary with matrix properties."""
    return MatrixStructureInfo(
        condition_number=np.round(condition_number(A)),
        is_symmetric=is_symmetric(A),
        is_positive_definite=is_positive_definite(A),
        is_diagonally_dominant=is_diagonally_dominant(A),
    )
