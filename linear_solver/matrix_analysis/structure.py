from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class MatrixStructureInfo:
    condition_number: float
    is_symmetric: bool
    is_positive_definite: bool
    is_diagonally_dominant: bool

def is_symmetric(A: csr_matrix, tol: float = 1e-8) -> bool:
    """Check if sparse matrix A is symmetric."""
    rows, cols = A.shape
    if rows != cols:
        return False
    return (A - A.T).nnz <= tol * A.nnz


def is_positive_definite(A: csr_matrix ) -> bool:
    """Check if sparse matrix A is positive definite (all eigenvalues > 0).
    This is only reliable for symmetric matrices.
    """
    if not is_symmetric(A):
        raise ValueError("Matrix must be symmetric to check for positive definiteness reliably with eigh.")
    try:
        eigenvalues = eigsh(A, k=min(A.shape) - 1, which='SA', return_eigenvectors=False)
        return np.all(eigenvalues > 0)
    except Exception as e:
        print(f"Error calculating eigenvalues: {e}")
        return False

def condition_number(A: csr_matrix) -> float:
    """Estimate condition number of sparse matrix A."""
    return np.linalg.cond(A.toarray())

def is_diagonally_dominant(A: csr_matrix) -> bool:
    """Check if sparse matrix A is diagonally dominant."""
    rows, cols = A.shape
    if rows != cols:
        return False
    for i in range(rows):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        row_indices = A.indices[row_start:row_end]
        row_values = A.data[row_start:row_end]

        diag_val = 0
        off_diag_sum = 0

        for j_idx, col_idx in enumerate(row_indices):
            if col_idx == i:
                diag_val = abs(row_values[j_idx])
            else:
                off_diag_sum += abs(row_values[j_idx])

        if diag_val < off_diag_sum:
            return False
    return True

def has_zero_in_diagonal(U: csr_matrix) -> bool:
    """Check if sparse matrix U has zero in diagonal."""
    rows, cols = U.shape
    if rows != cols:
        return False
    for i in range(rows):
        found_diag = False
        row_start = U.indptr[i]
        row_end = U.indptr[i + 1]
        row_indices = U.indices[row_start:row_end]
        row_values = U.data[row_start:row_end]
        for j_idx, col_idx in enumerate(row_indices):
            if col_idx == i and abs(row_values[j_idx]) < 1e-10:
                return True
            elif col_idx == i:
                found_diag = True
                break
        if not found_diag:
            return True  # Implicit zero if no diagonal entry in the sparse representation
    return False

def analyze_matrix(A: csr_matrix) -> MatrixStructureInfo:
    """Return a dictionary with matrix properties."""
    return MatrixStructureInfo(
        condition_number=np.round(condition_number(A)),
        is_symmetric=is_symmetric(A),
        is_positive_definite=is_positive_definite(A),
        is_diagonally_dominant=is_diagonally_dominant(A),
    )
