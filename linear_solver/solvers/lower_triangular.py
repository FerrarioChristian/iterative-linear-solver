import numpy as np


def lower_triangular_solve(A, b):
    data = A.data
    indices = A.indices
    indptr = A.indptr
    n = len(b)
    y = np.zeros_like(b, dtype=float)

    for j in range(n):
        sum_ = 0.0
        diag = None
        for idx in range(indptr[j], indptr[j + 1]):
            i = indices[idx]
            val = data[idx]

            if i < j and i != 0:
                sum_ += val * y[i]
            elif i == j:
                diag = val

        y[j] = (b[j] - sum_) / diag

    return y
