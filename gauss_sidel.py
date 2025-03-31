import numpy as np

from utils import criterioDiArresto


def solve(A, b, tol, maxIter):
    n = A.shape[0]
    D = np.diag(np.diag(A))

    B = D - A

    b = np.random.rand(n)
    xnew = np.random.rand(n)
    xold = xnew + 1
    k = 0

    while criterioDiArresto(xnew, xold, tol, k, maxIter):
        xold = xnew
        xnew = np.linalg.inv(D) @ (B @ xold + b)
        print(xnew)
        k += 1

    print("Errore: ", np.linalg.norm(xnew - xold, np.inf))
    print("risultato:\n", xnew)
    return False


def main():
    return False


if __name__ == "__main__":
    main()
