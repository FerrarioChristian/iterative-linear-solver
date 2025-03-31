import numpy as np

import gradient
import jacobi


def main():

    print("Insert size of matrix")
    n = int(input())

    A = np.random.rand(n, n)
    A = A + np.transpose(A)
    A = A + n * np.eye(n)
    print(A)

    b = np.random.rand(n)

    tol = 1e-8

    print("Insert max iterations: ")
    maxIter = int(input())

    res_jacobi, err_jacobi = jacobi.solve(A, b, tol, maxIter)
    res_grad, err_grad = gradient.solve(A, b, tol, maxIter)

    print("Jacobi:")
    print("Risultato:\n", res_jacobi)
    print("Errore: ", err_jacobi)

    print("\n\nGradient:")
    print("Risultato:\n", res_grad)
    print("Errore: ", err_grad)


if __name__ == "__main__":
    main()
