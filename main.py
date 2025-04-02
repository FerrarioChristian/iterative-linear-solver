import numpy as np

import conjugate_gradient
import gauss_sidel
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
    print(b)

    tol = 1e-8

    print("Insert max iterations: ")
    maxIter = int(input())

    res_jacobi, err_jacobi = jacobi.solve(A, b, tol, maxIter)
    res_sidel, err_sidel = gauss_sidel.solve(A, b, tol, maxIter)
    res_grad, err_grad = gradient.solve(A, b, tol, maxIter)
    res_conj, err_conj = conjugate_gradient.solve(A, b, tol, maxIter)

    print("Jacobi:")
    print("Risultato:\n", res_jacobi)
    print("Errore: ", err_jacobi)

    print("\n\nGauss-Sidel:")
    print("Risultato:\n", res_sidel)
    print("Errore: ", err_sidel)

    print("\n\nGradient:")
    print("Risultato:\n", res_grad)
    print("Errore: ", err_grad)

    print("\n\nConjugate gradient:")
    print("Risultato:\n", res_conj)
    print("Errore: ", err_conj)


if __name__ == "__main__":
    main()
