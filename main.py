import time

import numpy as np
from scipy.io import mmread

import conjugate_gradient
import gauss_sidel
import gradient
import jacobi


def main():
    A = mmread("./matrici/spa1.mtx").toarray()
    # print("Matrix A:\n", A)
    n = A.shape[0]

    x = np.array([1] * n)
    b = A @ x
    # print(b)

    tol = [1e-4, 1e-6, 1e-8, 1e-10]

    print("Insert max iterations: ")
    maxIter = int(input())

    for i in range(len(tol)):
        print("\n\nTolleranza: ", tol[i])
        start_computation(A, b, tol[i], maxIter)


def start_computation(A, b, tol, maxIter):
    res_jacobi = jacobi.solve(A, b, tol, maxIter)
    print_result("Jacobi", res_jacobi)

    res_sidel = gauss_sidel.solve(A, b, tol, maxIter)
    print_result("Gauss-Sidel", res_sidel)

    res_grad = gradient.solve(A, b, tol, maxIter)
    print_result("Gradient", res_grad)

    res_conj = conjugate_gradient.solve(A, b, tol, maxIter)
    print_result("Conjugate Gradient", res_conj)


def print_result(method, res):
    print("\n\n", method)
    # print("Risultato:\n", res)
    print("Errore: ", calculate_error(res))
    print("Tempo: ", res[1])
    print("Iterazioni: ", res[2])


def calculate_error(res):
    x = np.array([1] * res[0].shape[0])
    return np.linalg.norm(res[0] - x) / np.linalg.norm(x)


if __name__ == "__main__":
    main()
