import numpy as np
from scipy.io import mmread

from linear_solver.solvers import *


def main():
    # Caricamento della matrice A e del vettore b
    A = mmread("./matrici/spa1.mtx").toarray()
    n = A.shape[0]

    x = np.array([1] * n)
    b = A @ x

    tol = [1e-4, 1e-6, 1e-8, 1e-10]

    print("Insert max iterations: ")
    maxIter = int(input())

    # Creazione delle istanze dei solutori
    jacobi = JacobiSolver(A, b)
    gauss_sidel = GaussSeidelSolver(A, b)
    gradient = GradientSolver(A, b)
    conjugate_gradient = ConjugateGradientSolver(A, b)

    # Esecuzione dei solutori con diverse tolleranze
    for t in tol:
        print(f"\n\nTolleranza: {t}")
        start_computation(jacobi, gauss_sidel, gradient, conjugate_gradient, t, maxIter)


def start_computation(jacobi, gauss_sidel, gradient, conjugate_gradient, tol, maxIter):
    # Esecuzione e stampa dei risultati per ogni metodo
    res_jacobi = jacobi.solve(tol, maxIter)
    print_result("Jacobi", res_jacobi)

    res_sidel = gauss_sidel.solve(tol, maxIter)
    print_result("Gauss-Sidel", res_sidel)

    res_grad = gradient.solve(tol, maxIter)
    print_result("Gradient", res_grad)

    res_conj = conjugate_gradient.solve(tol, maxIter)
    print_result("Conjugate Gradient", res_conj)


def print_result(method, res):
    print(f"\n\n{method}")
    print("Errore:", calculate_error(res))
    print("Tempo:", res[1])
    print("Iterazioni:", res[1])


def calculate_error(res):
    # Calcolo dell'errore relativo
    x = np.array([1] * res[0].shape[0])
    return np.linalg.norm(res[0] - x) / np.linalg.norm(x)


if __name__ == "__main__":
    main()
