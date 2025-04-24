import numpy as np
import pandas as pd
from scipy.io import mmread
from linear_solver.utils import check_dominance
from linear_solver.analysis.benchmark import benchmark
from linear_solver.solvers import *

def main():

    

    all_results = []

    print("Insert max iterations: ")
    maxIter = int(input())

    TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
    SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
    MATRICES = [
        "matrices/spa1.mtx",
        "matrices/spa2.mtx",
        "matrices/vem1.mtx",
        "matrices/vem2.mtx",
    ]

    
    for matrix in MATRICES:
        A = mmread(matrix).toarray()
        n = A.shape[0]
        x = np.array([1] * n)
        b = A @ x
        print(f"\n📄 Matrice: {matrix}", "|", "Condizionamneto: ", np.linalg.cond(A), "|", "Simmetria: ", np.allclose(A, A.T), "|", "Positività: ", np.all(np.linalg.eigvals(A) > 0), "|", "Dominanza: ", check_dominance(A), "|")
        
        for tol in TOLERANCES:
            print(f"\n### Tolleranza: {tol:.0e} ###")
            for solver_class in SOLVERS:
                res = benchmark(solver_class, A, b, tol=tol, max_iter=maxIter)
                res["matrix"] = matrix.split("/")[-1]
                all_results.append(res)
                print_result(res, x)
        
    df = pd.DataFrame(all_results)
    df.to_csv("results.csv", index=False)
    df.to_json("results.json", orient="records", lines=True)

    print("\n--- Risultati ---")
    print(df)



def print_result(result: dict, x_true: np.ndarray):
    x = result["solution"][0]
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    print(f"\n--- {result['solver_class']} ---")
    print(f"Errore relativo: {err}")
    print(f"Iterazioni:     {result['iterations']}")
    print(f"Tempo (s):      {result['execution_time']:.6f}")


if __name__ == "__main__":
    main()
