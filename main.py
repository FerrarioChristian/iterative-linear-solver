import numpy as np
import pandas as pd
from scipy.io import mmread
from linear_solver.utils import info_matrice
from linear_solver.analysis.benchmark import benchmark
from linear_solver.solvers import *
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from linear_solver.utils import chiedi_esame_proprieta
from linear_solver.utils import num_iterazioni
def main():

    

    all_results = []

    maxIter = num_iterazioni()
    siono = chiedi_esame_proprieta()
    TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
    
    SOLVERS = [JacobiSolver, GaussSeidelSolver, GradientSolver, ConjugateGradientSolver]
    
    MATRICES = [
        "matrices/spa1.mtx",
        "matrices/spa2.mtx",
        "matrices/vem1.mtx",
        "matrices/vem2.mtx",
    ]

    
    for matrix in MATRICES:
        A = mmread(matrix).tocsr()
        n = A.shape[0]
        x = np.array([1] * n)
        b = A @ x
        if(siono):
            info = info_matrice(A)
            print(f"\n📄 Matrice: {matrix}", "|", "Condizionamento: ", info["condizionamento"], "|", "Simmetria: ", info["simmetria"], "|", "Positività: ", info["positività"], "|", "Dominanza: ", info["dominanza"], "|")
        else:
            print(f"\n📄 Matrice: {matrix} ", "--------------------------------------------------------------------------------------- |")
        for tol in TOLERANCES:
            print(f"\n### Tolleranza: {tol:.0e} ###")
            for solver_class in SOLVERS:
                res = benchmark(solver_class, A, b, tol=tol, max_iter=maxIter)
                res["matrix"] = matrix.split("/")[-1]
                all_results.append(res)
                print_result(res, x, maxIter)
        
    df = pd.DataFrame(all_results)
    df.to_csv("results.csv", index=False)
    df.to_json("results.json", orient="records", lines=True)

    print("\n--- Risultati ---")
    print(df)



def print_result(result: dict, x_true: np.ndarray, maxIter):
    x = result["solution"]
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    print(f"\n--- {result['solver_class']} ---")
    print(f"Errore relativo: {err:.0e}")
    print(f"Iterazioni:     {result['iterations']}")
    if(result['iterations']==maxIter):
        print("errore: non converge")
    print(f"Tempo (s):      {result['execution_time']:.6f}")


if __name__ == "__main__":
    main()
