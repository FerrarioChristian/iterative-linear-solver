# Linear Solver

> A Python library for solving linear systems using iterative methods, analyzing matrix properties, benchmarking solvers, and visualizing performance.

---

## Features

- Solve linear systems using:
  - **Jacobi method**
  - **Gauss-Seidel method**
  - **Gradient Descent**
  - **Conjugate Gradient**
- Analyze structural matrix properties:
  - Symmetry
  - Positive-definiteness
  - Diagonal dominance
  - Condition number
- Benchmark solvers (execution time, number of iterations, convergence behavior)
- Generate comparison plots (execution time, relative error, iterations)

---

## Installation

```bash
pip install -e .
```

## Usage
### Solving a system
You can use any solver directly:
```python
from linear_solver.solvers import JacobiSolver

solver = JacobiSolver(A, b, tolerance=1e-6, max_iterations=10000)
solution = solver.solve()
```

### Benchmarking solvers
You can benchmark any solver using the `benchmark_solver` function:
```python
from linear_solver.benchmark import benchmark_solver
from linear_solver.solvers import JacobiSolver

result = benchmark_solver(JacobiSolver, A, b, tol=1e-8, max_iter=5000)
print(result.execution_time, result.iterations)
```

### Analyzing matrix properties
You can analyze matrix properties using the `MatrixAnalyzer` class:
```python
from linear_solver.matrix_analysis.structure import analyze_matrix

properties = analyze_matrix(A)
print(properties.is_symmetric, properties.is_positive_definite, properties.is_diagonally_dominant)
```

## Command line tool
You can run the full benchmark from the command line:
```bash
python main.py --max-iter 20000
```

### Options
`--max-iter`: Set maximum iterations (default 20000)  
`--skip-check`: Skip matrix property analysis  
`--spy`: Generate a spy plot of the matrix  

Results will be saved in restuls.csv and plots will be generated in the `plots` directory

## Project Structure
``````
linear_solver/    
│  
├── analysis/  
│   ├── benchmark.py       # Benchmarking utilities  
│   ├── compare_plot.py     # Plotting utilities  
│  
├── matrix_analysis/  
│   ├── structure.py        # Matrix property analysis  
│  
├── solvers/  
│   ├── base.py             # Solver base class  
│   ├── jacobi.py           # Jacobi method  
│   ├── gauss_seidel.py     # Gauss-Seidel method  
│   ├── gradient.py         # Gradient Descent method  
│   ├── conjugate_gradient.py # Conjugate Gradient method  
│  
├── utils/  
│   ├── stopping_criteria.py # Stopping conditions  
│  
main.py                     # CLI interface  
constants.py                 # Solver settings (matrices, tolerances, etc.)
```
