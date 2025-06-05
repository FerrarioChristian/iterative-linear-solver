# Linear Solver

> A Python library for solving linear systems using iterative methods, analyzing matrix properties, benchmarking solvers, and visualizing performance.  The library is designed to solve large sparse linear systems efficiently, provided in sparse .mtx format.

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
# Clone the project
git clone git@github.com:FerrarioChristian/iterative-linear-solver.git
cd iterative-linear-solver

# Create the virtual environment
python3 -m venv .venv

# Install the dependencies
pip install -e .

```

## Usage
You can run the full benchmark from the command line:
```bash
linear-solver-demo [-h] [-it MAX_ITER] [-sc] [--spy] [-t TOLERANCES ...]

```

### Options
`-h, --help`: Show this help message and exit  
`-it, --max-iter`: Set maximum iterations (default 20000)  
`-sc, --skip-check`: Skip matrix property check  
`--spy`: Generate a spy plot of the matrices  
`-t, --tolerances`: Set a list of tolerances (default [1e-4, 1e-6, 1e-8, 1e-10])  

Results will be saved in restuls.csv and plots will be generated in the `results/:plots` directory

## Library functions

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


## Project Structure
``````
linear-iterative-solver
├── demo
│   ├── cli.py                       # Parameters options
│   ├── constants.py                 # Solver settings (matrices, tolerances, etc.)
│   ├── logger.py                    # Funtions to print the status of the comutation
│   └── main.py                      # CLI interface
├── linear_solver
│   ├── analysis
│   │   ├── benchmark.py             # Benchmarking utilities  
│   │   ├── compare_plot.py          # Plotting utilities
│   │   └── plot.py
│   ├── convergence
│   │   └── criteria.py              # Stopping conditions  
│   ├── matrix_analysis
│   │   └── structure.py             # Matrix property analysis
│   └── solvers
│       ├── base_solver.py
│       ├── conjugate_gradient.py
│       ├── gauss_seidel.py
│       ├── gradient.py
│       ├── jacobi.py
│       └── lower_triangular.py
├── matrices
│   ├── spa1.mtx
│   ├── spa2.mtx
│   ├── vem1.mtx
│   └── vem2.mtx
└── pyproject.toml
