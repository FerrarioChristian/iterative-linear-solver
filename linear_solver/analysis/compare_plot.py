import matplotlib.pyplot as plt


def plot_execution_time(df):
    plt.figure(figsize=(10, 6))
    methods = df["solver_class"].unique()

    for matrix in df["matrix"].unique():
        data = df[df["matrix"] == matrix]
        plt.bar(
            [f"{matrix}\n{method}" for method in data["solver_class"]],
            data["execution_time"],
            label=matrix,
        )

    plt.xlabel("Matrix and Solver")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
