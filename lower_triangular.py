import numpy as np


def solve(L, b):
    n = L.shape[0]

    x = np.zeros(n)

    x[0] = b[0] / L[0, 0]

    for i in range(1, n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    return x


def main():
    return False


if __name__ == "__main__":
    main()
