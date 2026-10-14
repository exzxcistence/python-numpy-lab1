
import matplotlib.pyplot as plt

from numpy_tasks import matrix_statistics
from grader_contracts.numpy_tasks import RandomMatrixInput


data = RandomMatrixInput(
    rows=5,
    columns=5,
    mean=0.0,
    std=1.0,
    seed=42
)

statistics = matrix_statistics(data)
matrix = statistics.matrix

# Гистограммы значений каждой строки
for i, row in enumerate(matrix):
    plt.figure()
    plt.hist(row, bins=5, edgecolor="black")
    plt.title(f"Гистограмма строки {i + 1}")
    plt.xlabel("Значение")
    plt.ylabel("Частота")

# Гистограммы значений каждого столбца
for j, column in enumerate(matrix.T):
    plt.figure()
    plt.hist(column, bins=5, edgecolor="black")
    plt.title(f"Гистограмма столбца {j + 1}")
    plt.xlabel("Значение")
    plt.ylabel("Частота")

plt.show()
