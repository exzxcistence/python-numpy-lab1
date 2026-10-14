"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)



def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors

    result = np.zeros(len(vectors[0]), dtype=float)

    for matrix, vector in zip(matrices, vectors):
        matrix = np.asarray(matrix)
        vector = np.asarray(vector)

        result += matrix @ vector

    return result



def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold

    matrix = np.asarray(matrix)

    result = (matrix > threshold).astype(int)

    return result




def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    result = []
    seen = set()

    for row in matrix:
        row_tuple = tuple(row)

        if row_tuple not in seen:
            seen.add(row_tuple)
            result.append(row.tolist())

    return result




def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    unique_indices = []
    seen = set()

    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        column_tuple = tuple(column)

        if column_tuple not in seen:
            seen.add(column_tuple)
            unique_indices.append(column_index)

    result = matrix[:, unique_indices]

    return result.tolist()



def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed
    )

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=mean,
        scale=std,
        size=(rows, columns)
    )

    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)

    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances
    )



def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = (
        data.rows,
        data.columns,
        data.first,
        data.second
    )

    row_indices, column_indices = np.indices((rows, columns))

    mask = (row_indices + column_indices) % 2 == 0

    result = np.where(mask, first, second)

    return result




def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros(
        (image_height, image_width, 3),
        dtype=np.uint8
    )

    image[:] = background_color

    x_start = (image_width - width) // 2
    y_start = (image_height - height) // 2

    x_end = x_start + width
    y_end = y_start + height

    image[y_start:y_end, x_start:x_end] = shape_color

    return image






def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.zeros(
        (image_height, image_width, 3),
        dtype=np.uint8
    )

    image[:] = background_color

    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2

    y, x = np.ogrid[:image_height, :image_width]

    mask = (
        ((x - center_x) / semi_axis_x) ** 2
        + ((y - center_y) / semi_axis_y) ** 2
        <= 1
    )

    image[mask] = shape_color

    return image




def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window

    values = np.asarray(values, dtype=float)

    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    local_maxima_indices = np.where(
        (values[1:-1] > values[:-2]) &
        (values[1:-1] > values[2:])
    )[0] + 1

    local_minima_indices = np.where(
        (values[1:-1] < values[:-2]) &
        (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average
    )



def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count

    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = int(np.max(labels)) + 1 if labels.size > 0 else 0

    result = np.zeros(
        (len(labels), class_count),
        dtype=int
    )

    result[np.arange(len(labels)), labels] = 1

    return result
