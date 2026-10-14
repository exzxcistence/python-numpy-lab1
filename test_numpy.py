import numpy as np

from numpy_tasks import sum_prod
from grader_contracts.numpy_tasks import MatrixVectorBatchInput
from numpy_tasks import binarize
from grader_contracts.numpy_tasks import BinarizeInput
from numpy_tasks import unique_rows
from grader_contracts.numpy_tasks import MatrixInput
from numpy_tasks import unique_columns
from numpy_tasks import matrix_statistics
from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import chess
from grader_contracts.numpy_tasks import ChessInput
from numpy_tasks import draw_rectangle
from grader_contracts.numpy_tasks import RectangleInput
from numpy_tasks import draw_ellipse
from grader_contracts.numpy_tasks import EllipseInput
from numpy_tasks import analyze_time_series
from grader_contracts.numpy_tasks import TimeSeriesInput
from numpy_tasks import one_hot
from grader_contracts.numpy_tasks import OneHotInput




def test_sum_prod():
    data = MatrixVectorBatchInput(
        matrices=[
            [[1, 2], [3, 4]],
            [[2, 0], [0, 2]]
        ],
        vectors=[
            [1, 2],
            [3, 4]
        ]
    )

    result = sum_prod(data)

    np.testing.assert_array_equal(result, [11, 19])


def test_sum_prod_identity():
    data = MatrixVectorBatchInput(
        matrices=[
            [[1, 0], [0, 1]]
        ],
        vectors=[
            [5, 7]
        ]
    )

    result = sum_prod(data)

    np.testing.assert_array_equal(result, [5, 7])


def test_sum_prod_negative():
    data = MatrixVectorBatchInput(
        matrices=[
            [[-1, 0], [0, -2]]
        ],
        vectors=[
            [3, 4]
        ]
    )

    result = sum_prod(data)

    np.testing.assert_array_equal(result, [-3, -8])


def test_binarize():
    data = BinarizeInput(
        matrix=[
            [1, 6, 3],
            [8, 5, 9]
        ],
        threshold=5
    )

    result = binarize(data)

    expected = np.array([
        [0, 1, 0],
        [1, 0, 1]
    ])

    np.testing.assert_array_equal(result, expected)


def test_binarize_all_zeros():
    data = BinarizeInput(
        matrix=[
            [1, 2],
            [3, 4]
        ],
        threshold=10
    )

    result = binarize(data)

    np.testing.assert_array_equal(
        result,
        [[0, 0], [0, 0]]
    )


def test_binarize_all_ones():
    data = BinarizeInput(
        matrix=[
            [5, 6],
            [7, 8]
        ],
        threshold=0
    )

    result = binarize(data)

    np.testing.assert_array_equal(
        result,
        [[1, 1], [1, 1]]
    )


def test_binarize_negative():
    data = BinarizeInput(
        matrix=[
            [-5, -1],
            [0, 3]
        ],
        threshold=-1
    )

    result = binarize(data)

    np.testing.assert_array_equal(
        result,
        [[0, 0], [1, 1]]
    )




def test_unique_rows():
    data = MatrixInput(
        matrix=[
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6]
        ]
    )

    result = unique_rows(data)

    assert result == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]


def test_unique_rows_all_same():
    data = MatrixInput(
        matrix=[
            [1, 1],
            [1, 1],
            [1, 1]
        ]
    )

    assert unique_rows(data) == [[1, 1]]


def test_unique_rows_all_different():
    data = MatrixInput(
        matrix=[
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    )

    assert unique_rows(data) == [
        [1, 2],
        [3, 4],
        [5, 6]
    ]


def test_unique_rows_negative():
    data = MatrixInput(
        matrix=[
            [-1, 2],
            [3, -4],
            [-1, 2]
        ]
    )

    assert unique_rows(data) == [
        [-1, 2],
        [3, -4]
    ]


def test_unique_rows_order():
    data = MatrixInput(
        matrix=[
            [9, 8],
            [1, 2],
            [9, 8],
            [5, 6]
        ]
    )

    assert unique_rows(data) == [
        [9, 8],
        [1, 2],
        [5, 6]
    ]



def test_unique_columns():
    data = MatrixInput(
        matrix=[
            [1, 2, 1, 3],
            [4, 5, 4, 6],
            [7, 8, 7, 9]
        ]
    )

    assert unique_columns(data) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]


def test_unique_columns_all_same():
    data = MatrixInput(
        matrix=[
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ]
    )

    assert unique_columns(data) == [
        [1],
        [2],
        [3]
    ]


def test_unique_columns_all_different():
    data = MatrixInput(
        matrix=[
            [1, 2, 3],
            [4, 5, 6]
        ]
    )

    assert unique_columns(data) == [
        [1, 2, 3],
        [4, 5, 6]
    ]


def test_unique_columns_negative():
    data = MatrixInput(
        matrix=[
            [-1, 2, -1],
            [3, -4, 3]
        ]
    )

    assert unique_columns(data) == [
        [-1, 2],
        [3, -4]
    ]


def test_unique_columns_order():
    data = MatrixInput(
        matrix=[
            [9, 1, 9, 5],
            [8, 2, 8, 6]
        ]
    )

    assert unique_columns(data) == [
        [9, 1, 5],
        [8, 2, 6]
    ]


def test_matrix_statistics_shape():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        mean=0,
        std=1,
        seed=42
    )

    result = matrix_statistics(data)

    assert result.matrix.shape == (3, 4)
    assert result.row_means.shape == (3,)
    assert result.column_means.shape == (4,)
    assert result.row_variances.shape == (3,)
    assert result.column_variances.shape == (4,)


def test_matrix_statistics_means():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        seed=42
    )

    result = matrix_statistics(data)

    np.testing.assert_allclose(
        result.row_means,
        np.mean(result.matrix, axis=1)
    )

    np.testing.assert_allclose(
        result.column_means,
        np.mean(result.matrix, axis=0)
    )


def test_matrix_statistics_variances():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        seed=42
    )

    result = matrix_statistics(data)

    np.testing.assert_allclose(
        result.row_variances,
        np.var(result.matrix, axis=1)
    )

    np.testing.assert_allclose(
        result.column_variances,
        np.var(result.matrix, axis=0)
    )


def test_matrix_statistics_seed():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        seed=123
    )

    result1 = matrix_statistics(data)
    result2 = matrix_statistics(data)

    np.testing.assert_array_equal(
        result1.matrix,
        result2.matrix
    )


def test_matrix_statistics_zero_std():
    data = RandomMatrixInput(
        rows=2,
        columns=3,
        mean=5,
        std=0,
        seed=42
    )

    result = matrix_statistics(data)

    np.testing.assert_array_equal(
        result.matrix,
        np.full((2, 3), 5.0)
    )

    np.testing.assert_array_equal(
        result.row_variances,
        [0, 0]
    )

    np.testing.assert_array_equal(
        result.column_variances,
        [0, 0, 0]
    )



def test_chess_4x4():
    data = ChessInput(
        rows=4,
        columns=4,
        first=0,
        second=1
    )

    result = chess(data)

    expected = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]

    np.testing.assert_array_equal(result, expected)


def test_chess_rectangle():
    data = ChessInput(
        rows=3,
        columns=5,
        first=1,
        second=0
    )

    result = chess(data)

    expected = [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ]

    np.testing.assert_array_equal(result, expected)


def test_chess_custom_values():
    data = ChessInput(
        rows=2,
        columns=3,
        first=5,
        second=9
    )

    result = chess(data)

    np.testing.assert_array_equal(
        result,
        [
            [5, 9, 5],
            [9, 5, 9]
        ]
    )


def test_chess_single_cell():
    data = ChessInput(
        rows=1,
        columns=1,
        first=7,
        second=3
    )

    result = chess(data)

    np.testing.assert_array_equal(result, [[7]])


def test_chess_single_column():
    data = ChessInput(
        rows=4,
        columns=1,
        first=2,
        second=8
    )

    result = chess(data)

    np.testing.assert_array_equal(
        result,
        [
            [2],
            [8],
            [2],
            [8]
        ]
    )



def test_draw_rectangle_shape():
    data = RectangleInput(
        width=4,
        height=2,
        image_height=6,
        image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_rectangle(data)

    assert result.shape == (6, 8, 3)
    assert result.dtype == np.uint8


def test_draw_rectangle_center():
    data = RectangleInput(
        width=4,
        height=2,
        image_height=6,
        image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_rectangle(data)

    np.testing.assert_array_equal(
        result[2:4, 2:6],
        np.full((2, 4, 3), (255, 0, 0), dtype=np.uint8)
    )

    np.testing.assert_array_equal(
        result[0, 0],
        [0, 0, 0]
    )


def test_draw_rectangle_full_image():
    data = RectangleInput(
        width=3,
        height=3,
        image_height=3,
        image_width=3,
        shape_color=(0, 255, 0),
        background_color=(0, 0, 0)
    )

    result = draw_rectangle(data)

    expected = np.full(
        (3, 3, 3),
        (0, 255, 0),
        dtype=np.uint8
    )

    np.testing.assert_array_equal(result, expected)


def test_draw_rectangle_single_pixel():
    data = RectangleInput(
        width=1,
        height=1,
        image_height=3,
        image_width=3,
        shape_color=(0, 0, 255),
        background_color=(255, 255, 255)
    )

    result = draw_rectangle(data)

    np.testing.assert_array_equal(
        result[1, 1],
        [0, 0, 255]
    )

    np.testing.assert_array_equal(
        result[0, 0],
        [255, 255, 255]
    )


def test_draw_rectangle_colors():
    data = RectangleInput(
        width=2,
        height=2,
        image_height=4,
        image_width=4,
        shape_color=(10, 20, 30),
        background_color=(100, 110, 120)
    )

    result = draw_rectangle(data)

    np.testing.assert_array_equal(
        result[1, 1],
        [10, 20, 30]
    )

    np.testing.assert_array_equal(
        result[0, 0],
        [100, 110, 120]
    )


def test_draw_ellipse_shape():
    data = EllipseInput(
        semi_axis_x=3,
        semi_axis_y=2,
        image_height=9,
        image_width=11,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    assert result.shape == (9, 11, 3)
    assert result.dtype == np.uint8


def test_draw_ellipse_center():
    data = EllipseInput(
        semi_axis_x=3,
        semi_axis_y=2,
        image_height=9,
        image_width=11,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    np.testing.assert_array_equal(
        result[4, 5],
        [255, 0, 0]
    )

    np.testing.assert_array_equal(
        result[0, 0],
        [0, 0, 0]
    )


def test_draw_ellipse_axes():
    data = EllipseInput(
        semi_axis_x=3,
        semi_axis_y=2,
        image_height=9,
        image_width=11,
        shape_color=(0, 255, 0),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    np.testing.assert_array_equal(result[4, 2], [0, 255, 0])
    np.testing.assert_array_equal(result[4, 8], [0, 255, 0])
    np.testing.assert_array_equal(result[2, 5], [0, 255, 0])
    np.testing.assert_array_equal(result[6, 5], [0, 255, 0])


def test_draw_ellipse_outside():
    data = EllipseInput(
        semi_axis_x=2,
        semi_axis_y=2,
        image_height=7,
        image_width=7,
        shape_color=(255, 255, 255),
        background_color=(10, 20, 30)
    )

    result = draw_ellipse(data)

    np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(result[0, 6], [10, 20, 30])
    np.testing.assert_array_equal(result[6, 0], [10, 20, 30])
    np.testing.assert_array_equal(result[6, 6], [10, 20, 30])


def test_draw_ellipse_exact_mask():
    data = EllipseInput(
        semi_axis_x=2,
        semi_axis_y=1,
        image_height=5,
        image_width=5,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    expected_mask = np.array([
        [False, False, False, False, False],
        [False, False, True,  False, False],
        [True,  True,  True,  True,  True],
        [False, False, True,  False, False],
        [False, False, False, False, False]
    ])

    actual_mask = np.all(result == [255, 0, 0], axis=2)

    np.testing.assert_array_equal(actual_mask, expected_mask)



def test_time_series_statistics():
    data = TimeSeriesInput(
        values=[1, 2, 3, 4, 5],
        window=2
    )

    result = analyze_time_series(data)

    assert np.isclose(result.mean, 3.0)
    assert np.isclose(result.variance, 2.0)
    assert np.isclose(result.std, np.sqrt(2))


def test_time_series_local_maxima():
    data = TimeSeriesInput(
        values=[1, 5, 2, 7, 3],
        window=2
    )

    result = analyze_time_series(data)

    np.testing.assert_array_equal(
        result.local_maxima_indices,
        [1, 3]
    )


def test_time_series_local_minima():
    data = TimeSeriesInput(
        values=[5, 1, 6, 2, 7],
        window=2
    )

    result = analyze_time_series(data)

    np.testing.assert_array_equal(
        result.local_minima_indices,
        [1, 3]
    )


def test_time_series_moving_average():
    data = TimeSeriesInput(
        values=[1, 2, 3, 4, 5],
        window=3
    )

    result = analyze_time_series(data)

    np.testing.assert_allclose(
        result.moving_average,
        [2, 3, 4]
    )


def test_time_series_constant():
    data = TimeSeriesInput(
        values=[4, 4, 4, 4],
        window=2
    )

    result = analyze_time_series(data)

    assert result.mean == 4
    assert result.variance == 0
    assert result.std == 0

    assert len(result.local_maxima_indices) == 0
    assert len(result.local_minima_indices) == 0

    np.testing.assert_allclose(
        result.moving_average,
        [4, 4, 4]
    )



def test_one_hot_basic():
    data = OneHotInput(
        labels=[0, 2, 1, 2],
        class_count=3
    )

    result = one_hot(data)

    expected = [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ]

    np.testing.assert_array_equal(result, expected)


def test_one_hot_auto_class_count():
    data = OneHotInput(
        labels=[0, 2, 1]
    )

    result = one_hot(data)

    expected = [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]

    np.testing.assert_array_equal(result, expected)


def test_one_hot_extra_classes():
    data = OneHotInput(
        labels=[0, 1],
        class_count=4
    )

    result = one_hot(data)

    expected = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]

    np.testing.assert_array_equal(result, expected)


def test_one_hot_single_class():
    data = OneHotInput(
        labels=[0, 0, 0],
        class_count=1
    )

    result = one_hot(data)

    np.testing.assert_array_equal(
        result,
        [[1], [1], [1]]
    )


def test_one_hot_empty():
    data = OneHotInput(
        labels=[],
        class_count=3
    )

    result = one_hot(data)

    assert result.shape == (0, 3)
    assert result.dtype == np.dtype(int)
