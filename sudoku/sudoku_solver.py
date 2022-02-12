import math
from itertools import product
from typing import List, Tuple

import numpy as np


def _contains_duplicates(X):
    return np.sum(np.unique(X)) != np.sum(X)


def contains_duplicates(sol):
    return (
        any(_contains_duplicates(sol[r, :]) for r in range(9))
        or any(_contains_duplicates(sol[:, r]) for r in range(9))
        or any(
            _contains_duplicates(sol[r : r + 3 :, c : c + 3])
            for r in range(0, 9, 3)
            for c in range(0, 9, 3)
        )
    )


def valid_solution(sol):
    return (
        not contains_duplicates(sol)
        and np.sum(sol) == (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 9
    )


def exact_cover_matrix(
    sudoku: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Transform standard format sudoku in exact cover matrix.

    Parameters
    ----------
    sudoku : np.ndarray
        Sudoku in standard format.

    Returns
    -------
    exact_cover_matrix: ndarray
        The mapped matrix.
    row_labels : List of tuple of ints
        Name of each row in format (row, column, digit).

    References
    ----------
    https://en.wikipedia.org/wiki/Exact_cover#Sudoku

    """

    rows, columns = sudoku.shape
    assert rows == columns
    possibilities_count = rows ** 3
    cells_count = rows ** 2

    row_col_matrix = np.zeros((possibilities_count, cells_count))
    row_num_matrix = np.zeros((possibilities_count, cells_count))
    col_num_matrix = np.zeros((possibilities_count, cells_count))
    box_num_matrix = np.zeros((possibilities_count, cells_count))

    row_labels = []
    for r, c, d in product(range(rows), range(rows), range(1, rows + 1)):
        row_labels.append((r, c, d))
        if sudoku[r, c] > 0 and not d == sudoku[r, c]:
            continue
        block_count = int(math.sqrt(rows))
        cell_r = math.floor(r / rows * block_count)
        cell_c = math.floor(c / rows * block_count)

        block_index = int(cell_c + cell_r * block_count)
        d -= 1
        row_index = (r * rows ** 2) + (c * rows) + d

        row_col_const = r * rows + c
        row_num_const = r * rows + d
        col_num_const = c * rows + d
        box_num_const = rows * block_index + d

        row_col_matrix[row_index, row_col_const] = 1
        row_num_matrix[row_index, row_num_const] = 1
        col_num_matrix[row_index, col_num_const] = 1
        box_num_matrix[row_index, box_num_const] = 1
    return (
        np.hstack((row_col_matrix, row_num_matrix, col_num_matrix, box_num_matrix)),
        row_labels,
    )


def algorithm_x(
    name_grid: Tuple[np.ndarray, np.ndarray], solution: List[int]
) -> Tuple[List[int], bool]:
    """
    Apply the Knuth's Algorithm X to an exact cover problem.

    Parameters
    ----------
    name_grid : Tuple[np.ndarray, np.ndarray]
        The first element contains the names of the rows.
        The second element is the exact cover matrix.
    solution : List[int]
        The current solution set.

    Returns
    -------
    solution: List[int], bool
        List of rows that solve the exact cover problem. The bool indicates if the solution is valid.

    References
    ----------
    https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf
    https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X

    """
    name, grid = name_grid

    # Matrix is empty. Terminate with success.
    if grid.size == 0:
        return solution, True

    if len(grid.shape) == 1:
        grid = grid.reshape(1, grid.shape[0])

    # Select c with fewest 1s.
    col_index = np.argmin(grid.sum(axis=0))
    # Select the rows with a 1 in col_index position
    rows_index = np.nonzero(grid[:, col_index] == 1)[0]

    # If there are only zero in col_index, terminate unsuccessfully.
    if rows_index.size == 0:
        return [], False

    for r in np.nditer(rows_index):
        name_copy = name.copy()
        copy = grid.copy()

        # Add r to partial solution
        solution.append(name_copy[r])

        # Select columns of r with 1s
        col_to_rem = copy[r] == 1
        # Select rows with 1s in col_to_rem
        rows_to_rem = np.any(copy[:, col_to_rem] == 1, axis=1)

        # Remove rows and columns from matrix
        copy = copy[np.logical_not(rows_to_rem)]
        copy = copy[:, np.logical_not(col_to_rem)]

        name_copy = name_copy[np.logical_not(rows_to_rem)]

        # Recursive call
        result = algorithm_x((name_copy, copy), solution)
        if result[1]:
            return solution, True
    return [], False
