import numpy as np
from sudoku_solver import (algorithm_x, contains_duplicates,
                           exact_cover_matrix, valid_solution)


def print_sudoku(sudoku):
    print("+-------+-------+-------+")
    for b in range(0, 9, 3):
        for r in range(3):
            print(
                "|",
                " | ".join(
                    " ".join(str(_) for _ in sudoku[b + r, c : c + 3])
                    for c in range(0, 9, 3)
                ),
                "|",
            )
        print("+-------+-------+-------+")


def sudoku_generator(sudokus=1, *, kappa=5, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    for puzzle in range(sudokus):
        sudoku = np.zeros((9, 9), dtype=np.int8)
        for cell in range(np.random.randint(kappa)):
            for p, val in zip(np.random.randint(0, 8, size=(9, 2)), range(1, 10)):
                tmp = sudoku.copy()
                sudoku[tuple(p)] = val
                if contains_duplicates(sudoku):
                    sudoku = tmp
        yield sudoku.copy()


if __name__ == "__main__":
    for sudoku in sudoku_generator(sudokus=122):
        m, lab = exact_cover_matrix(sudoku)
        names = np.arange(0, m.shape[0], 1)
        res = algorithm_x((names, m), [])
        if not res[1]:
            print("No solution")
            continue
        solution = np.empty_like(sudoku)
        for r, c, d in [lab[i] for i in res[0]]:
            solution[r, c] = d
        assert valid_solution(solution)
        print_sudoku(sudoku)
        print_sudoku(solution)
        print("\n")
