from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
from collections import Counter, defaultdict, deque
import random
import asyncio
from colorama import Fore


class BaseSolver(ABC):
    def __init__(self, columns: int = 7, column_height: int = 6) -> None:
        self.board = np.zeros((columns, column_height), dtype=np.int8)
        self.turn = random.choice([1, -1])

    @abstractmethod
    def state_eval(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def pass_turn(self):
        self.turn = -self.turn

    def valid_moves(self):
        """Returns columns where a disc may be played"""
        return [
            n
            for n in range(self.board.shape[0])
            if self.board[n, self.board.shape[1] - 1] == 0
        ]

    def play(self, column: int, player: int):
        """Updates `board` as `player` drops a disc in `column`"""
        (index,) = next((i for i, v in np.ndenumerate(self.board[column]) if v == 0))
        self.board[column, index] = player

    def take_back(self, column: int):
        """Updates `board` removing top disc from `column`"""
        (index,) = [i for i, v in np.ndenumerate(self.board[column]) if v != 0][-1]
        self.board[column, index] = 0

    def four_in_a_row(self, player: int):
        """Checks if `player` has a 4-piece line"""
        num_columns, column_height = self.board.shape
        return (
            any(
                all(self.board[c, r] == player)
                for c in range(num_columns)
                for r in (list(range(n, n + 4)) for n in range(column_height - 4 + 1))
            )
            or any(
                all(self.board[c, r] == player)
                for r in range(column_height)
                for c in (list(range(n, n + 4)) for n in range(num_columns - 4 + 1))
            )
            or any(
                np.all(self.board[diag] == player)
                for diag in (
                    (range(ro, ro + 4), range(co, co + 4))
                    for ro in range(0, num_columns - 4 + 1)
                    for co in range(0, column_height - 4 + 1)
                )
            )
            or any(
                np.all(self.board[diag] == player)
                for diag in (
                    (range(ro, ro + 4), range(co + 4 - 1, co - 1, -1))
                    for ro in range(0, num_columns - 4 + 1)
                    for co in range(0, column_height - 4 + 1)
                )
            )
        )

    def print_board(self):
        to_print = ""
        count = 0
        for row in self.board.tolist():
            to_print += f"{Fore.RESET}{count}: "
            for el in row:
                if el == 1:
                    to_print += f"{Fore.YELLOW}B "
                elif el == -1:
                    to_print += f"{Fore.RED}P "
                else:
                    to_print += f"{Fore.RESET}- "
            to_print += f"\n"
            count += 1
        print(to_print + Fore.RESET)


class MinMaxSolver(BaseSolver):
    def __init__(
        self, columns: int = 7, column_height: int = 6, max_depth: int = 3
    ) -> None:
        super().__init__(columns=columns, column_height=column_height)
        self.max_depth = max_depth

    def _corrected_max(self, current: Tuple[int, int], new: Tuple[int, int]):
        # Better to control central row
        if current[0] >= 0 and current[0] - new[0] == 0:
            return min(current, new, key=lambda k: np.abs(k[1] - 3))
        return max(current, new, key=lambda k: k[0])

    def state_eval(self):
        if self.four_in_a_row(-1):
            return -1
        elif self.four_in_a_row(1):
            return 1
        return 0

    def solve(
        self, player: int, depth: int = 0, alpha: float = np.NINF, beta: float = np.inf
    ) -> Union[Tuple[int, int], Tuple[int, None]]:
        won = self.state_eval()
        possibilities = self.valid_moves()

        if not won == 0:
            return won, None
        elif not possibilities or depth == self.max_depth:
            return 0, None

        if player == 1:
            value = (np.NINF, -1)
            for column in possibilities:
                if value[0] == 1:
                    break
                self.play(column, player)
                result = self.solve(-1, depth + 1)[0]
                value = self._corrected_max(value, (result, column))
                alpha = max(value[0], alpha)
                self.take_back(column)
                if beta < alpha:
                    break
            return value
        else:
            value = (np.inf, -1)
            for column in possibilities:
                if value[0] == -1:
                    break
                self.play(column, player)
                result = self.solve(1, depth + 1)[0]
                value = min(value, (result, column), key=lambda v: v[0])
                beta = min(beta, value[0])
                self.take_back(column)
                if beta < alpha:
                    break
            return value


class MonteCarloSolver(BaseSolver):
    def __init__(self, columns: int = 7, column_height: int = 6) -> None:
        super().__init__(columns=columns, column_height=column_height)

    def _mc(self, player: int):
        p = -player
        result = 0
        moves = deque()
        while self.valid_moves():
            p = -p
            c = random.choice(self.valid_moves())
            self.play(c, p)
            moves.append(c)
            if self.four_in_a_row(p):
                result = p
                break
        while moves:
            self.take_back(moves.pop())
        return result

    def montecarlo_value(self, player: int, num_samples: int = 100):
        cnt = Counter(self._mc(player) for _ in range(num_samples))
        return (cnt[1] - cnt[-1]) / num_samples

    def state_eval(self):
        if self.four_in_a_row(-1):
            return -1
        elif self.four_in_a_row(1):
            return 1
        return self.montecarlo_value(1, 10)

    async def async_thread_func(
        self, board: np.ndarray, column: int, adv_column: int, player: int
    ):
        new_solver = MonteCarloSolver()
        new_solver.board = board
        new_solver.play(adv_column, player)
        return {"bot": column, "player": adv_column, "value": new_solver.state_eval()}

    async def solve(self, player: int) -> Tuple[float, int]:
        possibilities = self.valid_moves()
        results = defaultdict(lambda: [])
        tasks = []
        for column in possibilities:
            self.play(column, player)
            # We won
            if self.four_in_a_row(1):
                self.take_back(column)
                return 1, column
            for adv_column in self.valid_moves():
                """Single Thread version
                self.play(adv_column, -player)
                results[column].append(self.state_eval())
                self.take_back(adv_column)
                """
                tasks.append(
                    asyncio.create_task(
                        self.async_thread_func(
                            self.board.copy(), column, adv_column, -player
                        )
                    )
                )
            self.take_back(column)

        async_res = await asyncio.gather(*tasks)

        taboo = set()
        for res in async_res:
            if res["value"] == -1:
                if res["bot"] == res["player"]:
                    taboo.add(res["bot"])
                elif res["bot"] not in taboo:
                    return -1, res["player"]
            results[res["bot"]].append(res["value"])

        for k in taboo:
            results.pop(k, None)
        for k in results:
            results[k] = np.mean(results[k])
        col = max(results, key=lambda k: results[k])

        return results[col], col
