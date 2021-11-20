import numpy as np
import random
from connect_four_solver import MinMaxSolver, MonteCarloSolver
import asyncio

solver = MonteCarloSolver()

solver.play(0, -1)
solver.play(0, -1)
solver.play(0, -1)

while (
    not solver.four_in_a_row(-1)
    and not solver.four_in_a_row(1)
    and solver.valid_moves()
):
    print(f"Turn for {solver.turn}")
    if solver.turn == -1:
        print(solver.board)
        col = int(input())
        if col not in solver.valid_moves():
            continue
    else:
        value, col = asyncio.run(solver.solve(1))
        # value, col = solver.solve(1)
    solver.play(col, solver.turn)
    solver.pass_turn()

result = solver.state_eval()
print(solver.board)
if result == -1:
    print(f"Player won!!!")
elif result == 1:
    print("Bot won!!!")
else:
    print("Draw")
