"""
Schrodinger's Sudoku
=====================

Sudoku as "the dungeon" for the same Wave Function Collapse idea from
schrodingers_dungeon.py - but Sudoku has a property the dungeon didn't:
it has *hard* constraints. An "organic but slightly wrong" dungeon
layout was fine; an "organic but slightly wrong" Sudoku board is just
wrong. That difference forces a real architectural change, not just a
bigger grid.

Why this can't be "one qubit-encoded cell per Sudoku cell" like the
dungeon: a 9x9 board needs 9 possible digits per cell, so a one-hot
quantum encoding would need roughly 4 qubits per cell (81 cells x 4
qubits = ~324 qubits total) to represent full per-cell superposition
over all 81 cells simultaneously. A dense simulator needs 2^n
amplitudes for n qubits - 2^324 is beyond astronomically infeasible,
nowhere close to anything any computer could simulate. This is the
same exponential wall from earlier in this project, just hit much
harder than the dungeon's 25-qubit grid.

The fix: real Sudoku-solving-via-WFC already works by repeatedly (1)
finding the most-constrained ("lowest entropy") empty cell, (2)
collapsing it to one of its remaining valid candidates, and (3)
propagating that constraint to the rest of the board - backtracking
whenever a cell runs out of valid candidates entirely. Steps 1-3 are
ordinary classical bookkeeping (checking rows/columns/3x3 boxes for
already-used digits), and they're what actually *guarantees* a correct
final board. The only place genuine quantum randomness belongs is step
2's "which of the remaining candidates do we pick" - a real, tiny,
per-decision quantum coin flip (or several-sided die, via a small
qubit register), rather than Python's random module. Every other cell
that's already been narrowed down to exactly one valid candidate is
just placed directly - there's no real "choice" left to make there, so
no quantum call happens for it (matching the same "don't fake quantum
randomness where there's nothing left to decide" principle used
throughout this project).

Backtracking guarantees a correct board regardless of what the quantum
die rolls come up as - if a quantum-chosen digit later leads to a
contradiction elsewhere on the board, that specific choice gets undone
and a different candidate is tried. The solver reports how many
genuine multi-candidate decisions it had to make, and how many of
those had to be backtracked out of.
"""

import math

import cudaq

# Arto Inkala's widely-circulated "world's hardest sudoku" (2012).
# Deliberately picked over an easier puzzle: a first attempt with a
# standard easy/medium example solved correctly but made *zero*
# genuine quantum decisions - pure constraint propagation forced every
# cell to a single candidate the whole way through, so the quantum die
# never actually got rolled. This puzzle is hard specifically because
# it requires real guessing (multiple valid candidates at some point,
# with no way to logically narrow it to one), so it actually exercises
# the quantum decision-making and backtracking machinery below.
PUZZLE = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0],
]


@cudaq.kernel
def quantum_die_roll(num_qubits: int) -> list[bool]:
    """A genuinely fair quantum die: equal superposition over all
    2^num_qubits outcomes, collapsed by one measurement."""
    qubits = cudaq.qvector(num_qubits)
    for i in range(num_qubits):
        h(qubits[i])
    return mz(qubits)


def quantum_choice(n: int) -> int:
    """Picks a uniformly random index in [0, n) using quantum_die_roll.
    n isn't generally a power of 2, so this uses rejection sampling:
    roll a register sized for the next power of 2 above n, and re-roll
    (a fresh quantum measurement each time) whenever the result lands
    outside [0, n) - the standard way to get an unbiased choice among
    a non-power-of-2 number of options from a binary source.
    """
    if n == 1:
        return 0
    num_qubits = max(1, math.ceil(math.log2(n)))
    while True:
        bits = cudaq.run(quantum_die_roll, num_qubits, shots_count=1)[0]
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        if value < n:
            return value


def candidates(grid, row, col):
    """Every digit not already used in this cell's row, column, or
    3x3 box - the classical constraint-checking that guarantees
    correctness, independent of whatever the quantum die rolls."""
    used = set()
    for i in range(9):
        used.add(grid[row][i])
        used.add(grid[i][col])
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            used.add(grid[r][c])
    return [d for d in range(1, 10) if d not in used]


def find_lowest_entropy_cell(grid):
    """The real WFC heuristic: always collapse the most-constrained
    empty cell next. This happens to also be the standard "minimum
    remaining values" heuristic real constraint-satisfaction Sudoku
    solvers use - WFC-style thinking and efficient classical solving
    landing on the same idea from different directions."""
    best_cell = None
    best_candidates = None
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                cs = candidates(grid, row, col)
                if best_cell is None or len(cs) < len(best_candidates):
                    best_cell, best_candidates = (row, col), cs
    return best_cell, best_candidates


def solve(grid, stats):
    cell, cs = find_lowest_entropy_cell(grid)
    if cell is None:
        return True  # no empty cells left - solved
    if not cs:
        return False  # this cell has no valid digits left - dead end, backtrack

    row, col = cell
    remaining = list(cs)
    while remaining:
        if len(remaining) == 1:
            choice = remaining[0]  # forced - nothing to actually choose
        else:
            stats["quantum_decisions"] += 1
            choice = remaining[quantum_choice(len(remaining))]

        grid[row][col] = choice
        if solve(grid, stats):
            return True

        # That choice led to a contradiction somewhere down the line -
        # undo it and remove it from consideration at this cell.
        grid[row][col] = 0
        remaining.remove(choice)
        if len(cs) > 1:
            stats["backtracks"] += 1

    return False


def verify_solution(grid) -> bool:
    """Independently checks the finished board satisfies every Sudoku
    rule - this is the actual correctness proof, not just "the solver
    said it succeeded"."""
    full_set = set(range(1, 10))
    for i in range(9):
        if set(grid[i]) != full_set:
            return False
        if {grid[r][i] for r in range(9)} != full_set:
            return False
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = {grid[r][c] for r in range(box_row, box_row + 3)
                   for c in range(box_col, box_col + 3)}
            if box != full_set:
                return False
    return True


def print_grid(grid) -> None:
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("  ------+-------+------")
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            row_str += f"{grid[r][c]} "
        print("  " + row_str)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(1)

    print("=== Starting puzzle ===")
    print_grid(PUZZLE)

    grid = [row[:] for row in PUZZLE]
    stats = {"quantum_decisions": 0, "backtracks": 0}
    solved = solve(grid, stats)

    print()
    print("=== Solved board ===")
    print_grid(grid)

    print()
    print(f"Solver reported success: {solved}")
    print(f"Independently verified correct: {verify_solution(grid)}")
    print(f"Genuine quantum decisions made (cells with >1 valid candidate): "
          f"{stats['quantum_decisions']}")
    print(f"Of those, later backtracked out of (a quantum guess turned out")
    print(f"wrong and had to be undone): {stats['backtracks']}")
