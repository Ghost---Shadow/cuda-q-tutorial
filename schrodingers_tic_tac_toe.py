"""
Schrodinger's Tic-Tac-Toe
===========================

A real variant exists: Quantum Tic-Tac-Toe (Allan Goff). A move can be
placed in TWO squares at once, in superposition ("a spooky mark") -
each mark stays undetermined until a "cycle" of overlapping spooky
marks forces a collapse.

The board (standard numbering):

    0 1 2
    3 4 5
    6 7 8

Square 3 already has a classical X on it. Three spooky marks remain
unresolved, forming a cycle:

    X1: squares {4, 6}
    O1: squares {6, 5}
    X2: squares {5, 4}

Every pair of marks shares exactly one square (X1-O1 share 6, O1-X2
share 5, X2-X1 share 4) - a closed 3-cycle, exactly the structure that
forces a collapse in the real game: you can't independently pick a
square for each mark without risking two marks landing on the same
square.

The key discovery this script is built around: enumerate all 8
independent square-choices for the 3 marks, and only 2 avoid a
double-booked square - the two "everyone rotates the same direction"
resolutions. That is *exactly* the structure of a 3-qubit GHZ state
(H then two CNOTs - the very first multi-qubit entanglement built in
this whole project, tutorials/01_building_kernels.py's ghz_state) -
which only ever collapses to |000> or |111>, never anything else.
So a quantum tic-tac-toe cycle isn't just "quantum-flavored" - it's
mechanically the same entangled state we started this project with,
repurposed.

The two resolutions have genuinely different consequences here, by
design: one wins the game for X outright, the other doesn't. Which one
happens is decided by a real GHZ-state measurement.
"""

from collections import Counter

import cudaq

# Board: 0-8, standard tic-tac-toe numbering. Square 3 is already X.
PRE_PLACED = {3: "X"}

# The cycle. Each spooky mark's two possible squares, and which player
# made it. q=0 picks the first square, q=1 picks the second - and
# because these three qubits form a GHZ state, q0, q1, and q2 are
# always equal (all 0 or all 1) once measured.
SPOOKY_MARKS = [
    ("X1", "X", 4, 6),
    ("O1", "O", 6, 5),
    ("X2", "X", 5, 4),
]

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


@cudaq.kernel
def ghz_cycle_collapse() -> list[bool]:
    """The exact same GHZ-state construction as
    tutorials/01_building_kernels.py's ghz_state - repurposed here to
    represent the 3-cycle's entangled resolution. Only |000> and |111>
    are physically possible outcomes; every other combination would
    double-book a square, and the entanglement structure makes those
    combinations literally unreachable, not just unlikely.
    """
    qubits = cudaq.qvector(3)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    x.ctrl(qubits[0], qubits[2])
    return mz(qubits)


@cudaq.kernel
def naive_independent_marks() -> list[bool]:
    """For comparison: what happens WITHOUT entanglement - three
    independent coin flips instead of a GHZ state. This is the
    "naive" version that a non-quantum (or improperly-entangled)
    implementation would produce."""
    qubits = cudaq.qvector(3)
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    return mz(qubits)


def resolve_board(bits) -> dict:
    board = dict(PRE_PLACED)
    for (name, player, square_if_0, square_if_1), bit in zip(SPOOKY_MARKS, bits):
        square = square_if_1 if bit else square_if_0
        board[square] = player
    return board


def is_conflicting(bits) -> bool:
    squares_used = []
    for (name, player, square_if_0, square_if_1), bit in zip(SPOOKY_MARKS, bits):
        squares_used.append(square_if_1 if bit else square_if_0)
    return len(set(squares_used)) != len(squares_used)


def render(board: dict) -> str:
    cells = [board.get(i, ".") for i in range(9)]
    rows = ["  " + " ".join(cells[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n".join(rows)


def check_winner(board: dict) -> str:
    for a, b, c in WIN_LINES:
        if a in board and board[a] == board.get(b) == board.get(c):
            return board[a]
    return None


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== The pending cycle ===")
    print("Board before collapse (X1/O1/X2 still undetermined between their squares):")
    print(render(PRE_PLACED))
    for name, player, sq0, sq1 in SPOOKY_MARKS:
        print(f"  {name} ({player}): squares {{{sq0}, {sq1}}}")

    print()
    print("=== One collapse, right now ===")
    bits = cudaq.run(ghz_cycle_collapse, shots_count=1)[0]
    board = resolve_board(bits)
    winner = check_winner(board)
    print(f"Measured: {[int(b) for b in bits]}")
    print(render(board))
    print(f"Winner: {winner if winner else 'no winner yet'}")

    print()
    print("=== Verifying the physics over many collapses ===")
    shots = 20000
    ghz_results = cudaq.run(ghz_cycle_collapse, shots_count=shots)
    outcome_counts = Counter(tuple(b) for b in ghz_results)
    conflict_count = sum(1 for bits in ghz_results if is_conflicting(bits))
    win_count = sum(1 for bits in ghz_results if check_winner(resolve_board(bits)) == "X")

    print("Outcomes observed (only (F,F,F) and (T,T,T) should ever appear):")
    for outcome, count in outcome_counts.most_common():
        print(f"  {outcome}: {count}/{shots} ({count/shots:.1%})")
    print(f"Double-booked (physically invalid) collapses: {conflict_count}/{shots}")
    print(f"X wins the game outright on collapse: {win_count}/{shots} ({win_count/shots:.1%})")

    print()
    print("=== Without entanglement: three independent coin flips instead ===")
    naive_results = cudaq.run(naive_independent_marks, shots_count=shots)
    naive_conflicts = sum(1 for bits in naive_results if is_conflicting(bits))
    print(f"Double-booked (invalid) outcomes with independent qubits: "
          f"{naive_conflicts}/{shots} ({naive_conflicts/shots:.1%})")
    print("Expected: about 75% (6 of the 8 equally-likely combinations")
    print("double-book a square) - this is exactly what entanglement fixes:")
    print("without it, most 'collapses' would be physically nonsensical.")
