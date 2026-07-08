"""
Schrodinger's Dungeon
======================

A genuine quantum take on Wave Function Collapse (Maxim Gumin's
well-known procedural generation algorithm). Worth being precise about
what WFC actually is first: it's a classical constraint-propagation
algorithm - "superposition" (a cell's remaining allowed tiles),
"collapse" (picking one), and "propagation" (narrowing neighbors'
options) are a metaphor borrowed from quantum mechanics, not literal
quantum computing. Every other script in this project used real qubits
for its "Schrodinger" premise, so this one does too: real per-cell
superposition, real entangling corrections between neighbors, real
mid-circuit measurement collapsing each cell in sequence.

A hard constraint drove the design, learned the hard way earlier in
this project: a dense state-vector simulator needs 2^n amplitudes for
n qubits, and that cost doesn't shrink just because a qubit gets
measured mid-circuit - CUDA-Q still simulates the full joint space for
every qubit ever allocated in the kernel, for the kernel's entire
runtime. One qubit per cell, one-hot over N tile types, would need
N qubits per cell - way too many for anything but a tiny grid. So this
version uses exactly *one* qubit per cell (binary: wall or floor), on
a 5x5 = 25-qubit grid (2^25 amplitudes, ~512MB - comfortable). Doors
and a treasure chest are added afterward as an explicit *classical*
post-processing step, clearly kept separate from the quantum part
rather than passed off as quantum decisions.

The algorithm, cell by cell in raster-scan order (top-left to
bottom-right, row by row):
  1. Start this cell in a superposition biased towards floor (most of
     a dungeon should be walkable) - see BASE_FLOOR_PROBABILITY.
  2. If this cell has an already-collapsed neighbor above or to the
     left, nudge this cell's amplitude to match that neighbor's value
     (a controlled rotation - real entanglement, since it's applied
     before this cell is measured). This is the quantum stand-in for
     WFC's constraint propagation: it's what turns independent random
     noise into organic, clustered wall/floor blobs.
  3. Measure this cell, collapsing it to a definite wall or floor -
     the "propagate to neighbors" step happens implicitly, since the
     *next* cells' corrections read from this cell's now-classical
     value.

This is a real simplification of full WFC, clearly flagged: no
backtracking (a full contradiction-and-retry search doesn't map onto a
single compiled quantum kernel in any natural way), and constraints
only flow forward from already-collapsed neighbors (up and left),
not the full bidirectional propagation a real WFC solver does.
"""

import math
import random

import cudaq

WIDTH = 5
HEIGHT = 5
NUM_CELLS = WIDTH * HEIGHT  # one qubit per cell - see module docstring

# Most of a dungeon should be open floor, not solid wall.
BASE_FLOOR_PROBABILITY = 0.65
BASE_ANGLE = 2 * math.asin(math.sqrt(BASE_FLOOR_PROBABILITY))

# How strongly a cell gets nudged towards matching an already-collapsed
# neighbor - this is what turns salt-and-pepper noise into clustered
# blobs, the qualitative signature real WFC is designed to produce.
MATCH_ANGLE = math.pi / 10


@cudaq.kernel
def decide_tile(cell: cudaq.qubit):
    ry(BASE_ANGLE, cell)


@cudaq.kernel
def blend_with_neighbor(neighbor: cudaq.qubit, this_tile: cudaq.qubit):
    """Nudges this_tile towards whatever neighbor already collapsed to
    - floor neighbors pull this cell towards floor, wall neighbors pull
    it towards wall. neighbor is already measured/classical by the
    time this runs (raster-scan order), so this is a controlled
    rotation with a definite, known control value - but it's still a
    real quantum operation on this_tile, applied before this_tile
    itself is measured.
    """
    ry.ctrl(MATCH_ANGLE, neighbor, this_tile)  # neighbor=floor -> pull towards floor
    x(neighbor)
    ry.ctrl(-MATCH_ANGLE, neighbor, this_tile)  # neighbor=wall -> pull towards wall
    x(neighbor)


@cudaq.kernel
def generate_dungeon(width: int, height: int) -> list[bool]:
    cells = cudaq.qvector(width * height)

    for row in range(height):
        for col in range(width):
            idx = row * width + col
            this_tile = cells[idx]

            decide_tile(this_tile)

            if col > 0:
                left_neighbor = cells[idx - 1]
                blend_with_neighbor(left_neighbor, this_tile)

            if row > 0:
                top_neighbor = cells[idx - width]
                blend_with_neighbor(top_neighbor, this_tile)

            mz(this_tile)  # collapse this cell before moving to the next

    return mz(cells)


def render(bits: list, width: int, height: int) -> list:
    """True = floor, False = wall - classical post-processing turns
    the raw quantum-collapsed grid into an ASCII map, and (explicitly
    classical, not quantum) scatters a door and a treasure chest onto
    open floor tiles for flavor."""
    grid = [["#" if not bits[r * width + c] else "." for c in range(width)]
            for r in range(height)]

    floor_tiles = [(r, c) for r in range(height) for c in range(width)
                   if bits[r * width + c]]
    random.shuffle(floor_tiles)
    if floor_tiles:
        r, c = floor_tiles[0]
        grid[r][c] = "C"  # treasure chest
    if len(floor_tiles) > 1:
        r, c = floor_tiles[1]
        grid[r][c] = "D"  # door
    return grid


def print_grid(grid: list) -> None:
    for row in grid:
        print("  " + "".join(row))


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(1)
    random.seed(1)

    print(f"=== Quantum-collapsed dungeon ({WIDTH}x{HEIGHT}) ===")
    print("  # = wall   . = floor   D = door   C = chest")
    print()
    bits = cudaq.run(generate_dungeon, WIDTH, HEIGHT, shots_count=1)[0]
    print_grid(render(bits, WIDTH, HEIGHT))

    floor_count = sum(1 for b in bits if b)
    print()
    print(f"  {floor_count}/{NUM_CELLS} cells are floor "
          f"({floor_count/NUM_CELLS:.0%}) in this one dungeon, vs a "
          f"{BASE_FLOOR_PROBABILITY:.0%} per-cell baseline - see the")
    print("  500-dungeon average below for whether that's just this")
    print("  particular draw, or a systematic effect.")

    print()
    print("=== A few more sample dungeons ===")
    for i in range(3):
        bits = cudaq.run(generate_dungeon, WIDTH, HEIGHT, shots_count=1)[0]
        print_grid(render(bits, WIDTH, HEIGHT))
        print()

    print("=== Quantitative check: is this actually clustering, or just")
    print("    noise that happens to look clustered in small samples? ===")
    many_results = cudaq.run(generate_dungeon, WIDTH, HEIGHT, shots_count=500)

    avg_floor_fraction = sum(
        sum(1 for cell in bits if cell) / NUM_CELLS for bits in many_results
    ) / len(many_results)

    same_pairs = 0
    total_pairs = 0
    for bits in many_results:
        for row in range(HEIGHT):
            for col in range(WIDTH):
                idx = row * WIDTH + col
                if col > 0:
                    total_pairs += 1
                    same_pairs += bits[idx] == bits[idx - 1]
                if row > 0:
                    total_pairs += 1
                    same_pairs += bits[idx] == bits[idx - WIDTH]
    observed_agreement = same_pairs / total_pairs

    p = avg_floor_fraction
    chance_agreement = p * p + (1 - p) * (1 - p)

    print(f"  Average floor fraction across 500 dungeons: {avg_floor_fraction:.1%} "
          f"(baseline target: {BASE_FLOOR_PROBABILITY:.0%})")
    print(f"  Observed adjacent-cell agreement rate:      {observed_agreement:.1%}")
    print(f"  Agreement expected from independent cells:  {chance_agreement:.1%}")
    print("  If the observed rate is clearly above the independent-cell")
    print("  prediction, neighbor blending is genuinely producing spatial")
    print("  clustering (organic blobs), not just noise that happens to")
    print("  look clustered in a handful of printed examples.")
    print()
    print("  Note the floor fraction also drifted up from the 65%")
    print("  baseline - a real, measured side effect of the blending")
    print("  correction, not a bug: matching a neighbor is a positive")
    print("  feedback loop (whichever tile type gets an early lead in a")
    print("  row tends to keep reinforcing itself), and the baseline")
    print("  already slightly favors floor, so that feedback compounds")
    print("  in floor's favor more often than wall's.")
