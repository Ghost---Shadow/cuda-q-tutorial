"""
Schrodinger's Game of Life
===========================

Conway's Game of Life, rendered live with pygame, with a genuinely
quantum twist on exactly one part of it: the random initial board.

Where quantum computing honestly belongs here, and where it doesn't:
Game of Life's own evolution rules (a cell lives, dies, or is born
based on its neighbor count) are a fully deterministic classical
cellular automaton - given a starting grid, every future generation is
completely determined. There's nothing for quantum superposition to
usefully do there, and pretending otherwise would just be quantum-
flavored theater. The one place genuine randomness actually enters
Game of Life is the traditional "random soup" starting board - and
that's exactly where this script uses real qubits: every starting
cell's alive/dead state comes from an actual quantum measurement,
not Python's random module.

Why the whole board isn't one big quantum circuit: this project has
run into the same wall repeatedly (the epidemic script, the dungeon,
Sudoku) - a dense state-vector simulator needs 2^n amplitudes for n
qubits, so one qubit per cell across a whole Game of Life grid (often
hundreds or thousands of cells) is completely infeasible in one shot.
The fix, same as Sudoku's approach: keep every individual quantum
circuit small (QUBITS_PER_BATCH qubits, well within budget) and call
it repeatedly, concatenating batches of real quantum-measured bits
until the whole board is filled - the board is quantum-seeded, without
ever needing an infeasibly large circuit.

Controls: close the window to stop. The console also prints a
generation counter, and a screenshot is saved to
schrodingers_game_of_life_screenshot.png every SCREENSHOT_INTERVAL
generations, so the simulation's visual state can be checked without
needing to watch the live window.
"""

import math

import cudaq
import pygame

GRID_WIDTH = 40
GRID_HEIGHT = 30
CELL_SIZE = 15

# Traditional Game of Life "random soup" density - too sparse and
# nothing interesting happens; too dense and it mostly just dies out
# in the first few generations. ~25-30% alive is the well-known sweet
# spot for producing gliders, oscillators, and other emergent patterns.
ALIVE_PROBABILITY = 0.28
ALIVE_ANGLE = 2 * math.asin(math.sqrt(ALIVE_PROBABILITY))

# Keep every individual quantum circuit small and call it repeatedly,
# rather than one infeasibly large circuit for the whole board.
QUBITS_PER_BATCH = 25

FRAMES_PER_SECOND = 10
SCREENSHOT_INTERVAL = 20
SCREENSHOT_PATH = "schrodingers_game_of_life_screenshot.png"


@cudaq.kernel
def quantum_random_cells(num_qubits: int) -> list[bool]:
    """A batch of independent, genuinely random alive/dead qubits,
    biased towards ALIVE_PROBABILITY - real quantum measurements, not
    pseudo-random numbers."""
    qubits = cudaq.qvector(num_qubits)
    for i in range(num_qubits):
        ry(ALIVE_ANGLE, qubits[i])
    return mz(qubits)


def generate_initial_board(width: int, height: int) -> list:
    """Fills the whole board by calling the small quantum kernel
    repeatedly, QUBITS_PER_BATCH cells at a time, until every cell has
    a real quantum-measured value."""
    total_cells = width * height
    bits = []
    while len(bits) < total_cells:
        remaining = total_cells - len(bits)
        batch_size = min(QUBITS_PER_BATCH, remaining)
        bits.extend(cudaq.run(quantum_random_cells, batch_size, shots_count=1)[0])

    grid = [[bits[row * width + col] for col in range(width)] for row in range(height)]
    return grid


def count_live_neighbors(grid, row, col, width, height) -> int:
    """Toroidal (wrap-around) neighbor counting - the classic Game of
    Life topology, avoiding edge-of-board artifacts."""
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r = (row + dr) % height
            c = (col + dc) % width
            if grid[r][c]:
                count += 1
    return count


def step(grid, width, height) -> list:
    """The standard, fully deterministic Game of Life rules - no
    quantum randomness here or anywhere past the initial board."""
    new_grid = [[False] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            alive = grid[row][col]
            neighbors = count_live_neighbors(grid, row, col, width, height)
            if alive:
                new_grid[row][col] = neighbors in (2, 3)
            else:
                new_grid[row][col] = neighbors == 3
    return new_grid


def draw(screen, grid, width, height) -> None:
    screen.fill((10, 10, 10))
    for row in range(height):
        for col in range(width):
            if grid[row][col]:
                rect = (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
                pygame.draw.rect(screen, (80, 220, 120), rect)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(1)

    print("Generating quantum-seeded initial board...")
    grid = generate_initial_board(GRID_WIDTH, GRID_HEIGHT)
    alive_count = sum(sum(1 for c in row if c) for row in grid)
    print(f"Initial board: {alive_count}/{GRID_WIDTH * GRID_HEIGHT} cells alive "
          f"({alive_count / (GRID_WIDTH * GRID_HEIGHT):.1%}, target "
          f"{ALIVE_PROBABILITY:.0%})")

    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
    pygame.display.set_caption("Schrodinger's Game of Life")
    clock = pygame.time.Clock()

    generation = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw(screen, grid, GRID_WIDTH, GRID_HEIGHT)
        pygame.display.flip()

        if generation % SCREENSHOT_INTERVAL == 0:
            pygame.image.save(screen, SCREENSHOT_PATH)
            print(f"Generation {generation}: {sum(sum(1 for c in r if c) for r in grid)} "
                  f"cells alive (screenshot saved)")

        grid = step(grid, GRID_WIDTH, GRID_HEIGHT)
        generation += 1
        clock.tick(FRAMES_PER_SECOND)

    pygame.quit()
