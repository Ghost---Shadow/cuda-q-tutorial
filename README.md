# cuda-q-tutorial

Learning [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/) by writing and
verifying real quantum circuits on a local GPU (RTX 4070, via WSL2 + Miniconda).
Every script here was actually run against the `nvidia` GPU simulator target and
checked against a known-correct answer (brute force, classical ground truth, or
hand-derived math) rather than just assumed to work.

## Setup

CUDA-Q has no native Windows build - everything runs inside WSL2 Ubuntu, in a
conda `base` environment:

```
wsl -d Ubuntu -- bash -lc "source ~/miniconda3/bin/activate base && python your_script.py"
```

## `hello_cudaq.py`

A from-scratch Bell-pair walkthrough: superposition, entanglement, measurement,
verbose wrapper kernels for `h`/`x`/`mz` (since those are recognized by CUDA-Q's
compiler purely by literal spelling, not as real Python objects), a fixed random
seed for reproducibility, and hand-worked comments explaining the state at each
step.

## `nvidia-cudaq-tutorials/`

NVIDIA's official CUDA-Q tutorials, rewritten by hand (not copy-pasted) for
readability, each one run and verified:

| File | Topic |
|---|---|
| `01_building_kernels.py` | Kernel composition, GHZ state, adjoint gates |
| `02_measuring_kernels.py` | Mid-circuit measurement, classical feed-forward |
| `03_executing_kernels.py` | `sample()` vs `run()` vs `observe()` vs `get_state()` |
| `04_operators.py` | Spin operators, a from-scratch mini-VQE |
| `05_optimizers_gradients.py` | Adam, SPSA, and SciPy COBYLA on a VQE problem |
| `06_ptsbe.py` | Pre-Trajectory Sampling with Batch Execution (noisy circuits) |
| `07_quantum_teleportation.py` | Teleporting a qubit via a Bell pair + classical bits |
| `08_grovers_algorithm.py` | Grover's search, verified against the classic overshoot curve |
| `09_shors_algorithm.py` | Shor's algorithm factoring 15 = 3x5, via a custom modular-exponentiation gate |

## `superdense_coding.py`

The mirror image of `07_quantum_teleportation.py`: instead of moving 1 unknown
qubit with a Bell pair + 2 classical bits, this moves 2 known classical bits
with a Bell pair + 1 physically-transmitted qubit. Verified against all 4
possible message pairs, every shot, on the `nvidia` simulator.

## `teleportation.py`

A standalone deep-dive into `07_quantum_teleportation.py`'s protocol, built to
answer a series of "wait, but why" questions rather than just show the happy
path: proves the Bell-pair setup must precede Alice's CNOT (swapping the two
breaks teleportation), proves Bob's qubit carries zero information before the
classical bits arrive (no-signaling), proves both classical bits are
structurally required (dropping either is invisible in one measurement basis
but breaks the other), and isolates Bob's correction step into a fully
standalone proof (1 qubit + 2 bits, no message qubit, no sender qubit, no
entanglement in scope at all).

## `quantum_sorting_network.py`

A real answer to "can you sort qubits while they're in superposition?" A
3-element sorting network (compare (e0,e1), (e1,e2), (e0,e1) again) built
entirely from reversible gates - Toffoli to compute "x > y" into an ancilla,
Fredkin (`swap.ctrl`) to conditionally swap - so the whole sort is one
unitary with no measurement in the middle. Run on a full superposition over
all 64 possible 3-tuple inputs at once; verified across 2000 shots that
every single sampled shot's output is the correctly sorted version of that
same shot's input, and that the comparator's helper ancillas uncompute back
to clean `|0>` every time.

## `novel_gates/`

Custom gates built via `cudaq.register_operation` or plain kernel
composition, each one analyzed for what it actually does mechanically:

| File | What it demonstrates |
|---|---|
| `maxwells_demon.py` | A coherent, reversible Maxwell's demon contrasted with an irreversible measurement |
| `quantum_wall.py` | Quantum tunneling through a wall using real WKB transmission math |
| `fresnel_lens.py` | A 5-zone Fresnel lens built as a genuine interference/recombination circuit, verified with exact `get_state()` checks |

## `jans_life_quantum_sim.py`

A fun one: encodes the real branching life-decisions of Jan Dolski (from the
video game *The Alters*) as an entangled decision tree, and samples it to find
"the most likely Jan."

## `schrodingers_sales_funnel.py`

A B2B sales funnel modeled as a chain of conditional quantum coin flips
(Lead -> MQL -> SQL -> Opportunity -> Closed-Won), verified against classical
funnel math - plus a second part reusing `schrodingers_detective.py`'s quantum
eraser mechanism as an honest, explicitly-labeled *analogy* for multi-touch
marketing attribution ambiguity (same underlying data, different "lens,"
different credit assigned) - not a claim that CRM data behaves quantum
mechanically.

## `executive_summary.md`

A business-facing summary of what this project's demos actually show about
quantum computing's real, current capabilities (true randomness, amplitude
amplification, correlated-uncertainty modeling) versus hype - source material
for a follow-up slide deck.

## The Schrodinger's \* series

Each script pairs a genuine CUDA-Q quantum mechanism (superposition, entangling
gates, phase, mid-circuit measurement) with a story that makes the mechanism
intuitive - and each one is verified against classical ground truth, brute
force, or hand-derived math, not just run once and eyeballed.

| File | What it demonstrates |
|---|---|
| `schrodingers_christmas.py` | A dynamically-sized entangled "graph" (N kids, each correlated with their own gift box) |
| `schrodingers_sudoku.py` | Solving a genuinely hard Sudoku via classical backtracking + a real quantum coin flip at each ambiguous decision |
| `schrodingers_dungeon.py` | A from-scratch, honestly-scoped quantum take on Wave Function Collapse procedural generation |
| `schrodingers_game_of_life.py` | Conway's Game of Life with a quantum-seeded (not pseudo-random) initial board, rendered live with pygame |
| `schrodingers_detective.py` | A real quantum eraser: measuring in a different basis destroys a correlation without undoing any gate |
| `schrodingers_epidemic.py` | Infection spread on a real graph with a cycle - showing gate order along a cycle genuinely changes the outcome |
| `schrodingers_higher_order.py` | 2nd- and 3rd-order "uncertainty about uncertainty," verified via nested conditional probabilities |
| `schrodingers_interval_scheduling.py` | Weighted interval scheduling solved by a Grover oracle built directly from the interval conflict graph (no DP table) |
| `schrodingers_shipping.py` | The (NP-hard) multi-berth Berth Allocation Problem - and a real, honest demonstration of Grover's oscillation failure mode when the marked fraction exceeds 50% |
| `schrodingers_tic_tac_toe.py` | Quantum Tic-Tac-Toe's "spooky mark" cycle collapse, shown to be mechanically identical to a GHZ state |

Two more that don't fit the table (they're not story-first, they're a
mechanism-first pair building on each other):

### `quantum_attention.py`

LLM attention translated as literally as possible into gates: a query and 3
key vectors get scored by dot product and softmax'd - plain classical math,
same as a real attention layer - but the softmax weights are then
amplitude-encoded directly into a 2-qubit key register's superposition
(`sqrt(weight_i)` on each branch, so sampling that register up front IS
sampling from the softmax, exactly). Then one value qubit gets a controlled
`RY(arccos(value_i))` rotation per key branch, and - crucially - the key
register is never conditioned on. That makes the value qubit's marginal
state a genuine probabilistic mixture of the 3 branches, and `<Z>` on that
mixture is provably `sum_i softmax(query.key_i) * value_i` - the literal
attention output - because expectation values are linear over mixtures.
Verified against the classical softmax-attention formula, and against each
branch's conditional `<Z | key=i>` individually equaling `value_i`.

### `quantum_hash_set.py`

A "hash set" needs no values, just membership - built by importing
`quantum_attention.py` and treating membership itself as the blended
value: `+1.0` for members, `-1.0` for non-members. A query vector's
"contains?" answer is the attention output: a softmax-weighted blend of
those +-1 scores by similarity to each stored key. Grounded in a real
result (modern continuous Hopfield networks are provably one attention
step - Ramsauer et al., "Hopfield Networks is All You Need"), not a made-up
analogy. Verified on 3 queries: one that confidently matches a member
(score -> +1), one that confidently matches a non-member (score -> -1),
and one deliberately ambiguous between a member and a non-member (score ->
0) - a graceful "not sure" a classical hash set has no way to express.

## Weighted interval scheduling, three ways

A single problem solved three times, each version verified against the one
before it - CPU, to real GPU (SIMT) parallelism, to quantum:

- `01_cpu_interval_scheduling.py` - plain sequential Python DP (the reference).
- `02_cuda_simt_interval_scheduling.py` - a real CUDA C kernel (compiled via
  NVRTC, through CuPy's `RawKernel`) parallelizes the genuinely-parallelizable
  part (computing each interval's compatibility pointer), while the DP
  recurrence itself stays sequential since it has a real data dependency.
- `03_cudaq_interval_scheduling.py` - the quantum version: no DP table, an
  entangling gate per conflicting pair of intervals, Grover amplification.
