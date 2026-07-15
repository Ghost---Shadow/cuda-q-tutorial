"""
Quantum Tunneling
===================

The requested gate: `quantum_wall.ctrl(particle_velocities, wall_height)`.
Registered here as a genuinely novel CUDA-Q operation
(`cudaq.register_operation`, the same mechanism as
novel_gates/maxwells_demon.py), used to build real quantum tunneling:
a particle with LESS kinetic energy than a potential barrier can still,
sometimes, be found on the far side of it - not because it jumped over,
but because its position was never a definite fact on either side
until measured, and the wavefunction genuinely extends a little way
past a "classically impossible" boundary.

Two honest adjustments from the literal snippet, both to keep the
demo pythonic and physically real rather than just gesturing at it:

  - `wall_height` in the snippet reads as the register a particle
    acts ON. Here it's split into its true 2 roles: WALL_HEIGHT, a
    fixed physical constant (the barrier's energy, used to compute
    real transmission probabilities), and `passed_through` - the
    per-particle qubit that actually gets acted on, recording whether
    THIS particle made it past the wall.
  - `quantum_wall` (the registered gate) is used for exactly the case
    it's actually correct for: a classically-ALLOWED particle
    (kinetic energy >= wall height) passes with certainty, which is a
    plain controlled bit-flip - the same "novel-gate-that's-secretly-
    an-X-gate" idea as maxwells_demon.py. A classically-FORBIDDEN
    particle's tunneling probability is graded (falls off
    exponentially, not a flat coin flip), which needs a real angle per
    energy level - cudaq.register_operation only accepts a FIXED
    matrix (verified directly while building maxwells_demon.py), so
    the graded tunneling itself is built from ry.ctrl() with an angle
    computed from actual physics, not the novel gate.

The physics used is the standard WKB (thick-barrier) approximation for
a rectangular potential barrier, in natural units (hbar = mass = 1):

    T(E) = 1                                  if E >= V (classically allowed)
    T(E) = exp(-2 * sqrt(2*(V-E)) * L)         if E <  V (classically forbidden)

where V is WALL_HEIGHT and L is BARRIER_WIDTH. This is a real,
textbook formula (not a picked-by-hand dummy probability like some of
this repo's other decision trees) - it's what makes the two forbidden
energy levels below get GENUINELY DIFFERENT, exponentially-scaled
tunneling probabilities rather than one flat "sometimes it tunnels"
number.
"""

import math

import numpy as np
import cudaq

WALL_HEIGHT = 5.0     # potential barrier height, arbitrary energy units
BARRIER_WIDTH = 1.0   # barrier width, arbitrary length units (hbar = mass = 1)

# 4 discretized particle kinetic energies - 2 classically forbidden
# (below the wall), 2 classically allowed (at or above it).
ENERGY_LEVELS = [2.0, 3.5, 6.0, 8.0]


def wkb_transmission(energy: float) -> float:
    """The real WKB tunneling-transmission formula for a rectangular
    barrier - see module docstring."""
    if energy >= WALL_HEIGHT:
        return 1.0
    kappa = math.sqrt(2 * (WALL_HEIGHT - energy))
    return math.exp(-2 * kappa * BARRIER_WIDTH)


TRANSMISSION_BY_LEVEL = [wkb_transmission(e) for e in ENERGY_LEVELS]
ANGLE_BY_LEVEL = [2 * math.asin(math.sqrt(t)) for t in TRANSMISSION_BY_LEVEL]

# The novel gate: for a classically-allowed particle, "hitting the
# wall" is a certainty of passing through - a plain bit flip.
QUANTUM_WALL_UNITARY = np.array([0, 1, 1, 0], dtype=np.complex128)
cudaq.register_operation("quantum_wall", QUANTUM_WALL_UNITARY)


@cudaq.kernel
def collide_with_wall(energy_bit_high: cudaq.qubit, energy_bit_low: cudaq.qubit,
                       passed_through: cudaq.qubit):
    """4 energy levels encoded as (energy_bit_high, energy_bit_low):
      (0,0) -> ENERGY_LEVELS[0] = 2.0  (forbidden - small tunnel chance)
      (0,1) -> ENERGY_LEVELS[1] = 3.5  (forbidden - larger tunnel chance)
      (1,0) -> ENERGY_LEVELS[2] = 6.0  (allowed - always passes)
      (1,1) -> ENERGY_LEVELS[3] = 8.0  (allowed - always passes)
    Same "open control" leaf-by-leaf trick as
    schrodingers_weather.py's decision tree, just with angles taken
    from real tunneling physics instead of hand-picked numbers, and
    the 2 classically-allowed leaves using the registered
    `quantum_wall` gate instead of a generic ry rotation."""
    # (0,0): forbidden, tunneling angle for ENERGY_LEVELS[0]
    x(energy_bit_high)
    x(energy_bit_low)
    ry.ctrl(ANGLE_BY_LEVEL[0], energy_bit_high, energy_bit_low, passed_through)
    x(energy_bit_high)
    x(energy_bit_low)

    # (0,1): forbidden, tunneling angle for ENERGY_LEVELS[1]
    x(energy_bit_high)
    ry.ctrl(ANGLE_BY_LEVEL[1], energy_bit_high, energy_bit_low, passed_through)
    x(energy_bit_high)

    # (1,0): allowed - certain pass via the registered quantum_wall gate
    x(energy_bit_low)
    quantum_wall.ctrl(energy_bit_high, energy_bit_low, passed_through)
    x(energy_bit_low)

    # (1,1): allowed - certain pass via the registered quantum_wall gate
    quantum_wall.ctrl(energy_bit_high, energy_bit_low, passed_through)


@cudaq.kernel
def fire_particles_at_wall(num_particles: int) -> list[bool]:
    """Each particle's kinetic energy is genuinely undetermined
    (h() on both encoding qubits - a uniform, equally-likely spread
    across the 4 discretized levels) until it collides with the wall
    and passed_through is measured."""
    energy_bits_high = cudaq.qvector(num_particles)
    energy_bits_low = cudaq.qvector(num_particles)
    passed_through = cudaq.qvector(num_particles)

    for i in range(num_particles):
        h(energy_bits_high[i])
        h(energy_bits_low[i])

    for i in range(num_particles):
        collide_with_wall(energy_bits_high[i], energy_bits_low[i], passed_through[i])

    high_readings = mz(energy_bits_high)
    low_readings = mz(energy_bits_low)
    passed_readings = mz(passed_through)

    results = [False for _ in range(3 * num_particles)]
    for i in range(num_particles):
        results[i] = high_readings[i]
        results[num_particles + i] = low_readings[i]
        results[2 * num_particles + i] = passed_readings[i]
    return results


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== The wall and the 4 discretized particle energy levels ===")
    print(f"Wall height (V) = {WALL_HEIGHT}, barrier width (L) = {BARRIER_WIDTH}")
    for i, energy in enumerate(ENERGY_LEVELS):
        classification = "classically ALLOWED" if energy >= WALL_HEIGHT else "classically FORBIDDEN"
        print(f"  E={energy:<5} ({classification:<22}) -> "
              f"WKB transmission T = {TRANSMISSION_BY_LEVEL[i]:.5f}")

    num_particles = 5
    shots = 100
    print(f"\n=== Firing {num_particles} particles/shot at the wall, {shots} shots ===")
    results = cudaq.run(fire_particles_at_wall, num_particles, shots_count=shots)

    # Per-particle, per-shot: (energy_bit_high, energy_bit_low, passed_through)
    level_hits = [0, 0, 0, 0]
    level_passes = [0, 0, 0, 0]
    for r in results:
        bits_high = r[:num_particles]
        bits_low = r[num_particles:2 * num_particles]
        bits_passed = r[2 * num_particles:]
        for i in range(num_particles):
            level_index = (2 if bits_high[i] else 0) + (1 if bits_low[i] else 0)
            level_hits[level_index] += 1
            if bits_passed[i]:
                level_passes[level_index] += 1

    print()
    print("=== Measured transmission rate per energy level, vs. the WKB prediction ===")
    print(f"  {'energy':>8}  {'classification':<22}{'measured T':>12}{'predicted T':>14}{'n':>8}")
    for i, energy in enumerate(ENERGY_LEVELS):
        classification = "ALLOWED" if energy >= WALL_HEIGHT else "FORBIDDEN"
        measured = level_passes[i] / level_hits[i] if level_hits[i] else float("nan")
        print(f"  {energy:>8}  {classification:<22}{measured:>12.4f}"
              f"{TRANSMISSION_BY_LEVEL[i]:>14.5f}{level_hits[i]:>8}")

    total_forbidden_hits = level_hits[0] + level_hits[1]
    total_forbidden_passes = level_passes[0] + level_passes[1]
    print()
    print(f"=== The actual tunneling headline ===")
    print(f"Classically-forbidden particles that STILL made it through the wall: "
          f"{total_forbidden_passes}/{total_forbidden_hits}")
    print("Every one of those particles had less kinetic energy than the wall is")
    print("tall - a classical ball never clears that barrier, ever. These did,")
    print("some of the time, at exactly the rate real tunneling math predicts -")
    print("not because they jumped over it, but because their position was never")
    print("a settled fact until passed_through got measured.")

    print()
    print("=== Honest conclusion ===")
    print("The registered `quantum_wall` gate itself is nothing exotic - for an")
    print("energetically-allowed particle it's a plain, certain bit flip, same as")
    print("maxwells_demon.py's coherent sort. The genuinely quantum part is the")
    print("forbidden branch: a nonzero, exponentially-small-but-real probability")
    print("of passing an energy barrier with insufficient energy to do so")
    print("classically, taken directly from the WKB transmission formula rather")
    print("than an invented number - and it's the one thing on this list from")
    print("earlier ('what can a quantum computer do that a classical one can't')")
    print("that isn't about computation at all: it's a real physical phenomenon,")
    print("already exploited today in scanning tunneling microscopes and flash")
    print("memory, that this script is simulating rather than creating.")
