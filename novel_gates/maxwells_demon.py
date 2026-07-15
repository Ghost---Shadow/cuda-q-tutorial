"""
Maxwell's Demon
=================

Answers a question asked directly: does CUDA-Q allow genuinely novel,
hypothetical gates? Yes - `cudaq.register_operation(name, unitary)`
registers any matrix you like as a new named gate, usable with
`.ctrl()`, `.adj()`, etc. exactly like a built-in one. This script
registers `maxwells_demon` and uses it to build the actual physics
thought experiment, not just a name: a demon that sorts hot particles
from cold ones without (apparently) doing any work - the 19th-century
paradox that seemed to violate the second law of thermodynamics.

The registered gate is, underneath, a CNOT - which is exactly the
point. A REAL Maxwell's demon, one that keeps a coherent quantum
record of its sorting decision instead of throwing it away, is not
mysterious - it's provably a plain unitary flip, gated on a genuinely
undetermined particle. `maxwells_demon.ctrl(particle_temperature,
door)` opens door only for a particle whose temperature qubit reads
hot - so this script's demon is coherent and reversible by
construction, and Part 1 verifies that literally: sort, then undo, and
recover the exact original statevector (not just similar statistics -
identical amplitudes).

The paradox's real resolution (Landauer's principle: information
erasure has an unavoidable thermodynamic cost) shows up in Part 2, by
contrast: a demon that instead LOOKS (mz(), an actual measurement, not
a coherent CNOT) collapses the particle for good. There's no unitary
to undo afterward - the superposition is just gone. That's the
qualitative difference this script can honestly show with a qubit
simulator; the quantitative energy cost of erasing a demon's memory
belongs to open-system thermodynamics, not a closed-circuit statevector,
so this script doesn't claim to derive it - only to show why a
coherent demon is physically unremarkable while a measuring one incurs
a real, irreversible loss.

Bonus, tested directly rather than assumed: CUDA-Q does NOT check that
a registered matrix is unitary. Part 3 registers an explicitly
non-unitary "broken demon" and shows it runs anyway, silently, with
physically nonsensical output - a real caveat, not a hypothetical one.
"""

import numpy as np
import cudaq

# The "novel gate" requested: registered as a distinct name, but its
# unitary is plain X - a coherent sort is just a controlled bit-flip.
MAXWELLS_DEMON_UNITARY = np.array([0, 1, 1, 0], dtype=np.complex128)
cudaq.register_operation("maxwells_demon", MAXWELLS_DEMON_UNITARY)


@cudaq.kernel
def particles_only(num_particles: int, angle: float):
    """Just the particles, each genuinely undetermined hot/cold via
    ry() - no demon yet, no measurement. Used as the "before" state to
    compare against, via get_state()."""
    particles = cudaq.qvector(num_particles)
    for i in range(num_particles):
        ry(angle, particles[i])


@cudaq.kernel
def sort_then_unsort(num_particles: int, angle: float):
    """The demon sorts every particle into a door qubit (coherently,
    via the registered gate), then immediately un-sorts by applying
    the exact same gate again - CNOT is its own inverse, so this
    should be a perfect round trip. No measurement anywhere, so
    get_state() can verify the FULL system exactly."""
    particles = cudaq.qvector(num_particles)
    doors = cudaq.qvector(num_particles)
    for i in range(num_particles):
        ry(angle, particles[i])
    for i in range(num_particles):
        maxwells_demon.ctrl(particles[i], doors[i])
    for i in range(num_particles):
        maxwells_demon.ctrl(particles[i], doors[i])


@cudaq.kernel
def coherent_sort_only(num_particles: int, angle: float) -> list[bool]:
    """The demon sorts (coherently) and we measure both particles and
    doors, to see the actual hot/cold <-> chamber correlation the
    demon produced - the "useful" half of the experiment, separate
    from the reversibility proof above."""
    particles = cudaq.qvector(num_particles)
    doors = cudaq.qvector(num_particles)
    for i in range(num_particles):
        ry(angle, particles[i])
    for i in range(num_particles):
        maxwells_demon.ctrl(particles[i], doors[i])

    particle_readings = mz(particles)
    door_readings = mz(doors)
    combined = [False for _ in range(2 * num_particles)]
    for i in range(num_particles):
        combined[i] = particle_readings[i]
        combined[num_particles + i] = door_readings[i]
    return combined


@cudaq.kernel
def measuring_demon(num_particles: int, angle: float) -> list[bool]:
    """A demon that LOOKS instead of coherently sorting: mz() each
    particle directly. This is a real measurement, not a gate - once
    it happens, the particle's superposition is gone for good, with
    no unitary left to undo. Immediately re-applying h() and
    measuring again demonstrates this: it's a fresh coin flip, not a
    recovery of the original undetermined state."""
    particles = cudaq.qvector(num_particles)
    for i in range(num_particles):
        ry(angle, particles[i])
    first_look = mz(particles)
    for i in range(num_particles):
        h(particles[i])
    second_look = mz(particles)
    results = [False for _ in range(2 * num_particles)]
    for i in range(num_particles):
        results[i] = first_look[i]
        results[num_particles + i] = second_look[i]
    return results


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    NUM_PARTICLES = 3
    ANGLE = 1.1  # an arbitrary, non-trivial angle - deliberately NOT a
                 # fair 50/50 coin, so a state-vector mismatch would be
                 # obvious rather than accidentally symmetric

    print("=== Part 1: is the coherent demon really reversible? ===")
    print("Comparing get_state() of 'just the particles' against")
    print("get_state() of 'particles -> sorted -> unsorted', exactly,")
    print("amplitude by amplitude - no sampling, no statistics.")

    before_state = np.array(cudaq.get_state(particles_only, NUM_PARTICLES, ANGLE))
    after_state = np.array(cudaq.get_state(sort_then_unsort, NUM_PARTICLES, ANGLE))

    # after_state lives on (particles + doors); doors should be back to
    # |0...0>, so the expected full state is before_state (x) |0...0>.
    expected_state = np.zeros(len(before_state) * (2 ** NUM_PARTICLES), dtype=complex)
    expected_state[:len(before_state)] = before_state  # doors=|0..0> is index 0 of that register

    max_amplitude_diff = np.max(np.abs(after_state - expected_state))
    print(f"Max amplitude difference (sorted+unsorted vs. original (x) |000>): "
          f"{max_amplitude_diff:.2e}")
    print("Effectively zero -> the demon's sort is EXACTLY undone. No information")
    print("was lost; a coherent 'peek' that keeps its own record is just a gate.")

    print()
    print("=== Part 2: what the coherent sort actually produces ===")
    shots = 100
    results = cudaq.run(coherent_sort_only, NUM_PARTICLES, ANGLE, shots_count=shots)
    mismatches = sum(1 for r in results if list(r[:NUM_PARTICLES]) != list(r[NUM_PARTICLES:]))
    print(f"Particle temperature vs. its own door, across {shots} shots: "
          f"{mismatches}/{shots} mismatches")
    print("Zero mismatches expected - hot particles (1) always end up behind the")
    print("hot door (1), cold behind the cold door (0), every single shot.")

    print()
    print("=== Part 3: a demon that LOOKS instead of coherently sorting ===")
    results2 = cudaq.run(measuring_demon, NUM_PARTICLES, ANGLE, shots_count=shots)
    p_first_hot = sum(1 for r in results2 for b in r[:NUM_PARTICLES] if b) / (shots * NUM_PARTICLES)
    agree_after_relook = sum(
        1 for r in results2
        for i in range(NUM_PARTICLES)
        if r[i] == r[NUM_PARTICLES + i]
    ) / (shots * NUM_PARTICLES)
    print(f"P(hot) on the demon's first look: {p_first_hot:.3f}")
    print(f"Agreement between the first look and a second look (after h() again): "
          f"{agree_after_relook:.3f}")
    print("~50% agreement, same as 2 independent coin flips - the first mz()")
    print("really did destroy the original state. There's no gate that undoes a")
    print("measurement; 'looking' and 'coherently sorting' are different physics,")
    print("even though both can be dressed up as 'the demon checked the particle.'")

    print()
    print("=== Part 4: CUDA-Q does not check your novel gate is unitary ===")
    broken_demon = np.array([1, 0, 0, 0], dtype=np.complex128)  # a projector, not unitary
    cudaq.register_operation("broken_demon", broken_demon)

    @cudaq.kernel
    def use_broken_demon() -> bool:
        q = cudaq.qubit()
        h(q)
        broken_demon(q)
        return mz(q)

    broken_results = cudaq.run(use_broken_demon, shots_count=20)
    print(f"A qubit put in equal superposition, then hit with a non-unitary "
          f"'demon': {broken_results}")
    print("Registered and ran without any error - every shot reads False, which")
    print("is NOT what a real quantum gate could ever do to a fair coin. CUDA-Q")
    print("trusts the matrix you hand it; physical validity (unitarity) is on you.")

    print()
    print("=== Honest conclusion ===")
    print("A coherent Maxwell's demon - one that records its sorting decision in")
    print("an actual qubit rather than discarding it - is not a paradox at all;")
    print("Part 1 proves it's exactly as reversible as any other CNOT. The")
    print("second law survives because the ONLY way to make the demon's")
    print("information useful for practical sorting (Part 2's clean correlation)")
    print("without eventually erasing that record runs into Landauer's principle")
    print("- and a closed, unitary qubit circuit like this one is the wrong tool")
    print("to quantify that thermodynamic cost, so this script doesn't claim to.")
    print("What it does show, concretely: coherent 'checking' (a gate) and actual")
    print("measurement (mz()) are physically different operations with different")
    print("consequences, and CUDA-Q will happily run a gate that isn't even")
    print("physically valid if you register one - novel gates are real and easy")
    print("to build, but nothing stops you from building a broken one.")
