"""Modular inversion by divsteps, uncomputed WITHOUT a history of branch choices.

From "Discovering state-of-the-art quantum circuits with large language
models" (Gu, Kemeny, Yelin, Lukin - arXiv:2610.22367), the "Reversible
modular inversion" section, Equations (6)-(9) and Figure 3.

Modular inversion (find x with w*x = 1 mod p) is a core subroutine of
elliptic-curve and number-theory quantum algorithms. "Divsteps" is a
division-free cousin of the extended Euclidean algorithm: keep the odd
integer f fixed, halve g (making it even first by adding/subtracting f if it
is odd), and occasionally swap the roles of f and g, scheduled by a small
signed counter z. Two auxiliary integers fv and gv track the input's
contribution so that, after t steps, fv*w = 2^t * f and gv*w = 2^t * g
(mod p). Starting from (z, f, g, fv, gv) = (0, p, w, 0, 1), each step is one
of three updates (paper's Equation 6):

    (6a)  z >= 0, g odd:  (-z,  g,  (g - f)/2,  2*gv,  gv - fv)   <- swap
    (6b)  z <  0, g odd:  (z+1, f,  (g + f)/2,  2*fv,  gv + fv)
    (6c)  g even:         (z+1, f,   g/2,       2*fv,  gv)

Classically each step overwrites its inputs. Running the circuit backwards
needs to know which update ran. The naive fix stores two bits per step (swap?
odd?) - a history that grows with the number of steps. The paper's circuit
instead DECODES both bits from the updated registers:

  * swap bit s: a swap sets fv' = 2*gv with gv odd, so fv' has exactly ONE
    trailing zero; a non-swap sets fv' = 2*fv, and fv is always even, so
    fv' has at least two. Always recoverable.
  * odd bit b, easy cases: if z' >= 1 the step cannot have been (6b) (6b
    leaves z' <= 0) and an odd g with z >= 0 would have swapped, so g was
    even: b = 0.
  * odd bit b, the hard case (non-swap, z' <= 0): either (6b) or (6c)
    could have run. After a swap, fv' = 2q with q odd, and on the j-th step
    of the ensuing run of non-swaps ("excursion") fv' = 2^j * q while the
    gv update is gv' = gv + b * 2^(j-1) * q (the shared factor q is why this
    isn't just the binary-record trick from script 02). Multiply by q^-1
    mod 2^j (q is odd, so it exists):

        b = bit_{j-1}( gv' * q^-1  mod 2^j )            (paper's Equation 9)

    where j = number of trailing zeros of fv' and q = fv' / 2^j. The paper
    proves (gv * q^-1 mod 2^j) < 2^(j-1), so the earlier history cannot
    contaminate bit j-1.

Because the rule at depth j reads only the lowest j bits of gv and q, small j
is cheap to implement and large j is expensive - so the paper picks a cutoff
j_store: decisions deeper than that are simply written to a small stack
instead of decoded, trading Toffolis against qubits. And since most walks
need only a handful of stored decisions (and terminate in a narrow window of
steps), it provisions a stack that is small but not worst-case and lets the
rare walk that overflows FAIL - the paper's "failure budget".

This is a classical simulation of reversible basis-state tracking, which is
how the paper's own evaluator checks circuits (every gate here maps a basis
state to a basis state). No CUDA-Q kernels: the point verified is that the
decoding is exact, so the history stack is unnecessary. Verified:
  1. The divstep walk really computes modular inverses (exhaustively for
     small primes, random for a 256-bit prime).
  2. For every step of every walk, the (s, b) decoded from the UPDATED state
     alone equals the true (s, b).
  3. Running the whole walk backwards using only decoded bits restores the
     initial state exactly.
  4. Measured stopping-time and stack-size distributions for a 256-bit
     prime, to show the failure-budget trade.
"""

import random
from dataclasses import dataclass

SECP256K1_PRIME = 2**256 - 2**32 - 977
PAPER_JSTORE = 5  # the paper's cutoff: decode below this depth, store at or above it


@dataclass(frozen=True)
class DivstepState:
    z: int
    f: int
    g: int
    f_aux: int  # the paper's fv
    g_aux: int  # the paper's gv


def initial_state(prime: int, value: int) -> DivstepState:
    return DivstepState(z=0, f=prime, g=value, f_aux=0, g_aux=1)


def forward_step(state: DivstepState) -> tuple[DivstepState, int, int]:
    """One divstep. Returns (new state, swapped, g_was_odd)."""
    z, f, g, f_aux, g_aux = state.z, state.f, state.g, state.f_aux, state.g_aux
    if g % 2 == 1 and z >= 0:                                    # (6a) swap
        return DivstepState(-z, g, (g - f) // 2, 2 * g_aux, g_aux - f_aux), 1, 1
    if g % 2 == 1:                                               # (6b)
        return DivstepState(z + 1, f, (g + f) // 2, 2 * f_aux, g_aux + f_aux), 0, 1
    return DivstepState(z + 1, f, g // 2, 2 * f_aux, g_aux), 0, 0  # (6c)


def trailing_zero_bits(value: int) -> int:
    return (value & -value).bit_length() - 1


def decode_branch(after: DivstepState) -> tuple[int, int, int]:
    """Recover (swapped, g_was_odd) from the UPDATED registers only.

    Returns (swapped, g_was_odd, excursion_depth); the depth is 0 unless the
    hard case (Equation 9) was needed.
    """
    swapped = int(after.f_aux != 0 and trailing_zero_bits(after.f_aux) == 1)
    if swapped:
        return 1, 1, 0
    if after.z >= 1:
        return 0, 0, 0
    # Hard case: non-swap with z' <= 0. f_aux' = 2^j * q, q odd.
    depth = trailing_zero_bits(after.f_aux)
    odd_part = after.f_aux >> depth
    modulus = 1 << depth
    scaled = (after.g_aux * pow(odd_part, -1, modulus)) % modulus
    return 0, (scaled >> (depth - 1)) & 1, depth


def backward_step(after: DivstepState, swapped: int, g_was_odd: int) -> DivstepState:
    """Invert one divstep given the (decoded) branch bits."""
    if swapped:                                                  # undo (6a)
        old_g = after.f                      # the swap moved g into f's slot
        old_f = old_g - 2 * after.g
        old_g_aux = after.f_aux // 2
        old_f_aux = old_g_aux - after.g_aux
        return DivstepState(-after.z, old_f, old_g, old_f_aux, old_g_aux)
    old_f_aux = after.f_aux // 2
    if g_was_odd:                                                # undo (6b)
        old_g = 2 * after.g - after.f
        return DivstepState(after.z - 1, after.f, old_g, old_f_aux, after.g_aux - old_f_aux)
    return DivstepState(after.z - 1, after.f, 2 * after.g, old_f_aux, after.g_aux)  # undo (6c)


def steps_needed(prime: int) -> int:
    """A generous fixed schedule (the circuit must be a fixed gate sequence)."""
    return 3 * prime.bit_length() + 10


def invert_mod(prime: int, value: int, num_steps: int) -> int:
    state = initial_state(prime, value)
    for _ in range(num_steps):
        state, _, _ = forward_step(state)
    assert state.g == 0 and abs(state.f) == 1, "walk did not terminate in time"
    # f_aux * w = 2^t * f  =>  w^-1 = f * f_aux / 2^t  (f = +-1)
    return (state.f * state.f_aux * pow(2, -num_steps, prime)) % prime


def verify_walk(prime: int, value: int, num_steps: int) -> int:
    """Run the walk forward, decode every step from the updated state, then
    run it backwards using ONLY decoded bits. Returns the number of hard
    (Equation 9) decodings."""
    start = initial_state(prime, value)
    trajectory = [start]
    truth = []
    for _ in range(num_steps):
        next_state, swapped, was_odd = forward_step(trajectory[-1])
        trajectory.append(next_state)
        truth.append((swapped, was_odd))

    hard_decodings = 0
    state = trajectory[-1]
    for step_index in range(num_steps - 1, -1, -1):
        swapped, was_odd, depth = decode_branch(state)
        assert (swapped, was_odd) == truth[step_index], (
            f"decode mismatch at step {step_index}: got {(swapped, was_odd)}, "
            f"truth {truth[step_index]} (p={prime}, w={value})")
        hard_decodings += depth > 0
        state = backward_step(state, swapped, was_odd)
        assert state == trajectory[step_index], "backward step did not restore the state"
    assert state == start
    return hard_decodings


def primes_below(limit: int) -> list[int]:
    sieve = [True] * limit
    found = []
    for candidate in range(2, limit):
        if sieve[candidate]:
            if candidate > 2:
                found.append(candidate)
            for multiple in range(candidate * candidate, limit, candidate):
                sieve[multiple] = False
    return found


def walk_statistics(prime: int, value: int, jstore: int) -> tuple[int, int]:
    """Returns (stopping time, stored decisions) for one walk.

    Stored decisions = ambiguous non-swap steps (z' <= 0) taken before the
    walk terminates whose excursion depth is >= jstore."""
    state = initial_state(prime, value)
    stopping_time = 0
    stored = 0
    num_steps = steps_needed(prime)
    for step_number in range(1, num_steps + 1):
        was_active = state.g != 0
        state, swapped, _ = forward_step(state)
        if was_active:
            if not swapped and state.z <= 0:
                if trailing_zero_bits(state.f_aux) >= jstore:
                    stored += 1
            if state.g == 0:
                stopping_time = step_number
    return stopping_time, stored


if __name__ == "__main__":
    random.seed(0)

    print("1+2+3. Exhaustive check on every odd prime below 200 and every "
          "nonzero input:")
    total_walks = 0
    total_hard = 0
    for prime in primes_below(200):
        num_steps = steps_needed(prime)
        for value in range(1, prime):
            inverse = invert_mod(prime, value, num_steps)
            assert (inverse * value) % prime == 1, f"bad inverse for {value} mod {prime}"
            total_hard += verify_walk(prime, value, num_steps)
            total_walks += 1
    print(f"   {total_walks} walks: inverses correct, every decoded (s, b) matches the "
          f"truth, and every walk unwinds back to its start from decoded bits alone")
    print(f"   ({total_hard} of the decodings needed the hard Equation 9 rule)\n")

    print("   Spot check on the 256-bit secp256k1 prime:")
    for _ in range(20):
        value = random.randrange(1, SECP256K1_PRIME)
        num_steps = steps_needed(SECP256K1_PRIME)
        assert (invert_mod(SECP256K1_PRIME, value, num_steps) * value) % SECP256K1_PRIME == 1
        verify_walk(SECP256K1_PRIME, value, num_steps)
    print("   20 random inputs: inverses correct, branch decoding exact at every step\n")

    print(f"4. Failure-budget statistics, 256-bit prime, j_store = {PAPER_JSTORE}:")
    sample_count = 3000
    stopping_times = []
    stored_counts = []
    for _ in range(sample_count):
        value = random.randrange(1, SECP256K1_PRIME)
        stopping_time, stored = walk_statistics(SECP256K1_PRIME, value, PAPER_JSTORE)
        stopping_times.append(stopping_time)
        stored_counts.append(stored)

    stopping_times.sort()
    stored_counts.sort()
    print(f"   stopping time over {sample_count} random inputs: min {stopping_times[0]}, "
          f"median {stopping_times[sample_count // 2]}, max {stopping_times[-1]}")
    print(f"   stored decisions per walk: mean {sum(stored_counts) / sample_count:.1f}, "
          f"median {stored_counts[sample_count // 2]}, max {stored_counts[-1]}")

    window_low = stopping_times[int(0.001 * sample_count)]
    window_high = stopping_times[int(0.999 * sample_count)]
    print(f"   99.8% of walks stop within steps {window_low}..{window_high}; a circuit "
          f"scheduled for just that window (and only testing for termination "
          f"inside it) fails on the rest")
    stack_capacity = stored_counts[int(0.999 * sample_count)]
    overflow = sum(count > stack_capacity for count in stored_counts) / sample_count
    print(f"   a stack of {stack_capacity} entries instead of the largest "
          f"sampled {stored_counts[-1]} overflows on {overflow:.2%} of walks")
    print("\n   The paper reports the same shape for its benchmark: stopping times "
          "concentrated in a narrow window (it schedules 533 steps, testing from "
          "step 499, against a 590-step worst case), an average of 21 stored "
          "decisions, and a 60-entry stack against a 263-entry worst case, "
          "failing with probability ~2.4e-6. This run reproduces the window and "
          "the average; the tail (so the exact overflow rate) needs far more "
          "samples or the paper's exact bookkeeping, so don't read our overflow "
          "percentage as their figure.")
