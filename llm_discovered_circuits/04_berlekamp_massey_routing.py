"""Reed-Solomon error location: Berlekamp-Massey, routed so it stays reversible.

From "Discovering state-of-the-art quantum circuits with large language
models" (Gu, Kemeny, Yelin, Lukin - arXiv:2610.22367), the "Reversible
syndrome decoding for DQI" section, Equations (3)-(5) and Figure 2.

Decoded quantum interferometry needs a Reed-Solomon decoder running
coherently on a superposition of inputs. The decoder finds the error
positions from "syndromes" S_1..S_2t by building a locator polynomial
sigma(z) whose roots mark the errors. Berlekamp-Massey (BM) does it
incrementally. Each round computes a discrepancy

    Delta = sum_k sigma_k * S_{r-k}                         (paper's Eq. 3)

and, if Delta != 0, corrects sigma using a saved template polynomial tau
(with its own old discrepancy delta), scaled by alpha = Delta / delta:

    retain:   (sigma, tau, delta) <- (sigma - alpha*tau,  tau,    delta)   (4a)
    replace:  (sigma, tau, delta) <- (sigma - alpha*tau,  sigma,  Delta)   (4b)

The "replace" case is the reversibility problem: it OVERWRITES the old tau
and delta. A naive reversible circuit has to stash the displaced polynomial
every time - a history of whole polynomials.

The paper's discovered circuit exploits a freedom of the task: the locator
only matters up to a nonzero scale factor (rescaling doesn't move roots). So
replace can be rewritten as

    replace:  (sigma, tau, delta) <- (tau - alpha^-1 * sigma,  sigma,  Delta)   (5b)

and retain stays (5a). Note tau - alpha^-1*sigma = -alpha^-1 * (sigma - alpha*tau):
the SAME correction as 4b, just rescaled. Now both branches are the shape
"first - ratio * second" on swapped inputs, so conditional routing (script
01) applies: swap (sigma, tau) and (Delta, delta) when replacing, run ONE
shared update, and the old sigma is already sitting in tau's slot as the
new template - nothing displaced, nothing to store.

The one thing still lost by replace is the old delta, but it can be DECODED
from the updated polynomials' constant coefficients. tau enters each round
shifted by z, so its constant coefficient is 0; hence after a replace

    delta_old = sigma'_0 * tau'_0^-1 * delta_new          (char 2: no signs)

so replace rounds add nothing to the history. Only retain rounds need to
remember something (their Delta), and those are rare.

Everything here is an ordinary classical simulation over GF(16) - the
decoder is a reversible-circuit design idea, and (as in the paper's own
evaluator) its correctness lives in tracking basis states, not amplitudes.
No CUDA-Q kernels in this one; script 01 covers the routing primitive in a
real circuit. Verified: both BM variants find the true error positions for
every error pattern of weight <= 2 and thousands of random weight-3 ones,
the routed locator equals the standard one up to a nonzero scalar, and the
delta-decoding identity holds in every replace round.
"""

import itertools
import random
from dataclasses import dataclass

FIELD_BITS = 4
FIELD_SIZE = 2**FIELD_BITS          # GF(16)
PRIMITIVE_POLYNOMIAL = 0b10011      # x^4 + x + 1
CODE_LENGTH = FIELD_SIZE - 1        # 15
MAX_ERRORS = 3
NUM_SYNDROMES = 2 * MAX_ERRORS      # 6 syndromes locate up to 3 errors


def _build_tables() -> tuple[list[int], list[int]]:
    exp_table = [0] * (2 * CODE_LENGTH)
    log_table = [0] * FIELD_SIZE
    value = 1
    for power in range(CODE_LENGTH):
        exp_table[power] = value
        log_table[value] = power
        value <<= 1
        if value & FIELD_SIZE:
            value ^= PRIMITIVE_POLYNOMIAL
    for power in range(CODE_LENGTH, 2 * CODE_LENGTH):
        exp_table[power] = exp_table[power - CODE_LENGTH]
    return exp_table, log_table


EXP_TABLE, LOG_TABLE = _build_tables()


def field_multiply(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]


def field_inverse(a: int) -> int:
    assert a != 0
    return EXP_TABLE[CODE_LENGTH - LOG_TABLE[a]]


def field_divide(a: int, b: int) -> int:
    return field_multiply(a, field_inverse(b))


# Polynomials are lists of GF(16) coefficients, lowest degree first. Addition
# and subtraction are both XOR in characteristic 2.

def poly_add(a: list[int], b: list[int]) -> list[int]:
    length = max(len(a), len(b))
    padded_a = a + [0] * (length - len(a))
    padded_b = b + [0] * (length - len(b))
    return [x ^ y for x, y in zip(padded_a, padded_b)]


def poly_scale(poly: list[int], factor: int) -> list[int]:
    return [field_multiply(coefficient, factor) for coefficient in poly]


def poly_shift(poly: list[int]) -> list[int]:
    """Multiply by z."""
    return [0] + poly


def poly_trim(poly: list[int]) -> list[int]:
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    return poly


def poly_evaluate(poly: list[int], point: int) -> int:
    result = 0
    for coefficient in reversed(poly):
        result = field_multiply(result, point) ^ coefficient
    return result


def make_syndromes(error_positions: list[int], error_values: list[int]) -> list[int]:
    """S_r = sum_i Y_i * (alpha^p_i)^r for r = 1..NUM_SYNDROMES.

    Returned with a dummy entry at index 0 so syndromes[r] is S_r."""
    syndromes = [0]
    for r in range(1, NUM_SYNDROMES + 1):
        total = 0
        for position, value in zip(error_positions, error_values):
            total ^= field_multiply(value, EXP_TABLE[(position * r) % CODE_LENGTH])
        syndromes.append(total)
    return syndromes


def discrepancy(locator: list[int], syndromes: list[int], round_number: int) -> int:
    total = 0
    for k, coefficient in enumerate(locator):
        if round_number - k >= 1:
            total ^= field_multiply(coefficient, syndromes[round_number - k])
    return total


@dataclass
class DecodeResult:
    locator: list[int]
    num_errors: int
    num_retain_rounds: int
    num_replace_rounds: int
    delta_decode_checks: int


def berlekamp_massey_standard(syndromes: list[int]) -> DecodeResult:
    """Textbook BM: Equations (4a) / (4b)."""
    locator, template, template_discrepancy, num_errors = [1], [1], 1, 0
    retains = replaces = 0
    for round_number in range(1, NUM_SYNDROMES + 1):
        current_discrepancy = discrepancy(locator, syndromes, round_number)
        template = poly_shift(template)
        if current_discrepancy == 0:
            continue
        alpha = field_divide(current_discrepancy, template_discrepancy)
        corrected = poly_add(locator, poly_scale(template, alpha))
        if 2 * num_errors <= round_number - 1:   # replace (4b)
            template = locator
            template_discrepancy = current_discrepancy
            num_errors = round_number - num_errors
            replaces += 1
        else:                                    # retain (4a)
            retains += 1
        locator = corrected
    return DecodeResult(poly_trim(locator), num_errors, retains, replaces, 0)


def berlekamp_massey_routed(syndromes: list[int]) -> DecodeResult:
    """Routed BM: Equations (5a) / (5b) as ONE shared update.

    Both branches compute  first - ratio * second  on routed operands:
      retain : (first, second, numerator, denominator) = (sigma, tau, Delta, delta)
      replace: (first, second, numerator, denominator) = (tau, sigma, delta, Delta)
    i.e. replace just swaps the roles, and the old sigma lands in the
    template slot for free.
    """
    locator, template, template_discrepancy, num_errors = [1], [1], 1, 0
    retains = replaces = delta_checks = 0
    for round_number in range(1, NUM_SYNDROMES + 1):
        current_discrepancy = discrepancy(locator, syndromes, round_number)
        template = poly_shift(template)
        if current_discrepancy == 0:
            continue
        is_replace = 2 * num_errors <= round_number - 1

        # --- conditional routing: swap roles when replacing ---
        if is_replace:
            first, second = template, locator
            numerator, denominator = template_discrepancy, current_discrepancy
        else:
            first, second = locator, template
            numerator, denominator = current_discrepancy, template_discrepancy

        # --- one shared arithmetic update ---
        ratio = field_divide(numerator, denominator)
        updated = poly_add(first, poly_scale(second, ratio))

        # --- the slot that held `second` is the new template; the
        #     denominator slot is the new template discrepancy ---
        new_locator, new_template, new_template_discrepancy = updated, second, denominator

        if is_replace:
            # Decode the displaced delta from constant coefficients instead of storing it.
            decoded_old_delta = field_multiply(
                field_multiply(new_locator[0], field_inverse(new_template[0])),
                new_template_discrepancy,
            )
            assert decoded_old_delta == template_discrepancy, "delta decode failed"
            delta_checks += 1
            num_errors = round_number - num_errors
            replaces += 1
        else:
            retains += 1
        locator, template, template_discrepancy = (
            new_locator, new_template, new_template_discrepancy)
    return DecodeResult(poly_trim(locator), num_errors, retains, replaces, delta_checks)


def locate_errors(locator: list[int]) -> list[int]:
    """Error positions p where the locator vanishes at alpha^-p."""
    return [position for position in range(CODE_LENGTH)
            if poly_evaluate(locator, EXP_TABLE[(-position) % CODE_LENGTH]) == 0]


def proportional(poly_a: list[int], poly_b: list[int]) -> bool:
    """True if poly_a = c * poly_b for some nonzero scalar c."""
    if len(poly_a) != len(poly_b) or poly_b[0] == 0:
        return False
    scale = field_divide(poly_a[0], poly_b[0])
    return all(field_multiply(scale, b) == a for a, b in zip(poly_a, poly_b))


def check_pattern(error_positions: list[int], error_values: list[int]) -> tuple[DecodeResult, DecodeResult]:
    syndromes = make_syndromes(error_positions, error_values)
    standard = berlekamp_massey_standard(syndromes)
    routed = berlekamp_massey_routed(syndromes)
    expected = sorted(error_positions)
    assert sorted(locate_errors(standard.locator)) == expected, "standard BM wrong"
    assert sorted(locate_errors(routed.locator)) == expected, "routed BM wrong"
    assert standard.num_errors == routed.num_errors == len(error_positions)
    assert proportional(routed.locator, standard.locator), "locators should differ only by a scalar"
    return standard, routed


if __name__ == "__main__":
    random.seed(0)
    nonzero_values = range(1, FIELD_SIZE)

    print(f"Reed-Solomon over GF({FIELD_SIZE}): length {CODE_LENGTH}, "
          f"{NUM_SYNDROMES} syndromes, up to {MAX_ERRORS} errors\n")

    patterns_checked = 0
    for num_errors in (0, 1, 2):
        for positions in itertools.combinations(range(CODE_LENGTH), num_errors):
            for values in itertools.product(nonzero_values, repeat=num_errors):
                check_pattern(list(positions), list(values))
                patterns_checked += 1
    print(f"exhaustive, weight <= 2: {patterns_checked} patterns, both BM variants "
          f"find the true positions")

    retain_counts, replace_counts = [], []
    decode_checks = 0
    random_patterns = 5000
    for _ in range(random_patterns):
        positions = random.sample(range(CODE_LENGTH), MAX_ERRORS)
        values = [random.choice(list(nonzero_values)) for _ in range(MAX_ERRORS)]
        _, routed = check_pattern(positions, values)
        retain_counts.append(routed.num_retain_rounds)
        replace_counts.append(routed.num_replace_rounds)
        decode_checks += routed.delta_decode_checks
    print(f"random weight-3: {random_patterns} patterns, all correct; the routed locator "
          f"was a nonzero scalar multiple of the standard one every time")
    print(f"delta-decoding identity checked in {decode_checks} replace rounds, "
          f"zero failures\n")

    bits = FIELD_BITS
    poly_bits = (MAX_ERRORS + 1) * bits
    worst_replaces = max(replace_counts)
    worst_retains = max(retain_counts)
    naive_history_qubits = worst_replaces * (poly_bits + bits)   # displaced tau and delta
    routed_history_qubits = worst_retains * bits                 # one Delta per retain
    print("History qubits needed (field elements are 4 bits):")
    print(f"  naive BM  : every replace stashes a polynomial + delta -> up to "
          f"{worst_replaces} replaces x {poly_bits + bits} bits = {naive_history_qubits} qubits")
    print(f"  routed BM : replace stores nothing; each retain stores one Delta -> up to "
          f"{worst_retains} retains x {bits} bits = {routed_history_qubits} qubits")
    average_retains = sum(retain_counts) / len(retain_counts)
    print(f"  average retain rounds per decode: {average_retains:.3f}")
    print("\nHonest scope note: these toy inputs are all weight-3 patterns, where "
          "almost every round has a nonzero discrepancy, so retain rounds are "
          "common here. The paper's DQI instance (70 syndromes, up to 35 errors, "
          "its own input distribution) needs a second stack entry with "
          "probability only ~1e-4, which is why it provisions a single entry and "
          "accepts that rare failure. That failure-budget trade is the paper's "
          "third principle; see 05_divsteps_modular_inversion.py for it measured.")
