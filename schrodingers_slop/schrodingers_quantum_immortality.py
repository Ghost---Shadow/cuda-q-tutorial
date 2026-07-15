"""
Schrodinger's Quantum Immortality
===================================

The premise: a brake pedal is asymptomatic-Schrodinger too - genuinely
undetermined ("will work" and "will fail" both real, superposed)
until the exact moment it's pressed. A car drives through N
intersections; at each one, the brake pedal gets "measured" (pressed).
This is the classic "quantum suicide / quantum immortality" thought
experiment (Hugh Everett / Max Tegmark), reskinned as a car instead of
a gun: if the world genuinely branches on every measurement (a
many-worlds reading, not something this script argues for or against),
then in every branch where the driver is still alive to ask "did the
brake ever fail?", the honest answer from INSIDE that branch is
always "no, never" - no matter how many intersections came before, and
no matter how unreliable the brake actually is.

That's not a claim the brake got more reliable. It's a claim about
which branches contain someone left to ask the question - the same
selection effect as "survivorship bias," just applied to the survivor's
own existence instead of someone else's outcome. This script makes that
distinction concrete and measured: it computes the REAL aggregate
survival probability (which decays exponentially with N, exactly like
classical reliability chains - verified against p**N below), and
separately shows that every single surviving shot, by construction, has
zero recorded brake failures - the tautology at the heart of the
"immortality" intuition, not evidence of anything special about the
brake.

Dummy numbers: a 95% reliable brake per intersection is absurdly
unsafe for a real car - picked here purely so the exponential decay is
visible at N as low as 10-20 instead of requiring N in the millions.
"""

import math

import cudaq

BRAKE_RELIABILITY = 0.95  # P(brake works), per intersection - a dummy,
                          # deliberately-bad number so the effect shows
                          # up at a small N.
BRAKE_ANGLE = 2 * math.asin(math.sqrt(BRAKE_RELIABILITY))


@cudaq.kernel
def drive_through_intersections(num_intersections: int, angle: float) -> list[bool]:
    """One qubit per intersection - each one is a genuinely undetermined
    brake test (ry puts it in superposition of worked/failed) until
    mz() presses it. True = brake worked at that intersection."""
    q = cudaq.qvector(num_intersections)
    for i in range(num_intersections):
        ry(angle, q[i])
    return mz(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 100

    N = 20
    print(f"=== A drive through {N} intersections, brake reliability = "
          f"{BRAKE_RELIABILITY:.0%} per intersection ===")
    predicted_survival = BRAKE_RELIABILITY ** N
    print(f"Predicted P(brake works at EVERY intersection) = "
          f"{BRAKE_RELIABILITY:.2f}^{N} = {predicted_survival:.4f}")

    results = cudaq.run(drive_through_intersections, N, BRAKE_ANGLE, shots_count=shots)
    survivors = [r for r in results if all(r)]
    measured_survival = len(survivors) / shots
    print(f"Measured across {shots} simulated drives: "
          f"{len(survivors)}/{shots} = {measured_survival:.4f} made it through clean")

    print()
    print("=== The tautology: what does the brake's record look like FROM ===")
    print("    INSIDE a surviving drive? ===")
    total_intersections_among_survivors = len(survivors) * N
    failures_among_survivors = sum(1 for r in survivors for worked in r if not worked)
    print(f"Brake failures recorded across all {len(survivors)} surviving drives, "
          f"{N} intersections each: {failures_among_survivors} "
          f"(out of {total_intersections_among_survivors} presses)")
    print("Zero, always, by construction - a 'surviving drive' is DEFINED as one")
    print("where every single brake press worked. This is not the brake being")
    print("reliable; it's the definition of which drives get counted here.")

    all_failures = sum(1 for r in results for worked in r if not worked)
    total_presses = shots * N
    print(f"\nFor comparison, across ALL {shots} drives (survivors and crashes")
    print(f"alike): {all_failures}/{total_presses} presses failed "
          f"({all_failures/total_presses:.3f}, vs. predicted "
          f"{1-BRAKE_RELIABILITY:.3f}) - the brake's TRUE failure rate, ")
    print("visible only once you stop conditioning on making it all the way through.")

    print()
    print("=== Does the illusion get stronger as N grows? (it shouldn't - and doesn't) ===")
    print(f"  {'N':>4}  {'predicted p^N':>14}  {'measured survival':>18}  {'survivors':>10}")
    # Capped at 28: this GPU's statevector simulator needs 2^N complex
    # amplitudes regardless of how independent the qubits are, and 30+
    # already exceeds available memory ("requested size is too big").
    for n in [1, 5, 10, 20, 28]:
        predicted = BRAKE_RELIABILITY ** n
        r = cudaq.run(drive_through_intersections, n, BRAKE_ANGLE, shots_count=shots)
        survived = sum(1 for row in r if all(row))
        print(f"  {n:>4}  {predicted:>14.5f}  {survived/shots:>18.5f}  {survived:>7}/{shots}")

    print()
    print("=== So: is the car's brake quantum immortal yet? ===")
    print("No - and it never becomes 'more immortal' as N grows, even though the")
    print("intuition (and the tautology above) never goes away either. The real,")
    print(f"third-person survival probability keeps shrinking exponentially ")
    print(f"({BRAKE_RELIABILITY:.2f}^N), matching plain classical reliability math")
    print("exactly - nothing about superposition or measurement changes that number")
    print("even by an epsilon. What's genuinely true is only the tautology: IF a")
    print("branch survives to N intersections, its brake record shows zero")
    print("failures, for any N, however large - that statement is 100% true and")
    print("100% uninformative, since it's true by definition of 'survived' rather")
    print("than a fact about the brake's actual reliability. A driver (or an")
    print("insurance actuary) should care about the shrinking p^N number, not the")
    print("tautological one - and no version of many-worlds gives you a way to")
    print("subjectively 'select into' the surviving branch on purpose from the")
    print("outside. This script demonstrates the selection effect precisely; it")
    print("is not an argument for or against many-worlds being physically true.")
