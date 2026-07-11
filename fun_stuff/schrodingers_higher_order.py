"""
Schrodinger's Higher (2nd and 3rd order)
==========================================

Generalizes schrodingers_squared.py's "uncertainty about uncertainty"
idea into a reusable recipe: a *chain* of qubits, where each qubit's
own Hadamard only fires if the previous qubit in the chain collapsed
to 1. That single pattern, repeated, is what "higher order Schrodinger"
means here:

    h(chain[0])
    h.ctrl(chain[0], chain[1])
    h.ctrl(chain[1], chain[2])
    ...

Each qubit in the chain (except the last) is named after the *action*
it takes, not a static property - "does this level flip a coin to
decide whether the next level gets to flip its own coin," rather than
"is this level quantum" (which is ambiguous: every qubit in the chain
is itself always prepared with a real Hadamard, regardless of its own
outcome - what varies is whether *the next* qubit gets one). The final
qubit, does_cat_die, is named directly after what its raw measured
value means: True if the cat dies, False if it survives.

- 2nd order (2 qubits: does_box_flip_a_coin_for_cat, does_cat_die):
  does_box_flip_a_coin_for_cat decides whether does_cat_die's own
  Hadamard fires at all. If it collapses to True, the box genuinely
  flips a coin and the cat's fate is real 50/50 uncertainty. If it
  collapses to False, the box skips its coin flip entirely and the cat
  simply, deterministically survives.
- 3rd order (3 qubits: does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin,
  does_box_flip_a_coin_for_cat, does_cat_die): one level higher -
  does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin decides whether
  does_box_flip_a_coin_for_cat itself gets to be a real coin flip, or
  is forced to a fixed "no" (in which case the cat is guaranteed to
  survive, since the box below it never got to flip a coin either).

Both orders get their conditional-probability structure fully verified
below - the entire point of "higher order" is that a flat, single
aggregate probability can't distinguish these from an ordinary biased
coin. Only by conditioning on each level of the chain do the discrete,
telescoping tiers (exact certainty, then a real coin flip, then
another real coin flip nested inside that) become visible.
"""

import cudaq


@cudaq.kernel
def schrodinger_chain(order: int) -> list[bool]:
    """order=2 reproduces schrodingers_squared.py exactly
    (does_box_flip_a_coin_for_cat, does_cat_die). order=3 adds one more
    qubit above that (does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin,
    does_box_flip_a_coin_for_cat, does_cat_die). In general, chain[i]'s
    Hadamard only fires if chain[i-1] collapsed to True - each level's
    own coin flip is itself gated by whether the level above it
    decided to flip one.
    """
    # Qubit flow: chain[i] is written exactly once, either by an
    # unconditional h (chain[0] only) or by a CONTROLLED h reading
    # chain[i-1] as its control - so information only ever flows
    # "downward," from lower index to the next one up, never back.
    #
    # order=2, State tracking (chain = [box, cat]):
    #   |0>_box |0>_cat
    #   h(chain[0]):          (|0>+|1>)_box |0>_cat
    #   h.ctrl(chain[0],chain[1]):
    #     box=|0> branch: cat untouched -> |0>_box |0>_cat
    #     box=|1> branch: cat gets its own real H -> |1>_box (|0>+|1>)_cat
    #     Combined (unnormalized): |0,0> + |1,0> + |1,1>
    #   Reading off P(cat=1) = P(box=1)*P(cat=1|box=1) = 0.5*0.5 = 0.25 -
    #   matches p_does_cat_die's predicted 0.250 below.
    chain = cudaq.qvector(order)
    h(chain[0])
    for i in range(1, order):
        # chain[i-1] is CONTROL (read, not modified again after this
        # point), chain[i] is TARGET (its own H only fires along the
        # control's |1> branch - the |0> branch leaves chain[i] at a
        # deterministic |0>, which is exactly the "box never flipped a
        # coin, cat can't die" case).
        h.ctrl(chain[i - 1], chain[i])
    return mz(chain)


def run_second_order(shots: int) -> None:
    print("=== 2nd order: does_box_flip_a_coin_for_cat controls does_cat_die ===")
    results = cudaq.run(schrodinger_chain, 2, shots_count=shots)
    # chain = [does_box_flip_a_coin_for_cat, does_cat_die]
    p_does_box_flip = sum(1 for b in results if b[0]) / shots
    p_does_cat_die = sum(1 for b in results if b[1]) / shots

    shots_where_box_did_not_flip = [b for b in results if not b[0]]
    shots_where_box_flipped = [b for b in results if b[0]]
    p_dies_given_box_did_not_flip = sum(
        1 for b in shots_where_box_did_not_flip if b[1]
    ) / len(shots_where_box_did_not_flip)
    p_dies_given_box_flipped = sum(1 for b in shots_where_box_flipped if b[1]) / len(
        shots_where_box_flipped
    )

    print(
        f"  P(does_box_flip_a_coin_for_cat)     = {p_does_box_flip:.3f}  "
        f"(predicted 0.500)"
    )
    print(
        f"  P(does_cat_die), aggregate           = {p_does_cat_die:.3f}  "
        f"(predicted 0.250)"
    )
    print(
        f"  P(dies | box did not flip)           = "
        f"{p_dies_given_box_did_not_flip:.3f}  (predicted 0.000 - exact certainty "
        f"of survival)"
    )
    print(
        f"  P(dies | box flipped)                = "
        f"{p_dies_given_box_flipped:.3f}  (predicted 0.500 - a genuine coin flip)"
    )
    print()


def run_third_order(shots: int) -> None:
    print(
        "=== 3rd order: does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin controls "
        "does_box_flip_a_coin_for_cat controls does_cat_die ==="
    )
    results = cudaq.run(schrodinger_chain, 3, shots_count=shots)
    # chain = [does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin, does_box_flip_a_coin_for_cat, does_cat_die]
    p_does_reality_flip = sum(1 for b in results if b[0]) / shots
    p_does_box_flip = sum(1 for b in results if b[1]) / shots
    p_does_cat_die = sum(1 for b in results if b[2]) / shots

    shots_where_reality_did_not_flip = [b for b in results if not b[0]]
    shots_where_reality_flipped = [b for b in results if b[0]]
    p_dies_given_reality_did_not_flip = sum(
        1 for b in shots_where_reality_did_not_flip if b[2]
    ) / len(shots_where_reality_did_not_flip)
    p_dies_given_reality_flipped = sum(
        1 for b in shots_where_reality_flipped if b[2]
    ) / len(shots_where_reality_flipped)

    reality_flipped_and_box_did_not = [
        b for b in shots_where_reality_flipped if not b[1]
    ]
    reality_flipped_and_box_did = [b for b in shots_where_reality_flipped if b[1]]
    p_dies_given_reality_flipped_and_box_did_not = (
        sum(1 for b in reality_flipped_and_box_did_not if b[2])
        / len(reality_flipped_and_box_did_not)
        if reality_flipped_and_box_did_not
        else float("nan")
    )
    p_dies_given_reality_flipped_and_box_did = (
        sum(1 for b in reality_flipped_and_box_did if b[2])
        / len(reality_flipped_and_box_did)
        if reality_flipped_and_box_did
        else float("nan")
    )

    print(
        f"  P(does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin) = {p_does_reality_flip:.3f}  "
        f"(predicted 0.500)"
    )
    print(
        f"  P(does_box_flip_a_coin_for_cat), aggregate = "
        f"{p_does_box_flip:.3f}  (predicted 0.250: only gets a real coin flip"
    )
    print(f"                                             at all when reality flipped)")
    print(
        f"  P(does_cat_die), aggregate           = {p_does_cat_die:.3f}  "
        f"(predicted 0.125)"
    )
    print()
    print(
        "  Tier 1 - condition on does_reality_flip_a_coin_to_decide_if_box_can_flip_a_coin:"
    )
    print(
        f"    P(dies | reality did not flip) = "
        f"{p_dies_given_reality_did_not_flip:.3f}  (predicted 0.000 - the box never"
    )
    print(f"                                       got to flip its own coin either)")
    print(
        f"    P(dies | reality flipped)      = "
        f"{p_dies_given_reality_flipped:.3f}  (predicted 0.250 - this is the *entire"
    )
    print(
        f"                                       2nd-order result*, nested one level down)"
    )
    print(
        "  Tier 2 - within reality flipped, condition on does_box_flip_a_coin_for_cat too:"
    )
    print(
        f"    P(dies | reality flipped, box did not flip) = "
        f"{p_dies_given_reality_flipped_and_box_did_not:.3f}  (predicted 0.000)"
    )
    print(
        f"    P(dies | reality flipped, box flipped)      = "
        f"{p_dies_given_reality_flipped_and_box_did:.3f}  (predicted 0.500)"
    )
    print()
    print("  Notice tier 2's two numbers are exactly the 2nd-order script's")
    print("  own two numbers, reappearing unchanged inside the reality-flipped")
    print("  branch - each order doesn't replace the previous one, it wraps")
    print("  around it.")


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 100

    run_second_order(shots)
    run_third_order(shots)
