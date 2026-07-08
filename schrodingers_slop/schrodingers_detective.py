"""
Schrodinger's Detective
========================

A "quantum eraser" story: entangling two things doesn't permanently
lock in a correlation you can always recover later - what basis you
choose to measure in matters just as much as the entangling gate
itself, and choosing the "wrong" one can genuinely erase a
correlation that was really there a moment before.

The premise: a suspect's guilt is genuinely undetermined (a fair
coin, until Santa's naughty-list-style algorithm - a stand-in for
reality - resolves it). A piece of evidence (a fingerprint match, say)
gets entangled with that guilt via a CNOT, so if you look at both in
the obvious way, they match up perfectly - exactly the Bell-state
mechanic from hello_cudaq.py.

The twist: instead of always looking at the evidence "the obvious
way" (measuring it directly), this script also tries looking at it
"sideways" - applying an H gate to the evidence qubit before measuring
it, which measures it in a *different* (the X, not Z) basis. Nothing
about the guilt qubit changes, and no gate is undone - but the
correlation between guilt and evidence completely vanishes once you
measure this way. This is a real, well-known quantum phenomenon (the
basis for actual quantum eraser experiments), not a script gimmick:
measuring one half of an entangled pair in the basis conjugate to how
it was entangled destroys the very correlation that a same-basis
measurement would have revealed.
"""

import math

import cudaq


@cudaq.kernel
def establish_guilt(suspect: cudaq.qubit):
    h(suspect)


@cudaq.kernel
def plant_evidence(suspect: cudaq.qubit, evidence: cudaq.qubit):
    x.ctrl(suspect, evidence)


@cudaq.kernel
def look_sideways(evidence: cudaq.qubit):
    h(evidence)


@cudaq.kernel
def crime_scene(look_the_obvious_way: bool) -> list[bool]:
    suspect = cudaq.qubit()
    evidence = cudaq.qubit()

    # The suspect's guilt: genuinely undetermined, 50/50, until measured.
    establish_guilt(suspect)

    # Entangle the evidence with guilt - if you look at both directly,
    # they'll always match (guilty <-> evidence found; innocent <-> no
    # evidence), exactly like a Bell pair's 00/11 correlation.
    plant_evidence(suspect, evidence)

    # The detective's choice: look at the evidence the obvious way (in
    # the same basis it was entangled in), or look at it "sideways"
    # (apply H first, measuring in the conjugate basis instead). Only
    # the first choice preserves the correlation with guilt.
    if not look_the_obvious_way:
        look_sideways(evidence)

    return [mz(suspect), mz(evidence)]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 20000

    for look_the_obvious_way in [True, False]:
        label = "Direct look (same basis as entangled)" if look_the_obvious_way \
            else "Sideways look (H before measuring - the eraser)"
        results = cudaq.run(crime_scene, look_the_obvious_way, shots_count=shots)

        p_guilty = sum(1 for b in results if b[0]) / shots
        p_evidence = sum(1 for b in results if b[1]) / shots
        p_both = sum(1 for b in results if b[0] and b[1]) / shots
        p_independent_guess = p_guilty * p_evidence

        print(f"=== {label} ===")
        print(f"  P(guilty)                       = {p_guilty:.3f}")
        print(f"  P(evidence found)                = {p_evidence:.3f}")
        print(f"  P(guilty AND evidence found)     = {p_both:.3f}")
        print(f"  P(guilty)*P(evidence) if independent = {p_independent_guess:.3f}")
        if look_the_obvious_way:
            print("  Expected: evidence always matches guilt exactly -")
            print("  P(guilty AND evidence) should equal P(guilty) (~0.5),")
            print("  clearly NOT matching the independent-guess column.")
        else:
            print("  Expected: evidence is now a coin flip, totally")
            print("  uncorrelated with guilt - P(guilty AND evidence) should")
            print("  closely match the independent-guess column instead.")
            print("  The entangling CNOT still ran; only the measurement")
            print("  basis changed. That's the eraser: no gate was undone,")
            print("  but the correlation is genuinely gone.")
        print()
