"""
Schrodinger's Christmas
========================

A fun, made-up story circuit exploring *dynamic* entangled graphs -
the qubit count scales with a Python list you can freely edit, rather
than being a fixed hand-drawn tree like jans_life_quantum_sim.py.

The premise: every kid's behavior over the whole year is genuinely
undetermined until Santa's naughty-list algorithm runs on Christmas
Eve - it's not that we don't know whether they were good, it's that
"good" isn't decided yet, in exactly the same sense a Bell-state qubit
isn't secretly 0 or 1 before it's measured (see hello_cudaq.py for why
that's not just a knowledge gap - Bell's theorem rules out any
"secretly already decided" picture).

Every kid's gift box starts as a fixed "coal by default" state, and
Santa's elves wire it to that kid's own verdict via a CNOT - the
circuit given as the inspiration for this script:

    box_actually_has = mz(cnot(john_is_good_boy, box_contains))

being good flips the box from coal towards socks. Each kid gets their
own independent pair of qubits (good_qubit, box_qubit) - so this is N
disconnected Bell-pair-style entanglements, one per kid, where N comes
from len(CHILDREN) at the top of the file.

A real bug I hit and fixed while building this (kept as a comment in
the code below, since it's a genuinely useful lesson): an earlier
version started every box from h() - an independent fair coin - before
entangling it. That's wrong: XOR-ing a control bit into an *already
independently random* target leaves the two looking statistically
uncorrelated in aggregate, even though each individual shot is still
causally linked - the same math behind why a one-time pad looks like
pure noise. Starting the box at a fixed, deterministic state instead
(exactly how the real Bell state's target qubit works in
hello_cudaq.py) is what makes the correlation actually show up in the
sampled statistics, verified in the output below.
"""

import math

import cudaq

# Dynamic: add, remove, or rename kids here and everything below - the
# circuit's qubit count and the final story adjust automatically.
CHILDREN = ["Timmy", "Sarah", "Max"]

# Most kids are mostly good over the course of a year - biased, not a
# fair coin.
GOOD_PROBABILITY = 0.75
GOOD_ANGLE = 2 * math.asin(math.sqrt(GOOD_PROBABILITY))

NUM_CHILDREN = len(CHILDREN)


# Named wrapper kernels, same trick as hello_cudaq.py's apply_hadamard /
# apply_cnot / measure_all - h, x, mz, and ry themselves can't be
# renamed (they're recognized by CUDA-Q's compiler purely by their
# literal spelling, not as real Python objects - see the chat for why
# `measure = mz` fails before even reaching a kernel), but wrapping
# each one in its own named kernel gets the readability without
# fighting the compiler.


@cudaq.kernel
def decide_goodness(chance_of_good: float, kid: cudaq.qubit):
    ry(chance_of_good, kid)


@cudaq.kernel
def start_as_coal(box: cudaq.qubit):
    x(box)


@cudaq.kernel
def flip_box_if_good(kid: cudaq.qubit, box: cudaq.qubit):
    x.ctrl(kid, box)


@cudaq.kernel
def measure_everyone(qubits: cudaq.qview) -> list[bool]:
    return mz(qubits)


@cudaq.kernel
def christmas_eve(num_children: int) -> list[bool]:
    """Every kid's fate is decided in one coherent superposition, and
    collapses in a single measurement at the very end - true to the
    "measure once" spirit established earlier in this project.
    """
    qubits = cudaq.qvector(2 * num_children)

    for i in range(num_children):
        # Readable local names for this kid's two qubits, instead of
        # writing qubits[...] index arithmetic in every gate call below.
        good_qubit = qubits[2 * i]
        box_qubit = qubits[2 * i + 1]

        # This kid's year: genuinely undetermined, but biased mostly
        # good - see GOOD_PROBABILITY.
        decide_goodness(GOOD_ANGLE, good_qubit)

        # Every box starts the same way: assume coal, no randomness of
        # its own, until good behavior earns a flip to socks below.
        start_as_coal(box_qubit)

        # The literal inspiration for this whole script: entangle this
        # kid's own verdict with their own box - being good flips the
        # box from its coal default towards socks.
        flip_box_if_good(good_qubit, box_qubit)

    # One measurement, collapsing every kid's Christmas at once.
    return measure_everyone(qubits)


def tell_the_story(bits: list) -> None:
    for i, name in enumerate(CHILDREN):
        was_good = bits[2 * i]
        got_coal = bits[2 * i + 1]
        gift = "a lump of coal" if got_coal else "a pair of socks"
        behavior = "was a good kid" if was_good else "was a little rascal"
        print(f"  {name} {behavior} this year, and found {gift} under the tree.")


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== One Christmas Eve, sampled once ===")
    one_shot = cudaq.run(christmas_eve, NUM_CHILDREN, shots_count=1)[0]
    tell_the_story(one_shot)

    print()
    print("=== Now let's run 100 Christmases to check the physics ===")
    shots = 100
    all_results = cudaq.run(christmas_eve, NUM_CHILDREN, shots_count=shots)

    for i, name in enumerate(CHILDREN):
        coal_count = sum(1 for bits in all_results if bits[2 * i + 1])
        good_count = sum(1 for bits in all_results if bits[2 * i])
        print(
            f"{name}: good {good_count/shots:.1%} of Christmases, "
            f"coal {coal_count/shots:.1%} of Christmases"
        )

    print()
    print("Correlation check (this is the actual point of the script):")
    name0 = CHILDREN[0]
    p_good = sum(1 for b in all_results if b[0]) / shots
    p_coal = sum(1 for b in all_results if b[1]) / shots
    p_good_and_coal = sum(1 for b in all_results if b[0] and b[1]) / shots
    print(f"  P({name0}=good)                  = {p_good:.3f}")
    print(f"  P({name0}=coal)                  = {p_coal:.3f}")
    print(f"  P({name0}=good AND coal)         = {p_good_and_coal:.3f}")
    print(f"  P(good)*P(coal) if independent  = {p_good*p_coal:.3f}")
    print("These should NOT match: this kid's box is a deterministic")
    print("function of their own goodness (coal exactly when NOT good)")
    print("- a real, detectable correlation, unlike the pre-randomized-")
    print("box version described in the module docstring's bug note.")
