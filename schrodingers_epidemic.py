"""
Schrodinger's Epidemic
=======================

A dynamic graph experiment: this circuit's connectivity is built
directly from an arbitrary contact list (a real graph, edges and all -
including a genuine cycle), not a tree or a star like the earlier
Schrodinger scripts.

The premise: a small town's residents each carry a genuinely
undetermined "infected" qubit - not "we don't know yet," but truly
undecided until Santa's naughty-list-style algorithm (a stand-in for
"reality") resolves it at measurement time. Before any contact between
people, everyone has a small independent baseline chance of infection
(BASELINE_INFECTION_PROBABILITY) from unrelated community exposure.
Then a series of *chronologically ordered* contacts each entangle one
person's infection status with whoever they met, via a controlled
rotation (being exposed to an infected contact nudges your own
likelihood of infection upward, without ever making it certain).

The interesting experiment this script is built around: the contact
list below has a cycle - Alice meets Bob, Bob meets Carol, and then
Carol meets Alice again - closing the loop. On a real quantum circuit
(or any circuit, quantum or classical), gates apply in a strict
sequential order, so a "cycle" in the underlying social graph can't be
resolved all at once - it has to be unrolled into a specific
chronological sequence. This script runs the *exact same set of
contacts*, just reordered (the cycle-closing contact moved from last
to first), and shows that the resulting infection statistics
genuinely differ - order along a cycle matters, and this is the
concrete, measured proof rather than just an abstract claim.
"""

import math
from collections import Counter

import cudaq

PEOPLE = ["Alice", "Bob", "Carol", "Dave"]

# Chronological order: Alice -> Bob -> Carol -> Alice (closes the
# cycle!) -> Dave (a branch off the cycle, met only once, only at the
# very end).
CONTACTS_CHRONOLOGICAL = [
    ("Alice", "Bob"),
    ("Bob", "Carol"),
    ("Carol", "Alice"),
    ("Carol", "Dave"),
]

# The exact same 4 contacts, same edges - just with the cycle-closing
# contact moved from last to first.
CONTACTS_REORDERED = [
    ("Carol", "Alice"),
    ("Alice", "Bob"),
    ("Bob", "Carol"),
    ("Carol", "Dave"),
]

# Small independent chance of infection from the wider community,
# before any of the specific listed contacts happen.
BASELINE_INFECTION_PROBABILITY = 0.3
BASELINE_ANGLE = 2 * math.asin(math.sqrt(BASELINE_INFECTION_PROBABILITY))

# How much a single contact with an infected person nudges your own
# infection likelihood upward - a fixed rotation, not a certainty.
EXPOSURE_ANGLE = math.pi / 2


def contacts_to_indices(contacts):
    from_idx = [PEOPLE.index(a) for a, b in contacts]
    to_idx = [PEOPLE.index(b) for a, b in contacts]
    return from_idx, to_idx


@cudaq.kernel
def spread(num_people: int, from_idx: list[int], to_idx: list[int],
           num_contacts: int) -> list[bool]:
    residents = cudaq.qvector(num_people)

    # Everyone starts with a small, independent baseline chance of
    # infection - unrelated to any of the specific contacts below.
    for i in range(num_people):
        ry(BASELINE_ANGLE, residents[i])

    # Contacts applied in the exact order given - this is the whole
    # point of the experiment: the same edges, in a different order,
    # produce different final statistics, because a cyclic graph can't
    # be resolved all at once by a circuit that must apply gates
    # sequentially.
    for i in range(num_contacts):
        a = from_idx[i]
        b = to_idx[i]
        ry.ctrl(EXPOSURE_ANGLE, residents[a], residents[b])

    return mz(residents)


def run_scenario(label: str, contacts: list, shots: int) -> dict:
    from_idx, to_idx = contacts_to_indices(contacts)
    results = cudaq.run(spread, len(PEOPLE), from_idx, to_idx, len(contacts),
                         shots_count=shots)

    print(f"=== {label} ===")
    print("Contact order:", " -> ".join(f"{a}-{b}" for a, b in contacts))
    infection_rates = {}
    for i, name in enumerate(PEOPLE):
        rate = sum(1 for bits in results if bits[i]) / shots
        infection_rates[name] = rate
        print(f"  {name:6s} infected in {rate:.1%} of towns")

    alice_idx, carol_idx = PEOPLE.index("Alice"), PEOPLE.index("Carol")
    both = sum(1 for bits in results if bits[alice_idx] and bits[carol_idx]) / shots
    independent_guess = infection_rates["Alice"] * infection_rates["Carol"]
    print(f"  P(Alice AND Carol both infected)        = {both:.3f}")
    print(f"  P(Alice)*P(Carol) if independent         = {independent_guess:.3f}")
    print()
    return infection_rates


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 20000

    chronological_rates = run_scenario("Chronological order (cycle closes last)",
                                        CONTACTS_CHRONOLOGICAL, shots)
    reordered_rates = run_scenario("Reordered (cycle closes first)",
                                    CONTACTS_REORDERED, shots)

    print("=== Same edges, different order: do the results actually differ? ===")
    for name in PEOPLE:
        diff = reordered_rates[name] - chronological_rates[name]
        print(f"  {name:6s} chronological={chronological_rates[name]:.1%}  "
              f"reordered={reordered_rates[name]:.1%}  "
              f"difference={diff:+.1%}")
    print()
    print("Same 4 contacts in both runs, same total 'amount' of exposure -")
    print("just a different chronological order. If any of the differences")
    print("above are well outside sampling noise (roughly +-1% at 20,000")
    print("shots), that's the cycle-ordering effect: who 'already knew'")
    print("about an infection at the moment they met someone else depends")
    print("on what's happened so far in that specific sequence, and a")
    print("cycle forces a specific sequence to be chosen.")
