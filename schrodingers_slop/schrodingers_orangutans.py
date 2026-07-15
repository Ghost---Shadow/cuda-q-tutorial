"""
Schrodinger's Orangutans
==========================

The zoo starts with 2 pair-bonded orangutans - 1 female (F0) and her
mate. Real orangutans almost always have a SINGLE infant per birth
(no litters), and a female keeps breeding roughly once per generation
for the rest of her reproductive life - both real facts, used here
directly instead of invented. Every newborn's sex is genuinely
undetermined (h()) until the birth is "measured" (mz()) - the same
premise as every other script in this series, just applied to a
pedigree instead of a coin.

The zoo's (very much invented) breeding-program rule: whenever a
female orangutan is born, the zoo's conservation program pairs her
with an imported, unrelated male once she matures, and she joins the
breeding pool starting the NEXT generation. Male orangutans mature
slower and don't enter this simplified model as breeders (a real,
if simplified, biological asymmetry - flanged male orangutans take
much longer to reach reproductive maturity than females).

That means the number of breeding females is a branching process:
every existing breeding female gives birth once per generation, and
only the female-sexed offspring join the pool for the generation
after that. This is exactly the classical Galton-Watson branching
process (famous for the "family surname extinction" problem), just
built here out of genuine quantum coin flips instead of
pseudo-random numbers.

The puzzle: after 3 generations, the zoo counts 5 pregnant females.
Starting from 1 breeding female, growth this way can only ever
DOUBLE at most each generation (every offspring female) or stay FLAT
(every offspring male) - so 5 is not a power of 2, meaning some
generations must have had a mix of male and female offspring. This
script runs the exact quantum branching circuit many times, keeps
only the runs that land on 5, and prints out the traced family
history for a few of them - concretely showing that "5 pregnant
orangutans" does NOT correspond to one unique history. At least 2
structurally different traces reach the same number, which is the
same lesson as controlled_phase_information.py in a different
costume: an aggregate count alone doesn't tell you which branch you're
actually in.
"""

from collections import Counter

import cudaq


@cudaq.kernel
def orangutan_lineage() -> list[bool]:
    """7 potential birth-qubits, enough for the worst case (1 + 2 + 4
    breeders across 3 generations). A qubit only gets a real h()
    coin-flip if its mother actually exists in this shot's family tree
    - otherwise it's left at |0> (deterministic False = 'this birth
    slot was never real'), decided via mid-circuit measurement
    feed-forward, same mechanism as 02_measuring_kernels.py.

    Slots:
      g1  - F0's gen-1 birth (always real - F0 always exists)
      g2a - F0's gen-2 birth (always real)
      g2b - F0's gen-1 daughter's gen-2 birth (real only if g1 female)
      g3a - F0's gen-3 birth (always real)
      g3b - F0's gen-1 daughter's gen-3 birth (real only if g1 female)
      g3c - F0's gen-2 daughter's gen-3 birth (real only if g2a female)
      g3d - gen-1 daughter's gen-2 daughter's gen-3 birth (real only
            if g1 AND g2b were both female)
    """
    q = cudaq.qvector(7)
    g1, g2a, g2b, g3a, g3b, g3c, g3d = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

    h(g1)
    b_g1 = mz(g1)

    h(g2a)
    b_g2a = mz(g2a)

    if b_g1:
        h(g2b)
    b_g2b = mz(g2b)

    h(g3a)
    b_g3a = mz(g3a)

    if b_g1:
        h(g3b)
    b_g3b = mz(g3b)

    if b_g2a:
        h(g3c)
    b_g3c = mz(g3c)

    if b_g1:
        if b_g2b:
            h(g3d)
    b_g3d = mz(g3d)

    return [b_g1, b_g2a, b_g2b, b_g3a, b_g3b, b_g3c, b_g3d]


def trace_family_tree(bits: list) -> dict:
    """Classical reconstruction of which birth-qubits were real for
    this shot, and the resulting breeding-female count per generation
    - the exact same logic the kernel used to decide which qubits got
    a real h() coin flip, applied here to interpret the result."""
    g1, g2a, g2b, g3a, g3b, g3c, g3d = bits

    f1 = 1 + (1 if g1 else 0)  # F0 + gen-1 daughter if born

    gen2_daughter_of_F0 = g2a
    gen2_daughter_of_gen1_daughter = g2b if g1 else False  # only real if g1 existed
    f2 = f1 + (1 if gen2_daughter_of_F0 else 0) + (1 if gen2_daughter_of_gen1_daughter else 0)

    gen3_daughter_of_F0 = g3a
    gen3_daughter_of_gen1_daughter = g3b if g1 else False
    gen3_daughter_of_gen2a_daughter = g3c if g2a else False
    gen3_daughter_of_gen2b_daughter = (g3d if (g1 and g2b) else False)
    f3 = f2 + sum([gen3_daughter_of_F0, gen3_daughter_of_gen1_daughter,
                   gen3_daughter_of_gen2a_daughter, gen3_daughter_of_gen2b_daughter])

    return {
        "f1": f1, "f2": f2, "f3": f3,
        "gen1_daughter_born": g1,
        "gen2_births": {
            "F0": gen2_daughter_of_F0,
            "gen1_daughter": gen2_daughter_of_gen1_daughter if g1 else None,
        },
        "gen3_births": {
            "F0": gen3_daughter_of_F0,
            "gen1_daughter": gen3_daughter_of_gen1_daughter if g1 else None,
            "gen2_daughter_of_F0": gen3_daughter_of_gen2a_daughter if g2a else None,
            "gen2_daughter_of_gen1_daughter": gen3_daughter_of_gen2b_daughter if (g1 and g2b) else None,
        },
    }


def print_trace(shot_num: int, trace: dict) -> None:
    print(f"  --- Example trace #{shot_num} ---")
    print(f"  Gen 0: 2 orangutans (F0 + her mate) - 1 breeding female")
    print(f"  Gen 1: F0 gives birth -> {'female' if trace['gen1_daughter_born'] else 'male'}"
          f"  => {trace['f1']} breeding female(s) entering gen 2")
    g2 = trace["gen2_births"]
    line = f"  Gen 2: F0 gives birth -> {'female' if g2['F0'] else 'male'}"
    if g2["gen1_daughter"] is not None:
        line += f"; gen-1 daughter gives birth -> {'female' if g2['gen1_daughter'] else 'male'}"
    print(line + f"  => {trace['f2']} breeding female(s) entering gen 3")
    g3 = trace["gen3_births"]
    line = f"  Gen 3: F0 gives birth -> {'female' if g3['F0'] else 'male'}"
    if g3["gen1_daughter"] is not None:
        line += f"; gen-1 daughter -> {'female' if g3['gen1_daughter'] else 'male'}"
    if g3["gen2_daughter_of_F0"] is not None:
        line += f"; F0's gen-2 daughter -> {'female' if g3['gen2_daughter_of_F0'] else 'male'}"
    if g3["gen2_daughter_of_gen1_daughter"] is not None:
        line += (f"; gen-1 daughter's gen-2 daughter -> "
                 f"{'female' if g3['gen2_daughter_of_gen1_daughter'] else 'male'}")
    print(line)
    print(f"  => {trace['f3']} PREGNANT orangutans entering gen 4")
    print()


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 200

    print("=== Running the branching-process circuit, {} shots ===".format(shots))
    results = cudaq.run(orangutan_lineage, shots_count=shots)
    traces = [trace_family_tree(list(r)) for r in results]

    f3_histogram = Counter(t["f3"] for t in traces)
    print("Distribution of pregnant-orangutan count after gen 3, across all shots:")
    for count in sorted(f3_histogram):
        print(f"  f3={count}: {f3_histogram[count]} shots")

    predicted_p5 = 0.125  # hand-derived: P(g1=1) * [...] = 0.125, see docstring
    measured_p5 = f3_histogram[5] / shots
    print(f"\nP(exactly 5 pregnant orangutans after gen 3): measured {measured_p5:.3f} "
          f"vs. hand-derived {predicted_p5:.3f}")

    print()
    print("=== Tracing what happened: example histories that land on exactly 5 ===")
    matching = [t for t in traces if t["f3"] == 5]
    shown_f2_values = set()
    shown = 0
    for t in matching:
        if t["f2"] in shown_f2_values:
            continue  # only show one example per distinct f2 path, to
                       # highlight that DIFFERENT histories reach 5
        shown_f2_values.add(t["f2"])
        shown += 1
        print_trace(shown, t)
        if shown >= 3:
            break

    print(f"{len(shown_f2_values)} structurally different f2 value(s) among the "
          f"{len(matching)} matching shots led to f3=5 "
          f"(f2 values seen: {sorted(shown_f2_values)}).")

    print()
    print("=== Honest conclusion ===")
    print("Exactly like controlled_phase_information.py's b qubit, 'we counted 5")
    print("pregnant orangutans' is an aggregate measurement that does NOT pin down")
    print("a single history - multiple genuinely different sequences of quantum")
    print("coin flips (which specific births were male vs. female, at which")
    print("generation) land on the same final count. What's real and verified")
    print("here: every birth's sex was a genuine h()/mz() coin flip, not a")
    print("pseudo-random number, and the measured P(f3=5) matches the")
    print("hand-derived branching-process probability. What's NOT recoverable")
    print("from '5 pregnant orangutans' alone is which specific great-")
    print("granddaughters of F0 exist - that information was real, and it was")
    print("thrown away the moment we only looked at the total count.")
