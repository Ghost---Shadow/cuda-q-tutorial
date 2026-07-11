"""
Jan's Life Paths, as a Quantum Decision Tree
==============================================

A fun side-project inspired by "The Alters" (11 bit studios, 2025), a
game where protagonist Jan Dolski uses an in-fiction quantum computer
to "branch" alternate versions of himself (the "Alters") from pivotal
decision points in his own past. The game itself hand-waves the actual
quantum computer and instead makes you scavenge the game world for
physical qubits to power it - which is a funny bit of irony, since the
in-fiction machine doesn't actually run a quantum circuit anywhere.
This script does the part the game skips: an actual CUDA-Q simulation
that encodes Jan's real, documented branching points as a genuine
quantum circuit, then samples it to find "the most likely Jan."

SOURCED FACTS (see chat for citations - gamerant.com, thegamer.com,
gamespot.com, the-alters.fandom.com, TV Tropes, and general web search
results on "The Alters" branching/Alters mechanics):

  - Jan either stays home (with his ailing mother) or leaves to
    continue his education.
  - If he stays home: he either accepts his father's mine job offer
    (-> Miner), or refuses it - and if he refuses it, he separately
    ends up deciding whether to work with the criminals running his
    workshop (-> Guard) or not (-> Technician).
  - If he leaves to continue his education: depending on later events,
    he stays academically focused (-> Scientist), prioritizes his
    relationship with Lena and moves away with her (-> Botanist), drops
    out after his mother's death and moves to an oil rig with his
    partner (-> Refiner), or drops out to stay and fight for workers'
    rights (-> Worker).
  - Separately, at age 15: instead of looking away while a girl was
    being robbed, Jan chases the thieves and runs into a street doctor,
    Dr. Edgar, who sets him on a medical path (-> Doctor). Also
    separately, rather than pushing to advance his mother's medical
    treatment, professionally uninvolved Jan lets it go; a year after
    she dies he wanders, then returns to train as a therapist
    (-> Shrink).
  - Tabula Rasa is branched from the *earliest possible point* in
    Jan's life - before any of the above decisions even happen - and
    has none of Jan's memories, thoughts, or personality. It is
    explicitly a separate, special, hard-to-reach branch in the game,
    not a "normal" outcome of an everyday decision.
  - Two named Alters I found in general character lists - Biologist and
    Chemist - have no sourced backstory I could find, so they're left
    out entirely rather than invented. There's also no "Jan Cook" in
    any source found; the Doctor Alter's gameplay bonus of cooking food
    faster is likely the source of that mix-up.

MY OWN CREATIVE ADDITIONS (not sourced game data - flagged clearly so
these aren't confused with canon):

  1. The exact probabilities. The game doesn't assign numeric
     likelihoods to Jan's choices (it's a narrative, not a dice roll) -
     I picked concrete numbers to make this a runnable simulation.
  2. One deliberate entangling correlation: within the "stay home"
     branch, I've modeled refusing the mine job as making Jan *more*
     likely to also refuse the criminal workshop bosses (Technician,
     75%) than to go along with them (Guard, 25%) - a "someone who
     defies one authority is more likely to defy another" hypothesis.
     This matters mechanically, not just narratively: independent coin
     flips for every decision would make this whole exercise just a
     classical random number generator wearing a quantum costume -
     correlating two of Jan's choices via ry() conditioned on another
     qubit's state is what makes this a genuinely quantum (as opposed
     to classical-equivalent) piece of code.
  3. Doctor and Shrink's age-15/mother's-treatment forks are two
     separate, unrelated life events in the sourced material - I've
     modeled them as one combined "formative early intervention"
     branch point (fires rarely; splits 50/50 into Doctor vs Shrink
     when it does) purely to keep the circuit tractable, since the
     sources don't describe how these two moments relate to the rest
     of the timeline or to each other.
  4. Tabula Rasa and the formative-intervention branch are both given
     deliberately *low* probability, reflecting that they read as
     rarer/more special branches in the game than an everyday
     stay-or-leave decision - not numbers from any real data.
"""

import math
from collections import Counter

import cudaq

# Probability of branching a Tabula Rasa (blank-slate) Jan instead of
# following any of his documented life decisions - see note 4 above.
TABULA_RASA_PROBABILITY = 0.10
TABULA_RASA_ANGLE = 2 * math.asin(math.sqrt(TABULA_RASA_PROBABILITY))

# Probability of the age-15 / mother's-treatment fork firing at all
# (leading to Doctor or Shrink), given it didn't branch as Tabula Rasa.
# See note 3 and 4 above.
FORMATIVE_INTERVENTION_PROBABILITY = 0.12
FORMATIVE_INTERVENTION_ANGLE = 2 * math.asin(math.sqrt(FORMATIVE_INTERVENTION_PROBABILITY))

# Probability of refusing the criminal workshop bosses, GIVEN Jan
# already refused his father's mine job - see note 2 above.
REFUSE_CRIMINALS_GIVEN_REFUSED_MINE = 0.75
CORRELATION_ANGLE = 2 * math.asin(math.sqrt(1 - REFUSE_CRIMINALS_GIVEN_REFUSED_MINE))
CORRECTION_ANGLE = CORRELATION_ANGLE - math.pi / 2


@cudaq.kernel
def jans_life_paths() -> list[bool]:
    """Every branch point stays in superposition simultaneously - no
    qubit is measured (collapsed) until the very last line. Right up
    until that final joint measurement, this circuit's state is a
    genuine coherent superposition of all nine named Jans at once
    (with the appropriate relative amplitudes) - "Schrodinger's Jan,"
    rather than a chain of classical coin flips made one at a time and
    dressed up in quantum syntax.
    """
    # One register, one qubit per branch point - so the final
    # measurement below is a single collapse of Jan's whole life at
    # once, not several separate little collapses that happen to be
    # written on the same line.
    #
    # Qubit flow (each qubit is written by exactly one biased ry/h
    # below, then read once - by the final mz - except second_choice
    # and third_choice, which get a second, entangling touch from the
    # ry.ctrl correction near the end):
    #   qubits[is_tabula_rasa] : ry only, never touched again
    #   qubits[is_formative]   : ry only, never touched again
    #   qubits[stayed_home]    : h, then re-read as a CONTROL below
    #   qubits[second_choice]  : h, then briefly flipped and re-read as
    #                            a CONTROL below (net effect on its own
    #                            marginal: none - see the X-sandwich)
    #   qubits[third_choice]   : h, then as TARGET of the correction
    #
    # All five stay in one coherent superposition the whole time -
    # every "branch point" below is a gate, never a measurement, so
    # nothing here collapses anything.
    qubits = cudaq.qvector(5)
    is_tabula_rasa, is_formative, stayed_home, second_choice, third_choice = 0, 1, 2, 3, 4

    # Branch point 0: does this Alter get branched from the earliest
    # possible point (Tabula Rasa), bypassing Jan's life story
    # entirely? Rare - see TABULA_RASA_PROBABILITY.
    ry(TABULA_RASA_ANGLE, qubits[is_tabula_rasa])

    # Branch point 0': does the separate age-15 / mother's-treatment
    # fork fire instead, leading to Doctor or Shrink? Also rare - see
    # FORMATIVE_INTERVENTION_PROBABILITY. second_choice is reused as
    # the Doctor/Shrink coin flip when this fires.
    ry(FORMATIVE_INTERVENTION_ANGLE, qubits[is_formative])
    h(qubits[second_choice])  # doubles as Doctor(0)/Shrink(1) coin flip

    # Branch point 1: stay home with his mother, or leave to continue
    # his education? stayed_home will be read again below as a CONTROL
    # qubit - it isn't measured here, so both branches stay alive.
    h(qubits[stayed_home])

    # Branch point 2 (stay-home line) / 2' (leave line, first bit):
    # accept his father's mine job offer, or (if he left instead) the
    # first bit of which later event redirects his life. (second_choice
    # was already given its H above - it's a fair coin either way.)

    # Branch point 3 (stay-home line) / 2' (leave line, second bit):
    # starts as a plain, unbiased coin - this is exactly right for the
    # "leave" line's second bit, and serves as the baseline for the
    # stay-home line too, before the correction below narrows it.
    h(qubits[third_choice])

    # The one deliberate entangling correction (see note 2): within
    # ONLY the "stayed home AND refused the mine job" branch, nudge
    # third_choice away from its 50/50 baseline towards refusing the
    # criminals too. Implemented as a rotation controlled on
    # stayed_home=1 AND second_choice=0 (temporarily flipping
    # second_choice so the control fires on its "0" value - the same
    # X-sandwich trick used to retarget Grover's oracle earlier).
    # Because this is a genuine multi-controlled *rotation* rather than
    # a measurement-triggered classical `if`, all qubits stay entangled
    # and coherent through this step - nothing collapses.
    #
    # Qubit flow through these 3 lines: stayed_home and second_choice
    # are read as CONTROLS (never modified in the sense of changing
    # their own marginal probability - the x() before and after
    # second_choice is a sandwich that flips which of its two branches
    # (0 or 1) triggers the control, then flips it back so its own
    # H-prepared 50/50 statistics are undisturbed); third_choice is the
    # only qubit whose own state actually changes here, and only in the
    # stayed_home=1,second_choice=0 branch - this is what makes
    # third_choice's probability depend on stayed_home/second_choice,
    # i.e. the one deliberate entangling correlation in this script.
    x(qubits[second_choice])
    ry.ctrl(CORRECTION_ANGLE, qubits[stayed_home], qubits[second_choice],
            qubits[third_choice])
    x(qubits[second_choice])

    # A single measurement of the whole register - one collapse,
    # covering all five branch points simultaneously.
    return mz(qubits)


def decode_alter(bits: list) -> str:
    """Classical post-processing: turns the 5 raw measured bits into
    the named Alter they correspond to - the same
    "measure-then-classically-decode" pattern used throughout this
    project (Shor's algorithm's continued-fraction step, teleportation's
    corrections)."""
    is_tabula_rasa, is_formative, stayed_home, second_choice, third_choice = bits

    if is_tabula_rasa:
        return "Tabula Rasa"

    if is_formative:
        return "Shrink" if second_choice else "Doctor"

    if stayed_home:
        if second_choice:  # accepted the mine job
            return "Miner"
        return "Technician" if not third_choice else "Guard"

    # left home / continued education - all 4 combinations now map to
    # a real named Alter, no leftover-case hack needed.
    code = (second_choice, third_choice)
    return {
        (False, False): "Scientist",
        (False, True): "Botanist",
        (True, False): "Refiner",
        (True, True): "Worker",
    }[code]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    shots = 20000
    raw_results = cudaq.run(jans_life_paths, shots_count=shots)
    alters = [decode_alter(bits) for bits in raw_results]
    histogram = Counter(alters)

    print("=== Which Jan came out of the quantum computer? ===")
    for name, count in histogram.most_common():
        bar = "#" * (count * 60 // shots)
        print(f"{name:12s} {count / shots:6.2%}  {bar}")

    most_likely, count = histogram.most_common(1)[0]
    print()
    print(f"The most likely Jan, across {shots} branchings, is: {most_likely} "
          f"({count / shots:.2%})")

    print()
    print("Predicted probabilities (hand-computed from the circuit's")
    print("rotation angles, for comparison against the simulated ones):")
    p_tabula = TABULA_RASA_PROBABILITY
    p_formative = (1 - p_tabula) * FORMATIVE_INTERVENTION_PROBABILITY
    p_doctor = p_formative * 0.5
    p_shrink = p_formative * 0.5
    p_normal = (1 - p_tabula) * (1 - FORMATIVE_INTERVENTION_PROBABILITY)
    p_stay = 0.5 * p_normal
    p_leave = 0.5 * p_normal
    p_miner = p_stay * 0.5
    p_refused_mine = p_stay * 0.5
    p_technician = p_refused_mine * REFUSE_CRIMINALS_GIVEN_REFUSED_MINE
    p_guard = p_refused_mine * (1 - REFUSE_CRIMINALS_GIVEN_REFUSED_MINE)
    p_scientist = p_leave * 0.25
    p_botanist = p_leave * 0.25
    p_refiner = p_leave * 0.25
    p_worker = p_leave * 0.25
    predictions = {
        "Tabula Rasa": p_tabula,
        "Doctor": p_doctor,
        "Shrink": p_shrink,
        "Miner": p_miner,
        "Technician": p_technician,
        "Guard": p_guard,
        "Scientist": p_scientist,
        "Botanist": p_botanist,
        "Refiner": p_refiner,
        "Worker": p_worker,
    }
    for name, prob in sorted(predictions.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} predicted={prob:6.2%}")
