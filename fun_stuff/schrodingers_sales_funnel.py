"""
Schrodinger's Sales Funnel
============================

A B2B sales funnel, modeled as a chain of conditional quantum coin
flips (same "higher order Schrodinger" pattern as
schrodingers_higher_order.py), plus a second, independent trick: a
real quantum eraser (same mechanism as schrodingers_detective.py)
repurposed as an honest analogy for multi-touch marketing attribution.

PART 1 - the funnel itself: Lead -> MQL -> SQL -> Opportunity ->
Closed-Won. Each stage only gets a REAL chance to advance if the
previous stage happened - modeled with the exact same "controlled
rotation, gated on the previous qubit" chain as
schrodingers_higher_order.py's schrodinger_chain(), just with 4
different, realistic conversion rates instead of one repeated 50/50
coin. This is standard funnel math (P(closed) = product of per-stage
conversion rates) - the "quantum" part isn't that the math is
different, it's that every stage stays in superposition simultaneously
until one final measurement, rather than being resolved one classical
coin-flip at a time.

PART 2 - attribution, honestly labeled as an ANALOGY, not a claim that
CRM data is quantum: two marketing touchpoints (an email campaign, a
paid ad) get entangled with whether the deal closed, via the same CNOT
mechanism as schrodingers_detective.py's evidence-planting. Measuring
a touchpoint "the obvious way" shows it perfectly correlated with
closed deals - exactly what a naive last-touch attribution model would
report. Measuring the SAME underlying touchpoint in a different basis
makes that correlation vanish, without undoing any gate or changing
any recorded interaction. This is a genuine structural parallel to a
real, well-documented headache in marketing analytics: different
attribution models (first-touch, last-touch, multi-touch, data-driven)
routinely assign wildly different credit to the exact same underlying
touchpoint data. The parallel is about the STRUCTURE of the ambiguity
(same underlying facts, different "lens" produces different credit),
not a claim that attribution software runs on entangled qubits.

Honest disclaimer, matching this project's house style: nothing here
is a real quantum algorithm for running a sales team. It's two genuine
quantum mechanisms (conditional superposition, and basis-dependent
correlation) wearing a funnel-shaped costume, verified against
classical ground truth below - not a business case for a quantum CRM.
"""

import math
from collections import Counter

import cudaq

# Realistic-ish B2B SaaS funnel conversion rates, stage by stage.
P_LEAD_TO_MQL = 0.40
P_MQL_TO_SQL = 0.25
P_SQL_TO_OPPORTUNITY = 0.50
P_OPPORTUNITY_TO_WON = 0.30

ANGLE_MQL = 2 * math.asin(math.sqrt(P_LEAD_TO_MQL))
ANGLE_SQL = 2 * math.asin(math.sqrt(P_MQL_TO_SQL))
ANGLE_OPPORTUNITY = 2 * math.asin(math.sqrt(P_SQL_TO_OPPORTUNITY))
ANGLE_WON = 2 * math.asin(math.sqrt(P_OPPORTUNITY_TO_WON))

STAGE_NAMES = ["mql", "sql", "opportunity", "closed_won"]
STAGE_PROBABILITIES = [P_LEAD_TO_MQL, P_MQL_TO_SQL, P_SQL_TO_OPPORTUNITY, P_OPPORTUNITY_TO_WON]


@cudaq.kernel
def enters_mql(mql: cudaq.qubit):
    """Every lead gets a real, unconditional shot at becoming a
    Marketing-Qualified Lead - this is the top of the funnel, so
    there's no previous stage to gate on."""
    ry(ANGLE_MQL, mql)


@cudaq.kernel
def converts_to_sql(mql: cudaq.qubit, sql: cudaq.qubit):
    """Only fires a REAL rotation on sql if mql collapsed to 1 - same
    "controlled coin flip" mechanism as schrodingers_higher_order.py.
    If mql=0, sql's own ry never runs, and it stays a deterministic 0:
    a lead that never became an MQL has exactly zero chance of
    becoming an SQL, not some fuzzy small chance."""
    ry.ctrl(ANGLE_SQL, mql, sql)


@cudaq.kernel
def advances_to_opportunity(sql: cudaq.qubit, opportunity: cudaq.qubit):
    ry.ctrl(ANGLE_OPPORTUNITY, sql, opportunity)


@cudaq.kernel
def closes_won(opportunity: cudaq.qubit, closed_won: cudaq.qubit):
    ry.ctrl(ANGLE_WON, opportunity, closed_won)


@cudaq.kernel
def run_funnel() -> list[bool]:
    """The whole funnel stays in one coherent superposition - every
    lead is simultaneously "maybe an MQL, maybe an SQL, maybe closed"
    right up until the single measurement at the end. Nothing about
    any individual stage is resolved one at a time."""
    stages = cudaq.qvector(4)
    mql, sql, opportunity, closed_won = stages[0], stages[1], stages[2], stages[3]

    enters_mql(mql)
    converts_to_sql(mql, sql)
    advances_to_opportunity(sql, opportunity)
    closes_won(opportunity, closed_won)

    return mz(stages)


@cudaq.kernel
def plant_touchpoint(closed_won: cudaq.qubit, touchpoint: cudaq.qubit):
    """Records a touchpoint as associated with this deal's outcome -
    the same CNOT mechanism as schrodingers_detective.py's
    plant_evidence(). Entangles touchpoint with closed_won: if you
    look at both "the obvious way," they always agree."""
    x.ctrl(closed_won, touchpoint)


@cudaq.kernel
def view_through_different_lens(touchpoint: cudaq.qubit):
    """The attribution-model swap: measuring in a different basis
    (H before mz) instead of the obvious one. No gate is undone, no
    information is deleted from the touchpoint qubit - only the
    question we ask of it changes, and that alone erases the
    correlation with closed_won."""
    h(touchpoint)


@cudaq.kernel
def attribution_experiment(view_email_sideways: bool, view_ad_sideways: bool) -> list[bool]:
    """A fresh closed_won qubit (biased at the funnel's real aggregate
    win rate) with 2 touchpoints entangled to it, then measured through
    whichever "attribution lens" the caller asks for."""
    q = cudaq.qvector(3)
    closed_won, email_touch, ad_touch = q[0], q[1], q[2]

    ry(ANGLE_WON, closed_won)  # stand-in for "this deal's own win probability"
    plant_touchpoint(closed_won, email_touch)
    plant_touchpoint(closed_won, ad_touch)

    if view_email_sideways:
        view_through_different_lens(email_touch)
    if view_ad_sideways:
        view_through_different_lens(ad_touch)

    return mz(q)


def print_correlation(label: str, results: list) -> None:
    n = len(results)
    p_won = sum(1 for r in results if r[0]) / n
    p_email = sum(1 for r in results if r[1]) / n
    p_ad = sum(1 for r in results if r[2]) / n
    p_won_and_email = sum(1 for r in results if r[0] and r[1]) / n
    p_won_and_ad = sum(1 for r in results if r[0] and r[2]) / n
    print(f"=== {label} ===")
    print(f"  P(won)                    = {p_won:.3f}")
    print(f"  P(won AND email credited) = {p_won_and_email:.3f}  "
          f"(independent-guess baseline: {p_won*p_email:.3f})")
    print(f"  P(won AND ad credited)    = {p_won_and_ad:.3f}  "
          f"(independent-guess baseline: {p_won*p_ad:.3f})")
    print()


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    # The aggregate win rate is a product of 4 probabilities (as low as
    # 1.5%) - resolving that needs real shot count, not the 100-shot
    # default used elsewhere in fun_stuff/ for coarser comparisons (see
    # schrodingers_shipping.py's shots comment for the same lesson).
    shots = 20000

    print("=== Part 1: the funnel, verified against classical funnel math ===")
    print(f"Stage rates: MQL {P_LEAD_TO_MQL:.0%}, SQL {P_MQL_TO_SQL:.0%} (of MQL), "
          f"Opportunity {P_SQL_TO_OPPORTUNITY:.0%} (of SQL), "
          f"Won {P_OPPORTUNITY_TO_WON:.0%} (of Opportunity)")
    predicted_aggregate = math.prod(STAGE_PROBABILITIES)
    print(f"Predicted aggregate win rate (product of all 4 stages): {predicted_aggregate:.4%}")

    results = cudaq.run(run_funnel, shots_count=shots)
    measured_aggregate = sum(1 for bits in results if bits[3]) / shots
    print(f"Measured aggregate win rate: {measured_aggregate:.4%}")

    print()
    print("Tiered conditional check - conditioning on each stage in turn,")
    print("exactly like schrodingers_higher_order.py's tiered verification:")
    for i in range(1, 4):
        prev_true = [b for b in results if b[i - 1]]
        prev_false = [b for b in results if not b[i - 1]]
        p_given_prev_true = sum(1 for b in prev_true if b[i]) / len(prev_true) if prev_true else float("nan")
        p_given_prev_false = sum(1 for b in prev_false if b[i]) / len(prev_false) if prev_false else float("nan")
        print(f"  P({STAGE_NAMES[i]}=1 | {STAGE_NAMES[i-1]}=1) = {p_given_prev_true:.3f}  "
              f"(predicted {STAGE_PROBABILITIES[i]:.3f})")
        print(f"  P({STAGE_NAMES[i]}=1 | {STAGE_NAMES[i-1]}=0) = {p_given_prev_false:.3f}  "
              f"(predicted 0.000 - can't skip a stage)")

    print()
    print("=== Part 2: attribution ambiguity (a quantum-eraser ANALOGY, ===")
    print("    not a claim that CRM data is quantum) ===")
    print("Same underlying entangled touchpoints in every run below - only")
    print("the measurement basis ('attribution lens') changes.")
    print()

    both_obvious = cudaq.run(attribution_experiment, False, False, shots_count=shots)
    print_correlation("Both touchpoints viewed the obvious way (naive last-touch-style credit)",
                       both_obvious)

    ad_sideways = cudaq.run(attribution_experiment, False, True, shots_count=shots)
    print_correlation("Ad touchpoint viewed through a different lens - same data, same gates",
                       ad_sideways)

    print("Expected: in the first run, BOTH touchpoints show strong correlation")
    print("with a won deal - a naive attribution model would credit both. In")
    print("the second run, the ad touchpoint's correlation with 'won' collapses")
    print("to roughly the independent-guess baseline, while email's doesn't -")
    print("even though plant_touchpoint() ran identically for both, and nothing")
    print("about the deal's actual outcome changed. The only thing that changed")
    print("is which question ('lens') we asked of the ad touchpoint qubit.")

    print()
    print("=== Honest conclusion ===")
    print("This is not a pitch for quantum-powered CRM software. Part 1 is real")
    print("quantum superposition computing an outcome classical funnel math")
    print("already computes exactly as well (and far faster) - the interesting")
    print("part is mechanism, not speed or advantage. Part 2 borrows a genuine")
    print("physical phenomenon (basis-dependent correlation) as a STRUCTURAL")
    print("analogy for a real, well-documented marketing-analytics problem")
    print("(attribution-model disagreement) - the analogy is about the SHAPE of")
    print("the ambiguity, not a mechanistic claim about how attribution software")
    print("works. Nothing here processes a real lead faster or better than a")
    print("spreadsheet would.")
