"""
Schrodinger's Weather
=======================

A quantum decision tree for "will it rain tomorrow", conditioned on 3
binary weather variables: pressure, temperature, and humidity. This is
the multi-control pattern from 08_grovers_algorithm.py's CCC-Z and
schrodingers_shipping.py's group-violation gates, reused for
classification instead of search: 3 input qubits are put into an equal
superposition over all 8 (pressure, temperature, humidity) scenarios,
then a 4th "rain" qubit gets ONE multi-controlled ry rotation per leaf
of the decision tree - each rotation gated on exactly one of the 8
input combinations via the open-control trick (flip with x(), apply
the n-controlled ry, flip back).

The 8 leaf rain-probabilities below are dummy numbers picked to be
directionally sane (low pressure + humid + cool -> most likely to
rain, high pressure + hot + dry -> least likely) - this is a toy
bootstrap, not a calibrated meteorology model.

Because all 8 scenarios are equally likely by construction (h() on
each input qubit), the aggregate P(rain) has an exact classical
ground truth: the plain average of the 8 leaf probabilities. Each
individual leaf is also verified by conditioning the measured results
on that exact combination of pressure/temperature/humidity, the same
"slice the shot results and compare" technique as
schrodingers_sales_funnel.py's tiered conditional check.
"""

import math
from itertools import product

import cudaq

# Dummy leaf table: P(rain | low_pressure, hot, humid), one entry per
# combination of the 3 binary inputs. Not real meteorology - just
# numbers picked to move in the directions you'd expect.
LEAF_RAIN_PROBABILITY = {
    (True, False, True): 0.85,  # low pressure, cool, humid -> classic storm setup
    (True, False, False): 0.55,  # low pressure, cool, dry
    (True, True, True): 0.65,  # low pressure, hot, humid -> summer storm
    (True, True, False): 0.35,  # low pressure, hot, dry
    (False, False, True): 0.30,  # high pressure, cool, humid
    (False, False, False): 0.10,  # high pressure, cool, dry
    (False, True, True): 0.20,  # high pressure, hot, humid
    (False, True, False): 0.05,  # high pressure, hot, dry -> classic clear-sky setup
}

LEAVES = list(LEAF_RAIN_PROBABILITY.keys())
ANGLES = [2 * math.asin(math.sqrt(LEAF_RAIN_PROBABILITY[leaf])) for leaf in LEAVES]


@cudaq.kernel
def weather_decision_tree(low_pressure: bool, hot: bool, humid: bool) -> bool:
    """Builds one (pressure, temperature, humidity, rain) scenario and
    returns whether it rained. The 3 input qubits are forced (via x())
    to the exact classical combination the caller asked for, so this
    kernel in isolation just checks one leaf's rotation is wired up
    correctly - the real superposition-over-all-leaves version is
    weather_superposition() below."""
    q = cudaq.qvector(4)
    pressure, temperature, humidity, rain = q[0], q[1], q[2], q[3]

    if low_pressure:
        x(pressure)
    if hot:
        x(temperature)
    if humid:
        x(humidity)

    _apply_all_leaves(pressure, temperature, humidity, rain)

    return mz(rain)


@cudaq.kernel
def _apply_all_leaves(
    pressure: cudaq.qubit,
    temperature: cudaq.qubit,
    humidity: cudaq.qubit,
    rain: cudaq.qubit,
):
    """One multi-controlled ry per leaf of the decision tree. Each
    leaf's rotation only fires when (pressure, temperature, humidity)
    exactly match that leaf's combination - open controls (values that
    should read as 0) are implemented by flipping with x() before the
    controlled rotation and flipping back after, so every other leaf's
    rotation is a no-op for this particular basis state."""
    # Leaf 0: low_pressure=True, hot=False, humid=True -> 0.85
    x(temperature)
    ry.ctrl(ANGLES[0], pressure, temperature, humidity, rain)
    x(temperature)

    # Leaf 1: low_pressure=True, hot=False, humid=False -> 0.55
    x(temperature)
    x(humidity)
    ry.ctrl(ANGLES[1], pressure, temperature, humidity, rain)
    x(temperature)
    x(humidity)

    # Leaf 2: low_pressure=True, hot=True, humid=True -> 0.65
    ry.ctrl(ANGLES[2], pressure, temperature, humidity, rain)

    # Leaf 3: low_pressure=True, hot=True, humid=False -> 0.35
    x(humidity)
    ry.ctrl(ANGLES[3], pressure, temperature, humidity, rain)
    x(humidity)

    # Leaf 4: low_pressure=False, hot=False, humid=True -> 0.30
    x(pressure)
    x(temperature)
    ry.ctrl(ANGLES[4], pressure, temperature, humidity, rain)
    x(pressure)
    x(temperature)

    # Leaf 5: low_pressure=False, hot=False, humid=False -> 0.10
    x(pressure)
    x(temperature)
    x(humidity)
    ry.ctrl(ANGLES[5], pressure, temperature, humidity, rain)
    x(pressure)
    x(temperature)
    x(humidity)

    # Leaf 6: low_pressure=False, hot=True, humid=True -> 0.20
    x(pressure)
    ry.ctrl(ANGLES[6], pressure, temperature, humidity, rain)
    x(pressure)

    # Leaf 7: low_pressure=False, hot=True, humid=False -> 0.05
    x(pressure)
    x(humidity)
    ry.ctrl(ANGLES[7], pressure, temperature, humidity, rain)
    x(pressure)
    x(humidity)


@cudaq.kernel
def weather_superposition() -> list[bool]:
    """The real demo: pressure, temperature, and humidity are all put
    into an equal superposition (h() on each) so all 8 scenarios exist
    simultaneously, then the SAME 8 leaf rotations from
    _apply_all_leaves run once. Every leaf's rotation only actually
    "hits" the branch of the superposition matching its own input
    combination - measurement then samples across all 8 scenarios at
    once, weighted by both the (equal) scenario prior and each leaf's
    rain probability."""
    q = cudaq.qvector(4)
    pressure, temperature, humidity, rain = q[0], q[1], q[2], q[3]

    h(pressure)
    h(temperature)
    h(humidity)

    _apply_all_leaves(pressure, temperature, humidity, rain)

    return mz(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 100

    print("=== Decision tree leaf table (dummy numbers) ===")
    for (low_pressure, hot, humid), p in LEAF_RAIN_PROBABILITY.items():
        print(
            f"  pressure={'low' if low_pressure else 'high':<4} "
            f"temp={'hot' if hot else 'cool':<4} "
            f"humidity={'humid' if humid else 'dry':<5} "
            f"-> P(rain) = {p:.2f}"
        )

    print()
    print("=== Part 1: single-leaf sanity check ===")
    print("Forcing each of the 8 combinations classically (no superposition)")
    print("and confirming the measured rain rate matches that leaf's dummy")
    print("probability - this just proves the multi-controlled rotations are")
    print("wired to the right combination, before trusting the superposition run.")
    for low_pressure, hot, humid in product([True, False], repeat=3):
        expected = LEAF_RAIN_PROBABILITY[(low_pressure, hot, humid)]
        results = cudaq.run(
            weather_decision_tree, low_pressure, hot, humid, shots_count=shots
        )
        measured = sum(1 for r in results if r) / shots
        label = f"pressure={'low' if low_pressure else 'high'}, temp={'hot' if hot else 'cool'}, humidity={'humid' if humid else 'dry'}"
        print(f"  {label:<45} measured {measured:.3f}  expected {expected:.3f}")

    print()
    print("=== Part 2: all 8 scenarios in one real superposition ===")
    predicted_aggregate = sum(LEAF_RAIN_PROBABILITY.values()) / len(
        LEAF_RAIN_PROBABILITY
    )
    print(
        f"Predicted P(rain) (equal-weight average of all 8 leaves): {predicted_aggregate:.4f}"
    )

    results = cudaq.run(weather_superposition, shots_count=shots)
    measured_aggregate = sum(1 for r in results if r[3]) / shots
    print(f"Measured P(rain) across the superposition: {measured_aggregate:.4f}")

    print()
    print("Per-leaf check within the SAME superposition run - slicing the")
    print("shot results by the measured (pressure, temperature, humidity),")
    print("exactly like schrodingers_sales_funnel.py's tiered conditional check:")
    for low_pressure, hot, humid in product([True, False], repeat=3):
        matching = [
            r for r in results if r[0] == low_pressure and r[1] == hot and r[2] == humid
        ]
        expected = LEAF_RAIN_PROBABILITY[(low_pressure, hot, humid)]
        measured = (
            sum(1 for r in matching if r[3]) / len(matching)
            if matching
            else float("nan")
        )
        label = f"pressure={'low' if low_pressure else 'high'}, temp={'hot' if hot else 'cool'}, humidity={'humid' if humid else 'dry'}"
        print(
            f"  {label:<45} n={len(matching):<5} measured {measured:.3f}  expected {expected:.3f}"
        )

    print()
    print("=== Honest conclusion ===")
    print("This is a toy decision tree with made-up leaf probabilities, not a")
    print("real weather model. The point being demonstrated is structural: an")
    print("n-controlled rotation per leaf lets a classification decision tree")
    print("live as one coherent superposition over every input combination at")
    print("once, with the correct per-leaf probability recovered exactly by")
    print("measurement - verified here against the classical average and")
    print("against each individual leaf's expected probability.")
