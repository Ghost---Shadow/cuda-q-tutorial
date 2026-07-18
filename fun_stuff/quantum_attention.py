"""
LLM attention, as literally as it translates into gates: given a query
vector and 3 key vectors, compute scores = query . key_i, softmax them into
weights, and produce output = sum_i weight_i * value_i - a weighted blend
of value vectors. The softmax/dot-product part is plain classical math (no
reason to reinvent arithmetic in gates); the interesting quantum part is
the LAST step, turning "weighted sum" into something that falls out of
quantum mechanics for free instead of being computed digit by digit.

The mechanism:
  1. Amplitude-encode the 3 softmax weights directly into a 2-qubit "key"
     register's superposition: sqrt(w_A)|00> + sqrt(w_B)|01> + sqrt(w_C)|10>
     (a small binary decision tree of controlled RY rotations - the same
     cascading-conditional-rotation pattern as schrodingers_sales_funnel.py).
     Measuring this register alone IS sampling from the softmax
     distribution, exactly - not an approximation of it.
  2. For each key branch, apply a controlled RY(arccos(value_i)) to a
     shared "value" qubit, controlled on the key register matching that
     branch (multi-controlled rotation via the same temporary-X-flip trick
     used for the reversible comparators elsewhere in this repo, since
     CUDA-Q has no negative-control primitive). RY(arccos(v))|0> has
     <Z> = v exactly, so each branch writes its own value's <Z> into the
     value qubit, but only within that branch.
  3. Never condition on which key branch fired. Discarding (not measuring
     against) the key register means the value qubit's marginal state is a
     genuine quantum-mechanical MIXTURE of the three branches, weighted
     exactly by their probabilities - and <Z> on that mixture is provably
     sum_i w_i * v_i, because expectation values are linear over
     probabilistic mixtures. That's the whole trick: attention's weighted
     sum comes out of the mixture's statistics for free.

Honest caveat, same one as the rest of this repo's superposition demos:
this blend is a STATISTICAL property, visible over many shots (or as the
reduced density matrix) - one single shot still collapses to one
(key, value) outcome, not a blended number. Verified below two ways: the
per-branch conditional <Z | key=i> must equal value_i exactly (proving the
mixture is built correctly), and the unconditional <Z> must match the
classical softmax(query.key) . value formula.
"""

import math
from collections import defaultdict

import cudaq

KEYS = ("A", "B", "C")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def attention_weights(query: list[float], keys: dict) -> dict:
    """The classical part of attention: scores = query . key_i, softmax'd."""
    scores = [dot(query, keys[k]) for k in KEYS]
    weights = softmax(scores)
    return dict(zip(KEYS, weights))


def attention_output(weights: dict, values: dict) -> float:
    """The classical ground truth this whole circuit is trying to reproduce."""
    return sum(weights[k] * values[k] for k in KEYS)


def encoding_angles(weights: dict, values: dict) -> tuple[float, float, float, float, float]:
    """Turn softmax weights + values into the 5 rotation angles the kernel
    needs. theta_hi/theta_lo amplitude-encode the weights (P(hi=1)=w_C;
    P(lo=1|hi=0)=w_B/(w_A+w_B)); theta_a/b/c = arccos(value) so RY(theta)|0>
    has <Z> = value exactly."""
    w_a, w_b, w_c = weights["A"], weights["B"], weights["C"]
    theta_hi = 2 * math.asin(math.sqrt(w_c))
    theta_lo = 2 * math.asin(math.sqrt(w_b / (w_a + w_b))) if (w_a + w_b) > 0 else 0.0
    theta_a = math.acos(max(-1.0, min(1.0, values["A"])))
    theta_b = math.acos(max(-1.0, min(1.0, values["B"])))
    theta_c = math.acos(max(-1.0, min(1.0, values["C"])))
    return theta_hi, theta_lo, theta_a, theta_b, theta_c


@cudaq.kernel
def quantum_attention(theta_hi: float, theta_lo: float,
                       theta_a: float, theta_b: float, theta_c: float) -> list[bool]:
    """3 keys (A=|00>, B=|01>, C=|10>; |11> unused), amplitude-encoded by
    theta_hi/theta_lo into a superposition matching the softmax weights,
    then a controlled RY per branch blends that branch's value into a
    shared value qubit. Returns [hi, lo, val]."""
    q = cudaq.qvector(3)
    hi, lo, val = q[0], q[1], q[2]

    ry(theta_hi, hi)          # P(hi=1) = w_C
    x(hi)
    ry.ctrl(theta_lo, hi, lo)  # P(lo=1 | hi=0) = w_B / (w_A + w_B)
    x(hi)

    # branch A: hi=0, lo=0
    x(hi)
    x(lo)
    ry.ctrl(theta_a, hi, lo, val)
    x(lo)
    x(hi)

    # branch B: hi=0, lo=1
    x(hi)
    ry.ctrl(theta_b, hi, lo, val)
    x(hi)

    # branch C: hi=1, lo=0 (lo is guaranteed 0 here, but ctrl still needs a |1>)
    x(lo)
    ry.ctrl(theta_c, hi, lo, val)
    x(lo)

    return mz(q)


def _key_of(hi: bool, lo: bool) -> str:
    if not hi and not lo:
        return "A"
    if not hi and lo:
        return "B"
    return "C"


def run_quantum_attention(query: list[float], keys: dict, values: dict, shots: int) -> dict:
    """Runs the circuit and returns a dict with the measured per-branch and
    unconditional <Z> statistics, plus the classical reference values, for
    comparison. `keys` and `values` map "A"/"B"/"C" -> vector / scalar."""
    weights = attention_weights(query, keys)
    theta_hi, theta_lo, theta_a, theta_b, theta_c = encoding_angles(weights, values)

    results = cudaq.run(quantum_attention, theta_hi, theta_lo, theta_a, theta_b, theta_c,
                         shots_count=shots)

    z_by_key = defaultdict(list)
    overall_z = []
    key_counts = defaultdict(int)
    for hi, lo, val in results:
        k = _key_of(hi, lo)
        z = 1 if not val else -1
        z_by_key[k].append(z)
        overall_z.append(z)
        key_counts[k] += 1

    return {
        "weights": weights,
        "measured_key_freq": {k: key_counts[k] / shots for k in KEYS},
        "measured_conditional_z": {k: (sum(z_by_key[k]) / len(z_by_key[k]) if z_by_key[k] else None)
                                   for k in KEYS},
        "classical_output": attention_output(weights, values),
        "measured_output": sum(overall_z) / len(overall_z),
    }


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(5)

    query = [1.0, 0.0]
    keys = {"A": [0.9, 0.1], "B": [1.0, 0.0], "C": [-0.5, 0.5]}
    values = {"A": 0.8, "B": -0.3, "C": 0.5}

    shots = 20000
    result = run_quantum_attention(query, keys, values, shots)

    print(f"query={query}  keys={keys}  values={values}")
    print(f"softmax weights: {{{', '.join(f'{k}: {v:.3f}' for k, v in result['weights'].items())}}}")
    print(f"measured key sample frequency (should match weights): "
          f"{{{', '.join(f'{k}: {v:.3f}' for k, v in result['measured_key_freq'].items())}}}")

    print("\nPer-branch check (proves the mixture is built correctly):")
    all_ok = True
    for k in KEYS:
        measured = result["measured_conditional_z"][k]
        expected = values[k]
        ok = abs(measured - expected) < 0.05
        all_ok = all_ok and ok
        print(f"  <Z | key={k}> = {measured:.3f}  (expected {expected})  {'OK' if ok else 'MISMATCH'}")

    print("\nBlended output check (the actual attention claim):")
    print(f"  classical softmax(query.key).value = {result['classical_output']:.4f}")
    print(f"  quantum <Z> (key register discarded) = {result['measured_output']:.4f}")
    output_ok = abs(result["classical_output"] - result["measured_output"]) < 0.02
    all_ok = all_ok and output_ok
    print(f"  match within shot noise: {output_ok}")

    print("\nSUCCESS" if all_ok else "\nFAILURE")
