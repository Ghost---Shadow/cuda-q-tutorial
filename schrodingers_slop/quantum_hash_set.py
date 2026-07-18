"""
A "hash set" needs no values, just membership: contains(x) -> bool. Built
on top of quantum_attention.py (imported, not reimplemented) by treating
set membership itself as the thing being blended: each stored key gets a
value of +1.0 if it's a set member, -1.0 if it isn't, and a query vector's
"contains?" answer is exactly the attention output - a softmax-weighted
blend of those +-1 membership scores, driven by how closely the query
vector matches each stored key vector.

This isn't a made-up gimmick: attention-as-associative-memory (nearest-
neighbor-ish retrieval, weighted by similarity) is a real, published
correspondence - modern continuous Hopfield networks are provably
equivalent to a single attention step (Ramsauer et al., "Hopfield Networks
is All You Need"). What that buys a "hash set" that a classical one
doesn't have: classical set membership is a hard 0/1 with no notion of
"how close" a non-member query was. This one degrades gracefully - a query
that's a near-exact match to a member key comes back near +1 (confidently
in the set), a query near a non-member comes back near -1 (confidently
out), and a query that's genuinely ambiguous between a member and a
non-member comes back near 0 - an honest "not sure," not a coin flip
dressed up as an answer.

Same honest caveat as quantum_attention.py: the blended score is a
statistical property (many shots / the reduced density matrix), not a
single-shot readout - and the softmax/dot-product scoring is classical
math, same as it is in a real attention layer.
"""

import cudaq

from quantum_attention import KEYS, attention_output, attention_weights, run_quantum_attention

# 3 "buckets" a query can land near - spread far apart (large magnitude)
# so dot-product scores differ sharply and softmax actually peaks instead
# of staying close to uniform.
STORED_KEYS = {"A": [5.0, 0.0], "B": [-5.0, 0.0], "C": [0.0, 5.0]}

# Ground truth: A and C are members of the set, B is not.
MEMBER_SET = {"A", "C"}
MEMBERSHIP_VALUES = {k: (1.0 if k in MEMBER_SET else -1.0) for k in KEYS}

CONFIDENT_THRESHOLD = 0.5  # |score| below this counts as "ambiguous", not a hard yes/no


def quantum_contains(query: list[float], shots: int = 20000) -> dict:
    """Runs the query through quantum_attention's circuit with the set's
    membership values plugged in as the "values" being blended. Returns
    the same diagnostic dict as run_quantum_attention, plus a verdict."""
    result = run_quantum_attention(query, STORED_KEYS, MEMBERSHIP_VALUES, shots)
    score = result["measured_output"]
    if score > CONFIDENT_THRESHOLD:
        verdict = "IN the set"
    elif score < -CONFIDENT_THRESHOLD:
        verdict = "NOT in the set"
    else:
        verdict = "AMBIGUOUS (query doesn't clearly match a member or a non-member)"
    result["score"] = score
    result["verdict"] = verdict
    return result


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(9)

    print(f"Stored keys: {STORED_KEYS}")
    print(f"Set members: {sorted(MEMBER_SET)}  (membership values: {MEMBERSHIP_VALUES})\n")

    test_queries = {
        "near member A":          [5.0, 0.0],
        "near non-member B":      [-5.0, 0.0],
        "ambiguous (A/B tied, away from C)": [0.0, -5.0],
    }

    all_ok = True
    for label, query in test_queries.items():
        result = quantum_contains(query)
        weights = result["weights"]
        print(f"--- query={query}  ({label}) ---")
        print(f"  softmax weights: {{{', '.join(f'{k}: {v:.3f}' for k, v in weights.items())}}}")
        print(f"  classical reference score = {result['classical_output']:.4f}")
        print(f"  quantum measured score    = {result['score']:.4f}")
        print(f"  verdict: {result['verdict']}")
        match = abs(result["classical_output"] - result["score"]) < 0.03
        all_ok = all_ok and match
        print(f"  quantum matches classical within shot noise: {match}\n")

    # Sanity checks against ground truth for the two clear-cut queries.
    near_a = quantum_contains(test_queries["near member A"])
    near_b = quantum_contains(test_queries["near non-member B"])
    checks_ok = ("IN the set" in near_a["verdict"]) and ("NOT in the set" in near_b["verdict"])
    all_ok = all_ok and checks_ok
    print(f"Ground-truth verdicts correct (A->IN, B->NOT IN): {checks_ok}")

    print("\nSUCCESS" if all_ok else "\nFAILURE")
