# Quantum Computing for Business: Executive Summary

*Findings from a hands-on CUDA-Q exploration project. Source material for a
follow-up slide deck — see `fun_stuff/` and `schrodingers_slop/` in this repo
for the runnable code behind every claim below.*

---

## TL;DR

- Quantum computing is **not** a general speedup for business software today.
  Nothing in this project ran faster, cheaper, or "smarter" than a classical
  equivalent — every demo was cross-checked against classical ground truth,
  and the classical answer was always available, always cheaper to compute,
  and always correct.
- What quantum computing genuinely offers **right now** falls into three
  narrow, well-defined categories (below) — not "AI but quantum," not
  "unhackable everything," not "instant optimization of anything NP-hard."
- The realistic business posture in 2026 is **exploration and literacy**, not
  production deployment. Pilot small, low-stakes, well-understood problems;
  don't bet a roadmap on quantum advantage that hasn't been demonstrated for
  your problem class.

---

## What quantum computing actually offers today

### 1. Genuine, hardware-backed randomness

Classical "random" numbers are pseudo-random — deterministic algorithms that
merely look random. A measured qubit is a **physically undetermined**
outcome (not a knowledge gap — this is what Bell's-theorem-style experiments
rule out). This is real and immediately useful anywhere "true randomness"
has business value: cryptographic key generation, fraud-detection sampling,
fair lottery/allocation systems, simulation seeding that must resist
adversarial prediction.

*Demonstrated in: `hello_cudaq.py`, `schrodingers_christmas.py`,
`schrodingers_game_of_life.py`.*

### 2. Amplitude amplification (Grover-style search)

For search problems over an unstructured space of size *N*, Grover's
algorithm finds a marked answer in roughly `sqrt(N)` steps instead of `N` —
a real, proven quadratic speedup, **in principle**. The catch, demonstrated
directly in this project: the speedup only shows up at problem sizes far
beyond what today's hardware (or simulators) can run, the marked fraction
has to be known/estimated correctly, and overshooting the iteration count
actively hurts you (a real, non-obvious failure mode, not a hypothetical
one).

*Demonstrated in: `08_grovers_algorithm.py`, `schrodingers_shipping.py`
(includes a worked example of the overshoot failure), `interval_scheduling/`
(three-way CPU vs. GPU vs. quantum comparison on the same problem).*

### 3. Native modeling of joint/correlated uncertainty (entanglement)

Classical systems can *fake* correlation with shared random seeds or lookup
tables, but genuine joint uncertainty — where individual outcomes are
undetermined until observed, yet perfectly correlated with each other — is a
distinct computational primitive. Where a business problem is naturally
"many uncertain things whose outcomes are linked" (contagion/network
effects, multi-party dependent decisions), this is a structurally different
modeling tool, not just a faster classical one.

*Demonstrated in: `schrodingers_epidemic.py` (contact-graph contagion),
`schrodingers_sales_funnel.py` (funnel-stage superposition, and an
attribution-ambiguity analogy — see caveat below), `jans_life_quantum_sim.py`
(entangled decision trees).

---

## Candidate business use cases surfaced by this project

| Domain | What was actually tested | Real-world analog |
|---|---|---|
| Logistics / scheduling | Multi-berth allocation (NP-hard resource assignment under conflict constraints), weighted interval scheduling | Dock/gate scheduling, room/equipment booking, workforce shift assignment |
| Marketing / sales | Funnel-stage conversion modeling; attribution-ambiguity **analogy** (not a literal claim about CRM data) | Pipeline forecasting exploration, a talking point for why attribution models disagree |
| Risk / fraud | True hardware randomness for unpredictable sampling | Audit sample selection, fraud-detection holdout sampling |
| Network effects | Graph-structured correlated outcomes with cycles | Epidemiological modeling, viral/referral spread, cascading-failure risk |
| Procedural / scenario generation | Wave-function-collapse-style constrained generation | Scenario planning, constrained content/config generation |

**Caveat that applies to every row above:** these are structural analogies
and small-scale proofs of mechanism, run on a GPU-based *simulator*, not
production algorithms benchmarked against real classical solvers at business
scale. None of them currently outperform a classical solution to the same
problem.

---

## What this project does *not* claim

- No quantum advantage was demonstrated on any problem here — every quantum
  result was verified *against* a classical answer, and the classical answer
  was the source of truth, not the quantum one.
- "Quantum eraser" and similar entanglement demos used as marketing/business
  analogies (see `schrodingers_sales_funnel.py`, `schrodingers_detective.py`)
  are illustrative parallels in *structure*, not claims that real business
  systems (CRMs, attribution platforms, etc.) behave quantum-mechanically.
- Nothing here ran on real quantum hardware — this project uses NVIDIA
  CUDA-Q's classical GPU simulator throughout. Real NISQ-era hardware adds
  noise, decoherence, and qubit-count limits that a simulator does not
  reproduce.
- Grover's algorithm's quadratic speedup does not apply to every
  "optimization" problem — it applies specifically to unstructured search
  with a checkable oracle, and only pays off past a scale this project never
  actually reached.

---

## Recommended posture for a business audience

1. **Treat this as R&D literacy, not a roadmap item.** The honest
   conclusion of every demo in this repo is "mechanism proven, advantage not
   yet demonstrated." That's a fair summary of the field's actual state in
   2026, not a limitation specific to this project.
2. **Watch the categories above, not the hype cycle.** True randomness,
   amplitude-amplified search, and correlated-uncertainty modeling are the
   three concrete primitives worth tracking for your industry — ignore
   claims that don't map to one of these three.
3. **Pilot on toy-scale, well-understood problems first**, exactly as this
   project did — a small scheduling or allocation problem where you already
   have a fast classical solution, so you can honestly measure whether (and
   at what scale) a quantum approach starts to matter.
4. **Budget for verification, not just execution.** Every demo in this
   project spent as much effort checking the quantum answer against a
   classical one as it did building the quantum circuit. That discipline —
   not blind trust in "quantum = better" — is what should carry over to any
   real pilot.

---

*Appendix: full runnable source for every claim above lives in `fun_stuff/`
and `schrodingers_slop/` in this repository, each verified against the
`nvidia` CUDA-Q simulator target with a fixed random seed for
reproducibility.*
