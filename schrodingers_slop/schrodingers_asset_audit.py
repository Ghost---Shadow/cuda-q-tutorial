"""
Schrodinger's Asset Audit
===========================

5 assets, each genuinely undetermined - "secure future" and "impaired
future" both real, superposed - until an audit "measures" one (h() +
mz(), same mechanism as every other script in this series). An audit
costs real money per asset. 4 strategies, applied uniformly across all
5 assets:

  1. Sell without audit  - pooled/expected price, no audit cost.
  2. Audit then sell      - true value revealed, sold at that price,
                            minus the audit cost.
  3. Audit then hold      - true value revealed, held to maturity at
                            that outcome's later value, minus audit cost.
  4. Hold without audit   - held to maturity blind, no audit cost. The
                            true state still gets "measured" eventually
                            (reality resolves itself for free at
                            maturity) - paying for an audit only buys
                            you that information EARLY.

Dummy but internally-consistent assumptions:
  - P(secure) = 60% per asset, independent, genuinely quantum (h()-style
    coin, not a pseudo-random draw).
  - Secure assets are worth 100 now, 130 at maturity (they appreciate).
  - Impaired assets are worth 20 now, 5 at maturity (they decay further
    without intervention - the classic reason you'd want to sell a
    lemon early instead of holding it).
  - Audit cost: $10/asset, $50 total across all 5.
  - Unaudited sale price = the fair pooled expected value (nobody, not
    even the seller, knows more than anyone else here - this is
    DELIBERATELY not Akerlof's "market for lemons": there's no
    adverse selection because the ignorance is genuinely symmetric,
    quantum, not private information the seller is hiding).

The sharp, honest, and slightly counter-intuitive result this script
verifies: strategies 1-vs-2 and 4-vs-3 differ ONLY by the audit cost -
same action taken regardless of what the audit would have revealed, so
the audit buys literally nothing. A 5th strategy is added for
contrast: audit, then act DIFFERENTLY depending on what's revealed
(hold the secure ones, sell the impaired ones early). That's the only
strategy where information has a chance to pay for itself - and this
script computes, exactly, whether it actually does at these prices.
"""

import cudaq

NUM_ASSETS = 5
P_SECURE = 0.60

VALUE_SECURE_NOW = 100.0
VALUE_IMPAIRED_NOW = 20.0
VALUE_SECURE_LATER = 130.0
VALUE_IMPAIRED_LATER = 5.0
AUDIT_COST_PER_ASSET = 10.0

POOLED_PRICE_NOW = P_SECURE * VALUE_SECURE_NOW + (1 - P_SECURE) * VALUE_IMPAIRED_NOW

import math
SECURE_ANGLE = 2 * math.asin(math.sqrt(P_SECURE))


@cudaq.kernel
def audit_all_assets(num_assets: int, angle: float) -> list[bool]:
    """One qubit per asset - genuinely undetermined (secure/impaired)
    until mz(). Every strategy below uses the SAME underlying quantum
    draw; they differ only in whether/when they look at it and what
    they do with the answer."""
    assets = cudaq.qvector(num_assets)
    for i in range(num_assets):
        ry(angle, assets[i])
    return mz(assets)


def strategy_1_sell_without_audit(is_secure: list) -> float:
    """Nobody looks - every asset sells at the same pooled price,
    regardless of its (unobserved) true state."""
    return NUM_ASSETS * POOLED_PRICE_NOW


def strategy_2_audit_then_sell(is_secure: list) -> float:
    total = sum(VALUE_SECURE_NOW if secure else VALUE_IMPAIRED_NOW for secure in is_secure)
    return total - NUM_ASSETS * AUDIT_COST_PER_ASSET


def strategy_3_audit_then_hold(is_secure: list) -> float:
    total = sum(VALUE_SECURE_LATER if secure else VALUE_IMPAIRED_LATER for secure in is_secure)
    return total - NUM_ASSETS * AUDIT_COST_PER_ASSET


def strategy_4_hold_without_audit(is_secure: list) -> float:
    """Nobody looks now - but reality still resolves at maturity for
    free. The true (unobserved-until-now) state determines the payoff
    exactly as it would have if audited, just without the fee."""
    total = sum(VALUE_SECURE_LATER if secure else VALUE_IMPAIRED_LATER for secure in is_secure)
    return total


def strategy_5_audit_then_act_per_asset(is_secure: list) -> float:
    """Bonus strategy, not one of the 4 requested, added for contrast:
    audit, then hold the ones revealed secure (they appreciate) and
    sell the ones revealed impaired (they'd only decay further). This
    is the only strategy where the audit's information could possibly
    pay for itself, since it's the only one where the action actually
    changes based on what's revealed."""
    total = sum(VALUE_SECURE_LATER if secure else VALUE_IMPAIRED_NOW for secure in is_secure)
    return total - NUM_ASSETS * AUDIT_COST_PER_ASSET


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 100

    print(f"=== Assumptions ===")
    print(f"{NUM_ASSETS} assets, P(secure)={P_SECURE:.0%} each, independent, genuinely quantum")
    print(f"Secure:   now=${VALUE_SECURE_NOW:.0f}   later=${VALUE_SECURE_LATER:.0f} (appreciates)")
    print(f"Impaired: now=${VALUE_IMPAIRED_NOW:.0f}   later=${VALUE_IMPAIRED_LATER:.0f} (decays further)")
    print(f"Audit cost: ${AUDIT_COST_PER_ASSET:.0f}/asset (${AUDIT_COST_PER_ASSET*NUM_ASSETS:.0f} total)")
    print(f"Pooled (unaudited) sale price/asset: ${POOLED_PRICE_NOW:.2f}")

    results = cudaq.run(audit_all_assets, NUM_ASSETS, SECURE_ANGLE, shots_count=shots)

    strategies = {
        "1. Sell without audit": strategy_1_sell_without_audit,
        "2. Audit then sell": strategy_2_audit_then_sell,
        "3. Audit then hold": strategy_3_audit_then_hold,
        "4. Hold without audit": strategy_4_hold_without_audit,
        "5. Audit, act per-asset (bonus)": strategy_5_audit_then_act_per_asset,
    }

    print()
    print(f"=== Simulated across {shots} independent 5-asset portfolios ===")
    print(f"  {'strategy':<32}{'measured avg':>14}{'hand-derived E[.]':>20}")
    predicted = {
        "1. Sell without audit": NUM_ASSETS * POOLED_PRICE_NOW,
        "2. Audit then sell": NUM_ASSETS * (POOLED_PRICE_NOW - AUDIT_COST_PER_ASSET),
        "3. Audit then hold": NUM_ASSETS * (P_SECURE * VALUE_SECURE_LATER + (1 - P_SECURE) * VALUE_IMPAIRED_LATER - AUDIT_COST_PER_ASSET),
        "4. Hold without audit": NUM_ASSETS * (P_SECURE * VALUE_SECURE_LATER + (1 - P_SECURE) * VALUE_IMPAIRED_LATER),
        "5. Audit, act per-asset (bonus)": NUM_ASSETS * (P_SECURE * VALUE_SECURE_LATER + (1 - P_SECURE) * VALUE_IMPAIRED_NOW - AUDIT_COST_PER_ASSET),
    }
    measured_totals = {name: [] for name in strategies}
    for r in results:
        is_secure = list(r)
        for name, fn in strategies.items():
            measured_totals[name].append(fn(is_secure))

    for name in strategies:
        avg = sum(measured_totals[name]) / shots
        print(f"  {name:<32}{avg:>14.2f}{predicted[name]:>20.2f}")

    print()
    print("=== Ranked ===")
    ranked = sorted(predicted.items(), key=lambda kv: -kv[1])
    for i, (name, value) in enumerate(ranked, start=1):
        print(f"  {i}. {name}: ${value:.2f}")

    print()
    print("=== Does the audit ever pay for itself? ===")
    diff_1_2 = predicted["1. Sell without audit"] - predicted["2. Audit then sell"]
    diff_4_3 = predicted["4. Hold without audit"] - predicted["3. Audit then hold"]
    print(f"Option 1 vs 2 (sell either way): option 1 wins by exactly "
          f"${diff_1_2:.2f} = the audit cost, ${AUDIT_COST_PER_ASSET*NUM_ASSETS:.2f}. "
          f"Same action either way -> audit bought nothing.")
    print(f"Option 4 vs 3 (hold either way): option 4 wins by exactly "
          f"${diff_4_3:.2f} = the audit cost, ${AUDIT_COST_PER_ASSET*NUM_ASSETS:.2f}. "
          f"Same story - the audit only pays for a fee here, not a decision.")

    diff_4_5 = predicted["4. Hold without audit"] - predicted["5. Audit, act per-asset (bonus)"]
    informational_gain = NUM_ASSETS * (1 - P_SECURE) * (VALUE_IMPAIRED_NOW - VALUE_IMPAIRED_LATER)
    breakeven_audit_cost = informational_gain / NUM_ASSETS
    print(f"\nOption 5 (audit, THEN differ per asset) is the only one where audit")
    print(f"could pay off - it's worth exactly the value of selling impaired assets")
    print(f"early instead of holding them: {NUM_ASSETS}*{1-P_SECURE:.0%}*"
          f"(${VALUE_IMPAIRED_NOW:.0f}-${VALUE_IMPAIRED_LATER:.0f}) = ${informational_gain:.2f}.")
    print(f"At ${AUDIT_COST_PER_ASSET*NUM_ASSETS:.0f} total audit cost, that gain does NOT "
          f"cover the fee: option 4 (blind hold) still wins by ${diff_4_5:.2f}.")
    print(f"Break-even audit cost would need to be <= ${breakeven_audit_cost:.2f}/asset "
          f"(vs. the ${AUDIT_COST_PER_ASSET:.2f}/asset assumed here).")

    print()
    print("=== Honest conclusion ===")
    print("Auditing (measuring) is not free, and - like every measurement in this")
    print("series - it only creates value if what you DO afterward depends on the")
    print("outcome. Options 1-4 as posed only differ by whether the same fixed")
    print("action costs an audit fee or not, so the audit is pure overhead in 3 of")
    print("the 4 requested strategies. At this project's dummy prices, even the one")
    print("strategy built to exploit the information (option 5) doesn't clear its")
    print("own audit bill - blindly holding everything and letting reality resolve")
    print("itself for free, at maturity, wins. That's a real, computed answer, not")
    print("a rule of thumb: whether due diligence is worth paying for is a")
    print("quantitative break-even calculation against the price of the audit, not")
    print("a moral imperative to 'always do your homework.'")
