"""
master_flb_oos.py
=================

Real-data, out-of-sample driver for the **favorite-longshot (FLB)** stack
(``alpha_engine`` -> ``bias_analyzer`` -> ``fader_signal_engine``). Where the module
demos prove the pipeline on a *synthetic* bias injected with a known parameter, this
script points it at **real resolved Polymarket markets** and asks the only question
that matters: does the out-of-sample, net-of-cost fade edge actually clear zero?

REAL DATA ONLY -- no synthetic fallback. If the Gamma/CLOB pull fails it says so and
exits; it never fabricates a panel or a metric.

    pip install requests
    python master_flb_oos.py                 # pulls FLB_MAX_MARKETS resolved binaries
    FLB_MAX_MARKETS=3000 python master_flb_oos.py

What it does
------------
1. Pull a large panel of resolved binary markets via
   ``PolymarketDataPuller.fetch_resolved_markets`` (highest-volume first), build the
   time-series ``panel_frame``.
2. **Out-of-sample split by resolution date**: calibrate the bias on the *earlier*
   markets, fade the *later* ones. The fade gate's "historical realized rate" is fit
   only on history, so it never sees an outcome it is trading.
3. Print the **cluster-bootstrapped calibration table** (distinct-market counts per
   decile; deciles whose edge is within the bootstrap CI of zero are flagged) and the
   **net-of-cost fader tear sheet** with a bootstrap CI on the net edge.
4. **Cost sensitivity sweep** over ``base_half_spread`` x ``extreme_mult`` x
   ``impact_coef`` -- the whole edge lives at the thin extremes and the defaults are
   guesses -- showing how fast the net edge / net Sharpe die as costs rise toward
   something realistic for a sub-$0.10 contract.

What WOULD count as a real edge (and what wouldn't)
---------------------------------------------------
WOULD: a net-of-cost net-edge whose **market-clustered CI is strictly > 0 at realistic
(not frictionless) cost assumptions**, backed by **enough distinct markets** in the
fade deciles (not rows), with the bias **present in both the calibration and the trade
period** (it persists out-of-sample-in-time), and **not driven by one resolution
event**. WOULDN'T: positive only at zero/low cost; net-edge CI straddles zero; the
fade decile rests on a handful of distinct markets; the edge evaporates at a realistic
sub-$0.10 spread; or the calibration-period bias fails to replicate in the trade
period. A near-zero / negative honest result is the expected outcome and is worth more
than a manufactured Sharpe.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alpha_engine import PolymarketDataPuller  # noqa: E402
from bias_analyzer import BiasAnalyzer  # noqa: E402
from fader_signal_engine import FaderSignalEngine  # noqa: E402

# Config (env-overridable).
MAX_MARKETS = int(os.environ.get("FLB_MAX_MARKETS", "1500"))
SPLIT_FRAC = float(os.environ.get("FLB_SPLIT_FRAC", "0.60"))     # earlier frac -> calibration
N_DECILES = int(os.environ.get("FLB_N_DECILES", "10"))
N_BOOT = int(os.environ.get("FLB_N_BOOT", "2000"))
IMPLIED_THRESHOLD = float(os.environ.get("FLB_IMPLIED_THRESHOLD", "0.10"))
REALIZED_THRESHOLD = float(os.environ.get("FLB_REALIZED_THRESHOLD", "0.05"))
SEED = 12345

# Cost sweep grids (price units / cents).
BHS_GRID = (0.0, 0.005, 0.01, 0.02, 0.03)       # base half-spread
EM_GRID = (0.0, 1.0, 2.0, 3.0)                  # extreme widening multiplier
IC_GRID = (0.0, 0.003, 0.01, 0.02)              # linear impact

# Named cost scenarios (base_half_spread, extreme_mult, impact_coef).
SCENARIOS = [
    ("frictionless",            0.000, 0.0, 0.000),
    ("default (current guess)", 0.005, 1.0, 0.003),
    ("realistic thin (<$0.10)", 0.015, 2.0, 0.008),
    ("pessimistic",             0.030, 3.0, 0.020),
]

CONFOUNDERS = (
    "Confounders the code cannot see (a real edge has to survive these, not just the math above):\n"
    "  - Survivorship / selection in what Gamma returns. fetch_resolved_markets pulls the highest-volume,\n"
    "    cleanly-resolved binaries; quietly-delisted, voided, or zero-volume longshots are under-represented,\n"
    "    so the sampled longshot win-rate is a biased estimate of the true one -- and high-volume markets are\n"
    "    exactly the ones most likely to already be efficient.\n"
    "  - Resolution-date clustering. Many markets settle on the same macro event (an election night, an FOMC\n"
    "    meeting, a sports final). The cluster bootstrap treats each MARKET as independent, but markets that\n"
    "    resolve on the same event are not -- the effective sample is the number of independent EVENTS, often\n"
    "    far smaller, so these CIs are still optimistically tight and an OOS fade P&L can be one or two\n"
    "    correlated events wearing N trades' clothing.\n"
    "  - A 75-year-old, well-known bias. The favorite-longshot bias has been documented since Griffith (1949)\n"
    "    and is a staple of the betting-market literature; sophisticated Polymarket participants know it. Any\n"
    "    historical edge may already be arbitraged/compressed -- especially as volume exploded post-2024 -- so\n"
    "    a calibration-period edge need not persist into the trade period (watch the history-vs-trade slope).\n"
    "  - Reference vs executable price. The calibration keys on the path's mean implied price, but a real fade\n"
    "    enters at a specific time and pays the live bid/ask then; the price you calibrate on is not the price\n"
    "    you can actually trade, and end-of-life convergence can leak into that mean."
)


def _load_dotenv(filename: str = ".env") -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# Data acquisition (REAL ONLY)
# --------------------------------------------------------------------------- #
def pull_resolved_panel(max_markets: int, page_limit: int = 100):
    """Pull resolved binary markets from Polymarket (Gamma + CLOB). Real data only."""
    markets = PolymarketDataPuller().fetch_resolved_markets(
        max_markets=max_markets, page_limit=page_limit, with_prices=True, skip_errors=True)
    if not markets:
        raise RuntimeError("fetch_resolved_markets returned no markets.")
    return markets


def _parsed_date(market):
    try:
        return pd.to_datetime(market.resolved_at, utc=True)
    except Exception:
        return pd.NaT


def split_markets_by_date(markets, frac: float):
    """Sort by resolution date; earlier ``frac`` -> calibration, rest -> trades. Markets
    with no parseable resolution date are dropped (can't be ordered out-of-sample)."""
    dated = [(m, _parsed_date(m)) for m in markets]
    n_undated = sum(1 for _, d in dated if pd.isna(d))
    dated = sorted(((m, d) for m, d in dated if not pd.isna(d)), key=lambda md: md[1])
    if not dated:
        return [], [], n_undated, None
    cut = int(len(dated) * frac)
    cut = min(max(cut, 1), len(dated) - 1) if len(dated) > 1 else len(dated)
    hist = [m for m, _ in dated[:cut]]
    trade = [m for m, _ in dated[cut:]]
    cutoff = dated[cut][1] if cut < len(dated) else dated[-1][1]
    return hist, trade, n_undated, cutoff


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def bootstrap_mean_ci(values, n_boot: int = N_BOOT, ci: float = 0.95, seed: int = SEED):
    """Percentile bootstrap CI for the mean (resampling rows = distinct markets here)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(v.mean()), float(lo), float(hi)


def run_oos_fader(history_panel, trades_snapshot, cost_kwargs, thresholds):
    fader = FaderSignalEngine(
        implied_threshold=thresholds[0], realized_threshold=thresholds[1],
        n_deciles=N_DECILES, fee_rate=0.0, **cost_kwargs)
    fader.fit_calibration(history_panel)
    return fader, fader.backtest(trades_snapshot)


def _per_contract_net_edge(fader, res):
    """Per-fade net edge in price units: (p - y) - transaction_cost(p). Each fade is one
    distinct market, so the bootstrap over this array is market-clustered."""
    t = res.trades
    if t is None or len(t) == 0:
        return np.array([])
    p = t["entry_price"].to_numpy(dtype=float)
    y = t["resolution"].to_numpy(dtype=float)
    tc = np.asarray(fader.transaction_cost(p, fader.stake), dtype=float)
    return (p - y) - tc


def cost_sensitivity(history_panel, trades_snapshot, thresholds, n_boot, seed):
    """Re-run the OOS fader across the cost grid (the fade SET is cost-independent, only
    the P&L moves). Returns a row per (base_half_spread, extreme_mult, impact_coef)."""
    rows = []
    for bhs in BHS_GRID:
        for em in EM_GRID:
            for ic in IC_GRID:
                fader, res = run_oos_fader(
                    history_panel, trades_snapshot,
                    dict(base_half_spread=bhs, extreme_mult=em, impact_coef=ic), thresholds)
                arr = _per_contract_net_edge(fader, res)
                mean, lo, hi = bootstrap_mean_ci(arr, n_boot, 0.95, seed)
                rows.append({
                    "base_half_spread": bhs, "extreme_mult": em, "impact_coef": ic,
                    "n_fades": int(res.n_fades), "gross_edge": res.avg_gross_edge,
                    "avg_cost": res.avg_cost, "net_edge": res.avg_net_edge,
                    "net_lo": lo, "net_hi": hi, "net_sharpe": res.sharpe,
                    "sig_pos": bool(np.isfinite(lo) and lo > 0),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x, p=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def _print_calibration(full_panel, hist_snap, trade_snap):
    an = BiasAnalyzer(full_panel, n_deciles=N_DECILES)
    rep = an.analyze(n_boot=N_BOOT, ci=0.95, seed=SEED)
    t = rep.table.copy()
    # Edge (= realized - implied) is within the bootstrap CI of zero iff the implied
    # price sits inside the realized rate's CI.
    t["edge_in_ci0"] = (t["ci_low"] <= t["mean_implied"]) & (t["mean_implied"] <= t["ci_high"])

    print("== 1. Cluster-bootstrapped calibration (FULL panel, descriptive) ==\n")
    print(f"{'decile':>9} | {'n_obs':>7} | {'mkts':>5} | {'implied':>7} | {'realized':>8} | "
          f"{'95% CI':>15} | {'edge':>7} | sig?")
    print("-" * 86)
    for r in t.itertuples():
        ci = f"[{r.ci_low:.3f},{r.ci_high:.3f}]"
        sig = "  .  (~0)" if r.edge_in_ci0 else (" yes" if r.realized < r.mean_implied else " yes(+)")
        flag = "" if np.isfinite(r.realized) else "  (empty)"
        print(f"{r.decile:>9} | {int(r.n_obs):>7} | {int(r.n_markets):>5} | {r.mean_implied:>7.3f} | "
              f"{_fmt(r.realized,3):>8} | {ci:>15} | {_fmt(r.bias,3):>7} | {sig}{flag}")

    low = t[t["bin_high"] <= 0.20]
    thin = low[low["n_markets"] < 30]
    print(f"\nDistinct markets behind the low-price deciles (where the fade lives): "
          f"{', '.join(f'{r.decile}:{int(r.n_markets)}' for r in low.itertuples()) or 'none'}")
    if len(thin):
        print(f"  THIN (<30 distinct markets): {', '.join(r.decile for r in thin.itertuples())} "
              "-> treat those deciles' rates as noisy regardless of how many rows back them.")
    n_flagged = int(t["edge_in_ci0"].sum())
    print(f"Deciles whose edge is within the bootstrap CI of zero (no detectable bias): "
          f"{n_flagged}/{len(t)}.")

    # Bias persistence across the OOS boundary (a key honesty check).
    sl = lambda d: BiasAnalyzer.calibration_slope(d["implied_prob"], d["resolution"]) if len(d) else float("nan")
    print(f"Calibration slope (>0 ⇒ FLB): full={_fmt(rep.calibration_slope,3)} | "
          f"history={_fmt(sl(hist_snap),3)} | trade={_fmt(sl(trade_snap),3)}  "
          "(history≈trade ⇒ the bias persists OOS-in-time; trade≈0 ⇒ likely arbitraged/regime shift).")
    return rep


def _print_tearsheet(fader, res, n_boot):
    print("\n== 2. Out-of-sample fader tear sheet (default costs, net of cost) ==\n")
    if res.trades is None or res.n_fades == 0:
        print(f"NO MARKETS PASSED THE FADE GATE out-of-sample (implied<{res.implied_threshold}, "
              f"hist-realized<{res.realized_threshold}) among {res.n_candidates} trade-period markets.")
        print("There is nothing to trade on this panel -> no edge to report. (Try a larger panel, "
              "or loosen the thresholds, but a stricter gate failing is itself a finding.)")
        return False
    net = _per_contract_net_edge(fader, res)
    mean, lo, hi = bootstrap_mean_ci(net, n_boot, 0.95, SEED)
    for k, v in res.summary().items():
        print(f"  {k:>20}: {v}")
    print(f"\n  Fade universe: {res.n_fades} distinct markets faded out of {res.n_candidates} "
          f"trade-period candidates.")
    print(f"  Gross edge/contract: {res.avg_gross_edge*100:+.2f}c | cost: {res.avg_cost*100:.2f}c | "
          f"net edge/contract: {res.avg_net_edge*100:+.2f}c")
    print(f"  Net edge 95% bootstrap CI (clustered by market): "
          f"[{lo*100:+.2f}c, {hi*100:+.2f}c]")
    indistinct = not (np.isfinite(lo) and lo > 0)
    if indistinct:
        print("\n  >>> BLUNT VERDICT: the net-of-cost OOS edge is STATISTICALLY INDISTINGUISHABLE FROM ZERO\n"
              "      (95% CI includes 0 once clustered and costed). On this data there is no demonstrable edge.")
    else:
        print("\n  >>> The net-of-cost OOS edge is positive at the 95% level on this panel -- necessary, not\n"
              "      sufficient: read the cost sweep and confounders below before believing it.")
    return True


def _pivot_grid(sweep, em_value, value_col="net_edge"):
    sub = sweep[sweep["extreme_mult"] == em_value]
    piv = sub.pivot(index="base_half_spread", columns="impact_coef", values=value_col)
    sig = sub.pivot(index="base_half_spread", columns="impact_coef", values="sig_pos")
    print(f"\n  net edge (cents/contract) | extreme_mult={em_value} | rows=base_half_spread, cols=impact_coef")
    cols = list(piv.columns)
    print("       impact:" + "".join(f"{c:>9.3f}" for c in cols))
    for bhs in piv.index:
        cells = []
        for c in cols:
            v = piv.loc[bhs, c] * 100
            mark = "*" if bool(sig.loc[bhs, c]) else " "
            cells.append(f"{v:>8.2f}{mark}")
        print(f"  bhs {bhs:>6.3f}:" + "".join(cells))
    print("  (* = net-edge 95% CI strictly > 0; blank = indistinguishable from / below zero)")


def _print_cost_sweep(history_panel, trades_snapshot, thresholds):
    print("\n== 3. Cost sensitivity -- how fast does the edge die at the thin extremes? ==\n")
    base_fader, base_res = run_oos_fader(
        history_panel, trades_snapshot, dict(base_half_spread=0.0, extreme_mult=0.0, impact_coef=0.0), thresholds)
    if base_res.trades is None or base_res.n_fades == 0:
        print("  (no fades -> nothing to sweep.)")
        return
    g_mean, g_lo, g_hi = bootstrap_mean_ci(_per_contract_net_edge(base_fader, base_res))
    print(f"  Gross edge (the ceiling) = {g_mean*100:+.2f}c/contract  CI[{g_lo*100:+.2f},{g_hi*100:+.2f}] "
          f"over {base_res.n_fades} faded markets.")
    print(f"  Breakeven: net edge hits 0 once the average per-fade cost reaches {g_mean*100:.2f}c. "
          "Any cost assumption above that kills the alpha outright.\n")

    sweep = cost_sensitivity(history_panel, trades_snapshot, thresholds, N_BOOT, SEED)
    n_pos = int((sweep["net_edge"] > 0).sum())
    n_sig = int(sweep["sig_pos"].sum())
    print(f"  Across {len(sweep)} cost combos: net edge > 0 in {n_pos}, and 95%-significantly > 0 in "
          f"{n_sig}. Frictionless is the most favorable; realistic combos sit toward the bottom-right.")
    _pivot_grid(sweep, em_value=1.0)
    _pivot_grid(sweep, em_value=2.0)

    print("\n  Named scenarios (net of cost):")
    print(f"  {'scenario':>26} | {'fades':>5} | {'gross':>7} | {'cost':>6} | {'net':>7} | "
          f"{'net 95% CI':>16} | {'Sharpe':>7} | sig?")
    print("  " + "-" * 96)
    for name, bhs, em, ic in SCENARIOS:
        f, r = run_oos_fader(history_panel, trades_snapshot,
                             dict(base_half_spread=bhs, extreme_mult=em, impact_coef=ic), thresholds)
        if r.trades is None or r.n_fades == 0:
            continue
        m, lo, hi = bootstrap_mean_ci(_per_contract_net_edge(f, r))
        ci = f"[{lo*100:+.2f},{hi*100:+.2f}]"
        sig = "yes" if (np.isfinite(lo) and lo > 0) else "NO"
        print(f"  {name:>26} | {r.n_fades:>5} | {r.avg_gross_edge*100:>6.2f}c | {r.avg_cost*100:>5.2f}c | "
              f"{r.avg_net_edge*100:>+6.2f}c | {ci:>16} | {_fmt(r.sharpe,2):>7} | {sig}")


# --------------------------------------------------------------------------- #
# Orchestration (testable: takes a list of markets)
# --------------------------------------------------------------------------- #
def analyze_panel(markets, *, split_frac=SPLIT_FRAC, source_label="markets"):
    # Only markets with usable CLOB price history can be analyzed; the rest come back
    # with empty price series. The filter is itself a finding: price-history
    # availability is time-dependent (CLOB serves only recent markets), which severely
    # constrains any out-of-sample-by-date split.
    priced = [m for m in markets if m.n_obs > 0]
    n_drop = len(markets) - len(priced)

    print("== 0. Provenance ==\n")
    print(f"  Source: {source_label}")
    print(f"  Markets pulled: {len(markets)} | WITH usable CLOB price history: {len(priced)} "
          f"({len(priced) / max(1, len(markets)):.0%}); {n_drop} dropped (empty price series).")
    if len(priced) < 10:
        print("\n  Too few markets with price history to run the pipeline -- CLOB returned a price\n"
              "  series for almost nothing. No analysis possible on this pull.")
        return

    full_panel = PolymarketDataPuller.panel_frame(priced)
    full_snap = PolymarketDataPuller.snapshot_frame(priced)
    pr_dates = pd.to_datetime([_parsed_date(m) for m in priced], utc=True)
    span = (pr_dates.max() - pr_dates.min()).days
    print(f"  Panel rows (price obs): {len(full_panel):,} | reference-price range "
          f"[{full_snap['implied_prob'].min():.3f}, {full_snap['implied_prob'].max():.3f}] | "
          f"YES base rate {full_snap['resolution'].mean():.3f}")
    print(f"  Price-bearing resolution dates: {pr_dates.min().date()} -> {pr_dates.max().date()} "
          f"({span} days) -- the ONLY window CLOB serves here.")

    hist, trade, n_undated, cutoff = split_markets_by_date(priced, split_frac)
    hist_panel = PolymarketDataPuller.panel_frame(hist)
    hist_snap = PolymarketDataPuller.snapshot_frame(hist)
    trade_snap = PolymarketDataPuller.snapshot_frame(trade)
    fade_zone = int((full_snap["implied_prob"] < IMPLIED_THRESHOLD).sum())
    print(f"  OOS split @ {split_frac:.0%} by date: {len(hist)} calibrate (≤ "
          f"{cutoff.date() if cutoff is not None else '?'}) / {len(trade)} fade; {n_undated} undated dropped.")
    print(f"  Distinct price-bearing markets in the fade zone (ref price < {IMPLIED_THRESHOLD}): "
          f"{fade_zone}  <-- the whole strategy lives on this many markets.\n")
    if span < 14:
        print(f"  ⚠️ The entire price-bearing window is {span} days -- this is NOT a real out-of-sample-in-time\n"
              "     test (calibration and trade periods are days apart and likely share the same events).\n")

    if len(hist) < 2 or len(trade) < 1 or len(hist_panel) == 0:
        print("Not enough price-bearing dated markets to form an out-of-sample split.")
        return
    _print_calibration(full_panel, hist_snap, trade_snap)
    fader, res = run_oos_fader(
        hist_panel, trade_snap,
        dict(base_half_spread=0.005, extreme_mult=1.0, impact_coef=0.003),
        (IMPLIED_THRESHOLD, REALIZED_THRESHOLD))
    has_fades = _print_tearsheet(fader, res, N_BOOT)
    if has_fades:
        _print_cost_sweep(hist_panel, trade_snap, (IMPLIED_THRESHOLD, REALIZED_THRESHOLD))
    print("\n== 4. Confounders ==\n")
    print(CONFOUNDERS)


def main() -> None:
    import warnings
    warnings.simplefilter("ignore")
    _load_dotenv()
    try:
        markets = pull_resolved_panel(MAX_MARKETS)
    except Exception as exc:
        print("Could not pull real resolved Polymarket markets:\n  " + str(exc).replace("\n", "\n  "))
        print("\nThis driver is REAL-DATA-ONLY (Polymarket Gamma + CLOB) -- no synthetic fallback. "
              "Ensure `requests` is installed and the APIs are reachable, then re-run.")
        return
    analyze_panel(markets, split_frac=SPLIT_FRAC,
                  source_label=f"Polymarket Gamma/CLOB ({len(markets)} resolved binary markets)")


if __name__ == "__main__":
    main()
