"""
reversion_diagnostic.py
=======================

Settle a modeling tension with data: is the cross-venue trade a **mark-to-market
mean-reversion play** (z-score entries, exit at the rolling mean, stop-losses --
``signal_engine`` / ``backtester``) or a **static held-to-resolution convergence
arbitrage** (P&L fixed at entry = ``|spread| - fees`` -- ``execution_engine``)?
They are different worlds; only one is real for these venues.

The asymmetry this measures
---------------------------
A held-to-resolution pair locks ``|s_entry| - fees`` at entry and settles for free
at resolution -- path-independent, no exit. The mean-reversion play exits *early* at
the rolling mean, which only adds value if (a) the spread actually reverts **before
resolution**, and (b) you can sell back into the book, **crossing a bid/ask the data
doesn't show.** And if the rolling mean is ~0, "revert to mean" *is* "converge to
resolution" -- so exiting early merely pays an extra cost to capture what holding
captures for free. Mean-reversion's only legitimate edge is **capital recycling**:
one unit of capital does many round-trips while held-to-resolution locks that unit
until settlement.

What it computes
----------------
**Part A -- reversion dynamics** (a property of the data): for every entry the
``SignalEngine`` triggers, does the spread revert to its rolling mean before the
event resolves (= end of frame)? Reversion rate, the distribution of
*time-to-reversion vs time-to-resolution*, and the share of mean-reversion P&L that
exists **only** because of early exits.

**Part B -- exit-cost sensitivity** (the honest part): a capital-matched comparison,
1 unit of capital over the same event, reusing the existing fee parameters and
``VectorizedBacktester`` metric definitions:

* **MR(c):** the actual ``SignalEngine`` -> ``VectorizedBacktester`` strategy
  (recycling 1 unit through N round-trips), minus a round-trip **early-exit cost c**
  charged on each early exit.
* **H2R:** 1 unit deployed at the first dislocation and held to resolution,
  ``|s_entry| - fees`` (guaranteed convergence).

Sweep ``c`` from 0 up to a plausible thin-extreme spread and find the **crossover
``c*``** where mean-reversion stops out-earning held-to-resolution. The verdict is one
sentence backed by these numbers.

Scope note: this isolates the reversion/convergence economics on the spread itself;
it does **not** re-apply the ``is_tradeable`` execution mask (orthogonal; handled in
``master_oos``). Both strategies see the same bars, so the comparison is
apples-to-apples. The frame is assumed to end at/near resolution (terminal
``|spread|`` ~ 0); a warning fires if it does not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_pipeline import MarketDataPipeline
from stat_lab import StatLab
from signal_engine import SignalEngine
from backtester import VectorizedBacktester

__all__ = ["ReversionDiagnostic", "ReversionReport"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL
_S = MarketDataPipeline.SPREAD_COL

# Plausible round-trip early-exit cost at a thin 2-3c book (selling both legs back
# across the bid/ask). The verdict compares the crossover c* against this.
DEFAULT_REF_EXIT_COST = 0.02
DEFAULT_COST_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)


@dataclass
class ReversionReport:
    """Output of the reversion diagnostic + the one-sentence verdict."""

    # Part A
    n_entries: int
    n_reverted: int
    n_stopped: int
    n_held: int
    reversion_rate: float
    t_rev_hours: pd.Series = field(repr=False, default=None)
    t_res_hours: pd.Series = field(repr=False, default=None)
    median_rev_fraction: float = float("nan")   # median (t_rev / t_res)
    mean_abs_rolling_mean: float = float("nan")  # ~0 => reverting-to-mean == converging
    early_exit_dependence: float = float("nan")  # (MR0 - H2R) / MR0
    # Part B
    mr_total_pnl: float = float("nan")           # MR at c=0
    mr_sharpe: float = float("nan")
    mr_maxdd: float = float("nan")
    n_early_exits: int = 0
    h2r_total_pnl: float = float("nan")          # 1-unit, guaranteed convergence
    h2r_total_scaled: float = float("nan")       # all dislocations held (N units)
    crossover_cost: float = float("nan")
    ref_exit_cost: float = DEFAULT_REF_EXIT_COST
    sweep: pd.DataFrame = field(repr=False, default=None)
    bar_hours: float = float("nan")
    resolves_in_frame: bool = True
    verdict: str = ""

    def summary(self) -> dict:
        return {
            "n_entries": self.n_entries,
            "reversion_rate": round(self.reversion_rate, 3),
            "median_rev_fraction_of_horizon": round(self.median_rev_fraction, 3),
            "mean_abs_rolling_mean": round(self.mean_abs_rolling_mean, 4),
            "early_exit_dependence": round(self.early_exit_dependence, 3),
            "mr_total_pnl": round(self.mr_total_pnl, 4),
            "mr_sharpe": round(self.mr_sharpe, 2),
            "mr_maxdd": round(self.mr_maxdd, 4),
            "h2r_total_pnl": round(self.h2r_total_pnl, 4),
            "n_early_exits": self.n_early_exits,
            "crossover_cost": round(self.crossover_cost, 4),
            "ref_exit_cost": self.ref_exit_cost,
        }


class ReversionDiagnostic:
    """Measure spread reversion vs resolution and the exit-cost sensitivity of the
    mean-reversion framing against the held-to-resolution arb.

    Parameters
    ----------
    synced:
        A synchronized frame (``kalshi_price``, ``polymarket_price``, ``price_spread``),
        DatetimeIndex, assumed to run up to ~resolution (terminal ``|spread|`` ~ 0).
    window, entry_z, stop_z:
        Rolling z-score window and entry / stop thresholds (the existing machinery).
    kalshi_fee, poly_slippage:
        The existing per-fill friction parameters (reused for both strategies).
    ref_exit_cost:
        The plausible thin-extreme round-trip early-exit cost the verdict is judged
        against. Default 0.02 (2c).
    """

    def __init__(self, synced: pd.DataFrame, *, window: int = 60, entry_z: float = 2.0,
                 stop_z: float = 3.5, kalshi_fee: float = 0.01, poly_slippage: float = 0.005,
                 ref_exit_cost: float = DEFAULT_REF_EXIT_COST,
                 kalshi_col: str = _K, poly_col: str = _P, spread_col: str = _S) -> None:
        for c in (kalshi_col, poly_col, spread_col):
            if c not in synced.columns:
                raise KeyError(f"synced is missing column {c!r}; columns are {list(synced.columns)}")
        if not isinstance(synced.index, pd.DatetimeIndex):
            raise TypeError("synced must have a DatetimeIndex")
        self.synced = synced
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.stop_z = float(stop_z)
        self.kalshi_fee = float(kalshi_fee)
        self.poly_slippage = float(poly_slippage)
        self.ref_exit_cost = float(ref_exit_cost)
        self.kalshi_col, self.poly_col, self.spread_col = kalshi_col, poly_col, spread_col

        deltas = pd.Series(synced.index).diff().dropna().dt.total_seconds()
        self.bar_hours = float(deltas.median() / 3600.0) if len(deltas) else float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._lab = StatLab(synced, kalshi_col=kalshi_col,
                                polymarket_col=poly_col, spread_col=spread_col)
            self._sig = SignalEngine(
                risk_mode="stop_loss", entry_z=entry_z, stop_z_score=stop_z,
                kalshi_fee=kalshi_fee, poly_slippage=poly_slippage,
            ).run(self._lab, window=window)
            self._bt = VectorizedBacktester(
                kalshi_fee=kalshi_fee, poly_slippage=poly_slippage).run(self._sig)
        self._capital = self._bt.initial_capital

    # ------------------------------------------------------------------ #
    # Trade-level tracing (Part A)
    # ------------------------------------------------------------------ #
    def trades(self) -> pd.DataFrame:
        """One row per SignalEngine entry: entry/exit bar, outcome, timing, P&L.

        ``outcome`` is ``reverted`` (exited at the rolling mean), ``stopped`` (hit the
        stop), or ``held`` (still open at resolution). ``pnl_early`` is the
        frictionless MR capture; ``pnl_if_held`` is what holding that same position to
        resolution (spread -> 0) would have made.
        """
        ev = self._sig["trade_event"].fillna("").astype(str)
        s = self._sig["spread"].to_numpy(dtype=float)
        pos = self._sig["position"].to_numpy(dtype=float)
        is_enter = ev.str.startswith("ENTER").to_numpy()
        is_exit = ((ev != "") & ~ev.str.startswith("ENTER")).to_numpy()
        is_stop = ev.str.contains("STOP", case=False).to_numpy()
        enter_pos = np.flatnonzero(is_enter)
        exit_pos = np.flatnonzero(is_exit)
        n = len(s)
        last = n - 1

        rows = []
        for pe in enter_pos:
            later = exit_pos[exit_pos > pe]
            px = int(later[0]) if len(later) else None
            if px is None:
                outcome, pexit = "held", last
            else:
                outcome, pexit = ("stopped" if is_stop[px] else "reverted"), px
            q = float(np.sign(pos[pe])) if pos[pe] != 0 else float(-np.sign(s[pe]))
            t_rev = (pexit - pe) * self.bar_hours
            t_res = (last - pe) * self.bar_hours
            rows.append({
                "entry_bar": int(pe), "exit_bar": int(pexit), "outcome": outcome,
                "entry_time": self._sig.index[pe], "s_entry": s[pe], "position": q,
                "t_rev_hours": t_rev, "t_res_hours": t_res,
                "rev_fraction": (t_rev / t_res) if t_res > 0 else np.nan,
                "pnl_early": q * (s[pexit] - s[pe]),     # captured by early exit (frictionless)
                "pnl_if_held": q * (0.0 - s[pe]),        # same position held to resolution
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Held-to-resolution baseline + cost-swept MR (Part B)
    # ------------------------------------------------------------------ #
    def _entry_friction(self, bar: int) -> float:
        pk = abs(float(self._sig[self.kalshi_col].iloc[bar]))
        pp = abs(float(self._sig[self.poly_col].iloc[bar]))
        return self.kalshi_fee * pk + self.poly_slippage * pp

    def _h2r_totals(self, trades: pd.DataFrame):
        """1-unit H2R (first dislocation, held) and the all-dislocations scaled version."""
        if trades.empty:
            return float("nan"), float("nan")
        first = int(trades["entry_bar"].iloc[0])
        h2r_1u = abs(float(trades["s_entry"].iloc[0])) - self._entry_friction(first)
        scaled = float(sum(abs(r.s_entry) - self._entry_friction(int(r.entry_bar))
                           for r in trades.itertuples()))
        return h2r_1u, scaled

    def _mr_pnl_series(self, c: float) -> pd.Series:
        """Per-bar net P&L of MR with an extra round-trip cost ``c`` on each early exit."""
        net = self._bt.frame["net_pnl"].copy()
        if c:
            ev = self._sig["trade_event"].fillna("").astype(str)
            is_exit = (ev != "") & ~ev.str.startswith("ENTER")
            net = net.subtract(c * is_exit.astype(float).values, axis=0)
        return net

    def sweep_exit_cost(self, costs=DEFAULT_COST_GRID) -> pd.DataFrame:
        """MR metrics (via VectorizedBacktester definitions) across early-exit costs,
        with the constant held-to-resolution baseline for comparison."""
        trades = self.trades()
        h2r_1u, _ = self._h2r_totals(trades)
        ev = self._sig["trade_event"].fillna("").astype(str)
        n_exits = int(((ev != "") & ~ev.str.startswith("ENTER")).sum())

        rows = []
        for c in costs:
            net = self._mr_pnl_series(c)
            daily_ret = net.resample("1D").sum() / self._capital
            equity = self._capital + net.cumsum()
            rows.append({
                "exit_cost": float(c),
                "mr_total_pnl": float(net.sum()),
                "mr_sharpe": VectorizedBacktester.annualized_sharpe(daily_ret),
                "mr_maxdd": VectorizedBacktester.max_drawdown(equity),
                "h2r_total_pnl": h2r_1u,
                "mr_beats_h2r": bool(net.sum() > h2r_1u),
            })
        return pd.DataFrame(rows)

    def crossover_cost(self) -> float:
        """Round-trip early-exit cost at which MR total P&L drops to the H2R baseline.

        Closed form (MR total is linear in c): ``(MR0 - H2R) / n_early_exits``, clipped
        at 0 (if MR loses even at zero cost the machinery is never justified)."""
        trades = self.trades()
        h2r_1u, _ = self._h2r_totals(trades)
        ev = self._sig["trade_event"].fillna("").astype(str)
        n_exits = int(((ev != "") & ~ev.str.startswith("ENTER")).sum())
        mr0 = float(self._bt.total_pnl)
        if n_exits == 0 or not np.isfinite(h2r_1u):
            return float("nan")
        return max(0.0, (mr0 - h2r_1u) / n_exits)

    # ------------------------------------------------------------------ #
    # Full report + verdict
    # ------------------------------------------------------------------ #
    def analyze(self, costs=DEFAULT_COST_GRID) -> ReversionReport:
        trades = self.trades()
        n_entries = len(trades)
        rev = trades[trades["outcome"] == "reverted"] if n_entries else trades
        n_reverted = int((trades["outcome"] == "reverted").sum()) if n_entries else 0
        n_stopped = int((trades["outcome"] == "stopped").sum()) if n_entries else 0
        n_held = int((trades["outcome"] == "held").sum()) if n_entries else 0

        h2r_1u, h2r_scaled = self._h2r_totals(trades)
        mr0 = float(self._bt.total_pnl)
        ev = self._sig["trade_event"].fillna("").astype(str)
        n_exits = int(((ev != "") & ~ev.str.startswith("ENTER")).sum())
        c_star = self.crossover_cost()
        sweep = self.sweep_exit_cost(costs)
        s_last = abs(float(self._sig["spread"].iloc[-1]))
        resolves = s_last < 0.03

        report = ReversionReport(
            n_entries=n_entries, n_reverted=n_reverted, n_stopped=n_stopped, n_held=n_held,
            reversion_rate=(n_reverted / n_entries) if n_entries else float("nan"),
            t_rev_hours=rev["t_rev_hours"] if n_entries else None,
            t_res_hours=trades["t_res_hours"] if n_entries else None,
            median_rev_fraction=float(rev["rev_fraction"].median()) if n_reverted else float("nan"),
            mean_abs_rolling_mean=float(self._sig["rolling_mean"].abs().mean()),
            early_exit_dependence=((mr0 - h2r_1u) / mr0) if (np.isfinite(mr0) and mr0 != 0) else float("nan"),
            mr_total_pnl=mr0, mr_sharpe=float(self._bt.sharpe),
            mr_maxdd=float(self._bt.max_drawdown), n_early_exits=n_exits,
            h2r_total_pnl=h2r_1u, h2r_total_scaled=h2r_scaled,
            crossover_cost=c_star, ref_exit_cost=self.ref_exit_cost,
            sweep=sweep, bar_hours=self.bar_hours, resolves_in_frame=resolves,
        )
        report.verdict = self._verdict(report)
        return report

    def _verdict(self, r: ReversionReport) -> str:
        cents = lambda x: f"{x * 100:.1f}c"
        if r.n_entries < 2 or r.n_early_exits == 0:
            return (f"INSUFFICIENT ACTIVITY: {r.n_entries} entries / {r.n_early_exits} early exits — "
                    "the spread barely reverts before resolution, so the z-score/stop machinery has "
                    "nothing to monetize over a held-to-resolution arb; treat it as held-to-resolution.")
        if not np.isfinite(r.h2r_total_pnl) or r.mr_total_pnl <= r.h2r_total_pnl:
            return (f"DROP the machinery: even at ZERO exit cost, mean-reversion makes "
                    f"${r.mr_total_pnl:.3f} vs ${r.h2r_total_pnl:.3f} for held-to-resolution on one unit "
                    f"of capital (spread reverts before resolution only {r.reversion_rate:.0%} of the "
                    "time), so early exits can't out-earn simply locking |spread| at entry — treat this "
                    "purely as a held-to-resolution arb.")
        if r.crossover_cost >= r.ref_exit_cost:
            return (f"KEEP the machinery: mean-reversion out-earns held-to-resolution "
                    f"(${r.mr_total_pnl:.3f} vs ${r.h2r_total_pnl:.3f} at zero cost, reverting "
                    f"{r.reversion_rate:.0%} of the time at ~{r.median_rev_fraction:.0%} of the way to "
                    f"resolution) and stays ahead until the round-trip early-exit cost exceeds "
                    f"{cents(r.crossover_cost)} — above the ~{cents(r.ref_exit_cost)} plausible "
                    "thin-extreme cost, so the z-score/stop exits are justified.")
        return (f"TREAT AS HELD-TO-RESOLUTION ARB: mean-reversion only out-earns holding while the "
                f"round-trip early-exit cost stays below {cents(r.crossover_cost)}, but a realistic "
                f"thin-extreme exit costs ~{cents(r.ref_exit_cost)} — so once you pay to cross the book, "
                f"the machinery ({r.reversion_rate:.0%} reversion, ${r.mr_total_pnl:.3f} gross vs "
                f"${r.h2r_total_pnl:.3f}) no longer beats locking |spread| at entry.")

    def report(self, costs=DEFAULT_COST_GRID) -> str:
        r = self.analyze(costs)
        L = []
        L.append("# Mean-Reversion vs Held-to-Resolution — Diagnostic\n")
        L.append(f"- Bars: {len(self.synced)} @ ~{self.bar_hours:.2f}h | "
                 f"window={self.window}, entry_z={self.entry_z}, stop_z={self.stop_z} | "
                 f"fees: kalshi {self.kalshi_fee:.1%}, poly_slip {self.poly_slippage:.1%}")
        if not r.resolves_in_frame:
            L.append("- ⚠️ **Terminal |spread| is not ~0** — the frame may not reach resolution; "
                     "held-to-resolution P&L assumes convergence to 0 at the true settlement.")
        if r.n_entries < 2:
            L.append(f"\n**Only {r.n_entries} entries triggered — not enough to judge.**\n")
            L.append(f"\n## Verdict\n\n> {r.verdict}")
            return "\n".join(L)

        L.append("\n## Part A — does the spread revert before it resolves?\n")
        L.append(f"- Entries: **{r.n_entries}** → reverted to mean **{r.n_reverted}** "
                 f"({r.reversion_rate:.0%}), stopped {r.n_stopped}, held to resolution {r.n_held}")
        if r.n_reverted:
            tr, ts = r.t_rev_hours.dropna(), r.t_res_hours.dropna()
            L.append(f"- Time-to-reversion (reverting trades): median **{tr.median():.1f}h** "
                     f"[{tr.quantile(.25):.1f}–{tr.quantile(.75):.1f}h]")
            L.append(f"- Time-to-resolution (all entries): median **{ts.median():.1f}h** "
                     f"[{ts.quantile(.25):.1f}–{ts.quantile(.75):.1f}h]")
            L.append(f"- Reversion lands at **~{r.median_rev_fraction:.0%}** of the way to resolution "
                     "(small = reverts long before the event; ~100% = only converges at the end)")
        L.append(f"- Avg |rolling mean| = **{r.mean_abs_rolling_mean:.4f}** "
                 "(≈0 ⇒ 'revert to mean' is essentially 'converge to resolution', so early exit only "
                 "adds value via capital recycling, not via capturing something extra)")
        if r.mr_total_pnl > r.h2r_total_pnl and r.mr_total_pnl > 0:
            L.append(f"- Share of MR P&L that exists **only** because of early exits / recycling: "
                     f"**{r.early_exit_dependence:.0%}** (the rest holding would capture anyway)")
        else:
            L.append(f"- Early exits **destroy** value here: MR makes ${r.mr_total_pnl:.3f} vs "
                     f"${r.h2r_total_pnl:.3f} held on the same capital — the machinery subtracts, not adds")

        L.append("\n## Part B — how much early-exit cost can the edge survive?\n")
        L.append(f"Capital-matched, 1 unit, same event (reusing `VectorizedBacktester` metrics + fees). "
                 f"Held-to-resolution baseline = **${r.h2r_total_pnl:.3f}** "
                 f"(1 unit; all {r.n_entries} dislocations held would make ${r.h2r_total_scaled:.3f} "
                 f"but needs ~{r.n_entries}× the capital).\n")
        sw = r.sweep.copy()
        L.append(f"| {'exit cost':>9} | {'MR total':>9} | {'MR Sharpe':>9} | {'MR MaxDD':>8} | "
                 f"{'H2R total':>9} | beats H2R |")
        L.append(f"|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*10}|{'-'*11}|:---------:|")
        for row in sw.itertuples():
            L.append(f"| {row.exit_cost*100:>7.2f}c | {row.mr_total_pnl:>9.3f} | "
                     f"{row.mr_sharpe:>9.2f} | {row.mr_maxdd:>7.1%} | {row.h2r_total_pnl:>9.3f} | "
                     f"{'  yes' if row.mr_beats_h2r else '   no':>9} |")
        cstar = "never beats it" if r.crossover_cost == 0 else f"{r.crossover_cost*100:.2f}c"
        L.append(f"\n- **Crossover:** mean-reversion stops beating held-to-resolution at an early-exit "
                 f"cost of **{cstar}** (plausible thin-extreme reference: {r.ref_exit_cost*100:.1f}c).")
        L.append(f"\n## Verdict\n\n> {r.verdict}")
        return "\n".join(L)


# --------------------------------------------------------------------------- #
# Self-contained smoke test (synthetic; two regimes to show it discriminates)
# --------------------------------------------------------------------------- #
def _make_synced(kind: str, seed: int, days: int = 40, ppd: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * ppd
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    b = np.zeros(n)
    if kind == "reverting":          # fast reversion around 0 -> frequent round-trips
        for i in range(1, n):
            b[i] = 0.75 * b[i - 1] + rng.normal(0, 0.02)
    else:                            # persistent dislocation that only resolves at the end
        b[0] = 0.06
        for i in range(1, n):
            b[i] = 0.995 * b[i - 1] + rng.normal(0, 0.004)
    b[-ppd:] = np.linspace(b[-ppd - 1], 0.0, ppd)   # converge to resolution (spread -> 0)
    k = np.clip(0.5 + b / 2, 0.01, 0.99)
    p = np.clip(0.5 - b / 2, 0.01, 0.99)
    return pd.DataFrame({_K: k, _P: p, _S: np.abs(k - p)}, index=idx)


def _demo() -> None:
    reports = {}
    for kind, seed in (("reverting", 1), ("persistent", 7)):
        diag = ReversionDiagnostic(_make_synced(kind, seed), window=60, entry_z=2.0, stop_z=3.5)
        rep = diag.analyze()
        reports[kind] = rep
        print("=" * 78)
        print(f"REGIME: {kind}")
        print("=" * 78)
        print(diag.report())
        print()

    a, b = reports["reverting"], reports["persistent"]
    print("=" * 78)
    print("Smoke-test discrimination check:")
    print(f"  reversion rate    reverting={a.reversion_rate:.0%}  persistent={b.reversion_rate:.0%}")
    print(f"  crossover cost    reverting={a.crossover_cost*100:.2f}c  persistent={b.crossover_cost*100:.2f}c")
    ok = (a.reversion_rate >= b.reversion_rate) and (a.crossover_cost >= b.crossover_cost)
    print(f"  [{'OK' if ok else 'UNEXPECTED'}] the fast-reverting regime supports the machinery more "
          "than the persistent one.")


if __name__ == "__main__":
    _demo()
