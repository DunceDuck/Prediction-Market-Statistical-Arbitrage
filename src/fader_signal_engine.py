"""
fader_signal_engine.py
======================

Monetize the favorite-longshot bias. :class:`FaderSignalEngine` turns the
structural miscalibration measured by :class:`bias_analyzer.BiasAnalyzer` into a
systematic trading rule -- **fade over-priced longshots** -- and backtests it with
an honest cost model for the thin extremes of a prediction-market book.

The trade
---------
A longshot YES contract trades at implied probability ``p`` (small). The
calibration analysis says markets at this price historically resolve YES *less*
often than ``p`` -- YES is over-priced. To **fade** it we bet against YES; on a
prediction market you cannot natively short, so we **buy the NO contract** at
``1 - p`` and hold to resolution (the same convention as
:mod:`execution_engine`). Per NO contract, with YES resolution ``y in {0, 1}``:

* market resolves **NO** (``y = 0``): NO pays \\$1, profit ``= 1 - (1 - p) = p``
* market resolves **YES** (``y = 1``): NO pays \\$0, loss ``= -(1 - p)``

so the **gross P&L per contract is ``p - y``**. The edge is ``E[p - y] = p - q``
where ``q`` is the true win rate; for an over-priced longshot ``q < p`` and the
edge is positive. Note the **negative skew**: you win a little (``+p``) most of
the time and lose a lot (``-(1 - p)``) rarely -- so costs and drawdown, not just
mean edge, decide whether the alpha is real.

Signal
------
We do **not** fade every longshot -- only those where history says the bias is
both *present* and *large*. A market is faded when

    implied price  <  ``implied_threshold``      (e.g. 0.10)   AND
    historical realized win rate in its price bucket  <  ``realized_threshold``  (e.g. 0.05)

The realized rate is read from a calibration **fit on past markets only**
(:meth:`fit_calibration`), so applying it to later markets is genuinely
out-of-sample -- no look-ahead.

Cost model (the crucial bit at the extremes)
---------------------------------------------
Because we hold to resolution (settling at \\$0/\\$1, no exit spread), costs are paid
**once, on entry**. Per contract, in price units:

    cost(p) = half_spread(p)  +  impact_coef * size  +  fee_rate * min(p, 1 - p)

* **half_spread(p)** -- crossing the bid/ask, *widened toward the extremes* where
  the book is thin: ``base_half_spread * (1 + extreme_mult * (0.5 - min(p,1-p))/0.5)``.
  At ``p = 0.5`` it is the base; at ``p -> 0`` it is ``(1 + extreme_mult)x`` wider.
* **impact_coef * size** -- linear market impact: a larger order walks deeper into a
  thin book.
* **fee_rate * min(p, 1 - p)** -- Polymarket's fee mechanism is symmetric in price.
  Polymarket's CLOB has historically charged ~no explicit trading fee (the real
  cost is the spread), so ``fee_rate`` defaults to 0; set it to model a schedule.

Buying ``size/(1-p)`` ... we stake a fixed ``stake`` of collateral per fade (max loss
= ``stake``), so the per-contract cost scales into a return drag of
``cost(p)/(1-p)`` -- proportionally *largest* exactly where we trade, the extremes.

Metrics
-------
Reusing :class:`backtester.VectorizedBacktester`'s definitions for consistency:
annualized **Sharpe** (daily P&L by resolution date / fixed capital, x sqrt(252)),
**max drawdown** (equity vs running peak), and the **hit rate** (fraction of fades
with positive net P&L). Every metric is reported **net** of costs, with a **gross**
(cost-free) counterpart so the cost drag is explicit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bias_analyzer import BiasAnalyzer
from backtester import VectorizedBacktester

__all__ = ["FaderSignalEngine", "FaderResult"]


# --------------------------------------------------------------------------- #
# Structured result
# --------------------------------------------------------------------------- #
@dataclass
class FaderResult:
    """Backtest output for the fade-the-longshot strategy (net of costs)."""

    sharpe: float
    max_drawdown: float            # negative fraction
    hit_rate: float                # fraction of fades with net P&L > 0
    total_return: float            # net P&L / total capital deployed
    total_pnl: float
    n_candidates: int              # markets considered
    n_fades: int                   # markets actually faded (trades placed)
    avg_gross_edge: float          # mean (p - y) per contract, before costs
    avg_cost: float                # mean per-contract entry cost
    avg_net_edge: float            # avg_gross_edge - avg_cost
    total_costs: float
    capital_base: float
    n_days: int
    gross_sharpe: float            # cost-free counterfactual
    gross_return: float
    implied_threshold: float
    realized_threshold: float
    trades: pd.DataFrame = field(default=None, repr=False)
    equity_curve: pd.Series = field(default=None, repr=False)
    daily_returns: pd.Series = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_fades": self.n_fades,
            "hit_rate": round(self.hit_rate, 4),
            "sharpe_annualized": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 4),
            "total_pnl": round(self.total_pnl, 6),
            "avg_gross_edge": round(self.avg_gross_edge, 5),
            "avg_cost": round(self.avg_cost, 5),
            "avg_net_edge": round(self.avg_net_edge, 5),
            "total_costs": round(self.total_costs, 6),
            "gross_sharpe": round(self.gross_sharpe, 4),
            "gross_return": round(self.gross_return, 4),
        }


class FaderSignalEngine:
    """Systematic fade-the-longshot signal + backtest with an extremes-aware cost model.

    Parameters
    ----------
    implied_threshold:
        Fade only markets priced **below** this (default 0.10).
    realized_threshold:
        ...and whose price bucket's **historical** realized win rate is below this
        (default 0.05). Both conditions must hold -- we fade only where the bias is
        present *and* large.
    n_deciles:
        Number of price buckets for the calibration. Default 10.
    stake:
        Collateral risked per fade (max loss). Default 1.0.
    base_half_spread, extreme_mult:
        Half-spread model: ``base_half_spread`` at mid, scaled up to
        ``(1 + extreme_mult)x`` at the price extremes. Defaults 0.005 / 1.0.
    impact_coef:
        Linear market-impact cost per unit ``size``. Default 0.003.
    fee_rate:
        Polymarket-style symmetric fee rate on ``min(p, 1-p)``. Default 0.0
        (Polymarket's CLOB has historically charged no explicit trading fee).
    trading_days_per_year:
        Sharpe annualization factor. Default 252.
    prob_col, outcome_col, group_col, date_col:
        Column names in the input frames. Defaults match the data puller's output.
    """

    def __init__(
        self,
        implied_threshold: float = 0.10,
        realized_threshold: float = 0.05,
        n_deciles: int = 10,
        stake: float = 1.0,
        base_half_spread: float = 0.005,
        extreme_mult: float = 1.0,
        impact_coef: float = 0.003,
        fee_rate: float = 0.0,
        trading_days_per_year: int = 252,
        prob_col: str = "implied_prob",
        outcome_col: str = "resolution",
        group_col: str = "market_id",
        date_col: str = "resolved_at",
    ) -> None:
        if not (0.0 < implied_threshold <= 1.0):
            raise ValueError("implied_threshold must be in (0, 1]")
        if not (0.0 <= realized_threshold <= 1.0):
            raise ValueError("realized_threshold must be in [0, 1]")
        if stake <= 0:
            raise ValueError("stake must be positive")
        if base_half_spread < 0 or impact_coef < 0 or fee_rate < 0 or extreme_mult < 0:
            raise ValueError("cost parameters must be non-negative")
        self.implied_threshold = float(implied_threshold)
        self.realized_threshold = float(realized_threshold)
        self.n_deciles = int(n_deciles)
        self.stake = float(stake)
        self.base_half_spread = float(base_half_spread)
        self.extreme_mult = float(extreme_mult)
        self.impact_coef = float(impact_coef)
        self.fee_rate = float(fee_rate)
        self.trading_days_per_year = int(trading_days_per_year)
        self.prob_col = prob_col
        self.outcome_col = outcome_col
        self.group_col = group_col
        self.date_col = date_col

        self._realized_by_decile = None  # set by fit_calibration

    # ------------------------------------------------------------------ #
    # Calibration (the structural alpha -- fit on history only)
    # ------------------------------------------------------------------ #
    def fit_calibration(self, history_df: pd.DataFrame) -> "FaderSignalEngine":
        """Estimate the historical realized win rate per price decile.

        Pass a frame of *past* resolved markets (the time-series ``panel_frame`` or a
        ``snapshot_frame``). Stored and later applied to new markets, so the gate is
        strictly out-of-sample.
        """
        analyzer = BiasAnalyzer(
            history_df, prob_col=self.prob_col, outcome_col=self.outcome_col,
            group_col=self.group_col, n_deciles=self.n_deciles,
        )
        self._realized_by_decile = analyzer.win_rates()["realized"].to_numpy()
        return self

    # ------------------------------------------------------------------ #
    # Cost model
    # ------------------------------------------------------------------ #
    def transaction_cost(self, p, size: float = None):
        """Per-contract entry cost (price units) at price ``p`` for order ``size``.

        ``half_spread(p) + impact_coef*size + fee_rate*min(p, 1-p)``, with the
        half-spread widening toward the extremes. Accepts scalars or arrays.
        """
        size = self.stake if size is None else size
        p = np.asarray(p, dtype=float)
        m = np.minimum(p, 1.0 - p)                       # distance to nearest boundary
        half_spread = self.base_half_spread * (1.0 + self.extreme_mult * (0.5 - m) / 0.5)
        cost = half_spread + self.impact_coef * size + self.fee_rate * m
        return cost if cost.ndim else float(cost)

    # ------------------------------------------------------------------ #
    # Signal
    # ------------------------------------------------------------------ #
    def generate(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Attach the fade signal to a frame of candidate markets (one row each).

        Adds ``decile``, ``realized_hist`` (the bucket's historical realized rate),
        ``fade`` (bool), and ``signal`` (-1 = fade YES / buy NO, 0 = pass). Requires
        :meth:`fit_calibration` first.
        """
        if self._realized_by_decile is None:
            raise RuntimeError("call fit_calibration() before generating signals")
        if self.prob_col not in trades_df.columns:
            raise KeyError(f"{self.prob_col!r} not found in trades frame")

        out = trades_df.copy()
        p = pd.to_numeric(out[self.prob_col], errors="coerce").to_numpy(dtype=float)
        dcode = np.clip((np.nan_to_num(p, nan=0.5) * self.n_deciles).astype(int),
                        0, self.n_deciles - 1)
        realized_hist = self._realized_by_decile[dcode]

        # NaN-safe: an unknown/empty bucket (realized_hist NaN) is never faded.
        fade = (p < self.implied_threshold) & (realized_hist < self.realized_threshold)
        out["decile"] = dcode
        out["realized_hist"] = realized_hist
        out["fade"] = fade
        out["signal"] = np.where(fade, -1, 0)
        return out

    # ------------------------------------------------------------------ #
    # Backtest
    # ------------------------------------------------------------------ #
    def backtest(
        self, trades_df: pd.DataFrame, history_df: pd.DataFrame = None
    ) -> FaderResult:
        """Backtest the fade strategy and return Sharpe / max drawdown / hit rate.

        If ``history_df`` is given, the calibration is (re)fit on it first -- the
        clean out-of-sample setup (fit on history, fade ``trades_df``). If omitted
        and no calibration has been fit, it falls back to fitting on ``trades_df``
        itself (in-sample; a warning is issued).
        """
        if history_df is not None:
            self.fit_calibration(history_df)
        elif self._realized_by_decile is None:
            warnings.warn(
                "No history provided and calibration not fit: fitting on the trade "
                "set itself (in-sample -- realized rates peek at outcomes being "
                "traded). Pass history_df for an out-of-sample backtest.",
                stacklevel=2,
            )
            self.fit_calibration(trades_df)

        for col in (self.outcome_col, self.date_col):
            if col not in trades_df.columns:
                raise KeyError(f"{col!r} not found in trades frame")

        sig = self.generate(trades_df)
        faded = sig[sig["fade"]].copy()
        n_candidates = int(len(sig))
        n_fades = int(len(faded))
        if n_fades == 0:
            return self._empty_result(n_candidates)

        p = np.clip(pd.to_numeric(faded[self.prob_col]).to_numpy(float), 1e-6, 0.999999)
        y = pd.to_numeric(faded[self.outcome_col]).to_numpy(float)

        # Per-contract gross P&L = p - y. Stake `stake` of collateral (buy stake/(1-p)
        # NO contracts), so dollar P&L scales by stake/(1-p).
        tc = np.asarray(self.transaction_cost(p, self.stake), dtype=float)
        gross_pnl = self.stake * (p - y) / (1.0 - p)
        cost_dollars = self.stake * tc / (1.0 - p)
        net_pnl = gross_pnl - cost_dollars
        ret = net_pnl / self.stake

        faded["entry_price"] = p
        faded["gross_pnl"] = gross_pnl
        faded["cost"] = cost_dollars
        faded["net_pnl"] = net_pnl
        faded["ret"] = ret

        capital_base = self.stake * n_fades  # total capital deployed (one stake per fade)

        # Daily P&L by resolution date -> Sharpe & drawdown (reusing the stack's defs).
        day = pd.to_datetime(faded[self.date_col]).dt.floor("D")
        daily_net = pd.Series(net_pnl, index=day.to_numpy()).groupby(level=0).sum().sort_index()
        daily_gross = pd.Series(gross_pnl, index=day.to_numpy()).groupby(level=0).sum().sort_index()
        full = pd.date_range(daily_net.index.min(), daily_net.index.max(),
                             freq="1D", tz=daily_net.index.tz)
        daily_net = daily_net.reindex(full, fill_value=0.0)
        daily_gross = daily_gross.reindex(full, fill_value=0.0)

        daily_returns = daily_net / capital_base
        equity = capital_base + daily_net.cumsum()
        sharpe = VectorizedBacktester.annualized_sharpe(daily_returns, self.trading_days_per_year)
        max_dd = VectorizedBacktester.max_drawdown(equity)
        gross_sharpe = VectorizedBacktester.annualized_sharpe(
            daily_gross / capital_base, self.trading_days_per_year)

        return FaderResult(
            sharpe=sharpe,
            max_drawdown=max_dd,
            hit_rate=float((net_pnl > 0).mean()),
            total_return=float(net_pnl.sum() / capital_base),
            total_pnl=float(net_pnl.sum()),
            n_candidates=n_candidates,
            n_fades=n_fades,
            avg_gross_edge=float((p - y).mean()),
            avg_cost=float(tc.mean()),
            avg_net_edge=float((p - y).mean() - tc.mean()),
            total_costs=float(cost_dollars.sum()),
            capital_base=float(capital_base),
            n_days=int(len(daily_returns)),
            gross_sharpe=gross_sharpe,
            gross_return=float(gross_pnl.sum() / capital_base),
            implied_threshold=self.implied_threshold,
            realized_threshold=self.realized_threshold,
            trades=faded,
            equity_curve=equity,
            daily_returns=daily_returns,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _empty_result(self, n_candidates: int) -> FaderResult:
        nan = float("nan")
        return FaderResult(
            sharpe=nan, max_drawdown=nan, hit_rate=nan, total_return=nan, total_pnl=0.0,
            n_candidates=n_candidates, n_fades=0, avg_gross_edge=nan, avg_cost=nan,
            avg_net_edge=nan, total_costs=0.0, capital_base=0.0, n_days=0,
            gross_sharpe=nan, gross_return=nan,
            implied_threshold=self.implied_threshold,
            realized_threshold=self.realized_threshold,
        )


# --------------------------------------------------------------------------- #
# Self-contained demo (mock data; out-of-sample split; matplotlib not required)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from alpha_engine import PolymarketDataPuller

    puller = PolymarketDataPuller()
    # 5000 markets resolving across ~1.5 years, with a real favorite-longshot bias.
    markets = puller.generate_mock_markets(
        n_markets=5000, bias_strength=1.40, n_obs=24, noise=0.13,
        span_days=540, seed=20,
    )

    # Out-of-sample split by resolution date: calibrate on the first 60%, trade the
    # last 40% (the gate's "historical realized" never sees the traded outcomes).
    snap = PolymarketDataPuller.snapshot_frame(markets).sort_values("resolved_at")
    cutoff = snap["resolved_at"].quantile(0.60)
    hist_markets = [m for m in markets if m.resolved_at <= cutoff]
    trade_markets = [m for m in markets if m.resolved_at > cutoff]
    history = PolymarketDataPuller.panel_frame(hist_markets)     # time-series calibration
    trades = PolymarketDataPuller.snapshot_frame(trade_markets)  # one fade decision / market

    print("=== FaderSignalEngine: fade over-priced longshots (out-of-sample) ===")
    print(f"calibrate on {len(hist_markets)} markets resolving <= {cutoff.date()}; "
          f"trade {len(trade_markets)} resolving after\n")

    fader = FaderSignalEngine(
        implied_threshold=0.10, realized_threshold=0.05,
        base_half_spread=0.005, extreme_mult=1.0, impact_coef=0.003, fee_rate=0.02,
    )
    fader.fit_calibration(history)
    result = fader.backtest(trades)

    print("Strategy tear sheet (net of costs):")
    for k, v in result.summary().items():
        print(f"  {k:>20}: {v}")

    # Where do fades come from, and what does each cost?
    f = result.trades
    print(f"\nFades fired in price deciles: {sorted(f['decile'].unique())}  "
          f"(all below implied_threshold={fader.implied_threshold})")
    print(f"Mean entry price faded     : {f['entry_price'].mean():.3f}")
    print(f"Gross edge / contract      : {result.avg_gross_edge:+.4f}")
    print(f"Cost   / contract          : {result.avg_cost:.4f}  "
          f"({result.avg_cost / result.avg_gross_edge:.0%} of gross edge)")
    print(f"Net edge / contract        : {result.avg_net_edge:+.4f}")
    print(f"\nGross Sharpe {result.gross_sharpe:.2f} -> net Sharpe {result.sharpe:.2f}   "
          f"|   gross return {result.gross_return:+.1%} -> net {result.total_return:+.1%}")
    print(f"(costs at the thin extremes consume "
          f"{1 - result.avg_net_edge / result.avg_gross_edge:.0%} of the raw edge)")

    # --- Verification ---
    ok = (
        result.avg_gross_edge > 0           # the structural bias is real
        and result.avg_cost > 0             # costs were actually deducted
        and result.avg_net_edge < result.avg_gross_edge  # costs reduce the edge
        and result.hit_rate > 0.85          # fading longshots wins often (neg. skew)
        and result.gross_sharpe >= result.sharpe         # costs can only hurt Sharpe
    )
    surv = "survives" if result.avg_net_edge > 0 else "is ERASED by"
    print(f"\n[{'OK' if ok else 'UNEXPECTED'}] the alpha {surv} costs; "
          f"hit rate {result.hit_rate:.0%}, net Sharpe {result.sharpe:.2f}, "
          f"max drawdown {result.max_drawdown:.1%}.")


if __name__ == "__main__":
    _demo()
