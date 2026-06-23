"""
backtester.py
========================

Event-study backtester for the prediction-market stat-arb stack. It consumes the
signal frame from :class:`signal_engine.SignalEngine` and produces an equity
curve plus risk/return metrics.

Look-ahead bias (the critical bit)
----------------------------------
The signal at bar ``t`` is computed from bar ``t``'s information (its trailing
z-score). Acting on it within bar ``t`` would trade on a close we could not have
known until the bar ended. We therefore **shift the execution position forward by
one bar** (``position.shift(1)``): the signal decided at ``t-1`` is the position
held through bar ``t``, and the P&L of bar ``t`` is

    pnl_t = position_{t-1} * (spread_t - spread_{t-1}) - cost_t .

Because ``position_{t-1}`` depends only on information through ``t-1``, no future
information can leak into bar ``t``'s P&L. (Verified in the module's tests by
perturbing a signal and confirming no prior P&L changes.)

P&L model
---------
A "spread" position is 1 long Kalshi contract + 1 short Polymarket contract per
unit (a Long Spread gains when ``spread = kalshi - polymarket`` rises). Its P&L
is **additive** in price units: ``position * d(spread)``. The spread is signed and
crosses zero, so multiplicative/percentage returns on it are undefined -- additive
P&L on a fixed notional is the correct model.

Transaction costs are charged on each actual fill (entry, exit, or reversal) at
that bar's leg prices: ``|d position| * (kalshi_fee * P_kalshi + poly_slippage *
P_poly)``. An entry+exit therefore pays the round trip; a reversal pays for two
contracts.

Equity, returns, and metrics
----------------------------
* ``equity_t = initial_capital + cumsum(net_pnl)``  (additive; no profit
  compounding, matching fixed-notional sizing).
* **Daily returns**: the equity curve is additive P&L and can cross zero, so a
  ``pct_change`` of its running level is invalid -- once equity is negative a
  further loss reads as a positive "return" and corrupts the Sharpe. The correct
  daily return for a fixed-notional book is the day's P&L as a fraction of the
  (fixed, positive) committed capital, which also makes the Sharpe invariant to
  ``initial_capital``.
* **Annualized Sharpe** (rf = 0%): ``mean(daily) / std(daily) * sqrt(252)``.
* **Max drawdown**: track the running maximum of the equity curve and take the
  largest percentage drop, ``min(equity / cummax(equity) - 1)``.

Sharpe is (to first order) invariant to ``initial_capital``; the percentage
drawdown scales with it, so set ``initial_capital`` to your real per-lot
capital-at-risk (~``P_kalshi + (1 - P_poly)`` ~ 1.0 for binary contracts) for a
calibrated drawdown figure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_pipeline import MarketDataPipeline

__all__ = ["VectorizedBacktester", "BacktestResult"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL


@dataclass
class BacktestResult:
    """Container for backtest output and headline metrics."""

    sharpe: float
    max_drawdown: float            # negative fraction, e.g. -0.18 = -18%
    total_return: float
    total_pnl: float
    annualized_return: float
    annualized_vol: float
    n_trades: int
    total_costs: float
    n_days: int
    initial_capital: float
    peak_date: object = None
    trough_date: object = None
    equity_curve: pd.Series = field(default=None, repr=False)
    daily_returns: pd.Series = field(default=None, repr=False)
    drawdown_curve: pd.Series = field(default=None, repr=False)
    frame: pd.DataFrame = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "sharpe_annualized": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_return": round(self.total_return, 4),
            "total_pnl": round(self.total_pnl, 6),
            "annualized_return": round(self.annualized_return, 4),
            "annualized_vol": round(self.annualized_vol, 4),
            "n_trades": self.n_trades,
            "total_costs": round(self.total_costs, 6),
            "n_days": self.n_days,
        }


class VectorizedBacktester:
    """Vectorized backtest of SignalEngine signals into an equity curve + metrics.

    Parameters
    ----------
    initial_capital:
        Starting NAV / per-lot capital base. Default 1.0.
    kalshi_fee, poly_slippage:
        Per-transaction cost rates (fractions of each leg's price). Should match
        the SignalEngine that produced the signals. Defaults 0.01 / 0.005.
    trading_days_per_year:
        Annualization factor for the Sharpe ratio. Default 252.
    risk_free_rate:
        Annual risk-free rate. Default 0.0.
    """

    def __init__(
        self,
        initial_capital: float = 1.0,
        kalshi_fee: float = 0.01,
        poly_slippage: float = 0.005,
        trading_days_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.kalshi_fee = float(kalshi_fee)
        self.poly_slippage = float(poly_slippage)
        self.trading_days_per_year = int(trading_days_per_year)
        self.risk_free_rate = float(risk_free_rate)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        signals: pd.DataFrame,
        *,
        spread_col: str = "spread",
        position_col: str = "position",
        kalshi_col: str = _K,
        poly_col: str = _P,
    ) -> BacktestResult:
        """Backtest a SignalEngine signal frame and return metrics + curves."""
        required = [spread_col, position_col, kalshi_col, poly_col]
        missing = [c for c in required if c not in signals.columns]
        if missing:
            raise KeyError(f"VectorizedBacktester.run: missing columns {missing}")

        spread = signals[spread_col].astype(float)
        position = signals[position_col].astype(float)
        p_kalshi = signals[kalshi_col].abs()
        p_poly = signals[poly_col].abs()

        # --- CRITICAL: shift execution forward one bar to remove look-ahead. ---
        # The signal decided at t-1 is the position held through bar t.
        exec_position = position.shift(1).fillna(0.0)

        # Per-bar P&L = held position * change in spread (additive, price units).
        spread_change = spread.diff()
        gross_pnl = exec_position * spread_change

        # Costs on each actual fill, at that bar's leg prices.
        trade_size = exec_position.diff().abs().fillna(exec_position.abs())
        cost = trade_size * (self.kalshi_fee * p_kalshi + self.poly_slippage * p_poly)

        net_pnl = (gross_pnl - cost).fillna(0.0)
        equity = self.initial_capital + net_pnl.cumsum()
        if equity.min() <= 0:
            warnings.warn(
                "Equity crossed zero: the additive book lost more than "
                "initial_capital (a blow-up). Drawdowns beyond -100% mean the "
                "position would have been wiped out / margin-called in practice.",
                stacklevel=2,
            )

        # --- Daily returns from the equity curve ---
        # Additive P&L can cross zero, so pct_change() of the running level is
        # invalid; use the day's P&L over the FIXED committed capital instead.
        daily_pnl = net_pnl.resample("1D").sum()
        daily_returns = daily_pnl / self.initial_capital
        daily_equity = equity.resample("1D").last()

        # --- Metrics ---
        sharpe = self.annualized_sharpe(
            daily_returns, self.trading_days_per_year, self.risk_free_rate
        )
        drawdown_curve = equity / equity.cummax() - 1.0
        max_dd = float(drawdown_curve.min()) if len(drawdown_curve) else float("nan")
        trough_date = drawdown_curve.idxmin() if len(drawdown_curve) else None
        peak_date = (
            equity.loc[:trough_date].idxmax() if trough_date is not None else None
        )

        n_trades = int((trade_size > 0).sum())
        total_costs = float(cost.sum())
        total_pnl = float(net_pnl.sum())
        total_return = float(equity.iloc[-1] / self.initial_capital - 1.0)
        n_days = int(len(daily_returns))
        ann_vol = (
            float(daily_returns.std(ddof=1) * np.sqrt(self.trading_days_per_year))
            if len(daily_returns) > 1
            else float("nan")
        )
        ann_return = (
            float(daily_returns.mean() * self.trading_days_per_year)
            if len(daily_returns)
            else float("nan")
        )

        frame = pd.DataFrame(
            {
                "exec_position": exec_position,
                spread_col: spread,
                "spread_change": spread_change,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "equity": equity,
                "drawdown": drawdown_curve,
            }
        )

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            total_pnl=total_pnl,
            annualized_return=ann_return,
            annualized_vol=ann_vol,
            n_trades=n_trades,
            total_costs=total_costs,
            n_days=n_days,
            initial_capital=self.initial_capital,
            peak_date=peak_date,
            trough_date=trough_date,
            equity_curve=equity,
            daily_returns=daily_returns,
            drawdown_curve=drawdown_curve,
            frame=frame,
        )

    # ------------------------------------------------------------------ #
    # Reusable, independently-testable metric primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def annualized_sharpe(
        daily_returns,
        trading_days_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Annualized Sharpe of a daily-return series. rf is an annual rate."""
        r = pd.Series(daily_returns).dropna()
        if len(r) < 2:
            return float("nan")
        excess = r - risk_free_rate / trading_days_per_year
        sd = excess.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return float("nan")
        return float(excess.mean() / sd * np.sqrt(trading_days_per_year))

    @staticmethod
    def max_drawdown(equity_curve) -> float:
        """Largest percentage drop from a running peak of the equity curve.

        Returns a non-positive fraction (e.g. -0.2 for a 20% drawdown).
        """
        eq = pd.Series(equity_curve).dropna()
        if eq.empty:
            return float("nan")
        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        return float(drawdown.min())


# --------------------------------------------------------------------------- #
# Self-contained demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from stat_lab import StatLab
    from signal_engine import SignalEngine

    rng = np.random.default_rng(8)
    days = 30
    n = days * 1440  # 1-minute bars, ~24/7 prediction markets
    idx = pd.date_range("2026-05-01", periods=n, freq="1min", tz="UTC")

    fair = np.clip(0.50 + np.cumsum(rng.normal(0, 0.0008, n)), 0.10, 0.90)
    # FAST reversion (phi=0.90, half-life ~6.6 min, well inside the 60-min z window)
    # so the rolling mean is stable and entries revert cleanly; amplitude
    # (sigma~2.7c, 2-sigma entry ~5.5c) comfortably clears the ~1.5c round-trip fee.
    phi = 0.90
    basis = np.empty(n)
    basis[0] = 0.0
    shock = rng.normal(0, 0.012, n)
    jumps = rng.random(n) < 0.0025
    shock[jumps] += rng.normal(0, 0.05, int(jumps.sum()))  # rare big shocks -> stops
    for i in range(1, n):
        basis[i] = phi * basis[i - 1] + shock[i]
    kalshi = np.clip(fair + basis / 2.0, 0.01, 0.99)
    poly = np.clip(fair - basis / 2.0, 0.01, 0.99)
    lab = StatLab(pd.DataFrame({_K: kalshi, _P: poly}, index=idx).assign(
        price_spread=np.abs(kalshi - poly)))

    bt = VectorizedBacktester(initial_capital=1.0, kalshi_fee=0.01, poly_slippage=0.005)

    print(f"=== Backtest over {days} days of 1-min data ===\n")
    print(f"{'mode':>16} | {'Sharpe':>7} | {'MaxDD':>8} | {'TotRet':>8} | "
          f"{'trades':>6} | {'costs':>7}")
    print("-" * 70)
    results = {}
    for mode, kw in (("indefinite_hold", {}), ("stop_loss", {"stop_z_score": 3.5})):
        sig = SignalEngine(risk_mode=mode, entry_z=2.0, **kw).run(lab, window=60)
        res = bt.run(sig)
        results[mode] = res
        print(f"{mode:>16} | {res.sharpe:>7.2f} | {res.max_drawdown:>7.2%} | "
              f"{res.total_return:>7.2%} | {res.n_trades:>6} | {res.total_costs:>7.4f}")

    res = results["stop_loss"]
    print(f"\n=== stop_loss detail ===")
    for k, v in res.summary().items():
        print(f"  {k:>20}: {v}")
    print(f"  max drawdown peak->trough: {res.peak_date}  ->  {res.trough_date}")

    # Make the 1% Kalshi fee impact explicit: identical signals, zero costs.
    sig_sl = SignalEngine(risk_mode="stop_loss", stop_z_score=3.5, entry_z=2.0).run(lab, window=60)
    gross = VectorizedBacktester(kalshi_fee=0.0, poly_slippage=0.0).run(sig_sl)
    print(f"\n  Fee drag (stop_loss): gross {gross.total_return:+.0%} (Sharpe {gross.sharpe:.0f})"
          f"  ->  net {res.total_return:+.0%} (Sharpe {res.sharpe:.0f})")
    print(f"\n  Equity curve (daily, tail):")
    print(res.equity_curve.resample("1D").last().tail(5).round(5).to_string())


if __name__ == "__main__":
    _demo()
