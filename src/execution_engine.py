"""
execution_engine.py
====================

Realistic execution layer for prediction-market stat-arb.

Prediction markets (Kalshi, Polymarket) have no native short. To "short" a YES
contract you **buy the complementary NO contract**, which costs ``1 - YES_price``
and locks that cash as collateral until the event resolves. So a theoretical
spread position becomes two *long* contract purchases:

* **Long Spread**  -> Buy Kalshi **YES** + Buy Polymarket **NO**
* **Short Spread** -> Buy Kalshi **NO**  + Buy Polymarket **YES**

The arbitrage identity (the key to the whole model)
---------------------------------------------------
Both venues settle the *same* event, so a Long-Spread pair (Kalshi YES + Poly NO)
pays **exactly $1 at resolution regardless of outcome** -- whichever way the event
resolves, exactly one of the two legs pays $1. The same holds for Short Spread
(Kalshi NO + Poly YES). Therefore, held to resolution, the P&L is *fixed at entry*:

    cost(Long)  = p_kalshi + (1 - p_poly) + fees = 1 + spread + fees
    cost(Short) = (1 - p_kalshi) + p_poly + fees = 1 - spread + fees
    payoff      = $1   (guaranteed)
    pnl(Long)   = 1 - cost = -spread - fees
    pnl(Short)  = 1 - cost = +spread - fees      (spread = p_kalshi - p_poly)

i.e. each position locks in ``|raw spread| - fees`` *if entered on the correct
side*. This is a static convergence arbitrage, **not** a mark-to-market
mean-reversion trade -- the realized P&L depends on the **raw entry spread**, not
on any subsequent z-score reversion.

Capital & exits
---------------
* ``capital_tracker``: each entry deducts its **fully-collateralized cost** from
  ``available_capital``. When ``available_capital`` can no longer collateralize a
  new position, **new entries are blocked**. Because positions cannot be closed
  early (below), capital only frees up at resolution -- so it genuinely runs out.
* **No early close** by selling to the book *unless* the dataset carries bid/ask
  columns to price the slippage of crossing the spread. Otherwise the cash is
  **locked until the resolution date**, where each open position pays $1.

This is the honest counterpart to :class:`backtester.VectorizedBacktester`, whose
mark-to-market "exit at z=0" assumes a frictionless sale that this venue cannot
actually provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline import MarketDataPipeline

__all__ = ["ExecutionEngine", "ExecutionResult"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL


@dataclass
class ExecutionResult:
    """Outcome of an execution run."""

    trades: pd.DataFrame            # one row per opened position
    capital_curve: pd.Series        # available_capital after each event + resolution
    initial_capital: float
    final_capital: float
    realized_pnl: float
    n_opened: int
    n_blocked: int
    n_closed_early: int
    n_held_to_resolution: int
    resolution_date: object
    peak_capital_deployed: float    # max collateral locked at once

    def summary(self) -> dict:
        return {
            "initial_capital": round(self.initial_capital, 4),
            "final_capital": round(self.final_capital, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "return_on_capital": round(self.realized_pnl / self.initial_capital, 4),
            "positions_opened": self.n_opened,
            "entries_blocked": self.n_blocked,
            "closed_early": self.n_closed_early,
            "held_to_resolution": self.n_held_to_resolution,
            "peak_capital_deployed": round(self.peak_capital_deployed, 4),
        }


class ExecutionEngine:
    """Map spread signals to collateralized prediction-market positions.

    Parameters
    ----------
    initial_capital:
        Starting cash. Each position locks ~$1 of collateral, so this caps the
        number of simultaneous positions.
    kalshi_fee, poly_slippage:
        Per-transaction fee rates (fraction of each leg's traded price), charged on
        entry (and on early-close fills, if any).
    resolution_date:
        When the event settles and open positions pay $1. Defaults to the last
        timestamp in the signal frame.
    bid_ask:
        Optional dict mapping ``{"kalshi_bid","kalshi_ask","polymarket_bid",
        "polymarket_ask"}`` to column names (YES-side quotes). If supplied and
        present, EXIT/STOP signals close positions at the bid (crossing the spread,
        with explicit slippage). If absent, positions are held to resolution.
    """

    DIRECTION_LABELS = {1: "Long Spread", -1: "Short Spread"}

    def __init__(
        self,
        initial_capital: float = 100.0,
        kalshi_fee: float = 0.01,
        poly_slippage: float = 0.005,
        resolution_date=None,
        kalshi_col: str = _K,
        poly_col: str = _P,
        bid_ask: Optional[dict] = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.kalshi_fee = float(kalshi_fee)
        self.poly_slippage = float(poly_slippage)
        self.resolution_date = resolution_date
        self.kalshi_col = kalshi_col
        self.poly_col = poly_col
        self.bid_ask = bid_ask

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, signals: pd.DataFrame) -> ExecutionResult:
        """Execute SignalEngine signals under prediction-market constraints."""
        self._validate(signals)
        res_date = (pd.Timestamp(self.resolution_date)
                    if self.resolution_date is not None else signals.index[-1])
        can_close = self._has_bid_ask(signals)

        available = self.initial_capital
        open_positions: list[dict] = []
        trade_log: list[dict] = []
        n_blocked = n_closed = 0
        cap_points = [(signals.index[0], available)]
        peak_deployed = 0.0

        events = signals[signals["trade_event"].astype(str).str.len() > 0]
        for ts, row in events.iterrows():
            event = str(row["trade_event"])

            if event.startswith("ENTER") or event.startswith("REVERSE"):
                direction = int(row["position"])
                if direction == 0:
                    continue
                leg = self._legs_and_cost(direction, row[self.kalshi_col], row[self.poly_col])
                # capital_tracker: only open if the position can be FULLY collateralized.
                if available >= leg["cost"]:
                    available -= leg["cost"]
                    pos = {
                        "entry_time": ts,
                        "label": self.DIRECTION_LABELS[direction],
                        "kalshi_leg": leg["kalshi_leg"],
                        "polymarket_leg": leg["poly_leg"],
                        "entry_spread": float(row[self.kalshi_col] - row[self.poly_col]),
                        "cost": leg["cost"],
                        "fees": leg["fees"],
                        "kalshi_side": leg["kalshi_side"],
                        "poly_side": leg["poly_side"],
                        "exit_time": pd.NaT,
                        "payoff": np.nan,
                        "pnl": np.nan,
                        "outcome": "open",
                    }
                    open_positions.append(pos)
                    trade_log.append(pos)
                    peak_deployed = max(peak_deployed, self.initial_capital - available)
                else:
                    n_blocked += 1
                cap_points.append((ts, available))

            elif event in ("EXIT", "STOP LOSS"):
                if can_close and open_positions:
                    pos = open_positions.pop(0)  # FIFO close of the oldest open position
                    proceeds = self._close_proceeds(pos, row)
                    available += proceeds
                    pos.update(exit_time=ts, payoff=proceeds,
                               pnl=proceeds - pos["cost"], outcome="closed_early")
                    n_closed += 1
                    cap_points.append((ts, available))
                # else: cannot sell without bid/ask -> position stays locked to resolution

        # Resolution: every still-open position pays exactly $1.
        held = 0
        for pos in open_positions:
            available += 1.0
            pos.update(exit_time=res_date, payoff=1.0,
                       pnl=1.0 - pos["cost"], outcome="resolved")
            held += 1
        cap_points.append((res_date, available))

        trades = pd.DataFrame(trade_log)
        capital_curve = (pd.Series(dict(cap_points)).sort_index()
                         .rename("available_capital"))
        return ExecutionResult(
            trades=trades,
            capital_curve=capital_curve,
            initial_capital=self.initial_capital,
            final_capital=available,
            realized_pnl=available - self.initial_capital,
            n_opened=len(trade_log),
            n_blocked=n_blocked,
            n_closed_early=n_closed,
            n_held_to_resolution=held,
            resolution_date=res_date,
            peak_capital_deployed=peak_deployed,
        )

    # ------------------------------------------------------------------ #
    # Mechanics
    # ------------------------------------------------------------------ #
    def _legs_and_cost(self, direction: int, p_kalshi: float, p_poly: float) -> dict:
        """Map a spread direction to the two contracts bought and the collateral cost."""
        p_kalshi, p_poly = float(p_kalshi), float(p_poly)
        if direction > 0:  # Long Spread -> Buy Kalshi YES + Buy Polymarket NO
            k_side, k_price = "YES", p_kalshi
            p_side, p_price = "NO", 1.0 - p_poly
        else:              # Short Spread -> Buy Kalshi NO + Buy Polymarket YES
            k_side, k_price = "NO", 1.0 - p_kalshi
            p_side, p_price = "YES", p_poly
        fees = self.kalshi_fee * k_price + self.poly_slippage * p_price
        return {
            "kalshi_side": k_side, "poly_side": p_side,
            "kalshi_leg": f"Buy Kalshi {k_side} @ {k_price:.3f}",
            "poly_leg": f"Buy Polymarket {p_side} @ {p_price:.3f}",
            "fees": fees,
            # Fully-collateralized cost: the two contract prices + entry fees.
            "cost": k_price + p_price + fees,
        }

    def _close_proceeds(self, pos: dict, row: pd.Series) -> float:
        """Cash from selling both legs at the bid (crossing the spread). Bid/ask path."""
        ba = self.bid_ask
        k_bid, k_ask = float(row[ba["kalshi_bid"]]), float(row[ba["kalshi_ask"]])
        p_bid, p_ask = float(row[ba["polymarket_bid"]]), float(row[ba["polymarket_ask"]])
        # A NO contract is sold at NO_bid = 1 - YES_ask (the mirrored book).
        k_sell = k_bid if pos["kalshi_side"] == "YES" else 1.0 - k_ask
        p_sell = p_bid if pos["poly_side"] == "YES" else 1.0 - p_ask
        fees = self.kalshi_fee * k_sell + self.poly_slippage * p_sell
        return k_sell + p_sell - fees

    def _has_bid_ask(self, signals: pd.DataFrame) -> bool:
        if not self.bid_ask:
            return False
        return all(col in signals.columns for col in self.bid_ask.values())

    def _validate(self, signals: pd.DataFrame) -> None:
        required = {"position", "trade_event", self.kalshi_col, self.poly_col}
        missing = required - set(signals.columns)
        if missing:
            raise KeyError(f"ExecutionEngine: signal frame missing columns {sorted(missing)}")
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise TypeError("signals must have a DatetimeIndex")


# --------------------------------------------------------------------------- #
# Self-contained demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from stat_lab import StatLab
    from signal_engine import SignalEngine

    rng = np.random.default_rng(3)
    n = 4000
    idx = pd.date_range("2026-06-01", periods=n, freq="1min", tz="UTC")
    fair = np.clip(0.50 + np.cumsum(rng.normal(0, 0.0010, n)), 0.15, 0.85)
    phi, basis = 0.90, np.empty(n)
    basis[0] = 0.0
    eps = rng.normal(0, 0.012, n)
    for i in range(1, n):
        basis[i] = phi * basis[i - 1] + eps[i]
    kalshi = np.clip(fair + basis / 2.0, 0.02, 0.98)
    poly = np.clip(fair - basis / 2.0, 0.02, 0.98)
    synced = pd.DataFrame({_K: kalshi, _P: poly, "price_spread": np.abs(kalshi - poly)}, index=idx)

    lab = StatLab(synced)
    signals = SignalEngine(entry_z=2.0).run(lab, window=60)
    n_entries = int(signals["trade_event"].str.startswith("ENTER").sum())

    # Capital deliberately small so the lock-up forces blocking.
    eng = ExecutionEngine(initial_capital=15.0, kalshi_fee=0.01, poly_slippage=0.005)
    res = eng.run(signals)

    print("=== Signal mapping -> collateralized positions (no native short) ===")
    cols = ["entry_time", "label", "kalshi_leg", "polymarket_leg", "entry_spread", "cost", "pnl"]
    print(res.trades[cols].head(4).to_string(
        index=False, formatters={"entry_spread": "{:+.3f}".format,
                                  "cost": "{:.3f}".format, "pnl": "{:+.3f}".format}))

    print(f"\n=== capital_tracker (hold-to-resolution, fully collateralized) ===")
    print(f"  entry signals fired      : {n_entries}")
    print(f"  positions opened         : {res.n_opened}")
    print(f"  entries BLOCKED (no cash): {res.n_blocked}")
    print(f"  available_capital        : ${res.initial_capital:.2f} start  ->  "
          f"${res.capital_curve.min():.2f} min (locked)  ->  ${res.final_capital:.2f} post-resolution")
    print(f"  peak capital deployed    : ${res.peak_capital_deployed:.2f} "
          f"({res.peak_capital_deployed / res.initial_capital:.0%} of capital)")
    print(f"  held to resolution       : {res.n_held_to_resolution} positions @ $1 payoff each")
    print(f"  realized P&L             : ${res.realized_pnl:+.3f}  "
          f"({res.realized_pnl / res.initial_capital:+.1%} on capital)")

    # Bid/ask present -> early close becomes legal (with slippage from crossing).
    half = 0.004
    signals_ba = signals.assign(
        k_bid=signals[_K] - half, k_ask=signals[_K] + half,
        p_bid=signals[_P] - half, p_ask=signals[_P] + half,
    )
    eng_ba = ExecutionEngine(initial_capital=15.0, bid_ask={
        "kalshi_bid": "k_bid", "kalshi_ask": "k_ask",
        "polymarket_bid": "p_bid", "polymarket_ask": "p_ask"})
    res_ba = eng_ba.run(signals_ba)
    print(f"\n=== With bid/ask -> early close allowed (slippage priced) ===")
    print(f"  closed early at bid: {res_ba.n_closed_early} | held to resolution: "
          f"{res_ba.n_held_to_resolution} | blocked: {res_ba.n_blocked}")
    print(f"  realized P&L: ${res_ba.realized_pnl:+.3f}  "
          f"(early exit frees capital to recycle, but pays the crossing cost)")


if __name__ == "__main__":
    _demo()
