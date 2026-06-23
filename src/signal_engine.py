"""
signal_engine.py
================

Signal generation for the prediction-market stat-arb backtester.

:class:`SignalEngine` turns a rolling Z-score (plus the underlying spread, its
rolling mean, and the two leg prices) into a vectorized trading signal:

* **Long Spread** (buy Kalshi, sell Polymarket) when ``z <= -entry_z`` (default -2).
* **Short Spread** (sell Kalshi, buy Polymarket) when ``z >= +entry_z`` (default +2).
* **Exit** (flat) when the Z-score reverts to the mean (``z`` crosses ``exit_z``,
  default 0). Positions are *held* between entry and exit.

Transaction-friction model
---------------------------
* Kalshi: a fixed **1%** fee (fraction of the Kalshi leg's traded price).
* Polymarket: **0.5%** variable slippage (fraction of the Polymarket leg's price).
* A trade is a *round trip* (open + close), so each leg is charged twice:
  ``roundtrip_friction = 2 * (kalshi_fee * P_kalshi + poly_slippage * P_poly)``.

Risk modes
----------
* ``'indefinite_hold'`` (default): a trade is held until the z-score reverts to
  the mean (``exit_z``, default 0).
* ``'stop_loss'``: in addition to the mean-reversion exit, an *adverse* move that
  pushes the z-score beyond ``stop_z_score`` (e.g. 3.5) force-closes the position.
  After a stop, the strategy is **locked out** -- it cannot re-enter until the
  z-score resets to the mean (0). The lockout is implemented as a vectorized mask
  (no iteration); see :meth:`SignalEngine._apply_stop_loss`.

Profitability filter
--------------------
A position is taken **only if the expected profit from reverting to the moving
average strictly exceeds the combined round-trip friction**:

    expected_profit = |spread - rolling_mean|        (the reversion distance)
    take position  <=>  expected_profit > roundtrip_friction

This is an *ex-ante* entry gate: at the entry bar we estimate the round-trip cost
from the current leg prices (we cannot know the exit prices yet) and compare it to
the profit we would capture if the spread mean-reverts all the way to the rolling
mean. The friction gate is applied only at entry; once open, a position is held
until the z-score reverts.

Units & sizing
--------------
Everything is in the spread's price units (0-1 probability for pmxt-normalized
data). The model assumes a unit position (1 contract per leg), so ``|spread -
mean|`` (dollars per 1-lot) is directly comparable to the per-1-lot friction.
The friction rates are fractions of each leg's *price* (notional = price per
contract), which is the natural reading of "1% fee" / "0.5% slippage".

No look-ahead: the rolling z-score uses a trailing window, so the signal at bar
``t`` is known at ``t``. For a P&L backtest, apply the position to the *next*
bar's return (``position.shift(1)``) to avoid trading on the same close used to
compute the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import MarketDataPipeline

__all__ = ["SignalEngine"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL


class SignalEngine:
    """Generate vectorized, friction-filtered spread-trading signals.

    Parameters
    ----------
    entry_z:
        Absolute z-score entry threshold. Default 2.0.
    exit_z:
        Z-level toward the mean at which to close. Default 0.0 (the mean): a long
        (entered below) exits when ``z >= exit_z``; a short exits when
        ``z <= -exit_z``.
    kalshi_fee:
        Per-transaction Kalshi fee as a fraction of the Kalshi leg price. Default
        0.01 (1%).
    poly_slippage:
        Per-transaction Polymarket slippage as a fraction of the Polymarket leg
        price. Default 0.005 (0.5%).
    include_exit_cost:
        If True (default), charge friction for both the opening and closing trade
        (round trip = 2 transactions per leg). If False, charge one-way only.
    risk_mode:
        ``'indefinite_hold'`` (default) or ``'stop_loss'``.
    stop_z_score:
        Adverse |z| threshold that force-closes a position in ``'stop_loss'`` mode
        (e.g. 3.5). Must exceed ``entry_z``. Ignored in ``'indefinite_hold'``.
    """

    POSITION_LABELS = {1: "Long Spread", -1: "Short Spread", 0: "Flat"}
    VALID_MODES = ("indefinite_hold", "stop_loss")

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        kalshi_fee: float = 0.01,
        poly_slippage: float = 0.005,
        include_exit_cost: bool = True,
        risk_mode: str = "indefinite_hold",
        stop_z_score: float = 3.5,
    ) -> None:
        if entry_z <= 0:
            raise ValueError("entry_z must be positive")
        if abs(exit_z) >= entry_z:
            raise ValueError("exit_z must be strictly inside the entry band")
        if kalshi_fee < 0 or poly_slippage < 0:
            raise ValueError("fees/slippage must be non-negative")
        if risk_mode not in self.VALID_MODES:
            raise ValueError(f"risk_mode must be one of {self.VALID_MODES}")
        if risk_mode == "stop_loss":
            if stop_z_score is None:
                raise ValueError("stop_loss mode requires a stop_z_score")
            if stop_z_score <= entry_z:
                raise ValueError(
                    "stop_z_score must exceed entry_z (the stop sits beyond the entry band)"
                )
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.kalshi_fee = float(kalshi_fee)
        self.poly_slippage = float(poly_slippage)
        self.include_exit_cost = bool(include_exit_cost)
        self.risk_mode = risk_mode
        self.stop_z_score = None if stop_z_score is None else float(stop_z_score)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def generate(
        self,
        df: pd.DataFrame,
        *,
        z_col: str = "zscore",
        spread_col: str = "spread",
        mean_col: str = "rolling_mean",
        kalshi_col: str = _K,
        poly_col: str = _P,
    ) -> pd.DataFrame:
        """Produce the friction-filtered signal frame.

        Parameters
        ----------
        df:
            Frame containing the z-score, the spread, its rolling mean, and the two
            leg prices. Easiest path: join :meth:`StatLab.rolling_zscore` output
            with the synchronized price columns (see :meth:`run`).

        Returns
        -------
        pandas.DataFrame
            Echoes the inputs and adds: ``expected_profit``, ``roundtrip_friction``,
            ``tradeable`` (profit > friction), ``position`` (+1/-1/0),
            ``signal`` (the 'Long Spread'/'Short Spread'/'Flat' label) and
            ``trade_event`` (ENTER/EXIT/REVERSE markers).
        """
        required = [z_col, spread_col, mean_col, kalshi_col, poly_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"SignalEngine.generate: missing columns {missing}")

        z = df[z_col]
        spread = df[spread_col]
        mean = df[mean_col]
        p_kalshi = df[kalshi_col].abs()   # notional per contract = price
        p_poly = df[poly_col].abs()

        # --- Economics (fully vectorized) ---
        expected_profit = (spread - mean).abs()  # distance to revert to the mean
        legs = 2.0 if self.include_exit_cost else 1.0
        roundtrip_friction = legs * (
            self.kalshi_fee * p_kalshi + self.poly_slippage * p_poly
        )
        tradeable = expected_profit > roundtrip_friction  # strict inequality

        # --- Entry / exit events ---
        # Entries are gated by the profitability filter; exits are pure z reversion.
        long_entry = (z <= -self.entry_z) & tradeable
        short_entry = (z >= self.entry_z) & tradeable
        long_exit = z >= self.exit_z          # entered below -> revert up to mean
        short_exit = z <= -self.exit_z         # entered above -> revert down to mean

        base_position = self._build_positions(
            df.index, long_entry, short_entry, long_exit, short_exit
        )

        # Risk overlay.
        if self.risk_mode == "stop_loss":
            position, stop_triggered, locked = self._apply_stop_loss(base_position, z)
        else:
            position = base_position
            stop_triggered = pd.Series(False, index=df.index)
            locked = pd.Series(False, index=df.index)

        signal = position.map(self.POSITION_LABELS)
        trade_event = self._trade_events(position, signal)
        # A stop-loss begins a lockout: label that bar STOP LOSS. This covers both
        # a stop that closes an open position and a gap that blows through entry and
        # stop in a single bar (position never held, but the lockout still starts).
        lock_start = locked & ~locked.shift(1, fill_value=False)
        trade_event = trade_event.mask(lock_start, "STOP LOSS")

        out = pd.DataFrame(
            {
                z_col: z,
                spread_col: spread,
                mean_col: mean,
                kalshi_col: df[kalshi_col],
                poly_col: df[poly_col],
                "expected_profit": expected_profit,
                "roundtrip_friction": roundtrip_friction,
                "tradeable": tradeable,
                "position": position,
                "signal": signal,
                "trade_event": trade_event,
                "stop_triggered": stop_triggered,
                "locked": locked,
            }
        )
        return out

    # ------------------------------------------------------------------ #
    # Vectorized state machine
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_positions(
        index: pd.Index,
        long_entry: pd.Series,
        short_entry: pd.Series,
        long_exit: pd.Series,
        short_exit: pd.Series,
    ) -> pd.Series:
        """Build the held position via two independent forward-filled legs.

        Each leg is a {0,1} state: set to 1 at an entry, 0 at an exit, NaN
        otherwise, then forward-filled so the position persists between events.
        Entry and exit for a given leg are mutually exclusive (|z|>=entry vs the
        zero line), so no within-leg conflict. position = long_leg - short_leg,
        which is provably in {-1, 0, +1}.
        """
        long_leg = pd.Series(np.nan, index=index)
        long_leg[long_entry] = 1.0
        long_leg[long_exit] = 0.0
        long_leg = long_leg.ffill().fillna(0.0)

        short_leg = pd.Series(np.nan, index=index)
        short_leg[short_entry] = 1.0
        short_leg[short_exit] = 0.0
        short_leg = short_leg.ffill().fillna(0.0)

        return (long_leg - short_leg).astype(int)

    def _apply_stop_loss(self, base_position: pd.Series, z: pd.Series):
        """Force the position flat on an adverse stop and lock out re-entry until
        the z-score resets to the mean. Fully vectorized.

        Why two forward-filled masks suffice (no iteration): the base
        (indefinite-hold) position is a single continuous hold within each
        z-excursion -- it enters once and only exits when z crosses 0. Both the
        normal exit and the lockout reset occur at that same zero crossing, so the
        lockout can never span excursions. Within an excursion the only thing the
        lockout must prevent is re-entry *after a stop*, which the ffill captures.

        Returns
        -------
        (position, stop_triggered, locked) : Series, Series[bool], Series[bool]
        """
        sz = self.stop_z_score
        # A stop fires only against an OPEN position that moves further against it.
        # While long, z is negative; while short, z is positive -- so "exceeds the
        # stop" is directional. Strict '>' to match "exceeds the threshold".
        long_stop = (base_position == 1) & (z < -sz)
        short_stop = (base_position == -1) & (z > sz)
        stop_triggered = long_stop | short_stop

        # Long lockout: from a long stop until z climbs back to the mean (z >= 0).
        long_lock = pd.Series(np.nan, index=z.index)
        long_lock[long_stop] = 1.0
        long_lock[z >= 0.0] = 0.0
        long_locked = long_lock.ffill().fillna(0.0) > 0.0

        # Short lockout: from a short stop until z falls back to the mean (z <= 0).
        short_lock = pd.Series(np.nan, index=z.index)
        short_lock[short_stop] = 1.0
        short_lock[z <= 0.0] = 0.0
        short_locked = short_lock.ffill().fillna(0.0) > 0.0

        locked = long_locked | short_locked
        position = base_position.where(~locked, 0).astype(int)
        return position, stop_triggered, locked

    @staticmethod
    def _trade_events(position: pd.Series, signal: pd.Series) -> pd.Series:
        """Label the bar of each position change (vectorized)."""
        prev = position.shift(1).fillna(0).astype(int)
        changed = position != prev
        entered = changed & (prev == 0) & (position != 0)
        exited = changed & (position == 0)
        reversed_ = changed & (prev != 0) & (position != 0)

        event = pd.Series("", index=position.index, dtype=object)
        event[entered] = "ENTER " + signal[entered]
        event[exited] = "EXIT"
        event[reversed_] = "REVERSE " + signal[reversed_]
        return event

    # ------------------------------------------------------------------ #
    # Convenience: end-to-end from a StatLab instance
    # ------------------------------------------------------------------ #
    def run(self, statlab, window: int = 60, use_signed: bool = True) -> pd.DataFrame:
        """Build features from a :class:`StatLab` and generate signals end-to-end.

        Uses the *signed* spread by default (recommended for directional signals;
        ``price_spread`` is absolute and loses direction -- see StatLab docs).
        """
        spread = statlab.signed_spread if use_signed else None
        zf = statlab.rolling_zscore(window=window, spread=spread)
        prices = statlab.df[[statlab.kalshi_col, statlab.polymarket_col]]
        feat = zf.join(prices, how="left")
        return self.generate(
            feat, kalshi_col=statlab.kalshi_col, poly_col=statlab.polymarket_col
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @staticmethod
    def trade_summary(signals: pd.DataFrame, entry_z: float = 2.0) -> dict:
        """Summary stats, including how many entries the friction filter blocked."""
        ev = signals["trade_event"]
        z = signals["zscore"]
        # Bars where z breached the band but the trade was NOT taken because of friction.
        breached = z.abs() >= entry_z
        blocked = int((breached & ~signals["tradeable"]).sum())
        return {
            "n_bars": int(len(signals)),
            "entries_long": int(ev.str.startswith("ENTER Long").sum()),
            "entries_short": int(ev.str.startswith("ENTER Short").sum()),
            "exits": int((ev == "EXIT").sum()),
            "stop_losses": int((ev == "STOP LOSS").sum()),
            "reverses": int(ev.str.startswith("REVERSE").sum()),
            "bars_long": int((signals["position"] == 1).sum()),
            "bars_short": int((signals["position"] == -1).sum()),
            "bars_flat": int((signals["position"] == 0).sum()),
            "friction_blocked_entries": blocked,
        }


# --------------------------------------------------------------------------- #
# Self-contained demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from stat_lab import StatLab

    rng = np.random.default_rng(5)
    n = 6000
    idx = pd.date_range("2026-06-22 09:00", periods=n, freq="1min", tz="UTC")

    fair = np.clip(0.50 + np.cumsum(rng.normal(0, 0.0012, n)), 0.10, 0.90)

    # Persistent mean-reverting basis (phi=0.96) plus rare fat-tailed jumps, so
    # some excursions blow past the stop band (|z| > 3.5) -- letting us contrast
    # the two risk modes on the same data.
    phi = 0.96
    basis = np.empty(n)
    basis[0] = 0.0
    shock = rng.normal(0, 0.0030, n)
    jumps = rng.random(n) < 0.012
    shock[jumps] += rng.normal(0, 0.025, int(jumps.sum()))
    for i in range(1, n):
        basis[i] = phi * basis[i - 1] + shock[i]

    kalshi = np.clip(fair + basis / 2.0, 0.01, 0.99)
    poly = np.clip(fair - basis / 2.0, 0.01, 0.99)
    synced = pd.DataFrame({_K: kalshi, _P: poly}, index=idx)
    lab = StatLab(synced.assign(price_spread=np.abs(kalshi - poly)))

    common = dict(entry_z=2.0, exit_z=0.0, kalshi_fee=0.01, poly_slippage=0.005)
    hold = SignalEngine(risk_mode="indefinite_hold", **common).run(lab, window=60)
    stop = SignalEngine(risk_mode="stop_loss", stop_z_score=3.5, **common).run(lab, window=60)

    print("=== Mode comparison (same data, entry +/-2, stop 3.5) ===")
    for name, sig in (("indefinite_hold", hold), ("stop_loss", stop)):
        s = SignalEngine.trade_summary(sig, entry_z=2.0)
        print(f"  [{name:>15}] entries={s['entries_long']+s['entries_short']:>3}  "
              f"exits={s['exits']:>3}  stop_losses={s['stop_losses']:>3}  "
              f"bars_in_market={s['bars_long']+s['bars_short']:>4}")

    print(f"\n  Worst |z| while holding a position:")
    print(f"    indefinite_hold: {hold.loc[hold['position'] != 0, 'zscore'].abs().max():.2f}")
    print(f"    stop_loss      : {stop.loc[stop['position'] != 0, 'zscore'].abs().max():.2f}"
          f"   (capped near the 3.5 stop)")

    # Show a stop-loss episode: the stop firing and the lockout that follows.
    stop_bars = stop.index[stop["stop_triggered"]]
    if len(stop_bars):
        t0 = stop.index.get_loc(stop_bars[0])
        window = stop.iloc[max(0, t0 - 2): t0 + 8]
        print(f"\n=== A stop-loss episode (stop at {stop_bars[0]}) ===")
        cols = ["zscore", "position", "signal", "stop_triggered", "locked", "trade_event"]
        print(window[cols].round(3).to_string())
        print("  -> position forced to 0 at the stop; 'locked' stays True (no re-entry)")
        print("     until z resets to 0, even if z dips back below -2 / above +2.")


if __name__ == "__main__":
    _demo()
