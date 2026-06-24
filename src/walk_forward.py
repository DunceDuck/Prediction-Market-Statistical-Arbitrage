"""
walk_forward.py
===============

Walk-forward parameter optimization for the prediction-market stat-arb strategy.

Hardcoding the rolling-window size and the z-score entry threshold over-fits a
single regime. :class:`WalkForwardOptimizer` instead **re-chooses them on every
step**, using only past data, and applies them to the next unseen day:

1. **Split** the synchronized frame into an *expanding* in-sample training window
   (>= ``train_days``, default 30) and a ``test_days``-day (default 1)
   out-of-sample step.
2. **Per training window**, estimate the Ornstein-Uhlenbeck **half-life** of the
   spread (the natural mean-reversion timescale) and build a rolling-window grid
   from it (multiples of the half-life clipped to ``[min_window, max_window]``).
   Grid-search ``(rolling_window x entry_z)`` to **maximize the friction-adjusted
   Sharpe ratio in-sample** (the net-of-fees Sharpe from a full mini-backtest).
3. **Store** the winning ``(window*, entry_z*)`` and apply them *strictly* to the
   1-day out-of-sample step.
4. **Return** the concatenated, continuous out-of-sample signal series plus a
   per-fold parameter log.

No look-ahead: parameter selection for a fold sees only data strictly before its
out-of-sample day. The out-of-sample signal is generated over
``[anchor, oos_end]`` so the rolling z-score's trailing window is valid at the
day's open, but every value at time ``t`` uses information only up to ``t``.

Because it reuses :class:`StatLab`, :class:`SignalEngine`, and
:class:`VectorizedBacktester`, the optimizer's objective is exactly the same
friction model the live strategy trades under.
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

__all__ = ["WalkForwardOptimizer", "WalkForwardResult"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL
_S = MarketDataPipeline.SPREAD_COL


@dataclass
class WalkForwardResult:
    """Output of a walk-forward run."""

    oos_signals: pd.DataFrame      # continuous out-of-sample signal frame
    params: pd.DataFrame           # per-fold optimal parameters + in-sample score

    @property
    def signal_series(self) -> pd.Series:
        """The continuous out-of-sample position series (+1/-1/0)."""
        return self.oos_signals["position"]

    def summary(self) -> dict:
        p = self.params
        return {
            "folds": int(len(p)),
            "oos_start": self.oos_signals.index.min(),
            "oos_end": self.oos_signals.index.max(),
            "oos_minutes": int(len(self.oos_signals)),
            "entries": int(self.oos_signals["trade_event"].str.startswith("ENTER").sum()),
            "window_min_max": (int(p["best_window"].min()), int(p["best_window"].max())),
            "entry_z_min_max": (float(p["best_entry_z"].min()), float(p["best_entry_z"].max())),
            "mean_insample_sharpe": float(p["insample_sharpe"].replace([-np.inf], np.nan).mean()),
        }


class WalkForwardOptimizer:
    """Expanding-window walk-forward optimizer for ``(rolling_window, entry_z)``.

    Parameters
    ----------
    train_days:
        Minimum in-sample length (days). Default 30.
    test_days:
        Out-of-sample step length (days). Default 1.
    min_window, max_window:
        Bounds for the rolling-window search (periods). Default 10 / 120.
    window_multipliers:
        Multiples of the OU half-life used to form window candidates (each clipped
        to ``[min_window, max_window]``). Default ``(1, 2, 3, 4)``.
    z_grid:
        Z-score entry thresholds to search. Default ``(1.5, 2.0, 2.5, 3.0)``.
    risk_mode, stop_z_score:
        Passed through to :class:`SignalEngine`. Default ``"indefinite_hold"`` / 3.5.
    kalshi_fee, poly_slippage:
        Per-transaction friction (so the objective is *friction-adjusted*).
    expanding:
        If ``True`` (default) the training window is anchored at the data start and
        grows; if ``False`` it is a fixed ``train_days``-day rolling window.
    """

    def __init__(
        self,
        train_days: int = 30,
        test_days: int = 1,
        min_window: int = 10,
        max_window: int = 120,
        window_multipliers=(1.0, 2.0, 3.0, 4.0),
        z_grid=(1.5, 2.0, 2.5, 3.0),
        risk_mode: str = "indefinite_hold",
        stop_z_score: float = 3.5,
        kalshi_fee: float = 0.01,
        poly_slippage: float = 0.005,
        expanding: bool = True,
        kalshi_col: str = _K,
        poly_col: str = _P,
        spread_col: str = _S,
    ) -> None:
        if train_days < 1 or test_days < 1:
            raise ValueError("train_days and test_days must be >= 1")
        if not (1 <= min_window <= max_window):
            raise ValueError("require 1 <= min_window <= max_window")
        if len(z_grid) == 0 or len(window_multipliers) == 0:
            raise ValueError("z_grid and window_multipliers must be non-empty")
        self.train_days = train_days
        self.test_days = test_days
        self.min_window = min_window
        self.max_window = max_window
        self.window_multipliers = tuple(float(m) for m in window_multipliers)
        self.z_grid = tuple(float(z) for z in z_grid)
        self.risk_mode = risk_mode
        self.stop_z_score = stop_z_score
        self.kalshi_fee = kalshi_fee
        self.poly_slippage = poly_slippage
        self.expanding = expanding
        self.kalshi_col = kalshi_col
        self.poly_col = poly_col
        self.spread_col = spread_col

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def split(self, synced: pd.DataFrame):
        """Yield ``(train_df, oos_df)`` per fold (expanding train, ``test_days`` OOS)."""
        self._validate(synced)
        days = synced.index.normalize()
        unique_days = pd.DatetimeIndex(np.sort(days.unique()))
        if len(unique_days) <= self.train_days:
            raise ValueError(
                f"Need more than train_days={self.train_days} calendar days of data; "
                f"got {len(unique_days)}."
            )
        for i in range(self.train_days, len(unique_days), self.test_days):
            oos_days = unique_days[i: i + self.test_days]
            oos_start, oos_end = oos_days[0], oos_days[-1]
            if self.expanding:
                train_mask = days < oos_start
            else:
                train_lo = unique_days[i - self.train_days]
                train_mask = (days >= train_lo) & (days < oos_start)
            oos_mask = (days >= oos_start) & (days <= oos_end)
            train_df, oos_df = synced.loc[train_mask], synced.loc[oos_mask]
            if len(train_df) and len(oos_df):
                yield train_df, oos_df

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, synced: pd.DataFrame) -> WalkForwardResult:
        """Run the full walk-forward and return the continuous OOS signal + params."""
        self._validate(synced)
        oos_parts, rows = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # grid search intentionally probes many configs
            for train_df, oos_df in self.split(synced):
                anchor = train_df.index[0]
                lab = StatLab(train_df, kalshi_col=self.kalshi_col,
                              polymarket_col=self.poly_col, spread_col=self.spread_col)
                window, entry_z, in_sharpe, half_life = self._optimize(lab, train_df)
                oos_sig = self._oos_signals(synced, anchor, oos_df, window, entry_z)
                oos_parts.append(oos_sig)
                rows.append({
                    "oos_date": oos_df.index[0].normalize(),
                    "train_start": anchor,
                    "train_end": train_df.index[-1],
                    "n_train_days": int(train_df.index.normalize().nunique()),
                    "half_life": half_life,
                    "best_window": window,
                    "best_entry_z": entry_z,
                    "insample_sharpe": in_sharpe,
                })

        if not oos_parts:
            raise ValueError("No out-of-sample folds produced; check data length.")
        oos_signals = pd.concat(oos_parts)
        oos_signals = oos_signals[~oos_signals.index.duplicated(keep="last")].sort_index()
        params = pd.DataFrame(rows).set_index("oos_date")
        return WalkForwardResult(oos_signals=oos_signals, params=params)

    # ------------------------------------------------------------------ #
    # In-sample optimization (grid search on friction-adjusted Sharpe)
    # ------------------------------------------------------------------ #
    def _optimize(self, lab: StatLab, train_df: pd.DataFrame):
        """Return ``(window*, entry_z*, in_sample_sharpe, half_life)``."""
        spread = lab.signed_spread
        half_life = self._half_life(lab, spread)
        windows = self._window_grid(half_life)
        prices = train_df[[self.kalshi_col, self.poly_col]]

        best_sharpe, best_w, best_z = -np.inf, None, None
        for w in windows:
            zframe = lab.rolling_zscore(window=w, spread=spread)
            feat = zframe.join(prices, how="left")
            for z in self.z_grid:
                sharpe = self._friction_adjusted_sharpe(feat, z)
                if sharpe > best_sharpe:
                    best_sharpe, best_w, best_z = sharpe, w, z

        if best_w is None:  # no config produced a finite Sharpe -> principled fallback
            fallback = half_life if np.isfinite(half_life) else (self.min_window + self.max_window) / 2
            best_w = int(np.clip(round(fallback), self.min_window, self.max_window))
            best_z = float(np.median(self.z_grid))
            best_sharpe = np.nan
        return int(best_w), float(best_z), float(best_sharpe), float(half_life)

    def _friction_adjusted_sharpe(self, feat: pd.DataFrame, entry_z: float) -> float:
        """Net-of-fees annualized Sharpe of one (window-implied feat, entry_z) config."""
        sig = SignalEngine(
            entry_z=entry_z, risk_mode=self.risk_mode, stop_z_score=self.stop_z_score,
            kalshi_fee=self.kalshi_fee, poly_slippage=self.poly_slippage,
        ).generate(feat, kalshi_col=self.kalshi_col, poly_col=self.poly_col)
        res = VectorizedBacktester(
            kalshi_fee=self.kalshi_fee, poly_slippage=self.poly_slippage
        ).run(sig)
        return res.sharpe if np.isfinite(res.sharpe) else -np.inf

    @staticmethod
    def _half_life(lab: StatLab, spread: pd.Series) -> float:
        """OU half-life of the spread (fast ADF settings; only the half-life is used)."""
        try:
            ou = lab.estimate_ou(spread=spread, adf_autolag=None, adf_maxlag=1)
            return float(ou.half_life_periods)
        except Exception:
            return float("nan")

    def _window_grid(self, half_life: float) -> "list[int]":
        """Window candidates as multiples of the half-life, clipped to bounds."""
        if np.isfinite(half_life) and half_life > 0:
            raw = [m * half_life for m in self.window_multipliers]
        else:  # not mean-reverting in-sample: span the allowed range instead
            raw = [self.min_window, (self.min_window + self.max_window) / 2, self.max_window]
        grid = sorted({int(np.clip(round(w), self.min_window, self.max_window)) for w in raw})
        return grid

    # ------------------------------------------------------------------ #
    # Out-of-sample application (strict, causal)
    # ------------------------------------------------------------------ #
    def _oos_signals(self, synced, anchor, oos_df, window: int, entry_z: float) -> pd.DataFrame:
        """Generate signals for the OOS step with the chosen params (no look-ahead).

        Computed over ``[anchor, oos_end]`` so the rolling z-score's trailing window
        is populated at the OOS open; only the OOS rows are returned.
        """
        full = synced.loc[anchor: oos_df.index[-1]]
        lab = StatLab(full, kalshi_col=self.kalshi_col,
                      polymarket_col=self.poly_col, spread_col=self.spread_col)
        zframe = lab.rolling_zscore(window=window, spread=lab.signed_spread)
        feat = zframe.join(full[[self.kalshi_col, self.poly_col]], how="left")
        sig = SignalEngine(
            entry_z=entry_z, risk_mode=self.risk_mode, stop_z_score=self.stop_z_score,
            kalshi_fee=self.kalshi_fee, poly_slippage=self.poly_slippage,
        ).generate(feat, kalshi_col=self.kalshi_col, poly_col=self.poly_col)
        return sig.loc[oos_df.index[0]: oos_df.index[-1]]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self, synced: pd.DataFrame) -> None:
        missing = {self.kalshi_col, self.poly_col, self.spread_col} - set(synced.columns)
        if missing:
            raise KeyError(f"WalkForwardOptimizer: dataframe missing columns {sorted(missing)}")
        if not isinstance(synced.index, pd.DatetimeIndex):
            raise TypeError("synced must have a DatetimeIndex")
        if not synced.index.is_monotonic_increasing:
            raise ValueError("synced index must be sorted ascending")


# --------------------------------------------------------------------------- #
# Self-contained demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    rng = np.random.default_rng(4)
    days, ppd = 60, 96               # 60 days of 15-minute bars
    n = days * ppd
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    fair = np.clip(0.50 + np.cumsum(rng.normal(0, 0.0008, n)), 0.1, 0.9)

    # Regime shift halfway: fast reversion (short half-life) -> slow reversion
    # (long half-life). A correct walk-forward should grow the chosen window.
    t = np.arange(n)
    phi = np.where(t < n // 2, 0.88, 0.94)        # both regimes revert fast enough to trade
    innov = np.where(t < n // 2, 0.021, 0.015)    # comparable amplitude (sigma ~4.4c) clears fees
    basis = np.empty(n)
    basis[0] = 0.0
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        basis[i] = phi[i] * basis[i - 1] + innov[i] * eps[i]

    kalshi = np.clip(fair + basis / 2.0, 0.01, 0.99)
    poly = np.clip(fair - basis / 2.0, 0.01, 0.99)
    synced = pd.DataFrame(
        {_K: kalshi, _P: poly, _S: np.abs(kalshi - poly)}, index=idx
    )

    wfo = WalkForwardOptimizer(train_days=30, test_days=1, min_window=10, max_window=120,
                               z_grid=(1.5, 2.0, 2.5, 3.0))
    result = wfo.run(synced)

    p = result.params
    print(f"=== Walk-forward over {days} days (expanding 30d train, 1d OOS) ===")
    print(f"folds: {len(p)}\n")
    print("Chosen parameters per fold (head / tail):")
    cols = ["n_train_days", "half_life", "best_window", "best_entry_z", "insample_sharpe"]
    print(p[cols].head(4).round(2).to_string())
    print("  ...")
    print(p[cols].tail(4).round(2).to_string())

    early = p["best_window"].iloc[: len(p) // 3].mean()
    late = p["best_window"].iloc[-len(p) // 3:].mean()
    print(f"\nMean chosen window — early folds: {early:.1f}  vs  late folds: {late:.1f}")
    print("(window grows as the slow-reversion regime enters the expanding window)\n")

    print("Continuous out-of-sample signal series:")
    for k, v in result.summary().items():
        print(f"  {k:>22}: {v}")

    # The honest payoff: backtest the stitched OOS signals (true out-of-sample).
    oos_bt = VectorizedBacktester(kalshi_fee=0.01, poly_slippage=0.005).run(result.oos_signals)
    print(f"\nTrue out-of-sample backtest (no look-ahead): "
          f"Sharpe={oos_bt.sharpe:.2f}, total_return={oos_bt.total_return:+.1%}, "
          f"MaxDD={oos_bt.max_drawdown:.1%}, trades={oos_bt.n_trades}")
    is_mean = result.params["insample_sharpe"].replace([-np.inf], np.nan).mean()
    print(f"In-sample Sharpe averaged {is_mean:.1f} vs out-of-sample {oos_bt.sharpe:.1f} -- "
          "that gap is the overfit walk-forward exists to expose (in-sample selection is optimistic).")


if __name__ == "__main__":
    _demo()
