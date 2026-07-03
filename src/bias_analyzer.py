"""
bias_analyzer.py
================

Statistical core of the favorite-longshot alpha. :class:`BiasAnalyzer` consumes a
panel of resolved-market observations (the frame from
:meth:`alpha_engine.PolymarketDataPuller.panel_frame` or ``snapshot_frame``) and
measures how far market prices sit from realized outcome frequencies -- the
**calibration curve**.

What it does
------------
1. **Bucket** implied probabilities into deciles ``[0.0, 0.1), [0.1, 0.2), ...``.
2. **Realized win rate** per decile -- the empirical ``P(resolve YES | price)``.
3. **Bootstrap confidence intervals** per decile, so a decile backed by only a
   handful of markets is shown as *uncertain* rather than taken at face value.
4. **Calibration curve** (realized vs implied) with the 45-degree perfect-efficiency
   line and error bars highlighting the bias.

Why a *cluster* bootstrap (the statistically critical bit)
----------------------------------------------------------
A time-series panel has many price observations **per market**, and every one of a
market's observations shares the *same* binary outcome. Those rows are therefore
**not independent**: 50 points from one market carry roughly the information of a
*single* resolved coin flip, not 50. A naive row-level bootstrap would treat them
as independent and report intervals that are far too tight -- the exact opposite of
the "don't get fooled by small samples" goal.

So the bootstrap resamples **whole markets** (the cluster), with replacement, and
recomputes each decile's win rate from the resampled set. A decile whose mass comes
from few distinct markets then shows a wide interval, honestly reflecting how little
independent evidence backs it. (When no ``group_col`` is supplied -- e.g. a
one-row-per-market snapshot -- every row is its own cluster and this reduces to the
ordinary i.i.d. bootstrap.)

The resampling is vectorized: observations are pre-aggregated into a
(market x decile) count matrix and a matching outcome-sum matrix, then ``B``
multinomial market-weight draws are applied as two matrix products. Cost is
``O(B * n_markets * n_deciles)`` -- independent of the number of time-series points.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["BiasAnalyzer", "BiasReport"]


# --------------------------------------------------------------------------- #
# Structured result
# --------------------------------------------------------------------------- #
@dataclass
class BiasReport:
    """Per-decile calibration table plus headline diagnostics.

    ``table`` columns (indexed by decile 0..n-1):
    ``decile`` (label), ``bin_low``, ``bin_high``, ``n_obs`` (price observations),
    ``n_markets`` (distinct markets -- the *real* sample size), ``mean_implied``,
    ``realized`` (win rate), ``se`` (bootstrap std error), ``ci_low``/``ci_high``
    (percentile interval), and ``bias = realized - mean_implied`` (negative =
    over-priced longshots, positive = under-priced favorites).
    """

    table: pd.DataFrame = field(repr=False)
    n_obs: int
    n_markets: int
    n_deciles: int
    n_boot: int
    ci_level: float
    calibration_slope: float
    prob_col: str = "implied_prob"
    outcome_col: str = "resolution"
    group_col: object = "market_id"

    def summary(self) -> dict:
        t = self.table
        valid = t.dropna(subset=["realized", "mean_implied"])
        worst_long = valid.loc[valid["bias"].idxmin()] if not valid.empty else None
        worst_fav = valid.loc[valid["bias"].idxmax()] if not valid.empty else None
        return {
            "n_obs": self.n_obs,
            "n_markets": self.n_markets,
            "n_deciles": self.n_deciles,
            "n_boot": self.n_boot,
            "ci_level": self.ci_level,
            "calibration_slope": round(self.calibration_slope, 4),
            "populated_deciles": int(valid.shape[0]),
            "max_overpricing": None if worst_long is None else
                f"{worst_long['decile']}: {worst_long['bias']:+.3f}",
            "max_underpricing": None if worst_fav is None else
                f"{worst_fav['decile']}: {worst_fav['bias']:+.3f}",
        }


class BiasAnalyzer:
    """Decile calibration analysis with a market-clustered bootstrap.

    Parameters
    ----------
    df:
        A panel of observations. Each row needs an implied-probability column and a
        binary outcome column; an optional grouping column (e.g. ``market_id``) keys
        the cluster bootstrap. Accepts either the long time-series ``panel_frame``
        or the one-row-per-market ``snapshot_frame``.
    prob_col, outcome_col:
        Column names for the implied probability (0-1) and the binary resolution
        (0/1). Defaults ``"implied_prob"`` / ``"resolution"`` (the puller's names).
    group_col:
        Cluster key for the bootstrap (default ``"market_id"``). If absent from
        ``df``, each row is its own cluster (ordinary i.i.d. bootstrap).
    n_deciles:
        Number of equal-width probability buckets. Default 10 (deciles).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        prob_col: str = "implied_prob",
        outcome_col: str = "resolution",
        group_col: object = "market_id",
        n_deciles: int = 10,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)!r}")
        for col in (prob_col, outcome_col):
            if col not in df.columns:
                raise KeyError(f"column {col!r} not found; columns are {list(df.columns)}")
        if n_deciles < 2:
            raise ValueError("n_deciles must be >= 2")

        self.prob_col = prob_col
        self.outcome_col = outcome_col
        self.n_deciles = int(n_deciles)
        # group_col is optional: keep it only if the column is actually present.
        self.group_col = group_col if (group_col is not None and group_col in df.columns) else None
        self._clustered = self.group_col is not None

        self._prepare(df)

    # ------------------------------------------------------------------ #
    # Preparation: clean rows, assign deciles, pre-aggregate the matrices
    # ------------------------------------------------------------------ #
    def _prepare(self, df: pd.DataFrame) -> None:
        p = pd.to_numeric(df[self.prob_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[self.outcome_col], errors="coerce").to_numpy(dtype=float)
        if self._clustered:
            groups = df[self.group_col].to_numpy()
        else:
            groups = np.arange(len(df))

        keep = np.isfinite(p) & np.isfinite(y)
        if not keep.any():
            raise ValueError("no rows with both a finite probability and outcome")
        p, y, groups = p[keep], y[keep], groups[keep]

        # Outcome must be binary 0/1.
        uniq = np.unique(y)
        if not np.isin(uniq, (0.0, 1.0)).all():
            raise ValueError(f"{self.outcome_col!r} must be binary 0/1; saw values {uniq[:6]}")

        # Probabilities must lie in [0, 1]; clip tiny FP overshoot, warn if material.
        if p.min() < -1e-9 or p.max() > 1 + 1e-9:
            if p.min() < -1e-6 or p.max() > 1 + 1e-6:
                warnings.warn(
                    f"{self.prob_col!r} has values outside [0, 1] "
                    f"(min={p.min():.4g}, max={p.max():.4g}); clipping.",
                    stacklevel=3,
                )
            p = np.clip(p, 0.0, 1.0)

        # Decile code in 0..D-1 ([0,0.1) -> 0, ..., 1.0 -> D-1).
        D = self.n_deciles
        dcode = np.clip((p * D).astype(int), 0, D - 1)
        gcode, group_labels = pd.factorize(groups, sort=False)
        M = len(group_labels)

        # Pre-aggregate: per (market, decile) observation count and YES-sum.
        cnt = np.zeros((M, D), dtype=float)
        sumy = np.zeros((M, D), dtype=float)
        np.add.at(cnt, (gcode, dcode), 1.0)
        np.add.at(sumy, (gcode, dcode), y)
        # Per-decile implied-probability sum (for the mean implied price per bucket).
        psum = np.zeros(D, dtype=float)
        np.add.at(psum, dcode, p)

        self._p, self._y = p, y
        self._dcode = dcode
        self._cnt, self._sumy, self._psum = cnt, sumy, psum
        self._M, self._D = M, D
        self.n_obs = int(p.size)
        self.n_markets = int(M)

    # ------------------------------------------------------------------ #
    # 1) Bucketing
    # ------------------------------------------------------------------ #
    def bucket_deciles(self) -> pd.DataFrame:
        """Return the cleaned observations with their decile assignment.

        Columns: ``implied_prob``, ``resolution``, (``group``), ``decile_idx``
        (0..n-1) and ``decile`` (the ``"0.0-0.1"`` label).
        """
        lo = self._dcode / self._D
        hi = (self._dcode + 1) / self._D
        labels = np.array([f"{a:.1f}-{b:.1f}" for a, b in zip(lo, hi)])
        out = pd.DataFrame(
            {
                self.prob_col: self._p,
                self.outcome_col: self._y.astype(int),
                "decile_idx": self._dcode,
                "decile": labels,
            }
        )
        return out

    # ------------------------------------------------------------------ #
    # 2) Realized win rate per decile
    # ------------------------------------------------------------------ #
    def win_rates(self) -> pd.DataFrame:
        """Per-decile observation count, distinct-market count, mean implied price,
        and realized win rate. Empty deciles are kept (with ``NaN`` rates)."""
        den = self._cnt.sum(axis=0)                  # observations per decile
        num = self._sumy.sum(axis=0)                 # YES observations per decile
        n_markets = (self._cnt > 0).sum(axis=0)      # distinct markets per decile
        realized = _safe_div(num, den)
        mean_implied = _safe_div(self._psum, den)

        idx = np.arange(self._D)
        return pd.DataFrame(
            {
                "decile": [f"{i / self._D:.1f}-{(i + 1) / self._D:.1f}" for i in idx],
                "bin_low": idx / self._D,
                "bin_high": (idx + 1) / self._D,
                "n_obs": den.astype(int),
                "n_markets": n_markets.astype(int),
                "mean_implied": mean_implied,
                "realized": realized,
            },
            index=idx,
        )

    # ------------------------------------------------------------------ #
    # 3) Cluster bootstrap (per-decile standard error + confidence interval)
    # ------------------------------------------------------------------ #
    def bootstrap(
        self, n_boot: int = 2000, ci: float = 0.95, seed: int = 12345
    ) -> pd.DataFrame:
        """Bootstrap each decile's win rate by resampling **markets** (clusters).

        Returns a per-decile frame with ``realized`` (point estimate), ``se``
        (bootstrap standard error), and ``ci_low``/``ci_high`` (percentile interval
        at level ``ci``). Resampling whole markets -- not rows -- keeps correlated
        within-market observations from inflating the effective sample size.
        """
        if n_boot < 1:
            raise ValueError("n_boot must be >= 1")
        if not (0.0 < ci < 1.0):
            raise ValueError("ci must be in (0, 1)")

        den = self._cnt.sum(axis=0)
        realized = _safe_div(self._sumy.sum(axis=0), den)

        M, D = self._M, self._D
        rng = np.random.default_rng(seed)
        pvals = np.full(M, 1.0 / M)

        # Draw B resamples of M markets as multinomial selection weights, then apply
        # them as matrix products. Chunked over B to bound memory at ~1e7 weights.
        chunk = max(1, int(1e7 // max(M, 1)))
        boot = np.empty((n_boot, D), dtype=float)
        filled = 0
        with np.errstate(invalid="ignore", divide="ignore"):
            while filled < n_boot:
                b = min(chunk, n_boot - filled)
                weights = rng.multinomial(M, pvals, size=b).astype(float)  # (b, M)
                num_b = weights @ self._sumy                                # (b, D)
                den_b = weights @ self._cnt                                 # (b, D)
                boot[filled:filled + b] = np.where(den_b > 0, num_b / den_b, np.nan)
                filled += b

        lo_q, hi_q = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN empty deciles
            se = np.nanstd(boot, axis=0, ddof=1)
            ci_low = np.nanpercentile(boot, lo_q, axis=0)
            ci_high = np.nanpercentile(boot, hi_q, axis=0)
        # Empty deciles carry no estimate.
        empty = den == 0
        se[empty] = np.nan
        ci_low[empty] = np.nan
        ci_high[empty] = np.nan

        idx = np.arange(D)
        return pd.DataFrame(
            {"realized": realized, "se": se, "ci_low": ci_low, "ci_high": ci_high},
            index=idx,
        )

    # ------------------------------------------------------------------ #
    # Convenience: full report
    # ------------------------------------------------------------------ #
    def analyze(self, n_boot: int = 2000, ci: float = 0.95, seed: int = 12345) -> BiasReport:
        """Run bucketing + win rates + bootstrap and return a :class:`BiasReport`."""
        wr = self.win_rates()
        bs = self.bootstrap(n_boot=n_boot, ci=ci, seed=seed)
        table = wr.join(bs[["se", "ci_low", "ci_high"]])
        table["bias"] = table["realized"] - table["mean_implied"]
        table = table[
            ["decile", "bin_low", "bin_high", "n_obs", "n_markets",
             "mean_implied", "realized", "se", "ci_low", "ci_high", "bias"]
        ]
        return BiasReport(
            table=table,
            n_obs=self.n_obs,
            n_markets=self.n_markets,
            n_deciles=self.n_deciles,
            n_boot=int(n_boot),
            ci_level=float(ci),
            calibration_slope=self.calibration_slope(self._p, self._y),
            prob_col=self.prob_col,
            outcome_col=self.outcome_col,
            group_col=self.group_col,
        )

    # ------------------------------------------------------------------ #
    # 4) Calibration curve
    # ------------------------------------------------------------------ #
    def plot_calibration(
        self,
        report: "BiasReport | None" = None,
        ax=None,
        save_path: "str | None" = None,
        annotate: bool = True,
        title: str = "Polymarket calibration — realized vs. implied probability",
    ):
        """Plot the calibration curve: realized win rate vs implied price, with the
        45-degree efficiency line, bootstrap error bars, and shaded bias regions.

        Marker area scales with the number of distinct markets in the decile (so thin
        buckets read as small + wide-barred). Returns the matplotlib ``Figure``;
        writes a PNG if ``save_path`` is given. Imports matplotlib lazily.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("plot_calibration requires matplotlib (pip install matplotlib)") from exc

        if report is None:
            report = self.analyze()
        t = report.table
        valid = t.dropna(subset=["realized", "mean_implied"]).copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.2, 7.2))
        else:
            fig = ax.figure

        # Perfect-efficiency 45-degree line.
        ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.3,
                label="Perfect calibration (45°)", zorder=2)

        x = valid["mean_implied"].to_numpy()
        yv = valid["realized"].to_numpy()
        order = np.argsort(x)
        xs, ys = x[order], yv[order]

        # Shade the bias: below the line = over-priced longshots; above = under-priced
        # favorites. Two translucent fills between the realized curve and the diagonal.
        ax.fill_between(xs, xs, ys, where=ys < xs, interpolate=True,
                        color="tab:red", alpha=0.15, label="Over-priced (longshots)")
        ax.fill_between(xs, xs, ys, where=ys >= xs, interpolate=True,
                        color="tab:green", alpha=0.15, label="Under-priced (favorites)")

        # Asymmetric error bars from the bootstrap percentile interval.
        yerr = np.vstack([
            np.clip(valid["realized"] - valid["ci_low"], 0, None),
            np.clip(valid["ci_high"] - valid["realized"], 0, None),
        ])
        ax.errorbar(x, yv, yerr=yerr, fmt="none", ecolor="0.35", elinewidth=1.2,
                    capsize=3, zorder=3)
        sizes = 30 + 4.0 * np.sqrt(valid["n_markets"].to_numpy())
        ax.scatter(x, yv, s=sizes, c="tab:blue", edgecolor="white", linewidth=0.6,
                   zorder=4, label="Realized win rate (±95% CI, ∝ #markets)")
        ax.plot(xs, ys, color="tab:blue", lw=1.0, alpha=0.6, zorder=3)

        if annotate:
            ax.annotate("Longshots win LESS\nthan priced", xy=(0.16, 0.045),
                        color="tab:red", fontsize=9, ha="center", va="center")
            ax.annotate("Favorites win MORE\nthan priced", xy=(0.84, 0.955),
                        color="tab:green", fontsize=9, ha="center", va="center")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Implied probability (market price)")
        ax.set_ylabel("Realized win rate")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------ #
    # Pure, independently-testable statistic
    # ------------------------------------------------------------------ #
    @staticmethod
    def calibration_slope(implied, outcome) -> float:
        """Pooled slope of ``(outcome - implied)`` on ``(implied - 0.5)``.

        A single, binning-free miscalibration summary: ``> 0`` is the
        favorite-longshot bias, ``~ 0`` is calibrated.
        """
        p = np.asarray(implied, dtype=float)
        y = np.asarray(outcome, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        if p.size < 3:
            return float("nan")
        x = p - 0.5
        var_x = x.var()
        if var_x == 0:
            return float("nan")
        return float(np.cov(x, y - p, bias=True)[0, 1] / var_x)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num/den`` with ``NaN`` where ``den == 0``."""
    out = np.full(np.shape(num), np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


# --------------------------------------------------------------------------- #
# Self-contained demo (mock data; matplotlib optional, saved to a temp file)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import os
    import tempfile

    from alpha_engine import PolymarketDataPuller

    puller = PolymarketDataPuller()
    # 1500 markets, 48 hourly price points each -> a ~72k-row time-series panel with a
    # deliberate favorite-longshot bias. Low path noise keeps each path near its belief.
    markets = puller.generate_mock_markets(
        n_markets=1500, bias_strength=1.35, n_obs=48, noise=0.13, seed=11
    )
    panel = puller.panel_frame(markets)
    print(f"=== BiasAnalyzer on a time-series panel ===")
    print(f"panel rows (price observations): {len(panel):,}   "
          f"distinct markets: {panel['market_id'].nunique()}\n")

    analyzer = BiasAnalyzer(panel)  # prob_col=implied_prob, group_col=market_id
    report = analyzer.analyze(n_boot=2000, ci=0.95, seed=7)

    show = report.table.copy()
    for c in ("mean_implied", "realized", "se", "ci_low", "ci_high", "bias"):
        show[c] = show[c].round(3)
    print("Per-decile calibration (cluster-bootstrapped 95% CI):")
    print(show[["decile", "n_obs", "n_markets", "mean_implied",
                "realized", "ci_low", "ci_high", "bias"]].to_string(index=False))

    print("\nReport summary:")
    for k, v in report.summary().items():
        print(f"  {k:>20}: {v}")

    # --- Verification: bias is the right shape, and the cluster bootstrap is NOT
    # fooled -- its interval is much wider than a naive row-level bootstrap. ---
    valid = report.table.dropna(subset=["realized", "mean_implied"])
    low = valid[valid["bin_high"] <= 0.5]
    high = valid[valid["bin_low"] >= 0.5]
    lo_ok = (low["bias"] < 0).mean() if len(low) else float("nan")
    hi_ok = (high["bias"] > 0).mean() if len(high) else float("nan")
    print(f"\nLow deciles with realized < implied : {lo_ok:.0%}")
    print(f"High deciles with realized > implied: {hi_ok:.0%}")
    print(f"Pooled calibration slope            : {report.calibration_slope:+.3f}  (>0 => FLB)")

    naive = BiasAnalyzer(panel, group_col=None)  # treat every row as independent
    nb = naive.bootstrap(n_boot=2000, ci=0.95, seed=7)
    cb = analyzer.bootstrap(n_boot=2000, ci=0.95, seed=7)
    width_naive = float((nb["ci_high"] - nb["ci_low"]).mean())
    width_cluster = float((cb["ci_high"] - cb["ci_low"]).mean())
    print(f"\nMean 95%-CI width — naive row bootstrap: {width_naive:.3f}  "
          f"vs market-clustered: {width_cluster:.3f}  "
          f"({width_cluster / width_naive:.1f}x wider, the honest interval)")

    ok = (report.calibration_slope > 0.02) and (lo_ok >= 0.6) and (hi_ok >= 0.6) \
        and (width_cluster > width_naive)
    print(f"\n[{'OK' if ok else 'UNEXPECTED'}] favorite-longshot bias detected; "
          "clustering correctly widens the intervals.")

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe for the demo (no display needed)
        out = os.path.join(tempfile.gettempdir(), "polymarket_calibration.png")
        analyzer.plot_calibration(report=report, save_path=out)
        print(f"\nCalibration curve written to: {out}")
    except ImportError:
        print("\n(matplotlib not installed — skipping the calibration plot)")


if __name__ == "__main__":
    _demo()
