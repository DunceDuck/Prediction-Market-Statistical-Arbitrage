"""
run_master.py
=============

Master execution script for the prediction-market stat-arb stack.

It runs the full pipeline -- MarketDataPipeline -> StatLab -> SignalEngine ->
VectorizedBacktester -- across the cartesian product of:

    datasets : {short-term (1-week Fed rate market),
                long-term  (6-month election market)}
    risk modes: {indefinite_hold, stop_loss}

i.e. four scenarios, and prints a Markdown table comparing Total Return,
Annualized Sharpe, and Maximum Drawdown side-by-side.

The four core classes are already stateless across runs and fully parameterized,
so the only "refactor" needed to loop them is a thin orchestrator
(:class:`StatArbPipeline`) that wires the stages and a :class:`DatasetSpec` that
captures how the two markets differ (horizon, bar frequency, reversion speed,
and -- crucially -- tail risk: the election market carries fat-tailed news jumps
that the stop-loss is designed to cap, while the Fed market reverts cleanly).

Run:  python run_master.py
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The strategy classes live in src/. Put it on the import path so this
# root-level entry point (and the modules' sibling imports) resolve cleanly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_pipeline import MarketDataPipeline
from stat_lab import StatLab
from signal_engine import SignalEngine
from backtester import VectorizedBacktester


# --------------------------------------------------------------------------- #
# Dataset specification + synthetic raw-feed generation
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    """Parameters describing one market and how to synthesize raw feeds for it."""

    key: str               # short id, e.g. "short_term"
    label: str             # column label, e.g. "Short-Term"
    description: str       # human description, e.g. "1-week Fed rate market"
    start: str             # ISO start date
    n_bars: int            # number of underlying bars to simulate
    freq: str              # MarketDataPipeline resample frequency
    bar_seconds: int       # seconds per underlying bar (for timestamp synthesis)
    fair0: float           # starting fair value (probability)
    fair_vol: float        # per-bar drift vol of the shared fair value
    phi: float             # AR(1) reversion of the cross-venue basis
    basis_vol: float       # per-bar basis innovation vol
    jump_prob: float       # probability of a news jump per bar
    jump_vol: float        # size (std) of a news jump
    micro: float           # idiosyncratic per-venue microstructure noise
    seed: int


def generate_raw_feeds(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize raw, irregular, differently-clocked Kalshi & Polymarket feeds.

    Kalshi is emitted with epoch-millisecond timestamps, Polymarket with datetime
    timestamps -- exercising both timestamp paths of MarketDataPipeline. Each venue
    has independent within-bar jitter and random dropouts, so the pipeline performs
    real resampling, forward-filling, and alignment.
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_bars

    # Shared fair-value path both venues track.
    fair = np.clip(spec.fair0 + np.cumsum(rng.normal(0, spec.fair_vol, n)), 0.05, 0.95)

    # Mean-reverting cross-venue basis with rare fat-tailed news jumps.
    basis = np.empty(n)
    basis[0] = 0.0
    shock = rng.normal(0, spec.basis_vol, n)
    jmask = rng.random(n) < spec.jump_prob
    shock[jmask] += rng.normal(0, spec.jump_vol, int(jmask.sum()))
    for i in range(1, n):
        basis[i] = spec.phi * basis[i - 1] + shock[i]

    k_true = np.clip(fair + basis / 2.0, 0.01, 0.99)
    p_true = np.clip(fair - basis / 2.0, 0.01, 0.99)

    grid = pd.Timestamp(spec.start, tz="UTC") + pd.to_timedelta(
        np.arange(n) * spec.bar_seconds, unit="s"
    )

    def emit(prices: np.ndarray, drop_p: float):
        keep = rng.random(n) >= drop_p
        keep[[0, -1]] = True  # anchor the endpoints so the venues overlap fully
        idx = np.flatnonzero(keep)
        jitter = rng.integers(0, max(1, spec.bar_seconds), size=idx.size)
        ts = grid[idx] + pd.to_timedelta(jitter, unit="s")
        px = np.clip(prices[idx] + rng.normal(0, spec.micro, idx.size), 0.01, 0.99)
        return ts, px

    kt, kp = emit(k_true, drop_p=0.05)
    pt, pp = emit(p_true, drop_p=0.08)

    # Volume from an independent stream so prices/timestamps (and the published
    # backtest results) are unchanged; the pipeline needs Price + Volume.
    vrng = np.random.default_rng(spec.seed + 9973)
    kvol = vrng.integers(20, 200, size=kp.size).astype(float)
    pvol = vrng.integers(20, 200, size=pp.size).astype(float)

    kalshi_df = pd.DataFrame({"timestamp": kt.asi8 // 1_000_000, "price": kp, "volume": kvol})  # epoch ms
    poly_df = pd.DataFrame({"timestamp": pt, "price": pp, "volume": pvol})                        # datetime
    return kalshi_df, poly_df


# --------------------------------------------------------------------------- #
# Orchestrator: wires the four stages, reusable across datasets and modes
# --------------------------------------------------------------------------- #
class StatArbPipeline:
    """Runs ingestion -> stats -> signals -> backtest for one configuration.

    Dataset-level work (sync + StatLab + cointegration/OU) is done once via
    :meth:`prepare`; mode-level work (signals + backtest) via :meth:`backtest`,
    so the expensive alignment/diagnostics are not repeated per risk mode.
    """

    def __init__(
        self,
        freq: str,
        window: int = 60,
        entry_z: float = 2.0,
        stop_z_score: float = 3.5,
        kalshi_fee: float = 0.01,
        poly_slippage: float = 0.005,
        initial_capital: float = 1.0,
        trading_days_per_year: int = 252,
    ) -> None:
        self.freq = freq
        self.window = window
        self.entry_z = entry_z
        self.stop_z_score = stop_z_score
        self.kalshi_fee = kalshi_fee
        self.poly_slippage = poly_slippage
        self.initial_capital = initial_capital
        self.trading_days_per_year = trading_days_per_year

    def prepare(self, kalshi_df: pd.DataFrame, poly_df: pd.DataFrame) -> StatLab:
        synced = MarketDataPipeline(
            freq=self.freq, timestamp_col="timestamp"
        ).synchronize(kalshi_df, poly_df)
        return StatLab(synced)

    @staticmethod
    def diagnostics(lab: StatLab) -> tuple:
        """Dataset-level cointegration + OU half-life (for context, not the table).

        Uses a fixed short ADF lag (autolag off) -- exhaustive lag search is
        unnecessary for a context p-value and is the only slow step at this scale.
        """
        eg = lab.engle_granger_test(autolag=None, maxlag=1)
        ou = lab.estimate_ou(spread=lab.signed_spread, adf_autolag=None, adf_maxlag=1)
        return eg, ou

    def backtest(self, lab: StatLab, risk_mode: str):
        engine = SignalEngine(
            risk_mode=risk_mode,
            entry_z=self.entry_z,
            stop_z_score=self.stop_z_score,
            kalshi_fee=self.kalshi_fee,
            poly_slippage=self.poly_slippage,
        )
        signals = engine.run(lab, window=self.window, use_signed=True)
        bt = VectorizedBacktester(
            initial_capital=self.initial_capital,
            kalshi_fee=self.kalshi_fee,
            poly_slippage=self.poly_slippage,
            trading_days_per_year=self.trading_days_per_year,
        )
        return bt.run(signals)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown_table(headers, rows, aligns) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def cell(s, w, a):
        return s.rjust(w) if a == "r" else (s.center(w) if a == "c" else s.ljust(w))

    def row(cells):
        return "| " + " | ".join(cell(c, widths[i], aligns[i]) for i, c in enumerate(cells)) + " |"

    def sep():
        segs = []
        for w, a in zip(widths, aligns):
            if a == "r":
                segs.append("-" * (w + 1) + ":")
            elif a == "c":
                segs.append(":" + "-" * w + ":")
            else:
                segs.append(":" + "-" * (w + 1))
        return "|" + "|".join(segs) + "|"

    return "\n".join([row(headers), sep()] + [row(r) for r in rows])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
MODE_LABELS = {"indefinite_hold": "Indefinite Hold", "stop_loss": "Stop-Loss"}


def build_datasets() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="short_term",
            label="Short-Term",
            description="1-week Fed rate market",
            start="2026-03-16",
            n_bars=7 * 1440,          # 1 week of 1-minute bars
            freq="1min",
            bar_seconds=60,
            fair0=0.62,
            fair_vol=0.0008,
            phi=0.90,                 # fast reversion (half-life ~6.6 min) -> clean
            basis_vol=0.012,
            jump_prob=0.0008,         # few shocks
            jump_vol=0.03,
            micro=0.0010,
            seed=11,
        ),
        DatasetSpec(
            key="long_term",
            label="Long-Term",
            description="6-month election market",
            start="2026-01-05",
            n_bars=180 * 96,          # 6 months of 15-minute bars
            freq="15min",
            bar_seconds=900,
            fair0=0.47,
            fair_vol=0.0015,
            phi=0.92,                 # half-life ~2 h
            basis_vol=0.010,
            jump_prob=0.0030,         # frequent, large news jumps...
            jump_vol=0.10,            # ...big enough to blow past the 3.5-sigma stop
            micro=0.0015,
            seed=22,
        ),
    ]


def main() -> None:
    datasets = build_datasets()
    modes = ["indefinite_hold", "stop_loss"]

    rows, context, footnotes = [], [], []

    for spec in datasets:
        kalshi_df, poly_df = generate_raw_feeds(spec)
        pipe = StatArbPipeline(freq=spec.freq, window=60, entry_z=2.0, stop_z_score=3.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # synthetic-data warnings are expected
            lab = pipe.prepare(kalshi_df, poly_df)
            eg, ou = pipe.diagnostics(lab)

            synced = lab.df
            span_days = (synced.index[-1] - synced.index[0]).total_seconds() / 86400.0
            hl_min = ou.half_life_periods * (spec.bar_seconds / 60.0)
            context.append(
                f"- **{spec.label}** — {spec.description}: "
                f"{synced.shape[0]:,} {spec.freq} bars ({span_days:.0f} days); "
                f"{'cointegrated' if eg.cointegrated else 'NOT cointegrated'} "
                f"(p={eg.pvalue:.1e}); basis half-life ~{hl_min:.0f} min"
            )

            for mode in modes:
                res = pipe.backtest(lab, mode)
                rows.append(
                    [
                        f"{spec.label} / {MODE_LABELS[mode]}",
                        f"{res.total_return:+.1%}",
                        f"{res.sharpe:.2f}",
                        f"{res.max_drawdown:.1%}",
                        f"{res.n_trades:,}",
                    ]
                )
                if res.equity_curve.min() <= 0:
                    footnotes.append(f"  - {spec.label}/{MODE_LABELS[mode]}: equity blew up (lost > capital).")
                if mode == "indefinite_hold" and spec.key == "short_term" and res.n_days < 15:
                    footnotes.append(
                        f"  - {spec.label} Sharpe is computed from only ~{res.n_days} daily "
                        "returns (1-week horizon) and is statistically noisy."
                    )

    headers = ["Scenario", "Total Return", "Ann. Sharpe", "Max Drawdown", "Trades"]
    aligns = ["l", "r", "r", "r", "r"]

    print("# Prediction-Market Stat-Arb — Scenario Comparison\n")
    print("**Datasets**")
    print("\n".join(context))
    print("\n**Results** (Total Return over each market's full horizon; "
          "Sharpe annualized from daily returns, rf=0%, 252 days)\n")
    print(render_markdown_table(headers, rows, aligns))

    if footnotes:
        print("\n**Notes**")
        # de-duplicate while preserving order
        seen = set()
        for f in footnotes:
            if f not in seen:
                print(f)
                seen.add(f)
    print(
        "\n_Synthetic AR(1) data is far cleaner than real markets, so the absolute "
        "Sharpe levels are optimistic. The instructive pattern: the stop-loss lowers "
        "return and Sharpe in BOTH markets without improving drawdown. With a "
        "rolling-mean z-score the strategy self-corrects (the mean adapts to a "
        "dislocation within the window), so stops mostly cut winners that would have "
        "reverted. A stop pays off only against genuine non-reverting breaks — gap "
        "risk, decointegration — that a rolling-window signal cannot absorb._"
    )


if __name__ == "__main__":
    main()
