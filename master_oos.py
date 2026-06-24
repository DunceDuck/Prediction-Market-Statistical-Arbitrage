"""
master_oos.py
=============

Out-of-Sample Performance Tear Sheet on **real** prediction-market data.

Pipeline (no synthetic data anywhere in this script):

    pmxt (Kalshi + Polymarket matched market)
        -> MarketDataPipeline      (align, quote-age + rolling-volume -> is_tradeable)
        -> WalkForwardOptimizer    (expanding 30d train / 1d OOS; window + entry-z
                                    re-selected each fold from the OU half-life to
                                    maximize friction-adjusted Sharpe)
        -> VectorizedBacktester    (signals shifted 1 period for look-ahead, and
                                    INTERSECTED with is_tradeable so we never execute
                                    on a stale/illiquid quote)
        -> Markdown tear sheet     (OOS Sharpe, Max Drawdown, % Capital Locked, Win Rate)

Cross-venue matching and Kalshi history come from pmxt's hosted matcher, which
needs a pmxt.dev API key:

    export PMXT_API_KEY=pmxt_live_...
    python master_oos.py "government shutdown"

If the key (or a usable matched market) is unavailable, the script says so and
exits -- it never falls back to synthetic data and never fabricates metrics. All
numbers it prints are strictly out-of-sample, on real market data.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_pipeline import MarketDataPipeline, PmxtFeed  # noqa: E402
from walk_forward import WalkForwardOptimizer  # noqa: E402
from backtester import VectorizedBacktester  # noqa: E402

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL

# pmxt OHLCV resolution -> pandas resample frequency for the pipeline grid.
_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def _load_dotenv(filename: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (no dependency).

    Lets you keep PMXT_API_KEY in a git-ignored .env instead of exporting it.
    """
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
# Real data acquisition (pmxt)
# --------------------------------------------------------------------------- #
def _overlap_days(kdf: pd.DataFrame, pdf: pd.DataFrame) -> float:
    lo = max(kdf.index.min(), pdf.index.min())
    hi = min(kdf.index.max(), pdf.index.max())
    return max(0.0, (hi - lo).total_seconds() / 86400.0)


def fetch_matched_pair(query=None, *, kalshi_slug=None, polymarket_slug=None,
                       resolution="1h", lookback_days=180, limit=8000,
                       min_overlap_days=14):
    """Fetch a real Kalshi+Polymarket pair -> (kalshi_df, poly_df, meta).

    Two modes:
      * explicit ``kalshi_slug`` + ``polymarket_slug`` -> fetch those two markets
        directly (most reliable; no matching service needed);
      * else use pmxt's cross-venue matcher (needs PMXT_API_KEY) to discover a
        same-event pair, preferring the highest-volume cluster.

    OHLCV is pulled with the *direct* venue clients over a bounded time window --
    the hosted Router OHLCV path needs a wallet and rejects unbounded intervals.
    Raises a clear RuntimeError if no pair has >= ``min_overlap_days`` of
    overlapping history; it never fabricates data.
    """
    import time
    import datetime as dt
    import pmxt

    _load_dotenv()
    poly, kal, feed = pmxt.Polymarket(), pmxt.Kalshi(), PmxtFeed()
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=lookback_days)

    def ohlcv(client, market):
        if market is None:
            return None
        candles = client.fetch_ohlcv(market.yes, resolution=resolution,
                                     start=start, end=end, limit=limit)
        return feed.candles_to_frame(candles) if candles else None

    # --- Mode 1: explicit slugs (most reliable; the lever to target a liquid market)
    if kalshi_slug and polymarket_slug:
        kdf = ohlcv(kal, kal.fetch_market(slug=kalshi_slug))
        pdf = ohlcv(poly, poly.fetch_market(slug=polymarket_slug))
        if kdf is None or pdf is None:
            raise RuntimeError("No OHLCV history returned for one of the provided slugs.")
        ov = _overlap_days(kdf, pdf)
        if ov < min_overlap_days:
            raise RuntimeError(f"Only {ov:.1f} days of overlapping history for the given "
                               f"slugs (need >= {min_overlap_days}).")
        return kdf, pdf, {"event": f"{kalshi_slug} <-> {polymarket_slug}",
                          "resolution": resolution, "kalshi_candles": len(kdf),
                          "polymarket_candles": len(pdf), "overlap_days": round(ov, 1)}

    # --- Mode 2: cross-venue matcher (needs the pmxt.dev key) ---
    key = os.environ.get("PMXT_API_KEY")
    if not key:
        raise RuntimeError(
            "PMXT_API_KEY is not set and no explicit slugs were given. Cross-venue "
            "matching needs a pmxt.dev key:\n    export PMXT_API_KEY=pmxt_live_...\n"
            "or pass kalshi_slug=... polymarket_slug=... for a known liquid pair.")
    router = pmxt.Router(pmxt_api_key=key)

    clusters, last = [], None              # hosted matcher is intermittent -> retry
    for i in range(6):
        try:
            clusters = router.fetch_matched_market_clusters(
                query=query, relation="identity", venues=["kalshi", "polymarket"],
                min_venues=2, limit=25)
            break
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    if not clusters:
        raise RuntimeError(f"pmxt matcher returned nothing ({type(last).__name__ if last else 'empty'}). "
                           "Retry, or pass explicit slugs.")
    clusters.sort(key=lambda c: getattr(c, "volume_24h", 0) or 0, reverse=True)

    best = 0.0
    for cluster in clusters:
        vm = _venue_markets(cluster)
        if not ({"kalshi", "polymarket"} <= set(vm)):
            continue
        try:
            kdf = ohlcv(kal, kal.fetch_market(slug=getattr(vm["kalshi"], "slug", None)))
            pdf = ohlcv(poly, poly.fetch_market(slug=getattr(vm["polymarket"], "slug", None)))
        except Exception:
            continue
        if kdf is None or pdf is None:
            continue
        ov = _overlap_days(kdf, pdf)
        best = max(best, ov)
        if ov >= min_overlap_days:
            return kdf, pdf, {"event": getattr(cluster, "canonical_title", None) or query,
                              "resolution": resolution, "kalshi_candles": len(kdf),
                              "polymarket_candles": len(pdf), "overlap_days": round(ov, 1)}

    raise RuntimeError(
        f"No matched Kalshi+Polymarket market with >= {min_overlap_days} days of overlapping "
        f"history for query={query!r} (best found: {best:.1f} days). The live matcher mostly "
        "surfaces sparse, long-dated markets; pass explicit liquid slugs "
        "(kalshi_slug=..., polymarket_slug=...) or use a deeper data source.")


def _venue_markets(cluster) -> dict:
    """Map venue -> matched market from a MatchedMarketCluster.

    A cluster's ``.markets`` holds the per-venue matched markets; the venue is in
    ``.source_exchange`` (with ``.venue``/``.exchange`` as fallbacks across pmxt
    versions). Each market exposes ``.yes`` for the YES outcome.
    """
    members = (getattr(cluster, "markets", None)
               or getattr(cluster, "unified_markets", None)
               or getattr(cluster, "members", None) or [])
    out = {}
    for m in members:
        venue = (getattr(m, "source_exchange", None)
                 or getattr(m, "venue", None) or getattr(m, "exchange", None))
        if venue:
            out[str(venue).lower()] = m
    return out


# --------------------------------------------------------------------------- #
# Out-of-sample pipeline
# --------------------------------------------------------------------------- #
def run_oos(synced: pd.DataFrame, train_days: int = 30):
    """Walk-forward -> OOS signals (shift-1 + is_tradeable) -> backtest result."""
    n_days = int(synced.index.normalize().nunique())
    if n_days < 4:
        raise RuntimeError(f"Only {n_days} days of history; need a longer-lived market.")
    train_days = max(2, min(train_days, n_days - 2))   # adapt to available history

    result = WalkForwardOptimizer(train_days=train_days, test_days=1).run(synced)
    oos = result.oos_signals
    if "is_tradeable" in synced.columns:                # carry tradeability onto OOS rows
        oos = oos.join(synced["is_tradeable"])
    bt = VectorizedBacktester(kalshi_fee=0.01, poly_slippage=0.005).run(
        oos, tradeable_col="is_tradeable")
    return result, bt, train_days


def render_tear_sheet(bt, meta: dict, n_folds: int, train_days: int,
                      illustrative: bool = False) -> str:
    """Markdown 'Out-of-Sample Performance Tear Sheet'.

    With ``illustrative=True`` the sheet is loudly stamped as synthetic (not real).
    """
    rows = [
        ("OOS Annualized Sharpe Ratio", f"{bt.sharpe:.2f}"),
        ("Maximum Drawdown", f"{bt.max_drawdown:.1%}"),
        ("Percentage of Capital Locked", f"{bt.pct_capital_locked:.1%}"),
        ("Win Rate (executed trades)", f"{bt.win_rate:.1%}"),
    ]
    label_w = max(len(r[0]) for r in rows)
    val_w = max(len(r[1]) for r in rows)

    title = "# Out-of-Sample Performance Tear Sheet"
    if illustrative:
        title += "  —  ILLUSTRATIVE (SYNTHETIC DATA)"
    out = [title, ""]
    if illustrative:
        out += [
            "> ⚠️ **ILLUSTRATIVE ONLY — NOT REAL RESULTS.** These numbers are from "
            "**synthetic** AR(1) data and exist solely to show the tear-sheet format and that "
            "the methodology is strictly out-of-sample. For genuine figures, run "
            "`master_oos.py` on real pmxt data (`PMXT_API_KEY`, or explicit market slugs).",
            "",
        ]
    data_line = ("**SYNTHETIC** (illustrative) — not a real market"
                 if illustrative else
                 f"real pmxt market data ({meta.get('resolution', '?')} candles, "
                 f"Kalshi {meta.get('kalshi_candles', '?')} / Polymarket {meta.get('polymarket_candles', '?')})")
    out += [
        f"- **Event:** {meta.get('event', '?')}",
        f"- **Data source:** {data_line}",
        f"- **Walk-forward:** expanding {train_days}-day train / 1-day OOS, {n_folds} folds; "
        "window & entry-z re-selected per fold (OU half-life + friction-adjusted-Sharpe grid)",
        "- **Execution:** signals shifted 1 period (no look-ahead) and intersected with `is_tradeable`",
        "",
        f"| {'Metric'.ljust(label_w)} | {'Value'.rjust(val_w)} |",
        f"|:{'-' * (label_w + 1)}|{'-' * (val_w + 1)}:|",
    ]
    out += [f"| {k.ljust(label_w)} | {v.rjust(val_w)} |" for k, v in rows]
    footer = ("_Illustrative output on **synthetic** data — not real performance. The "
              "methodology is strictly out-of-sample: walk-forward parameter selection, a "
              "1-period execution shift, and `is_tradeable` intersection._"
              if illustrative else
              "_All metrics are strictly out-of-sample (walk-forward) on real market data. "
              "No synthetic data is used in this pipeline._")
    out += ["", footer]
    return "\n".join(out)


def _illustrative_tear_sheet() -> None:
    """Run the FULL OOS pipeline on SYNTHETIC data, loudly labeled, to show the format."""
    from run_master import DatasetSpec, generate_raw_feeds   # reuse the synthetic generator

    spec = DatasetSpec(
        key="illustrative", label="Illustrative",
        description="synthetic AR(1) cross-venue spread",
        start="2026-01-01", n_bars=60 * 24, freq="1h", bar_seconds=3600,
        fair0=0.50, fair_vol=0.0015, phi=0.85, basis_vol=0.020,
        jump_prob=0.0025, jump_vol=0.06, micro=0.003, seed=7,
    )
    kalshi_df, poly_df = generate_raw_feeds(spec)
    synced = MarketDataPipeline(freq="1h", timestamp_col="timestamp").synchronize(kalshi_df, poly_df)
    result, bt, train_days = run_oos(synced, train_days=30)
    meta = {"event": "synthetic mean-reverting spread", "resolution": "1h",
            "kalshi_candles": len(kalshi_df), "polymarket_candles": len(poly_df)}
    print(render_tear_sheet(bt, meta, n_folds=len(result.params),
                            train_days=train_days, illustrative=True))


def main() -> None:
    warnings.simplefilter("ignore")
    if "--illustrative" in sys.argv:
        _illustrative_tear_sheet()
        return
    _load_dotenv()
    query = " ".join(a for a in sys.argv[1:] if not a.startswith("--")).strip() or "presidential election"
    resolution = "1h"
    k_slug = os.environ.get("KALSHI_SLUG")
    p_slug = os.environ.get("POLYMARKET_SLUG")

    try:
        kalshi_df, poly_df, meta = fetch_matched_pair(
            query=query, kalshi_slug=k_slug, polymarket_slug=p_slug, resolution=resolution)
        synced = MarketDataPipeline(
            freq=_FREQ.get(resolution, "1h"), timestamp_col=None
        ).synchronize(kalshi_df, poly_df)
        result, bt, train_days = run_oos(synced, train_days=30)
    except Exception as exc:
        print("Could not produce a real out-of-sample tear sheet:\n")
        print("  " + str(exc).replace("\n", "\n  "))
        print("\nThis script reports STRICTLY real, out-of-sample results -- no synthetic "
              "fallback, nothing fabricated. Provide PMXT_API_KEY + a liquid matched event, "
              "or KALSHI_SLUG / POLYMARKET_SLUG for a known deep-history pair.")
        return

    print(render_tear_sheet(bt, meta, n_folds=len(result.params), train_days=train_days))


if __name__ == "__main__":
    main()
