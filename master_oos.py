"""
master_oos.py
=============

Out-of-Sample Performance Tear Sheet on **real** cross-venue prediction-market data
(Kalshi vs Polymarket spread). No synthetic data anywhere in the real path.

    pmxt (matched Kalshi + Polymarket market, or explicit slugs)
        -> MarketDataPipeline      (align, quote-age + rolling-volume -> is_tradeable)
        -> [hardening] trim dead head/tail (resolution, pre-open) + strict execution mask
        -> WalkForwardOptimizer    (expanding train / 1d OOS; window + entry-z re-selected
                                    each fold from the OU half-life, friction-adj Sharpe)
        -> VectorizedBacktester    (signals shift(1) for look-ahead, INTERSECTED with the
                                    execution mask so we never trade a stale/illiquid quote)
        -> Markdown tear sheet     (metrics + DATA PROVENANCE + a signal-vs-noise check)

Usage
-----
    export PMXT_API_KEY=pmxt_live_...

    # 1) Discover current, liquid, deep matched pairs and their exact slugs:
    python master_oos.py --discover "election"
    python master_oos.py --discover "fed rate"

    # 2) Run on a chosen pair (most reliable -- pin the exact markets):
    KALSHI_SLUG=<kalshi-slug> POLYMARKET_SLUG=<polymarket-slug> \
        python master_oos.py --resolution 1h --lookback 400

    # or let the matcher pick by query (less reliable -- it favours novelty markets):
    python master_oos.py "presidential election"

    # format demo only (synthetic, loudly labelled; needs no key/network):
    python master_oos.py --illustrative

It never falls back to synthetic data in the real path and never fabricates metrics.
If the data is too thin to estimate anything, it FAILS LOUDLY with what went wrong.

Reading the output  (read this BEFORE believing a Sharpe)
---------------------------------------------------------
Real cross-venue history is thin, so the tear sheet prints a **Data provenance**
block (tradeable vs total bars, overlap window, folds, fraction of OOS bars traded)
and a **signal-vs-noise** check. The honest reading rules:

* **Sample size first.** Below ~30 round-trip trades or ~20 OOS daily returns, the
  Sharpe is noise *whatever its value* -- report it as inconclusive.
* **Significance floor.** With ``n`` OOS daily returns, an annualized Sharpe smaller
  than ``2 * sqrt(252 / n)`` is under 2 standard errors from zero. On 20 days that
  floor is ~7; on 60 days ~4; on 120 days ~2.9. A Sharpe *below* the floor is not
  distinguishable from luck -- so a big Sharpe on little data means little.
* **Concentration.** If the single best bar is >30% of the absolute P&L, the result
  rides on one print, not a repeatable edge.
* **Activity.** If <~5% of OOS bars trade, there were too few events to conclude
  anything.
* **What "disappointing but honest" looks like:** a Sharpe roughly in [-1, +1] with a
  handful of trades, drawdown of the same order as (or larger than) total return, and
  most P&L from one or two bars. That is the *expected* outcome on thin real data and
  is the correct thing to report -- a negative/near-zero OOS result is a real finding.
  A "Sharpe 8 on 6 trades" is noise dressed as alpha; the noise floor will say so.

Recommended starting pairs  (as of early 2026 -- VERIFY currency with --discover)
---------------------------------------------------------------------------------
Slugs/tickers drift and markets get archived; I cannot verify these are live from
here. Use ``--discover`` (or the Polymarket Gamma API / Kalshi API) to confirm the
current slug before trusting a name. Deepest cross-venue history tends to be:

  * 2024 US Presidential winner (Trump)  -- the longest, most-liquid cross-venue book
  * 2024 Presidential party / popular-vote
  * Balance of power: Senate / House control 2024
  * A specific FOMC meeting's rate decision (recurring; weeks-to-months each)

Confirm a Polymarket slug:
    curl -s "https://gamma-api.polymarket.com/markets?closed=true&order=volumeNum&ascending=false&limit=50" | python -c "import sys,json;[print(m['slug'],'|',m.get('volume')) for m in json.load(sys.stdin)]"
Confirm a Kalshi ticker/slug: browse the market page (ticker is in the URL) or hit the
Kalshi REST `/markets` endpoint. Best of all: ``--discover`` prints exactly what to pass.
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
_T = MarketDataPipeline.TRADEABLE_COL
_KA = MarketDataPipeline.KALSHI_AGE
_PA = MarketDataPipeline.POLYMARKET_AGE

# pmxt OHLCV resolution -> pandas resample frequency for the pipeline grid.
_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
_RES_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# Hardening thresholds (override via env). These gate the REAL path only.
MAX_EXEC_AGE = int(os.environ.get("OOS_MAX_EXEC_AGE", "1"))          # bars of staleness allowed at execution
MIN_TRADEABLE_BARS = int(os.environ.get("OOS_MIN_TRADEABLE_BARS", "100"))  # below this, can't estimate
MIN_OOS_DAYS = int(os.environ.get("OOS_MIN_OOS_DAYS", "5"))         # OOS days needed beyond the train window
PINNED_EPS = 0.02                                                   # |price-0| or |price-1| <= this == "resolved"


def _load_dotenv(filename: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (no dependency)."""
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
def _overlap_window(kdf: pd.DataFrame, pdf: pd.DataFrame):
    """(start, end, days) of the time span where BOTH venues have candles."""
    lo = max(kdf.index.min(), pdf.index.min())
    hi = min(kdf.index.max(), pdf.index.max())
    days = max(0.0, (hi - lo).total_seconds() / 86400.0)
    return lo, hi, days


def _median_cadence_seconds(df: pd.DataFrame):
    """Median spacing (seconds) between a venue's *raw* candles, or None if <2 rows."""
    if df is None or len(df.index) < 2:
        return None
    deltas = pd.Series(df.index).sort_values().diff().dropna().dt.total_seconds()
    return float(deltas.median()) if len(deltas) else None


def _cadence_notes(kdf, pdf, resolution: str):
    """Warn when a venue's native candle cadence is coarser than requested (-> lots
    of forward-fill -> mostly-stale bars) or the two venues differ a lot."""
    notes = []
    want = _RES_SECONDS.get(resolution)
    ck, cp = _median_cadence_seconds(kdf), _median_cadence_seconds(pdf)
    fmt = lambda s: "?" if s is None else (f"{s/3600:.1f}h" if s >= 3600 else f"{s/60:.0f}m")
    if want:
        for name, c in (("Kalshi", ck), ("Polymarket", cp)):
            if c is not None and c > 1.75 * want:
                notes.append(f"{name} native candles are ~{fmt(c)} apart but you requested "
                             f"{resolution} -- most bars will be forward-filled (stale) and "
                             f"flagged untradeable. Consider --resolution {fmt(c)}.")
    if ck and cp and (max(ck, cp) / min(ck, cp) > 3.0):
        notes.append(f"Venue cadences differ a lot (Kalshi ~{fmt(ck)} vs Polymarket ~{fmt(cp)}); "
                     "alignment is dominated by the coarser venue.")
    return notes, ck, cp


def _ohlcv_df(client, market, feed, resolution, start, end, limit):
    if market is None:
        return None
    candles = client.fetch_ohlcv(market.yes, resolution=resolution, start=start, end=end, limit=limit)
    return feed.candles_to_frame(candles) if candles else None


def fetch_matched_pair(query=None, *, kalshi_slug=None, polymarket_slug=None,
                       resolution="1h", lookback_days=400, limit=8000,
                       min_overlap_days=14):
    """Fetch a real Kalshi+Polymarket pair -> (kalshi_df, poly_df, meta).

    Modes: explicit ``kalshi_slug`` + ``polymarket_slug`` (most reliable), else the
    pmxt cross-venue matcher (needs PMXT_API_KEY). OHLCV uses the direct venue clients
    over a bounded window. Raises a clear RuntimeError on missing markets, empty
    history, or < ``min_overlap_days`` of overlap. Never fabricates data.

    Note: ``lookback_days`` is measured back from *now*. A market that traded long ago
    (e.g. the 2024 election) needs a large lookback to be reached at all -- the default
    400 days is generous; raise it (``--lookback``) for older resolved markets.
    """
    import time
    import datetime as dt
    import pmxt

    _load_dotenv()
    poly, kal, feed = pmxt.Polymarket(), pmxt.Kalshi(), PmxtFeed()
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=lookback_days)

    def fetch_one(client, slug, venue):
        if not slug:
            return None
        market = client.fetch_market(slug=slug)
        if market is None:
            raise RuntimeError(f"{venue}: no market found for slug {slug!r}. "
                               "Run `--discover` to list current slugs.")
        df = _ohlcv_df(client, market, feed, resolution, start, end, limit)
        if df is None:
            raise RuntimeError(f"{venue}: market {slug!r} returned no OHLCV in the last "
                               f"{lookback_days}d. Raise --lookback (older market?) or pick "
                               "a market with recent trading.")
        return df

    def _pack(kdf, pdf, event):
        lo, hi, ov = _overlap_window(kdf, pdf)
        if ov < min_overlap_days:
            raise RuntimeError(f"Only {ov:.1f} days of overlapping history (need >= "
                               f"{min_overlap_days}). The two markets barely coexist in time.")
        notes, ck, cp = _cadence_notes(kdf, pdf, resolution)
        return kdf, pdf, {"event": event, "resolution": resolution,
                          "kalshi_candles": len(kdf), "polymarket_candles": len(pdf),
                          "overlap_days": round(ov, 1), "overlap_start": lo, "overlap_end": hi,
                          "cadence_kalshi_s": ck, "cadence_poly_s": cp, "notes": notes}

    # --- Mode 1: explicit slugs ---
    if kalshi_slug and polymarket_slug:
        kdf = fetch_one(kal, kalshi_slug, "Kalshi")
        pdf = fetch_one(poly, polymarket_slug, "Polymarket")
        return _pack(kdf, pdf, f"{kalshi_slug} <-> {polymarket_slug}")

    # --- Mode 2: cross-venue matcher ---
    key = os.environ.get("PMXT_API_KEY")
    if not key:
        raise RuntimeError(
            "PMXT_API_KEY is not set and no explicit slugs were given. Cross-venue "
            "matching needs a pmxt.dev key:\n    export PMXT_API_KEY=pmxt_live_...\n"
            "or pass KALSHI_SLUG=... POLYMARKET_SLUG=... (see --discover).")
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
                           "Retry, or pass explicit slugs (--discover lists them).")
    clusters.sort(key=lambda c: getattr(c, "volume_24h", 0) or 0, reverse=True)

    best = 0.0
    for cluster in clusters:
        vm = _venue_markets(cluster)
        if not ({"kalshi", "polymarket"} <= set(vm)):
            continue
        try:
            kdf = _ohlcv_df(kal, kal.fetch_market(slug=getattr(vm["kalshi"], "slug", None)),
                            feed, resolution, start, end, limit)
            pdf = _ohlcv_df(poly, poly.fetch_market(slug=getattr(vm["polymarket"], "slug", None)),
                            feed, resolution, start, end, limit)
        except Exception:
            continue
        if kdf is None or pdf is None:
            continue
        _, _, ov = _overlap_window(kdf, pdf)
        best = max(best, ov)
        if ov >= min_overlap_days:
            return _pack(kdf, pdf, getattr(cluster, "canonical_title", None) or query)

    raise RuntimeError(
        f"No matched Kalshi+Polymarket market with >= {min_overlap_days} days of overlapping "
        f"history for query={query!r} (best found: {best:.1f} days). The live matcher mostly "
        "surfaces sparse, long-dated markets; pass explicit liquid slugs via --discover.")


def _venue_markets(cluster) -> dict:
    """Map venue -> matched market from a MatchedMarketCluster (venue in
    ``.source_exchange``, with ``.venue``/``.exchange`` fallbacks). ``.yes`` = YES outcome."""
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


def discover(query=None, resolution="1h", lookback_days=400, limit=8000, top=10) -> None:
    """List current matched Kalshi+Polymarket pairs (slugs, volume, overlap) so you can
    pick a liquid, deep one and pass its slugs explicitly. Solves the slug-currency
    problem -- print exactly what to paste."""
    import time
    import datetime as dt
    import pmxt

    _load_dotenv()
    key = os.environ.get("PMXT_API_KEY")
    if not key:
        print("--discover needs PMXT_API_KEY (export PMXT_API_KEY=pmxt_live_...).")
        return
    router = pmxt.Router(pmxt_api_key=key)
    poly, kal, feed = pmxt.Polymarket(), pmxt.Kalshi(), PmxtFeed()
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=lookback_days)

    clusters, last = [], None
    for i in range(6):
        try:
            clusters = router.fetch_matched_market_clusters(
                query=query, relation="identity", venues=["kalshi", "polymarket"],
                min_venues=2, limit=40)
            break
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    if not clusters:
        print(f"pmxt matcher returned nothing ({type(last).__name__ if last else 'empty'}). Retry.")
        return
    clusters.sort(key=lambda c: getattr(c, "volume_24h", 0) or 0, reverse=True)

    print(f"Probing up to {top} highest-volume matched pairs for query={query!r} "
          f"(resolution={resolution}, lookback={lookback_days}d)...\n")
    rows = []
    for cluster in clusters[:top]:
        vm = _venue_markets(cluster)
        if not ({"kalshi", "polymarket"} <= set(vm)):
            continue
        ks = getattr(vm["kalshi"], "slug", None)
        ps = getattr(vm["polymarket"], "slug", None)
        title = (getattr(cluster, "canonical_title", None) or query or "?")[:38]
        ov = kc = pc = None
        try:
            kdf = _ohlcv_df(kal, kal.fetch_market(slug=ks), feed, resolution, start, end, limit)
            pdf = _ohlcv_df(poly, poly.fetch_market(slug=ps), feed, resolution, start, end, limit)
            if kdf is not None and pdf is not None:
                _, _, ov = _overlap_window(kdf, pdf)
                kc, pc = len(kdf), len(pdf)
        except Exception:
            pass
        rows.append({"title": title, "ks": ks, "ps": ps, "ov": ov, "kc": kc, "pc": pc,
                     "vol": getattr(cluster, "volume_24h", 0) or 0})

    rows.sort(key=lambda r: (r["ov"] if r["ov"] is not None else -1), reverse=True)
    print(f"{'overlap':>8} {'k.cdl':>6} {'p.cdl':>6}  event")
    print("-" * 64)
    for r in rows:
        ov = f"{r['ov']:.0f}d" if r["ov"] is not None else "  n/a"
        print(f"{ov:>8} {str(r['kc'] or '-'):>6} {str(r['pc'] or '-'):>6}  {r['title']}")
        print(f"           KALSHI_SLUG={r['ks']}")
        print(f"           POLYMARKET_SLUG={r['ps']}")
    deep = next((r for r in rows if r["ov"] is not None), None)
    if deep:
        print(f"\nDeepest pair -> run:\n  KALSHI_SLUG={deep['ks']} POLYMARKET_SLUG={deep['ps']} "
              f"python master_oos.py --resolution {resolution} --lookback {lookback_days}")
    else:
        print("\nNo pair had reachable overlapping OHLCV in this window; try a larger --lookback.")


# --------------------------------------------------------------------------- #
# Hardening: trim dead ends, strict execution mask
# --------------------------------------------------------------------------- #
def _trim_dead_ends(synced: pd.DataFrame):
    """Trim leading/trailing UNtradeable runs (pre-open warmup, post-resolution dead
    tail) so the stats and walk-forward see only the live-trading window.

    Returns ``(trimmed, n_head, n_tail, resolved)`` where ``resolved`` is True when the
    trimmed tail looks like a resolution (price pinned near 0/1)."""
    if _T not in synced.columns:
        return synced, 0, 0, False
    mask = synced[_T].to_numpy(dtype=bool)
    if not mask.any():
        return synced, 0, 0, False     # nothing tradeable -> let gating raise clearly
    first = int(mask.argmax())
    last = int(len(mask) - 1 - mask[::-1].argmax())
    n_head, n_tail = first, len(mask) - 1 - last

    resolved = False
    if n_tail > 0:
        tail = synced.iloc[last + 1:]
        near01 = lambda s: float((((s <= PINNED_EPS) | (s >= 1 - PINNED_EPS)).mean())) if len(s) else 0.0
        if near01(tail[_K]) > 0.8 or near01(tail[_P]) > 0.8:
            resolved = True
    return synced.iloc[first:last + 1], int(n_head), int(n_tail), resolved


def _execution_mask(synced: pd.DataFrame, max_exec_age: int) -> pd.Series:
    """Stricter-than-liquidity mask for EXECUTION: tradeable AND both venues' quotes are
    at most ``max_exec_age`` bars stale. Feeding this to the backtester means a position
    is only ever held/marked across consecutive *fresh* quotes, so a price jump printed
    when a venue resumes after a gap is never booked as P&L (the bar before it is forced
    flat). This is the fix for 'gaps that survive the is_tradeable mask'."""
    base = synced[_T].to_numpy(dtype=bool)
    fresh = (synced[_KA].to_numpy() <= max_exec_age) & (synced[_PA].to_numpy() <= max_exec_age)
    return pd.Series(base & fresh, index=synced.index, name=_T)


def _noise_floor_sharpe(n_days) -> float:
    """Annualized-Sharpe magnitude below which the result is < 2 SE from zero, given
    ``n_days`` daily returns. (t = SR_ann * sqrt(n/252); |t|>2 -> SR_ann > 2*sqrt(252/n).)"""
    if not n_days or n_days < 2:
        return float("inf")
    return 2.0 * np.sqrt(252.0 / n_days)


# --------------------------------------------------------------------------- #
# Out-of-sample pipeline
# --------------------------------------------------------------------------- #
def run_oos(synced: pd.DataFrame, train_days: int = 30, *,
            trim_dead_ends: bool = False, max_exec_age=None,
            min_tradeable_bars: int = 0, min_oos_days: int = 0):
    """Walk-forward -> OOS signals (shift-1 + execution mask) -> backtest + provenance.

    Defaults reproduce the original behaviour exactly (no trimming, no strict mask, no
    extra gating) so the ``--illustrative`` path is byte-identical. The real path turns
    the hardening on. Returns ``(result, bt, train_days, provenance)``.
    """
    prov = {"notes": []}
    prov["total_bars_raw"] = int(len(synced))

    # 1) Trim dead head/tail (resolution / pre-open) so stats aren't polluted.
    if trim_dead_ends:
        synced, n_head, n_tail, resolved = _trim_dead_ends(synced)
        prov["trimmed_head"], prov["trimmed_tail"], prov["resolved_tail"] = n_head, n_tail, resolved
        if resolved:
            prov["notes"].append(f"Market appears RESOLVED inside the window: trimmed {n_tail} "
                                 "trailing bars pinned near 0/1. Analyzing the pre-resolution span only.")
        if not len(synced):
            raise RuntimeError("No tradeable bars at all after alignment -- the two markets never "
                               "have a simultaneously fresh, liquid quote. Check slugs / resolution.")

    prov["total_bars"] = int(len(synced))
    prov["overlap_start"], prov["overlap_end"] = synced.index.min(), synced.index.max()
    prov["overlap_days"] = round((prov["overlap_end"] - prov["overlap_start"]).total_seconds() / 86400.0, 1)

    # 2) Liquidity-tradeable count (the as-synchronized mask), then the strict exec mask.
    if _T in synced.columns:
        prov["tradeable_bars"] = int(synced[_T].sum())
        prov["tradeable_frac"] = prov["tradeable_bars"] / max(1, len(synced))
    if max_exec_age is not None and _T in synced.columns:
        execm = _execution_mask(synced, max_exec_age)
        synced = synced.copy()
        synced[_T] = execm.values            # the backtester intersects against THIS
        prov["executable_bars"] = int(execm.sum())
        prov["executable_frac"] = float(execm.mean())
        prov["max_exec_age"] = int(max_exec_age)
    else:
        prov["executable_bars"] = prov.get("tradeable_bars")
        prov["executable_frac"] = prov.get("tradeable_frac")

    # 3) Thin-data gating -- fail LOUDLY with specifics rather than emit a fake number.
    n_days = int(synced.index.normalize().nunique())
    prov["calendar_days"] = n_days
    if n_days < 4:
        raise RuntimeError(f"Only {n_days} calendar days of history; need a longer-lived market "
                           "(a walk-forward needs many days).")
    eff = prov.get("executable_bars") or 0
    if eff < min_tradeable_bars:
        raise RuntimeError(
            f"Only {eff} executable bars (need >= {min_tradeable_bars}). After alignment, "
            "freshness, and liquidity filtering there is too little real data to estimate a "
            "z-score / OU half-life or run a walk-forward. This market is too thin -- pick a "
            "more liquid/deeper pair (--discover) or a coarser --resolution.")
    train_days = max(2, min(train_days, n_days - 2))
    if min_oos_days and n_days < train_days + min_oos_days:
        raise RuntimeError(f"{n_days} calendar days leaves < {min_oos_days} out-of-sample days after a "
                           f"{train_days}-day train window. Need a longer history for a meaningful OOS.")

    # 4) Walk-forward + look-ahead-safe, tradeability-intersected backtest.
    try:
        result = WalkForwardOptimizer(train_days=train_days, test_days=1).run(synced)
    except Exception as e:
        raise RuntimeError(f"Walk-forward could not run on this (thin) data: {e}") from e
    oos = result.oos_signals
    if _T in synced.columns:
        oos = oos.join(synced[_T])
    bt = VectorizedBacktester(kalshi_fee=0.01, poly_slippage=0.005).run(oos, tradeable_col=_T)

    # 5) OOS provenance + signal-vs-noise diagnostics.
    prov["n_folds"] = int(len(result.params))
    prov["oos_bars"] = int(len(oos))
    if _T in oos.columns:
        prov["oos_executable_bars"] = int(oos[_T].astype(bool).sum())
    frame = bt.frame if bt.frame is not None else pd.DataFrame()
    prov["oos_traded_bars"] = int((frame.get("exec_position", pd.Series(dtype=float)) != 0).sum())
    prov["traded_frac"] = float(bt.pct_capital_locked)
    prov["n_round_trips"] = int(bt.n_round_trips)
    prov["n_daily_returns"] = int(bt.n_days)
    abspnl = frame.get("net_pnl", pd.Series(dtype=float)).abs()
    tot = float(abspnl.sum())
    prov["pnl_top1_frac"] = float(abspnl.max() / tot) if tot > 0 else float("nan")
    prov["pnl_top3_frac"] = float(abspnl.nlargest(3).sum() / tot) if tot > 0 else float("nan")
    prov["noise_floor_sharpe"] = _noise_floor_sharpe(bt.n_days)
    return result, bt, train_days, prov


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_num(x):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.2f}"


def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.1%}"


def _fmt_dt(x):
    try:
        return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(x)


def _provenance_block(p: dict, bt) -> list:
    """Markdown: data provenance + a signal-vs-noise sanity check (real path only)."""
    out = ["", "### Data provenance"]
    if p.get("overlap_start") is not None:
        out.append(f"- **Overlap window:** {_fmt_dt(p['overlap_start'])} → {_fmt_dt(p['overlap_end'])} "
                   f"({p.get('overlap_days', '?')} days)")
    out.append(f"- **Bars:** {p.get('total_bars', '?')} total | "
               f"{p.get('tradeable_bars', '?')} liquidity-tradeable ({_fmt_pct(p.get('tradeable_frac'))}) | "
               f"{p.get('executable_bars', '?')} executable ({_fmt_pct(p.get('executable_frac'))}, "
               f"≤{p.get('max_exec_age', '?')}-bar-stale both venues)")
    if "trimmed_head" in p:
        extra = "  ⟵ market RESOLVED in tail" if p.get("resolved_tail") else ""
        out.append(f"- **Trimmed dead ends:** {p['trimmed_head']} leading + {p['trimmed_tail']} trailing bars{extra}")
    ck, cp = p.get("cadence_kalshi_s"), p.get("cadence_poly_s")
    cfmt = lambda s: "?" if s is None else (f"{s/3600:.1f}h" if s >= 3600 else f"{s/60:.0f}m")
    out.append(f"- **Candle cadence (native):** Kalshi ~{cfmt(ck)} | Polymarket ~{cfmt(cp)} (requested {p.get('resolution','?')})")
    out.append(f"- **Walk-forward folds:** {p.get('n_folds', '?')}")
    out.append(f"- **OOS:** {p.get('oos_bars', '?')} bars; {p.get('oos_traded_bars', '?')} traded "
               f"(**{_fmt_pct(p.get('traded_frac'))} of OOS bars**); "
               f"{p.get('n_round_trips', '?')} round-trip trades over {p.get('n_daily_returns', '?')} daily returns")
    out.append(f"- **P&L concentration:** top bar {_fmt_pct(p.get('pnl_top1_frac'))} of |P&L|, "
               f"top 3 {_fmt_pct(p.get('pnl_top3_frac'))}")

    # --- signal vs noise ---
    floor = p.get("noise_floor_sharpe", float("inf"))
    n_ret = p.get("n_daily_returns", 0) or 0
    n_tr = p.get("n_round_trips", 0) or 0
    sharpe = bt.sharpe
    if sharpe is None or (isinstance(sharpe, float) and sharpe != sharpe):
        verdict = "**no verdict** — too few trades to compute a Sharpe"
    elif n_tr < 30 or n_ret < 20:
        verdict = (f"**NOISE** — only {n_tr} trades / {n_ret} daily returns; a Sharpe on this "
                   "little data is not interpretable, whatever its value")
    elif abs(sharpe) < floor:
        verdict = (f"**NOISE** — |{sharpe:.2f}| is below the {floor:.1f} significance floor "
                   f"(< 2 SE from zero on {n_ret} daily returns)")
    else:
        verdict = (f"clears the {floor:.1f} significance floor on {n_ret} daily returns — "
                   "**necessary, not sufficient** (still check concentration & out-of-period stability)")
    out += [
        "",
        "### Signal or noise?",
        f"- **Significance floor:** with {n_ret} OOS daily returns, |Sharpe| under **{floor:.1f}** "
        "is < 2 SE from zero.",
        f"- **Verdict:** {verdict}.",
        f"- **Reminder:** below ~30 trades / ~20 daily returns, or with the top bar >30 % of |P&L| "
        f"({_fmt_pct(p.get('pnl_top1_frac'))} here), treat any Sharpe as noise. A near-zero / negative "
        "OOS result on thin real data is the *expected, honest* outcome — report it as such.",
    ]
    notes = p.get("notes") or []
    if notes:
        out += ["", "### ⚠️ Data caveats"] + [f"- {n}" for n in notes]
    return out


def render_tear_sheet(bt, meta: dict, n_folds: int, train_days: int,
                      provenance: dict = None, illustrative: bool = False) -> str:
    """Markdown 'Out-of-Sample Performance Tear Sheet'.

    ``illustrative=True`` loudly stamps it synthetic. ``provenance`` (real path only)
    appends a data-provenance + signal-vs-noise block. With ``provenance=None`` and
    ``illustrative=True`` the output is byte-identical to the original demo sheet.
    """
    rows = [
        ("OOS Annualized Sharpe Ratio", _fmt_num(bt.sharpe)),
        ("Maximum Drawdown", _fmt_pct(bt.max_drawdown)),
        ("Percentage of Capital Locked", _fmt_pct(bt.pct_capital_locked)),
        ("Win Rate (executed trades)", _fmt_pct(bt.win_rate)),
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

    if provenance is not None:
        out += _provenance_block(provenance, bt)

    footer = ("_Illustrative output on **synthetic** data — not real performance. The "
              "methodology is strictly out-of-sample: walk-forward parameter selection, a "
              "1-period execution shift, and `is_tradeable` intersection._"
              if illustrative else
              "_All metrics are strictly out-of-sample (walk-forward) on real market data. "
              "No synthetic data is used in this pipeline. Read the 'Signal or noise?' block "
              "before trusting the Sharpe._")
    out += ["", footer]
    return "\n".join(out)


def _illustrative_tear_sheet() -> None:
    """Run the FULL OOS pipeline on SYNTHETIC data, loudly labeled, to show the format.

    Uses run_oos's default (un-hardened) behaviour and passes provenance=None, so this
    output is byte-identical to the published illustrative tear sheet."""
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
    result, bt, train_days, _prov = run_oos(synced, train_days=30)   # defaults => identical to before
    meta = {"event": "synthetic mean-reverting spread", "resolution": "1h",
            "kalshi_candles": len(kalshi_df), "polymarket_candles": len(poly_df)}
    print(render_tear_sheet(bt, meta, n_folds=len(result.params),
                            train_days=train_days, illustrative=True))


def _parse_args(argv):
    out = {"flags": set(), "query": [], "resolution": None, "lookback": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--resolution" and i + 1 < len(argv):
            out["resolution"] = argv[i + 1]; i += 2; continue
        if a == "--lookback" and i + 1 < len(argv):
            out["lookback"] = argv[i + 1]; i += 2; continue
        if a.startswith("--"):
            out["flags"].add(a); i += 1; continue
        out["query"].append(a); i += 1
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    args = _parse_args(sys.argv[1:])

    if "--illustrative" in args["flags"]:
        _illustrative_tear_sheet()
        return

    _load_dotenv()
    resolution = args["resolution"] or os.environ.get("OOS_RESOLUTION", "1h")
    lookback = int(args["lookback"] or os.environ.get("LOOKBACK_DAYS", "400"))
    query = " ".join(args["query"]).strip() or None

    if "--discover" in args["flags"]:
        discover(query=query or "election", resolution=resolution, lookback_days=lookback)
        return

    query = query or "presidential election"
    k_slug = os.environ.get("KALSHI_SLUG")
    p_slug = os.environ.get("POLYMARKET_SLUG")

    try:
        kalshi_df, poly_df, meta = fetch_matched_pair(
            query=query, kalshi_slug=k_slug, polymarket_slug=p_slug,
            resolution=resolution, lookback_days=lookback)
        synced = MarketDataPipeline(
            freq=_FREQ.get(resolution, "1h"), timestamp_col=None).synchronize(kalshi_df, poly_df)
        result, bt, train_days, prov = run_oos(
            synced, train_days=30, trim_dead_ends=True, max_exec_age=MAX_EXEC_AGE,
            min_tradeable_bars=MIN_TRADEABLE_BARS, min_oos_days=MIN_OOS_DAYS)
        prov["resolution"] = resolution
        prov["cadence_kalshi_s"] = meta.get("cadence_kalshi_s")
        prov["cadence_poly_s"] = meta.get("cadence_poly_s")
        prov["notes"] = list(meta.get("notes", [])) + list(prov.get("notes", []))
    except Exception as exc:
        print("Could not produce a real out-of-sample tear sheet:\n")
        print("  " + str(exc).replace("\n", "\n  "))
        print("\nThis script reports STRICTLY real, out-of-sample results -- no synthetic "
              "fallback, nothing fabricated. Provide PMXT_API_KEY + a liquid matched event, "
              "or KALSHI_SLUG / POLYMARKET_SLUG for a known deep-history pair. "
              "Run `python master_oos.py --discover \"<topic>\"` to list current liquid pairs.")
        return

    print(render_tear_sheet(bt, meta, n_folds=prov["n_folds"], train_days=train_days, provenance=prov))


if __name__ == "__main__":
    main()
