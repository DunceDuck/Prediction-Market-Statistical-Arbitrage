"""
alpha_engine.py
===============

Alpha-forecasting layer for the prediction-market stack -- the *single-venue*
counterpart to the cross-venue spread engine. Where ``signal_engine.py`` trades
the **gap between** Kalshi and Polymarket, this module mines a documented
inefficiency **within one venue**: the **favorite-longshot bias (FLB)**.

The favorite-longshot bias
--------------------------
Across racetracks, sportsbooks, and prediction markets, prices are an *imperfect*
map of true probability in a very regular way:

* **Longshots are over-bet.** A contract trading at an implied 5% tends to resolve
  YES *less* than 5% of the time -- the crowd overpays for lottery-like payoffs.
* **Favorites are under-bet.** A contract at an implied 95% tends to resolve YES
  *more* than 95% of the time -- the crowd underpays for near-certainties.

The tradable edge is therefore to **back favorites and fade longshots**, sized by
the estimated miscalibration. Before any of that, we need data: a clean panel of
**resolved** markets, each with its implied-probability **price time-series** and
its binary **resolution** (1 = YES, 0 = NO). That is what this module's scaffolding
delivers.

What's here (this module is the data layer)
-------------------------------------------
* :class:`ResolvedMarket` -- one resolved binary market: its price path + outcome.
* :class:`PolymarketDataPuller` -- pulls **resolved** markets from Polymarket's
  public APIs (Gamma for market metadata + resolution, CLOB for price history),
  with production-grade **retry / backoff / rate-limit** handling. The live path
  lazily imports ``requests`` (optional, like ``pmxt`` for the spread stack); the
  mock path and the demo need only pandas + numpy.
* :meth:`PolymarketDataPuller.generate_mock_markets` -- a synthetic generator that
  *intentionally* bakes in a favorite-longshot bias, so the downstream calibration
  / sizing logic can be built and tested before any live API key is wired in.

The mock bias is a one-parameter **logit-scaling distortion**. With
``bias_strength = a``, the true win probability is

    true = sigmoid( a * logit(implied) ) ,

which pivots exactly at 0.50: for ``a > 1`` every implied probability below 0.5 is
pulled *down* (longshots win less than priced) and every one above 0.5 is pushed
*up* (favorites win more than priced) -- the favorite-longshot bias by
construction. ``a = 1`` is perfectly calibrated (no bias), which the demo uses as a
control to prove the generator is doing what it claims.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ResolvedMarket", "PolymarketDataPuller"]

# Public Polymarket endpoints. Gamma serves market metadata + resolution; the CLOB
# serves per-token price history. Both are read-only and need no key for public data.
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

_YES_LABELS = {"yes", "true", "y"}
_PRICE_COL = "price"


# --------------------------------------------------------------------------- #
# Structured result
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedMarket:
    """A single **resolved** binary market: its price history and final outcome.

    Attributes
    ----------
    market_id:
        Venue-native id (or ``"mock-#####"`` for synthetic markets).
    resolution:
        Final outcome of the YES contract: ``1`` if YES won, ``0`` if NO won.
    implied_prob:
        A single reference implied probability for this market (the price level the
        bias is measured against). For synthetic data this is the generating belief
        ``p0``; for live data it is :meth:`reference_price` of the path (mean by
        default). Always on a 0-1 scale.
    prices:
        The implied-probability **time-series** (0-1), tz-aware ``DatetimeIndex``.
    question:
        Human-readable market question, when available.
    token_id:
        CLOB token id of the YES outcome (live markets only).
    resolved_at:
        Resolution / close timestamp, when available.
    true_prob:
        The *generating* win probability. Known only for synthetic data (used to
        verify the bias); ``None`` for live markets, where the truth is unobserved.
    source:
        ``"mock"`` or ``"polymarket"``.
    """

    market_id: str
    resolution: int
    implied_prob: float
    prices: pd.Series = field(repr=False)
    question: str = ""
    token_id: Optional[str] = None
    resolved_at: object = None
    true_prob: Optional[float] = None
    source: str = "mock"

    @property
    def n_obs(self) -> int:
        """Number of (non-NaN) price observations in the path."""
        return int(self.prices.dropna().shape[0]) if self.prices is not None else 0

    @property
    def final_price(self) -> float:
        """Last observed price (the market's terminal implied probability)."""
        s = self.prices.dropna() if self.prices is not None else pd.Series(dtype=float)
        return float(s.iloc[-1]) if not s.empty else float("nan")

    def reference_price(self, method: str = "mean") -> float:
        """Summarize the price path to one reference implied probability.

        ``method`` is one of ``"mean"`` (default), ``"median"``, ``"last"``,
        ``"first"`` -- the price level the favorite-longshot calibration is keyed on.
        """
        s = self.prices.dropna() if self.prices is not None else pd.Series(dtype=float)
        if s.empty:
            return float("nan")
        if method == "mean":
            return float(s.mean())
        if method == "median":
            return float(s.median())
        if method == "last":
            return float(s.iloc[-1])
        if method == "first":
            return float(s.iloc[0])
        raise ValueError(f"unknown reference price method {method!r}")

    def summary(self) -> dict:
        return {
            "market_id": self.market_id,
            "source": self.source,
            "resolution": self.resolution,
            "implied_prob": round(self.implied_prob, 4),
            "true_prob": None if self.true_prob is None else round(self.true_prob, 4),
            "final_price": round(self.final_price, 4),
            "n_obs": self.n_obs,
            "resolved_at": self.resolved_at,
        }


class PolymarketDataPuller:
    """Pull resolved Polymarket markets (live) or synthesize them (mock).

    Parameters
    ----------
    gamma_url, clob_url:
        API base URLs. Defaults to the public Polymarket endpoints.
    reference_method:
        How a live market's ``implied_prob`` is summarized from its price path
        (passed to :meth:`ResolvedMarket.reference_price`). Default ``"mean"``.
    price_interval:
        CLOB ``prices-history`` interval window (``"max"``, ``"1m"``, ``"1w"``...).
        Default ``"max"`` (the market's whole life).
    price_fidelity:
        CLOB sampling granularity in minutes (e.g. ``60`` for hourly). ``None``
        lets the API choose. Default ``None``.
    max_retries:
        Retry budget per request for *transient* failures (429, 5xx, timeouts,
        empty/garbled bodies). Default 5.
    backoff_base, backoff_cap:
        Exponential-backoff base and ceiling in seconds (with jitter). Default
        0.5 / 30.0.
    min_request_interval:
        Minimum spacing between requests in seconds -- a simple client-side rate
        limiter so we stay well under the venue's limits. Default 0.2.
    timeout:
        Per-request timeout in seconds. Default 20.
    user_agent:
        ``User-Agent`` header sent with every request.
    session:
        An optional pre-built ``requests.Session`` (or any object with a
        compatible ``.get``). If ``None``, one is lazily created on first use.
    """

    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        clob_url: str = CLOB_API_URL,
        reference_method: str = "mean",
        price_interval: str = "max",
        price_fidelity: Optional[int] = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        min_request_interval: float = 0.2,
        timeout: float = 20.0,
        user_agent: str = "alpha-engine/1.0 (+prediction-market-research)",
        session=None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_base <= 0 or backoff_cap <= 0:
            raise ValueError("backoff_base and backoff_cap must be positive")
        if min_request_interval < 0:
            raise ValueError("min_request_interval must be non-negative")
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self.reference_method = reference_method
        self.price_interval = price_interval
        self.price_fidelity = price_fidelity
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.min_request_interval = float(min_request_interval)
        self.timeout = float(timeout)
        self.user_agent = user_agent

        self._session = session
        self._requests = None          # the lazily-imported module
        self._retry_exc: tuple = ()    # retryable transport exceptions
        self._last_request: Optional[float] = None
        self._rng = np.random.default_rng()  # backoff jitter only (not result-affecting)

    # ------------------------------------------------------------------ #
    # Live data: resolved markets
    # ------------------------------------------------------------------ #
    def fetch_resolved_markets(
        self,
        max_markets: int = 200,
        page_limit: int = 100,
        with_prices: bool = True,
        skip_errors: bool = True,
        order: str = "volumeNum",
        ascending: bool = False,
        resolved_tol: float = 0.05,
        **gamma_filters,
    ) -> "list[ResolvedMarket]":
        """Page through Gamma's closed markets and return resolved binary markets.

        Each returned :class:`ResolvedMarket` carries the YES-contract price history
        (unless ``with_prices=False``) and its 0/1 resolution. Markets that are
        closed but not cleanly resolved (YES price not within ``resolved_tol`` of
        0 or 1 -- e.g. voided / 50-50), non-binary, or otherwise unparseable are
        skipped (logged) when ``skip_errors`` is ``True``, else they raise.

        ``order`` / ``ascending`` and any extra ``gamma_filters`` are forwarded as
        query params (defaults pull the highest-volume markets first -- the ones
        with the richest, most informative price histories).
        """
        if max_markets < 1 or page_limit < 1:
            raise ValueError("max_markets and page_limit must be >= 1")
        out: "list[ResolvedMarket]" = []
        offset = 0
        while len(out) < max_markets:
            params = {
                "closed": "true",
                "limit": int(page_limit),
                "offset": int(offset),
                "order": order,
                "ascending": str(bool(ascending)).lower(),
                **gamma_filters,
            }
            page = self._get_json(f"{self.gamma_url}/markets", params)
            rows = page.get("data", page) if isinstance(page, dict) else page
            if not rows:
                break  # exhausted
            for obj in rows:
                if len(out) >= max_markets:
                    break
                try:
                    market = self._parse_market(obj, with_prices, resolved_tol)
                except Exception as exc:  # noqa: BLE001 - one bad market must not kill the pull
                    logger.warning("skipping market id=%s: %s",
                                   (obj or {}).get("id"), exc)
                    if not skip_errors:
                        raise
                    continue
                if market is not None:
                    out.append(market)
            if len(rows) < page_limit:
                break  # last page
            offset += len(rows)
        logger.info("fetched %d resolved binary markets", len(out))
        return out

    def fetch_price_history(
        self,
        token_id: str,
        interval: Optional[str] = None,
        fidelity: Optional[int] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pd.Series:
        """Fetch one YES token's implied-probability time-series from the CLOB.

        Returns a 0-1 ``price`` Series indexed by a tz-aware ``DatetimeIndex``.
        Pass either ``interval`` (default :attr:`price_interval`) or an explicit
        ``start_ts``/``end_ts`` (unix seconds) window.
        """
        params = {"market": str(token_id)}
        if start_ts is not None or end_ts is not None:
            if start_ts is not None:
                params["startTs"] = int(start_ts)
            if end_ts is not None:
                params["endTs"] = int(end_ts)
        else:
            params["interval"] = interval or self.price_interval
        fid = fidelity if fidelity is not None else self.price_fidelity
        if fid is not None:
            params["fidelity"] = int(fid)

        data = self._get_json(f"{self.clob_url}/prices-history", params)
        history = data.get("history", []) if isinstance(data, dict) else (data or [])
        if not history:
            return pd.Series(dtype=float, name=_PRICE_COL)
        ts = [int(pt["t"]) for pt in history]
        px = [float(pt["p"]) for pt in history]
        idx = pd.to_datetime(ts, unit="s", utc=True)
        s = pd.Series(px, index=idx, name=_PRICE_COL).sort_index()
        return s[~s.index.duplicated(keep="last")]

    # ------------------------------------------------------------------ #
    # Mock data: a synthetic panel with a built-in favorite-longshot bias
    # ------------------------------------------------------------------ #
    def generate_mock_markets(
        self,
        n_markets: int = 600,
        bias_strength: float = 1.25,
        n_obs: int = 96,
        obs_freq: str = "1h",
        start: Optional[object] = None,
        span_days: Optional[float] = None,
        prob_range: "tuple[float, float]" = (0.02, 0.98),
        persistence: float = 0.92,
        noise: float = 0.25,
        seed: int = 7,
    ) -> "list[ResolvedMarket]":
        """Synthesize resolved markets that **intentionally** exhibit the FLB.

        For each market: draw a "true belief" implied probability ``p0`` uniformly
        across ``prob_range``; distort it to a *true* win probability via the
        logit-scaling map (:meth:`favorite_longshot_distort`) with ``bias_strength``;
        draw the binary resolution as ``Bernoulli(true)``; and generate a price
        path that mean-reverts (in logit space) around ``p0`` so the path's average
        is an unbiased read of the market's belief.

        Parameters
        ----------
        bias_strength:
            ``a`` in ``true = sigmoid(a * logit(p0))``. ``> 1`` => favorite-longshot
            bias (the default 1.25 is a *slight* bias); ``= 1`` => calibrated (no
            bias), useful as a control.
        span_days:
            If set, stagger markets' resolution dates uniformly over this many days
            (each path keeps its ``obs_freq`` cadence). ``None`` (default) puts all
            markets on one shared grid. Staggered dates give a downstream fade
            strategy a genuine time-series P&L.
        persistence, noise:
            AR(1) coefficient and innovation std of the logit-space price path
            (path realism only; the calibration is keyed on ``p0``).

        Notes
        -----
        Real markets also converge to 0/1 as resolution nears; that end-of-life
        drift is deliberately *omitted* here because it is orthogonal to the
        price-level -> outcome calibration the FLB alpha studies, and including it
        would muddy the reference price. Add it downstream if modeling execution.
        """
        if n_markets < 1:
            raise ValueError("n_markets must be >= 1")
        if bias_strength <= 0:
            raise ValueError("bias_strength must be positive")
        if n_obs < 1:
            raise ValueError("n_obs must be >= 1")
        lo, hi = prob_range
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("prob_range must satisfy 0 < lo < hi < 1")
        if not (0.0 <= persistence < 1.0):
            raise ValueError("persistence must be in [0, 1)")

        rng = np.random.default_rng(seed)
        m = int(n_markets)

        # 1) Each market's "belief" implied probability, spanning the whole range.
        p0 = rng.uniform(lo, hi, size=m)
        # 2) Distort belief -> true win probability (the bias), then draw outcomes.
        true_p = self.favorite_longshot_distort(p0, bias_strength)
        outcomes = (rng.random(m) < true_p).astype(int)

        # 3) Logit-space AR(1) price paths centered on each market's belief.
        mu = _logit(p0)                                   # (m,)
        eps = rng.standard_normal((m, int(n_obs)))
        x = np.empty((m, int(n_obs)), dtype=float)
        x[:, 0] = mu
        for t in range(1, int(n_obs)):
            x[:, t] = mu + persistence * (x[:, t - 1] - mu) + noise * eps[:, t]
        paths = _sigmoid(x)                               # (m, n_obs) in (0, 1)

        start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp("2025-01-01")
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        base_grid = pd.date_range(start_ts, periods=int(n_obs), freq=obs_freq, tz="UTC")
        # Optionally stagger resolution dates across a span (each path keeps its
        # obs_freq cadence) so a downstream fade strategy has a real time-series P&L.
        if span_days is None:
            offsets = [pd.Timedelta(0)] * m
        elif span_days < 0:
            raise ValueError("span_days must be non-negative")
        else:
            offsets = pd.to_timedelta(rng.uniform(0.0, float(span_days), size=m), unit="D")

        markets: "list[ResolvedMarket]" = []
        for i in range(m):
            grid_i = base_grid + offsets[i]
            series = pd.Series(paths[i], index=grid_i, name=_PRICE_COL)
            markets.append(
                ResolvedMarket(
                    market_id=f"mock-{i:05d}",
                    resolution=int(outcomes[i]),
                    implied_prob=float(p0[i]),
                    prices=series,
                    question=f"Mock resolved market #{i} (belief p0={p0[i]:.3f})",
                    token_id=None,
                    resolved_at=grid_i[-1],
                    true_prob=float(true_p[i]),
                    source="mock",
                )
            )
        return markets

    # ------------------------------------------------------------------ #
    # Aggregation helpers (the inputs the calibration/sizing model consumes)
    # ------------------------------------------------------------------ #
    @classmethod
    def snapshot_frame(cls, markets: Sequence[ResolvedMarket]) -> pd.DataFrame:
        """One row per market: reference price, outcome, and diagnostics."""
        rows = [
            {
                "market_id": mk.market_id,
                "source": mk.source,
                "implied_prob": mk.implied_prob,
                "true_prob": mk.true_prob if mk.true_prob is not None else np.nan,
                "resolution": mk.resolution,
                "final_price": mk.final_price,
                "mean_price": mk.reference_price("mean"),
                "n_obs": mk.n_obs,
                "resolved_at": mk.resolved_at,
            }
            for mk in markets
        ]
        return pd.DataFrame(rows)

    @classmethod
    def panel_frame(cls, markets: Sequence[ResolvedMarket]) -> pd.DataFrame:
        """Long/tidy **time-series** panel: one row per (market, timestamp).

        Columns: ``market_id``, ``timestamp``, ``implied_prob`` (the market's price
        at that instant), ``resolution`` (constant within a market), ``source``.
        This is the frame :class:`bias_analyzer.BiasAnalyzer` buckets to build a
        *time-series* calibration curve -- and ``market_id`` is the cluster key its
        bootstrap resamples over, so correlated within-market points don't
        masquerade as independent samples.
        """
        cols = ["market_id", "timestamp", "implied_prob", "resolution", "source"]
        frames = []
        for mk in markets:
            s = mk.prices.dropna() if mk.prices is not None else pd.Series(dtype=float)
            if s.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "market_id": mk.market_id,
                        "timestamp": s.index,
                        "implied_prob": s.to_numpy(dtype=float),
                        "resolution": int(mk.resolution),
                        "source": mk.source,
                    }
                )
            )
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)[cols]

    @classmethod
    def calibration_table(
        cls,
        markets: Sequence[ResolvedMarket],
        bins: int = 10,
        price_col: str = "implied_prob",
    ) -> pd.DataFrame:
        """Bucket markets by implied price and compare to the empirical win rate.

        Columns: ``n``, ``mean_implied``, ``win_rate`` (empirical YES frequency),
        ``mean_true`` (only if synthetic), and ``edge = win_rate - mean_implied``.
        Under a favorite-longshot bias ``edge`` is **negative for low buckets**
        (longshots win less than priced) and **positive for high buckets**
        (favorites win more) -- and the sign of ``edge`` is the trade.
        """
        df = cls.snapshot_frame(markets)
        if df.empty:
            return pd.DataFrame()
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        df = df.assign(bucket=pd.cut(df[price_col], bins=edges, include_lowest=True))
        grp = df.groupby("bucket", observed=False)
        table = pd.DataFrame(
            {
                "n": grp.size(),
                "mean_implied": grp[price_col].mean(),
                "win_rate": grp["resolution"].mean(),
            }
        )
        if df["true_prob"].notna().any():
            table["mean_true"] = grp["true_prob"].mean()
        table["edge"] = table["win_rate"] - table["mean_implied"]
        return table

    # ------------------------------------------------------------------ #
    # Bias math (pure, vectorized, independently testable)
    # ------------------------------------------------------------------ #
    @staticmethod
    def favorite_longshot_distort(implied, bias_strength: float, eps: float = 1e-9):
        """Map implied probability -> true win probability via logit scaling.

        ``true = sigmoid(bias_strength * logit(implied))``. ``bias_strength > 1``
        produces the favorite-longshot bias (pivoting at 0.5); ``= 1`` is the
        identity (calibrated). Accepts scalars or arrays.
        """
        p = np.clip(np.asarray(implied, dtype=float), eps, 1.0 - eps)
        out = _sigmoid(bias_strength * _logit(p))
        return out if np.ndim(out) else float(out)

    @staticmethod
    def calibration_slope(implied, outcome) -> float:
        """Pooled slope of ``(outcome - implied)`` on ``(implied - 0.5)``.

        A single, binning-free summary of miscalibration: ``> 0`` is a
        favorite-longshot bias (favorites under-priced, longshots over-priced),
        ``~ 0`` is calibrated. More robust than any single bucket's win rate.
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

    # ------------------------------------------------------------------ #
    # HTTP internals: lazy session, throttle, retry/backoff, parsing
    # ------------------------------------------------------------------ #
    def _ensure_session(self):
        """Lazily import ``requests`` and build a session (live path only)."""
        if self._session is not None:
            return self._session
        try:
            import requests  # noqa: PLC0415 - optional dependency, imported on demand
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Live Polymarket pulls require the 'requests' package "
                "(pip install requests). The mock generator needs no network/deps."
            ) from exc
        self._requests = requests
        self._retry_exc = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})
        return self._session

    def _throttle(self) -> None:
        """Block until at least ``min_request_interval`` has passed since the last call."""
        if self.min_request_interval <= 0 or self._last_request is None:
            return
        wait = self.min_request_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float] = None) -> None:
        """Sleep before a retry: honor ``Retry-After`` if given, else jittered backoff."""
        if retry_after is not None:
            time.sleep(max(0.0, retry_after))
            return
        delay = min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempt - 1)))
        time.sleep(delay + float(self._rng.uniform(0.0, self.backoff_base)))

    def _get_json(self, url: str, params: dict):
        """GET ``url`` with retry/backoff/rate-limit handling; return parsed JSON.

        Retries transient failures -- HTTP 429 (respecting ``Retry-After``), 5xx,
        connection/timeout errors, and empty/garbled bodies -- up to
        ``max_retries`` with exponential, jittered backoff. Non-429 4xx are client
        errors and raise immediately (retrying won't help).
        """
        session = self._ensure_session()
        attempt = 0
        while True:
            self._throttle()
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                self._last_request = time.monotonic()
                status = resp.status_code

                if status == 429 or 500 <= status < 600:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(
                            f"GET {url} failed after {self.max_retries} retries "
                            f"(last status {status})"
                        )
                    retry_after = self._parse_retry_after(resp) if status == 429 else None
                    logger.warning("GET %s -> %s; retry %d/%d",
                                   url, status, attempt, self.max_retries)
                    self._sleep_backoff(attempt, retry_after)
                    continue
                if status >= 400:
                    raise RuntimeError(f"GET {url} -> client error {status}: {resp.text[:200]}")

                try:
                    return resp.json()
                except ValueError:  # empty / non-JSON body -> transient, retry
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(
                            f"GET {url} returned a non-JSON body after "
                            f"{self.max_retries} retries"
                        )
                    logger.warning("GET %s returned non-JSON body; retry %d/%d",
                                   url, attempt, self.max_retries)
                    self._sleep_backoff(attempt)
                    continue

            except self._retry_exc as exc:  # connection reset, timeout, etc.
                self._last_request = time.monotonic()
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"GET {url} failed after {self.max_retries} retries: {exc}"
                    ) from exc
                logger.warning("GET %s transport error (%s); retry %d/%d",
                               url, exc, attempt, self.max_retries)
                self._sleep_backoff(attempt)
                continue

    @staticmethod
    def _parse_retry_after(resp) -> Optional[float]:
        """Best-effort parse of a ``Retry-After`` header (seconds)."""
        raw = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None  # HTTP-date form: fall back to normal backoff

    def _parse_market(
        self, obj: dict, with_prices: bool, resolved_tol: float
    ) -> Optional[ResolvedMarket]:
        """Turn one Gamma market object into a :class:`ResolvedMarket`, or ``None``.

        Returns ``None`` (skip) for non-binary or not-cleanly-resolved markets.
        """
        outcomes = self._as_list(obj.get("outcomes"))
        out_prices = self._as_list(obj.get("outcomePrices"))
        token_ids = self._as_list(obj.get("clobTokenIds"))
        if not (outcomes and out_prices and token_ids):
            return None
        if len(outcomes) != 2 or len(out_prices) != 2 or len(token_ids) != 2:
            return None  # binary markets only

        yes_idx = next(
            (i for i, o in enumerate(outcomes) if str(o).strip().lower() in _YES_LABELS),
            0,  # fall back to the first listed outcome as "YES"
        )
        yes_price = float(out_prices[yes_idx])
        resolution = int(round(yes_price))
        # A genuinely resolved market settles its YES price at ~0 or ~1. Anything in
        # between (voided / 50-50 / still-pending) is not a clean label -> skip.
        if min(abs(yes_price - 0.0), abs(yes_price - 1.0)) > resolved_tol:
            return None

        token = str(token_ids[yes_idx])
        if with_prices:
            series = self.fetch_price_history(token)
        else:
            series = pd.Series(dtype=float, name=_PRICE_COL)

        market = ResolvedMarket(
            market_id=str(obj.get("id") or obj.get("conditionId") or token),
            resolution=resolution,
            implied_prob=float("nan"),
            prices=series,
            question=str(obj.get("question") or obj.get("slug") or ""),
            token_id=token,
            resolved_at=obj.get("closedTime") or obj.get("endDate"),
            true_prob=None,
            source="polymarket",
        )
        market.implied_prob = (
            market.reference_price(self.reference_method) if market.n_obs else float("nan")
        )
        return market

    @staticmethod
    def _as_list(value):
        """Coerce a Gamma field to a list (it returns JSON-encoded strings or lists)."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except (ValueError, TypeError):
                return [s]
            return parsed if isinstance(parsed, list) else [parsed]
        return list(value)


# --------------------------------------------------------------------------- #
# Small numeric helpers (kept module-level so the math is reused, not duplicated)
# --------------------------------------------------------------------------- #
def _logit(p):
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Self-contained demo (mock data only -- no network, no API key, pandas+numpy)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    puller = PolymarketDataPuller()

    # A biased panel (slight FLB) and a calibrated CONTROL from the same seed.
    n = 2500
    biased = puller.generate_mock_markets(n_markets=n, bias_strength=1.25, seed=7)
    control = puller.generate_mock_markets(n_markets=n, bias_strength=1.00, seed=7)

    print(f"=== PolymarketDataPuller mock panel: {len(biased)} resolved markets ===\n")

    # Show that each market carries a price *time-series* + a binary resolution.
    ex = next(mk for mk in biased if mk.implied_prob < 0.12)  # a longshot
    print(f"Example longshot market  id={ex.market_id}  resolution={ex.resolution}")
    print(f"  implied_prob (belief)  : {ex.implied_prob:.3f}")
    print(f"  true win prob (hidden) : {ex.true_prob:.3f}   <- below implied (over-priced longshot)")
    print(f"  price series ({ex.n_obs} obs), head:")
    print(ex.prices.head(4).round(4).to_string())

    # Calibration table: empirical win rate vs implied, by decile of implied price.
    table = PolymarketDataPuller.calibration_table(biased, bins=10)
    show = table[table["n"] > 0].copy()
    show["win_rate"] = show["win_rate"].round(3)
    show["mean_implied"] = show["mean_implied"].round(3)
    show["mean_true"] = show["mean_true"].round(3)
    show["edge"] = show["edge"].round(3)
    print("\nCalibration by implied-price decile (edge = empirical win_rate - implied):")
    print(show[["n", "mean_implied", "mean_true", "win_rate", "edge"]].to_string())

    # --- Verification: the bias is present in the BIASED panel and absent in the
    # control. Use a pooled slope (robust) plus the tercile edges (interpretable). ---
    bdf = PolymarketDataPuller.snapshot_frame(biased)
    cdf = PolymarketDataPuller.snapshot_frame(control)
    slope_b = PolymarketDataPuller.calibration_slope(bdf["implied_prob"], bdf["resolution"])
    slope_c = PolymarketDataPuller.calibration_slope(cdf["implied_prob"], cdf["resolution"])

    lo = bdf[bdf["implied_prob"] < 1 / 3]
    hi = bdf[bdf["implied_prob"] > 2 / 3]
    lo_edge = lo["resolution"].mean() - lo["implied_prob"].mean()
    hi_edge = hi["resolution"].mean() - hi["implied_prob"].mean()

    print("\n--- Favorite-longshot bias check ---")
    print(f"  Longshot tercile (p<1/3): win {lo['resolution'].mean():.3f} "
          f"vs implied {lo['implied_prob'].mean():.3f}  -> edge {lo_edge:+.3f}")
    print(f"  Favorite tercile (p>2/3): win {hi['resolution'].mean():.3f} "
          f"vs implied {hi['implied_prob'].mean():.3f}  -> edge {hi_edge:+.3f}")
    print(f"  Calibration slope        : biased {slope_b:+.3f}  vs  control {slope_c:+.3f}")

    ok = (lo_edge < 0 < hi_edge) and (slope_b > 0.02) and (slope_b > 2 * abs(slope_c))
    verdict = "OK" if ok else "UNEXPECTED"
    print(f"\n[{verdict}] longshots win LESS than priced, favorites win MORE; "
          f"the calibrated control shows ~no slope ({slope_c:+.3f}).")
    print("Tradable read: fade the longshots (sell YES), back the favorites (buy YES).")


if __name__ == "__main__":
    _demo()
