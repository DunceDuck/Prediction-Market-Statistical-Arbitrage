"""
data_pipeline.py
=======================

Data plumbing for a prediction-market statistical-arbitrage backtester.

The core deliverable is :class:`MarketDataPipeline`, which ingests two *raw*
historical price frames for the **same event/outcome** (e.g. the YES contract on
Kalshi and the YES contract on Polymarket), aligns them onto a uniform 1-minute
grid, forward-fills each venue independently, and emits a single synchronized
frame carrying a ``price_spread`` column (``|kalshi - polymarket|`` per minute).

A thin, *optional* adapter :class:`PmxtFeed` is also provided. It wraps the
``pmxt`` library ("CCXT for prediction markets", https://github.com/pmxt-dev/pmxt)
to pull OHLCV candles from either venue and return tidy price frames that drop
straight into :class:`MarketDataPipeline`.

Why pmxt as the data source
---------------------------
* One unified client covers **both** Kalshi and Polymarket (and others), so we
  do not maintain two bespoke API clients with two different auth schemes.
* It normalizes prices on every venue to a **0.0-1.0 probability scale**.
  Natively Kalshi quotes cents (0-100) and Polymarket quotes 0-1; pmxt removes
  that unit mismatch, which is precisely what makes a cross-venue price spread
  meaningful without manual rescaling.
* It ships a free historical OHLCV archive (archive.pmxt.dev) suitable for
  backtests.

The alternative -- ``kalshi-python`` + ``py-clob-client`` wired up separately --
works, but you pay for it with two clients, two auth flows, and a manual
cents->probability conversion that pmxt does for you. The pipeline below is
deliberately source-agnostic, so either path produces identical results.

Design notes that matter for a quant
-------------------------------------
* **Aggregation within a minute.** Sub-minute ticks are collapsed with ``last``
  (the closing quote of the minute) by default -- the price you could actually
  have transacted at the bar close. Configurable via ``agg``.
* **Forward-fill is per-venue and only fills *after* a venue's first print.**
  A stale quote is the best available estimate of the current price, but we
  never fabricate a price for a venue before it has traded (no look-back filling
  across the leading edge) or after it has closed (no filling past the last
  print).
* **Tradeable window.** With ``trim_to_overlap=True`` (default) the result is
  trimmed to the intersection where *both* venues are live, which is the only
  window in which a cross-venue spread is economically real. Inside that window
  the frame is gap-free, so ``price_spread`` is fully defined.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["MarketDataPipeline", "PmxtFeed"]

_VALID_AGGS = {"last", "first", "mean", "median", "max", "min"}


class MarketDataPipeline:
    """Synchronize two prediction-market price histories onto a common grid.

    Parameters
    ----------
    freq:
        Resampling frequency for the uniform grid. Defaults to ``"1min"``.
    price_col:
        Name of the price column to read from each input frame. Defaults to
        ``"price"``. (:class:`PmxtFeed` emits a ``"price"`` column = candle close.)
    timestamp_col:
        Name of the column holding timestamps. If ``None`` (default), the frame's
        existing index is used as the timestamp source.
    timestamp_unit:
        Unit for *epoch-integer* timestamps (``"s"``, ``"ms"``, ``"us"``, ``"ns"``).
        If ``None``, the unit is inferred from the magnitude of the values. Ignored
        for already-datetime or string timestamps.
    agg:
        Within-bin aggregation applied during resampling. One of
        ``{"last", "first", "mean", "median", "max", "min"}``. Defaults to ``"last"``.
    tz:
        Target timezone for the index. Defaults to ``"UTC"``.
    trim_to_overlap:
        If ``True`` (default), trim the output to the window where both venues
        have data, yielding a gap-free, fully-defined spread series. If ``False``,
        keep the full union of both ranges (leading/trailing minutes where only one
        venue exists will have ``NaN`` spread).
    scale_check:
        If ``True`` (default), emit a warning when the two price series appear to
        be on different scales (e.g. one in 0-1, the other in cents), which would
        make the spread meaningless.
    """

    KALSHI_COL = "kalshi_price"
    POLYMARKET_COL = "polymarket_price"
    SPREAD_COL = "price_spread"

    def __init__(
        self,
        freq: str = "1min",
        price_col: str = "price",
        timestamp_col: Optional[str] = None,
        timestamp_unit: Optional[str] = None,
        agg: str = "last",
        tz: str = "UTC",
        trim_to_overlap: bool = True,
        scale_check: bool = True,
    ) -> None:
        if agg not in _VALID_AGGS:
            raise ValueError(f"agg must be one of {sorted(_VALID_AGGS)}, got {agg!r}")
        self.freq = freq
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.timestamp_unit = timestamp_unit
        self.agg = agg
        self.tz = tz
        self.trim_to_overlap = trim_to_overlap
        self.scale_check = scale_check

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def synchronize(
        self,
        kalshi_df: pd.DataFrame,
        polymarket_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align two raw price frames and compute the per-minute price spread.

        Parameters
        ----------
        kalshi_df, polymarket_df:
            Raw historical price frames for the *same* outcome. Each must expose a
            price column (``price_col``) and a timestamp source (``timestamp_col``
            or the index).

        Returns
        -------
        pandas.DataFrame
            Indexed by a tz-aware 1-minute ``DatetimeIndex`` named ``timestamp``,
            with columns ``[kalshi_price, polymarket_price, price_spread]``.
        """
        kalshi = self._to_price_series(kalshi_df, "kalshi")
        poly = self._to_price_series(polymarket_df, "polymarket")

        if self.scale_check:
            self._warn_on_scale_mismatch(kalshi, poly)

        # Resample to the uniform grid and forward-fill EACH venue independently.
        kalshi_grid = self._resample_ffill(kalshi).rename(self.KALSHI_COL)
        poly_grid = self._resample_ffill(poly).rename(self.POLYMARKET_COL)

        # Outer-join on the union of the two minute grids. Both grids are anchored
        # to minute boundaries, so they align exactly on shared minutes.
        synced = pd.concat([kalshi_grid, poly_grid], axis=1)

        if self.trim_to_overlap:
            synced = self._trim_to_overlap(synced)

        # price_spread = |kalshi - polymarket| at every minute (numpy abs).
        # NaN on either side (only possible outside the overlap) propagates to NaN,
        # so we never report a spread against a non-existent quote.
        synced[self.SPREAD_COL] = np.abs(
            synced[self.KALSHI_COL] - synced[self.POLYMARKET_COL]
        )

        synced.index.name = "timestamp"
        return synced

    @staticmethod
    def spread_summary(synced: pd.DataFrame, spread_col: str = SPREAD_COL) -> dict:
        """Quick descriptive stats on the spread series (convenience for EDA)."""
        s = synced[spread_col].dropna()
        if s.empty:
            return {"n_minutes": 0}
        return {
            "n_minutes": int(s.shape[0]),
            "start": synced.index.min(),
            "end": synced.index.max(),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "max": float(s.max()),
            "p95": float(s.quantile(0.95)),
            "pct_gt_1c": float((s > 0.01).mean()),  # share of minutes wider than 1c
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _to_price_series(self, df: pd.DataFrame, label: str) -> pd.Series:
        """Return a clean, sorted, de-duplicated price Series on a DatetimeIndex."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{label}: expected a pandas DataFrame, got {type(df)!r}")
        if df.empty:
            raise ValueError(f"{label}: dataframe is empty")
        if self.price_col not in df.columns:
            raise KeyError(
                f"{label}: price_col {self.price_col!r} not found; "
                f"columns are {list(df.columns)}"
            )

        # 1) Resolve the timestamp source -> tz-aware DatetimeIndex.
        if self.timestamp_col is not None:
            if self.timestamp_col not in df.columns:
                raise KeyError(
                    f"{label}: timestamp_col {self.timestamp_col!r} not found; "
                    f"columns are {list(df.columns)}"
                )
            idx = self._coerce_datetime(df[self.timestamp_col].to_numpy(), label)
        else:
            idx = self._coerce_datetime(np.asarray(df.index), label)

        # 2) Coerce prices to numeric and assemble the Series.
        price = pd.to_numeric(df[self.price_col], errors="coerce").to_numpy()
        series = pd.Series(price, index=idx, name=label).sort_index()

        # 3) Drop unparseable prices and collapse duplicate timestamps (keep last).
        series = series[~series.isna()]
        series = series[~series.index.duplicated(keep="last")]
        if series.empty:
            raise ValueError(f"{label}: no valid numeric prices after cleaning")
        return series

    def _coerce_datetime(self, values: np.ndarray, label: str) -> pd.DatetimeIndex:
        """Convert an array of timestamps (datetime / string / epoch) to tz-aware index."""
        s = pd.Series(np.asarray(values))

        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s)
        elif pd.api.types.is_numeric_dtype(s):
            unit = self.timestamp_unit or self._infer_epoch_unit(s.to_numpy(), label)
            logger.debug("%s: treating numeric timestamps as epoch unit=%s", label, unit)
            dt = pd.to_datetime(s, unit=unit, utc=True)
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")

        dt = pd.DatetimeIndex(dt)
        if dt.isna().any():
            raise ValueError(
                f"{label}: {int(dt.isna().sum())} timestamp(s) could not be parsed"
            )

        # Localize naive indices, convert tz-aware ones, to the target tz.
        if dt.tz is None:
            dt = dt.tz_localize(self.tz)
        else:
            dt = dt.tz_convert(self.tz)
        return dt

    @staticmethod
    def _infer_epoch_unit(values: np.ndarray, label: str) -> str:
        """Infer the epoch unit of integer timestamps from their magnitude."""
        v = np.abs(np.asarray(values, dtype="float64"))
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"{label}: no finite timestamp values to infer unit from")
        median = float(np.median(v))
        # Rough boundaries (year ~2001-2033): s~1e9, ms~1e12, us~1e15, ns~1e18.
        if median >= 1e17:
            return "ns"
        if median >= 1e14:
            return "us"
        if median >= 1e11:
            return "ms"
        return "s"

    def _resample_ffill(self, series: pd.Series) -> pd.Series:
        """Resample to the uniform grid (within-bin ``agg``) then forward-fill gaps."""
        return series.resample(self.freq).agg(self.agg).ffill()

    def _trim_to_overlap(self, synced: pd.DataFrame) -> pd.DataFrame:
        """Trim to the minute range where both venues have a (ffilled) price."""
        both_live = synced[[self.KALSHI_COL, self.POLYMARKET_COL]].notna().all(axis=1)
        if not both_live.any():
            raise ValueError(
                "Kalshi and Polymarket price histories do not overlap in time; "
                "no cross-venue spread can be computed."
            )
        live_index = synced.index[both_live]
        return synced.loc[live_index[0] : live_index[-1]]

    def _warn_on_scale_mismatch(self, kalshi: pd.Series, poly: pd.Series) -> None:
        """Warn if the two series look like they are on different price scales."""
        k_max, p_max = float(kalshi.max()), float(poly.max())
        looks_prob = lambda x: x <= 1.5  # noqa: E731 - tiny local predicate
        if looks_prob(k_max) != looks_prob(p_max):
            warnings.warn(
                f"Possible price-unit mismatch: kalshi max={k_max:.4g}, "
                f"polymarket max={p_max:.4g}. One series looks like 0-1 probability "
                "and the other like cents/percent. price_spread will be meaningless "
                "unless both are on the same scale -- pre-scale your inputs (e.g. "
                "divide cents by 100) or pass already-normalized prices.",
                stacklevel=3,
            )


class PmxtFeed:
    """Optional adapter over the ``pmxt`` library to fetch venue price frames.

    This wrapper is intentionally tolerant of the exact candle representation
    (attribute-style ``PriceCandle`` objects, dicts, or ccxt-style lists), so it
    keeps working across minor pmxt versions.

    Example
    -------
    >>> import pmxt
    >>> kalshi = pmxt.Kalshi()          # public reads; pass keys for trading
    >>> poly = pmxt.Polymarket()
    >>> feed = PmxtFeed()
    >>> k_market = kalshi.fetch_market(slug="...")
    >>> p_market = poly.fetch_market(slug="...")
    >>> k_df = feed.fetch_ohlcv_frame(kalshi, k_market.yes, resolution="1m")
    >>> p_df = feed.fetch_ohlcv_frame(poly, p_market.yes, resolution="1m")
    >>> synced = MarketDataPipeline().synchronize(k_df, p_df)
    """

    _FIELDS = ("timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, tz: str = "UTC") -> None:
        self.tz = tz

    def fetch_ohlcv_frame(
        self,
        client,
        outcome_id,
        resolution: str = "1m",
        start=None,
        end=None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for one outcome and return a tidy price frame.

        Returns a DataFrame indexed by a tz-aware ``DatetimeIndex`` with columns
        ``[open, high, low, close, volume, price]`` where ``price`` is the close.
        """
        kwargs = {"resolution": resolution}
        if start is not None:
            kwargs["start"] = self._to_datetime_param(start)
        if end is not None:
            kwargs["end"] = self._to_datetime_param(end)
        if limit is not None:
            kwargs["limit"] = limit

        candles = client.fetch_ohlcv(outcome_id, **kwargs)
        return self.candles_to_frame(candles, tz=self.tz)

    def fetch_pair(
        self,
        kalshi_client,
        kalshi_outcome,
        polymarket_client,
        polymarket_outcome,
        resolution: str = "1m",
        start=None,
        end=None,
        limit: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch both venues' frames for the same event in one call."""
        k_df = self.fetch_ohlcv_frame(
            kalshi_client, kalshi_outcome, resolution, start, end, limit
        )
        p_df = self.fetch_ohlcv_frame(
            polymarket_client, polymarket_outcome, resolution, start, end, limit
        )
        return k_df, p_df

    @classmethod
    def candles_to_frame(cls, candles, tz: str = "UTC") -> pd.DataFrame:
        """Normalize a sequence of pmxt candles into a price DataFrame."""
        rows = [[cls._candle_get(c, f) for f in cls._FIELDS] for c in candles]
        df = pd.DataFrame(rows, columns=list(cls._FIELDS))
        if df.empty:
            raise ValueError("fetch_ohlcv returned no candles for this outcome/window")
        # pmxt timestamps are Unix milliseconds marking the start of each candle.
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if tz != "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        df = df.set_index("timestamp").sort_index()
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
        return df

    @classmethod
    def _candle_get(cls, candle, field: str):
        """Read one field from a candle regardless of its representation."""
        if hasattr(candle, field):
            return getattr(candle, field)
        if isinstance(candle, dict):
            return candle.get(field)
        if isinstance(candle, (list, tuple)):
            return candle[cls._FIELDS.index(field)]
        raise TypeError(f"Unsupported candle type: {type(candle)!r}")

    @staticmethod
    def _to_datetime_param(value):
        """Coerce a datetime-like value to a tz-aware ``datetime`` for pmxt.

        pmxt's ``fetch_ohlcv`` types ``start``/``end`` as ``datetime.datetime``
        (it does its own epoch conversion internally), so we hand it a real
        timezone-aware datetime rather than an epoch integer.
        """
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()


# ---------------------------------------------------------------------------- #
# Self-contained demo (runs with only pandas + numpy -- no network/keys needed)
# ---------------------------------------------------------------------------- #
def _demo() -> None:
    """Build two synthetic, irregular, partially-overlapping price feeds and sync."""
    rng = np.random.default_rng(7)

    # Kalshi: a print roughly every ~40s from 12:00, with random gaps, for ~90 min.
    base = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    k_offsets = np.cumsum(rng.integers(20, 70, size=160))  # seconds between prints
    k_times = base + pd.to_timedelta(k_offsets, unit="s")
    k_price = 0.50 + np.cumsum(rng.normal(0, 0.004, size=k_times.size))
    k_price = np.clip(k_price, 0.01, 0.99)
    kalshi_df = pd.DataFrame(
        {"timestamp": (k_times.view("int64") // 1_000_000), "price": k_price}  # epoch ms
    )

    # Polymarket: starts 15 min later, ends earlier, different cadence -> tests overlap.
    p_start = base + pd.Timedelta(minutes=15)
    p_offsets = np.cumsum(rng.integers(15, 90, size=110))
    p_times = p_start + pd.to_timedelta(p_offsets, unit="s")
    p_price = k_price[0] + 0.01 + np.cumsum(rng.normal(0, 0.005, size=p_times.size))
    p_price = np.clip(p_price, 0.01, 0.99)
    poly_df = pd.DataFrame({"timestamp": p_times, "price": p_price})  # datetime, not epoch

    pipeline = MarketDataPipeline(freq="1min", price_col="price", timestamp_col="timestamp")
    synced = pipeline.synchronize(kalshi_df, poly_df)

    pd.set_option("display.width", 120)
    print("Raw inputs:")
    print(f"  kalshi:     {len(kalshi_df):>4} ticks (epoch-ms timestamps)")
    print(f"  polymarket: {len(poly_df):>4} ticks (datetime timestamps, starts +15m)\n")
    print(f"Synchronized 1-minute frame: {synced.shape[0]} rows x {synced.shape[1]} cols")
    print(f"NaNs in spread (should be 0 within overlap): {int(synced['price_spread'].isna().sum())}\n")
    print("Head:")
    print(synced.head(5).round(4))
    print("\nTail:")
    print(synced.tail(5).round(4))
    print("\nSpread summary:")
    for key, val in MarketDataPipeline.spread_summary(synced).items():
        print(f"  {key:>10}: {val}")


if __name__ == "__main__":
    _demo()
