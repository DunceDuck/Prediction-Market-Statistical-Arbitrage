"""
data_pipeline.py
================

Data plumbing for a prediction-market statistical-arbitrage backtester.

:class:`MarketDataPipeline` ingests two *raw* tick-or-1-minute frames carrying
**Price and Volume** for the same event on two venues (Kalshi, Polymarket),
aligns them onto a uniform 1-minute grid, and emits a synchronized frame with a
per-minute **``is_tradeable`` mask**.

Eliminating "phantom liquidity"
-------------------------------
A naive ``ffill()`` is dangerous in a backtest: when a venue stops quoting, the
last price is carried forward indefinitely and *looks* like a live, fillable
quote. A strategy then "trades" against a price nobody was actually showing —
**phantom liquidity** — which silently inflates fills and returns.

This pipeline still carries the last price forward (so a value exists for marking
and spread computation), but it never lets that be mistaken for tradeable
liquidity. Two guards are tracked per venue, per minute:

1. **``quote_age``** — minutes since the last *fresh* print. A new print resets it
   to 0; each carried-forward minute increments it. A quote older than ``max_age``
   is stale.
2. **rolling volume** — the trailing ``vol_window``-minute traded volume. Zero
   recent volume means the book is dry even if a price is being quoted.

:meth:`filter_stale_quotes` combines them into ``is_tradeable``: a minute is
**untradeable (False)** if *either* venue's ``quote_age`` exceeds ``max_age`` *or*
*either* venue's rolling volume is 0. Both legs must be fresh **and** liquid to
trade the spread, since execution requires hitting both books.

A thin, optional :class:`PmxtFeed` adapter wraps the ``pmxt`` library to pull
OHLCV candles (which already carry volume) from either venue.
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
    """Synchronize two venue price/volume feeds and flag tradeable minutes.

    Parameters
    ----------
    freq:
        Resampling frequency for the uniform grid. Defaults to ``"1min"``.
    price_col, volume_col:
        Column names for price and volume in each input frame. Default
        ``"price"`` / ``"volume"``.
    timestamp_col:
        Column holding timestamps. If ``None`` (default), the frame's index is used.
    timestamp_unit:
        Unit for epoch-integer timestamps (``"s"``/``"ms"``/``"us"``/``"ns"``);
        inferred from magnitude if ``None``.
    agg:
        Within-minute price aggregation (default ``"last"`` -- the bar's closing
        quote). Volume is always **summed** within the minute.
    tz:
        Target timezone for the index. Defaults to ``"UTC"``.
    max_age:
        Maximum tolerated ``quote_age`` (minutes) before a quote is stale.
        Default 5.
    vol_window:
        Look-back (minutes) for the rolling traded-volume liquidity check.
        Default 5.
    scale_check:
        If ``True`` (default), warn when the two price series look like they are on
        different scales (0-1 vs cents).
    """

    KALSHI_COL = "kalshi_price"
    POLYMARKET_COL = "polymarket_price"
    SPREAD_COL = "price_spread"
    TRADEABLE_COL = "is_tradeable"
    KALSHI_AGE = "kalshi_quote_age"
    POLYMARKET_AGE = "polymarket_quote_age"
    KALSHI_VOL = "kalshi_volume"
    POLYMARKET_VOL = "polymarket_volume"
    KALSHI_RVOL = "kalshi_roll_vol"
    POLYMARKET_RVOL = "polymarket_roll_vol"

    def __init__(
        self,
        freq: str = "1min",
        price_col: str = "price",
        volume_col: str = "volume",
        timestamp_col: Optional[str] = None,
        timestamp_unit: Optional[str] = None,
        agg: str = "last",
        tz: str = "UTC",
        max_age: int = 5,
        vol_window: int = 5,
        scale_check: bool = True,
    ) -> None:
        if agg not in _VALID_AGGS:
            raise ValueError(f"agg must be one of {sorted(_VALID_AGGS)}, got {agg!r}")
        if max_age < 0:
            raise ValueError("max_age must be non-negative")
        if vol_window < 1:
            raise ValueError("vol_window must be >= 1")
        self.freq = freq
        self.price_col = price_col
        self.volume_col = volume_col
        self.timestamp_col = timestamp_col
        self.timestamp_unit = timestamp_unit
        self.agg = agg
        self.tz = tz
        self.max_age = max_age
        self.vol_window = vol_window
        self.scale_check = scale_check
        self._synced: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def synchronize(
        self,
        kalshi_df: pd.DataFrame,
        polymarket_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align two raw price/volume frames onto a 1-minute grid with staleness
        and liquidity tracking, and attach the ``is_tradeable`` mask.

        Returns
        -------
        pandas.DataFrame
            Indexed by a tz-aware 1-minute ``DatetimeIndex`` named ``timestamp``,
            with columns: ``kalshi_price``, ``polymarket_price``, ``price_spread``,
            ``kalshi_quote_age``, ``polymarket_quote_age``, ``kalshi_volume``,
            ``polymarket_volume``, ``kalshi_roll_vol``, ``polymarket_roll_vol``,
            and ``is_tradeable``.
        """
        legs = {}
        for label, raw in (("kalshi", kalshi_df), ("polymarket", polymarket_df)):
            price, volume = self._clean_leg(raw, label)
            legs[label] = (
                price.resample(self.freq).agg(self.agg),  # last price (NaN if empty)
                volume.resample(self.freq).sum(),          # total volume (0 if empty)
            )

        if self.scale_check:
            self._warn_on_scale_mismatch(
                legs["kalshi"][0].dropna(), legs["polymarket"][0].dropna()
            )

        # Common 1-minute grid spanning both venues (union of the per-leg grids).
        common = legs["kalshi"][0].index.union(legs["polymarket"][0].index)

        cols = {}
        names = {
            "kalshi": (self.KALSHI_COL, self.KALSHI_AGE, self.KALSHI_VOL, self.KALSHI_RVOL),
            "polymarket": (self.POLYMARKET_COL, self.POLYMARKET_AGE,
                           self.POLYMARKET_VOL, self.POLYMARKET_RVOL),
        }
        for label in ("kalshi", "polymarket"):
            price_1m, vol_1m = legs[label]
            price_1m = price_1m.reindex(common)          # NaN where this venue is silent
            vol_1m = vol_1m.reindex(common).fillna(0.0)   # no ticks -> 0 volume
            pcol, acol, vcol, rcol = names[label]
            cols[acol] = self._quote_age(price_1m)        # staleness BEFORE ffill
            cols[pcol] = price_1m.ffill()                 # value carried forward
            cols[vcol] = vol_1m
            cols[rcol] = vol_1m.rolling(self.vol_window, min_periods=1).sum()

        synced = pd.DataFrame(cols, index=common)
        synced.index.name = "timestamp"

        # Drop the leading region before BOTH venues have printed a first quote
        # (price is genuinely undefined there). Everything after is kept; stale or
        # illiquid minutes remain in the frame but are flagged untradeable.
        both_quoted = synced[[self.KALSHI_COL, self.POLYMARKET_COL]].notna().all(axis=1)
        if not both_quoted.any():
            raise ValueError("Kalshi and Polymarket histories never overlap in time.")
        synced = synced.loc[both_quoted.idxmax():]

        synced[self.SPREAD_COL] = np.abs(
            synced[self.KALSHI_COL] - synced[self.POLYMARKET_COL]
        )

        self._synced = synced
        synced[self.TRADEABLE_COL] = self.filter_stale_quotes(self.max_age)

        ordered = [
            self.KALSHI_COL, self.POLYMARKET_COL, self.SPREAD_COL,
            self.KALSHI_AGE, self.POLYMARKET_AGE,
            self.KALSHI_VOL, self.POLYMARKET_VOL,
            self.KALSHI_RVOL, self.POLYMARKET_RVOL,
            self.TRADEABLE_COL,
        ]
        synced = synced[ordered]
        self._synced = synced
        return synced

    def filter_stale_quotes(self, max_age: int = 5) -> pd.Series:
        """Boolean tradeability mask over the synchronized frame.

        A minute is ``False`` (untradeable) if **either** venue's ``quote_age``
        exceeds ``max_age`` **or** **either** venue's rolling volume is 0 -- i.e.
        a minute is tradeable only when both legs are simultaneously *fresh* and
        *liquid*. Call :meth:`synchronize` first.
        """
        if self._synced is None:
            raise RuntimeError("Call synchronize() before filter_stale_quotes().")
        df = self._synced
        fresh = (df[self.KALSHI_AGE] <= max_age) & (df[self.POLYMARKET_AGE] <= max_age)
        liquid = (df[self.KALSHI_RVOL] > 0) & (df[self.POLYMARKET_RVOL] > 0)
        return (fresh & liquid).rename(self.TRADEABLE_COL)

    @staticmethod
    def spread_summary(synced: pd.DataFrame, spread_col: str = SPREAD_COL,
                       tradeable_col: str = TRADEABLE_COL) -> dict:
        """Descriptive stats on the spread, restricted to tradeable minutes if present."""
        s = synced[spread_col]
        if tradeable_col in synced.columns:
            s = s[synced[tradeable_col]]
        s = s.dropna()
        if s.empty:
            return {"n_minutes": 0}
        return {
            "n_minutes": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "max": float(s.max()),
            "p95": float(s.quantile(0.95)),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _clean_leg(self, df: pd.DataFrame, label: str) -> "tuple[pd.Series, pd.Series]":
        """Return cleaned, datetime-indexed (price, volume) Series for one venue."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{label}: expected a pandas DataFrame, got {type(df)!r}")
        if df.empty:
            raise ValueError(f"{label}: dataframe is empty")
        for col in (self.price_col, self.volume_col):
            if col not in df.columns:
                raise KeyError(
                    f"{label}: column {col!r} not found; columns are {list(df.columns)}"
                )

        if self.timestamp_col is not None:
            if self.timestamp_col not in df.columns:
                raise KeyError(
                    f"{label}: timestamp_col {self.timestamp_col!r} not found; "
                    f"columns are {list(df.columns)}"
                )
            idx = self._coerce_datetime(df[self.timestamp_col].to_numpy(), label)
        else:
            idx = self._coerce_datetime(np.asarray(df.index), label)

        leg = pd.DataFrame(
            {
                "price": pd.to_numeric(df[self.price_col], errors="coerce").to_numpy(),
                "volume": pd.to_numeric(df[self.volume_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(),
            },
            index=idx,
        ).sort_index()

        leg = leg[leg["price"].notna()]  # drop ticks with no usable price
        if leg.empty:
            raise ValueError(f"{label}: no valid (priced) rows after cleaning")
        return leg["price"], leg["volume"]

    @staticmethod
    def _quote_age(price_1m: pd.Series) -> pd.Series:
        """Minutes since the last fresh print (0 = fresh this minute).

        Vectorized: group by the cumulative count of non-NaN prints; within each
        group the first row is the fresh print (age 0) and subsequent carried-
        forward rows increment. Minutes before the first-ever print are ``inf``
        (no quote yet -> always untradeable).
        """
        valid = price_1m.notna()
        grp = valid.cumsum()
        age = price_1m.groupby(grp).cumcount().astype("float64")
        return age.mask(grp == 0, np.inf)

    def _coerce_datetime(self, values: np.ndarray, label: str) -> pd.DatetimeIndex:
        """Convert timestamps (datetime / string / epoch) to a tz-aware index."""
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

    def _warn_on_scale_mismatch(self, kalshi: pd.Series, poly: pd.Series) -> None:
        """Warn if the two series look like they are on different price scales."""
        if kalshi.empty or poly.empty:
            return
        k_max, p_max = float(kalshi.max()), float(poly.max())
        looks_prob = lambda x: x <= 1.5  # noqa: E731 - tiny local predicate
        if looks_prob(k_max) != looks_prob(p_max):
            warnings.warn(
                f"Possible price-unit mismatch: kalshi max={k_max:.4g}, "
                f"polymarket max={p_max:.4g}. One series looks like 0-1 probability "
                "and the other like cents/percent. price_spread will be meaningless "
                "unless both are on the same scale.",
                stacklevel=3,
            )


class PmxtFeed:
    """Optional adapter over the ``pmxt`` library to fetch venue price/volume frames.

    Tolerant of the candle representation (attribute-style ``PriceCandle`` objects,
    dicts, or ccxt-style lists). The returned frame carries both ``price`` (close)
    and ``volume`` -- exactly what the pipeline now needs.

    Example
    -------
    >>> import pmxt
    >>> kalshi, poly = pmxt.Kalshi(), pmxt.Polymarket()
    >>> feed = PmxtFeed()
    >>> k_df = feed.fetch_ohlcv_frame(kalshi, kalshi.fetch_market(slug="...").yes, "1m")
    >>> p_df = feed.fetch_ohlcv_frame(poly, poly.fetch_market(slug="...").yes, "1m")
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
        """Fetch OHLCV candles for one outcome and return a tidy price/volume frame."""
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
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Fetch both venues' frames for the same event in one call."""
        k_df = self.fetch_ohlcv_frame(kalshi_client, kalshi_outcome, resolution, start, end, limit)
        p_df = self.fetch_ohlcv_frame(polymarket_client, polymarket_outcome, resolution, start, end, limit)
        return k_df, p_df

    @classmethod
    def candles_to_frame(cls, candles, tz: str = "UTC") -> pd.DataFrame:
        """Normalize a sequence of pmxt candles into a price/volume DataFrame."""
        rows = [[cls._candle_get(c, f) for f in cls._FIELDS] for c in candles]
        df = pd.DataFrame(rows, columns=list(cls._FIELDS))
        if df.empty:
            raise ValueError("fetch_ohlcv returned no candles for this outcome/window")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if tz != "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        df = df.set_index("timestamp").sort_index()
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        return df

    @classmethod
    def _candle_get(cls, candle, field: str):
        if hasattr(candle, field):
            return getattr(candle, field)
        if isinstance(candle, dict):
            return candle.get(field)
        if isinstance(candle, (list, tuple)):
            return candle[cls._FIELDS.index(field)]
        raise TypeError(f"Unsupported candle type: {type(candle)!r}")

    @staticmethod
    def _to_datetime_param(value):
        """Coerce a datetime-like value to a tz-aware ``datetime`` for pmxt."""
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()


# ---------------------------------------------------------------------------- #
# Self-contained demo (runs with only pandas + numpy -- no network/keys needed)
# ---------------------------------------------------------------------------- #
def _demo() -> None:
    """Two 1-minute feeds with Price + Volume, each carrying a phantom-liquidity trap."""
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2026-06-22 12:00:00", tz="UTC")
    n = 120
    minutes = base + pd.to_timedelta(np.arange(n), unit="min")

    k_price = np.clip(0.50 + np.cumsum(rng.normal(0, 0.003, n)), 0.05, 0.95)
    p_price = np.clip(k_price + 0.01 + rng.normal(0, 0.004, n), 0.05, 0.95)
    k_vol = rng.integers(40, 200, n).astype(float)
    p_vol = rng.integers(40, 200, n).astype(float)

    # Trap 1 -- Kalshi goes dark for 10 minutes (no prints): the last quote is
    #           carried forward and would look live under naive ffill.
    k_present = np.ones(n, dtype=bool)
    k_present[40:50] = False
    # Trap 2 -- Polymarket keeps printing a price but volume dries to 0: a fresh-
    #           looking quote with no book behind it (classic phantom liquidity).
    p_vol[80:96] = 0.0

    kalshi_df = pd.DataFrame({
        "timestamp": minutes[k_present].asi8 // 1_000_000,  # epoch ms
        "price": k_price[k_present],
        "volume": k_vol[k_present],
    })
    poly_df = pd.DataFrame({
        "timestamp": minutes,                                # datetime
        "price": p_price,
        "volume": p_vol,
    })

    pipe = MarketDataPipeline(timestamp_col="timestamp", max_age=5, vol_window=5)
    synced = pipe.synchronize(kalshi_df, poly_df)

    n_min = len(synced)
    n_ok = int(synced["is_tradeable"].sum())
    pd.set_option("display.width", 130)
    print(f"Synchronized minutes : {n_min}")
    print(f"Tradeable            : {n_ok}  ({n_ok / n_min:.0%})")
    print(f"Flagged UNTRADEABLE  : {n_min - n_ok}   "
          f"(a naive ffill would have called all {n_min} tradeable -> phantom liquidity)\n")

    print("Trap 1 — Kalshi goes dark (stale carried quote), 12:43–12:51:")
    c1 = ["kalshi_price", "kalshi_quote_age", "kalshi_roll_vol", "is_tradeable"]
    print(synced.loc[minutes[43]:minutes[51], c1].round(3).to_string())

    print("\nTrap 2 — Polymarket quoting but zero volume (fresh quote, dry book), 13:22–13:38:")
    c2 = ["polymarket_quote_age", "polymarket_volume", "polymarket_roll_vol", "is_tradeable"]
    print(synced.loc[minutes[82]:minutes[98], c2].round(3).to_string())

    strict = int(pipe.filter_stale_quotes(max_age=2).sum())
    print(f"\nfilter_stale_quotes(max_age=2) is stricter: {strict} tradeable (vs {n_ok} at max_age=5)")
    print("\nTradeable-only spread summary:")
    for k, v in MarketDataPipeline.spread_summary(synced).items():
        print(f"  {k:>10}: {v}")


if __name__ == "__main__":
    _demo()
