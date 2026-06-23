"""
stat_lab.py
===========

Statistical layer for the prediction-market stat-arb backtester.

:class:`StatLab` consumes the synchronized frame produced by
:class:`data_pipeline.MarketDataPipeline` (columns ``kalshi_price``,
``polymarket_price``, ``price_spread``) and provides:

* :meth:`StatLab.engle_granger_test` -- the Engle-Granger two-step cointegration
  test (OLS step-1 for the cointegrating vector + ADF unit-root test on the
  residual, via ``statsmodels.tsa.stattools.coint``).
* :meth:`StatLab.estimate_ou` -- Ornstein-Uhlenbeck parameter estimation on the
  spread via the discrete AR(1) representation, yielding the **half-life of mean
  reversion**.
* :meth:`StatLab.rolling_zscore` -- a fully vectorized 60-period rolling mean,
  rolling std, and rolling Z-score of the spread.
* :meth:`StatLab.analyze` -- convenience: run the cointegration test and, *only
  if cointegrated*, estimate the OU half-life.

A note on ``price_spread`` (read this before trading on it)
-----------------------------------------------------------
``MarketDataPipeline`` defines ``price_spread`` as the **absolute** difference
``|kalshi - polymarket|``. Cointegration is therefore run on the *raw* price
series (correct -- that is what the test operates on), and the cointegrating
residual is reported separately.

For OU mean-reversion and Z-score *signals*, the absolute value is a poor choice:
``abs(.)`` folds negative deviations onto positive ones, destroying the sign that
a stat-arb signal needs (positive Z = spread too wide = short the rich leg) and
biasing the OU fit. The methods below honour the requested ``price_spread`` column
by default, but :attr:`StatLab.signed_spread` (``kalshi_price - polymarket_price``)
is exposed so you can fit/standardize the signed series instead -- which is what
you almost certainly want downstream. See the ``__main__`` demo for the contrast.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from data_pipeline import MarketDataPipeline

__all__ = ["StatLab", "CointegrationResult", "OUParams"]

_K = MarketDataPipeline.KALSHI_COL
_P = MarketDataPipeline.POLYMARKET_COL
_S = MarketDataPipeline.SPREAD_COL


# --------------------------------------------------------------------------- #
# Structured results
# --------------------------------------------------------------------------- #
@dataclass
class CointegrationResult:
    """Outcome of the Engle-Granger two-step test."""

    dependent: str            # which series was the regressand in step 1
    independent: str
    test_stat: float          # Engle-Granger (ADF-on-residual) t-statistic
    pvalue: float             # MacKinnon p-value
    crit_values: dict         # {'1%','5%','10%'} critical values
    alpha: float              # step-1 intercept
    beta: float               # step-1 hedge ratio (cointegrating coefficient)
    significance: float
    cointegrated: bool
    n_obs: int

    @property
    def verdict(self) -> str:
        rel = "<" if self.cointegrated else ">="
        return (
            f"{'COINTEGRATED' if self.cointegrated else 'NOT cointegrated'} "
            f"(p={self.pvalue:.4f} {rel} {self.significance:g}); "
            f"hedge ratio beta={self.beta:.4f}"
        )


@dataclass
class OUParams:
    """Estimated Ornstein-Uhlenbeck parameters for dX = theta*(mu - X)dt + sigma*dW.

    ``theta`` and ``half_life_periods`` are expressed per *period* of the input
    grid (1 minute for a 1-minute synchronized frame), scaled by ``dt``.
    """

    spread_name: str
    ar1_phi: float            # AR(1) slope b in X_{t+1}=a+b*X_t+eps  (b=exp(-theta*dt))
    phi_stderr: float         # std error of the AR(1) slope
    adf_pvalue: float         # ADF unit-root p-value on the spread (stationarity check)
    theta: float              # mean-reversion speed (per period)
    half_life_periods: float  # ln(2)/theta, in grid periods
    mu: float                 # long-run mean
    sigma: float              # instantaneous diffusion volatility
    sigma_eq: float           # stationary (equilibrium) std of the spread
    mean_reverting: bool
    n_obs: int

    @property
    def verdict(self) -> str:
        if not self.mean_reverting:
            return f"NOT mean-reverting (phi={self.ar1_phi:.4f}); half-life undefined"
        return (
            f"mean-reverting: half-life={self.half_life_periods:.2f} periods, "
            f"mu={self.mu:.4f}, theta={self.theta:.4f} "
            f"(phi={self.ar1_phi:.4f}+/-{self.phi_stderr:.4f}, ADF p={self.adf_pvalue:.3g})"
        )


# --------------------------------------------------------------------------- #
# StatLab
# --------------------------------------------------------------------------- #
class StatLab:
    """Cointegration / OU / Z-score analytics on a synchronized price frame.

    Parameters
    ----------
    df:
        Synchronized frame (typically from :meth:`MarketDataPipeline.synchronize`).
    kalshi_col, polymarket_col, spread_col:
        Column names. Default to the constants emitted by ``MarketDataPipeline``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        kalshi_col: str = _K,
        polymarket_col: str = _P,
        spread_col: str = _S,
    ) -> None:
        missing = {kalshi_col, polymarket_col, spread_col} - set(df.columns)
        if missing:
            raise KeyError(f"StatLab: dataframe is missing columns {sorted(missing)}")

        self.df = df
        self.kalshi_col = kalshi_col
        self.polymarket_col = polymarket_col
        self.spread_col = spread_col

        # Clean, jointly-aligned price pair for the cointegration regression.
        self._pair = df[[kalshi_col, polymarket_col]].dropna()
        if len(self._pair) < 20:
            warnings.warn(
                f"Only {len(self._pair)} aligned observations; cointegration/OU "
                "estimates will be unreliable.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Convenience views
    # ------------------------------------------------------------------ #
    @property
    def signed_spread(self) -> pd.Series:
        """``kalshi_price - polymarket_price`` (the *signed* spread)."""
        s = self._pair[self.kalshi_col] - self._pair[self.polymarket_col]
        return s.rename("signed_spread")

    def _spread_series(self, spread: Optional[pd.Series]) -> pd.Series:
        """Resolve the spread series to operate on (default = ``spread_col``)."""
        if spread is None:
            return self.df[self.spread_col].dropna()
        return spread.dropna()

    # ------------------------------------------------------------------ #
    # 1) Engle-Granger two-step cointegration test
    # ------------------------------------------------------------------ #
    def engle_granger_test(
        self,
        dependent: str = "kalshi",
        trend: str = "c",
        significance: float = 0.05,
        autolag: Optional[str] = "aic",
        maxlag: Optional[int] = None,
    ) -> CointegrationResult:
        """Run the Engle-Granger two-step cointegration test.

        Step 1 regresses one price series on the other (OLS) to recover the
        cointegrating vector (intercept ``alpha``, hedge ratio ``beta``). Step 2
        tests the regression residual for a unit root. ``statsmodels.coint``
        performs the full two-step (augmented Engle-Granger) procedure and returns
        the MacKinnon p-value.

        Parameters
        ----------
        dependent:
            ``"kalshi"`` (default) regresses Kalshi on Polymarket; ``"polymarket"``
            reverses it. Engle-Granger is mildly order-dependent, so the choice is
            explicit.
        trend:
            Deterministic term in the test ('c', 'ct', 'ctt', 'n'). Default 'c'.
        significance:
            Threshold for the ``cointegrated`` flag. Default 0.05.
        """
        if dependent == "kalshi":
            y, x = self._pair[self.kalshi_col], self._pair[self.polymarket_col]
            yname, xname = self.kalshi_col, self.polymarket_col
        elif dependent == "polymarket":
            y, x = self._pair[self.polymarket_col], self._pair[self.kalshi_col]
            yname, xname = self.polymarket_col, self.kalshi_col
        else:
            raise ValueError("dependent must be 'kalshi' or 'polymarket'")

        # Step 1: OLS for the cointegrating vector (alpha, beta).
        design = sm.add_constant(x.to_numpy())
        ols = sm.OLS(y.to_numpy(), design).fit()
        alpha, beta = float(ols.params[0]), float(ols.params[1])

        # Two-step Engle-Granger (ADF on the cointegrating residual).
        test_stat, pvalue, crit = coint(
            y.to_numpy(), x.to_numpy(), trend=trend, maxlag=maxlag, autolag=autolag
        )
        crit_values = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}

        return CointegrationResult(
            dependent=yname,
            independent=xname,
            test_stat=float(test_stat),
            pvalue=float(pvalue),
            crit_values=crit_values,
            alpha=alpha,
            beta=beta,
            significance=significance,
            cointegrated=bool(pvalue < significance),
            n_obs=int(len(y)),
        )

    # ------------------------------------------------------------------ #
    # 2) Ornstein-Uhlenbeck estimation + half-life
    # ------------------------------------------------------------------ #
    def estimate_ou(
        self,
        spread: Optional[pd.Series] = None,
        dt: float = 1.0,
        adf_autolag: Optional[str] = "aic",
        adf_maxlag: Optional[int] = None,
    ) -> OUParams:
        """Estimate OU parameters and the half-life of mean reversion.

        Uses the exact discrete-time AR(1) representation of an OU process,
            X_{t+1} = a + b * X_t + eps,   b = exp(-theta * dt),
        fitted by a single (vectorized) OLS. From the fit:
            theta = -ln(b) / dt,   half-life = ln(2) / theta,   mu = a / (1 - b).

        Parameters
        ----------
        spread:
            Series to fit. Defaults to the ``price_spread`` column. Pass
            :attr:`signed_spread` to fit the signed equilibrium error instead
            (recommended for trading signals -- see module docstring).
        dt:
            Length of one step in the time units you want ``theta``/half-life in.
            Default 1.0 = "per grid period" (e.g. per minute for a 1-min frame).
        """
        s = self._spread_series(spread)
        if len(s) < 3:
            raise ValueError("Need at least 3 observations to estimate an OU process")

        # Vectorized AR(1) OLS: regress X_{t+1} on X_t. No Python loops.
        x_t = s.to_numpy()[:-1]
        x_next = s.to_numpy()[1:]
        design = sm.add_constant(x_t)
        res = sm.OLS(x_next, design).fit()
        a, b = float(res.params[0]), float(res.params[1])
        phi_se = float(res.bse[1])              # std error of the AR(1) slope
        sigma_eps = float(np.sqrt(res.scale))   # std of the AR(1) innovation

        # ADF unit-root test on the spread itself: the half-life is only
        # meaningful if the spread is stationary. This uses the Dickey-Fuller
        # distribution -- a plain normal t-test on (phi-1) is invalid near a unit
        # root and would falsely flag random walks as mean-reverting.
        try:
            adf_pvalue = float(
                adfuller(s.to_numpy(), maxlag=adf_maxlag, autolag=adf_autolag)[1]
            )
        except (ValueError, np.linalg.LinAlgError):
            adf_pvalue = float("nan")

        mean_reverting = 0.0 < b < 1.0
        if mean_reverting:
            theta = -np.log(b) / dt
            half_life = np.log(2.0) / theta
            mu = a / (1.0 - b)
            # Map innovation std to the OU diffusion and stationary std.
            sigma = sigma_eps * np.sqrt(2.0 * theta / (1.0 - b**2))
            sigma_eq = sigma_eps / np.sqrt(1.0 - b**2)
            # Stationarity guard: a spread that fails ADF is (near) a unit root,
            # so the half-life is untrustworthy even though 0 < phi < 1.
            if not (adf_pvalue <= 0.10):  # True for p>0.10 and for NaN
                warnings.warn(
                    f"Spread fails the ADF stationarity test (p={adf_pvalue:.3g} "
                    f"> 0.10): it is statistically near a unit root, so the OU "
                    f"half-life (~{half_life:.0f} periods) is unreliable.",
                    stacklevel=2,
                )
        else:
            # b>=1: random-walk/explosive -> no finite half-life.
            # b<=0: anti-persistent -> OU half-life not well-defined.
            theta = (-np.log(b) / dt) if b > 0.0 else np.nan
            half_life = np.inf if b >= 1.0 else np.nan
            mu = (a / (1.0 - b)) if b != 1.0 else np.nan
            sigma = np.nan
            sigma_eq = np.nan
            warnings.warn(
                f"AR(1) slope phi={b:.4f} is outside (0, 1); the spread is not "
                "mean-reverting, so the OU half-life is not finite/defined.",
                stacklevel=2,
            )

        return OUParams(
            spread_name=str(s.name) if s.name is not None else "spread",
            ar1_phi=b,
            phi_stderr=phi_se,
            adf_pvalue=adf_pvalue,
            theta=float(theta),
            half_life_periods=float(half_life),
            mu=float(mu),
            sigma=float(sigma),
            sigma_eq=float(sigma_eq),
            mean_reverting=bool(mean_reverting),
            n_obs=int(len(x_t)),
        )

    # ------------------------------------------------------------------ #
    # 3) Rolling Z-score (strictly vectorized)
    # ------------------------------------------------------------------ #
    def rolling_zscore(
        self,
        window: int = 60,
        spread: Optional[pd.Series] = None,
        ddof: int = 1,
        min_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Vectorized rolling mean, rolling std, and rolling Z-score of the spread.

        Z_t = (spread_t - rolling_mean_t) / rolling_std_t, computed over a trailing
        ``window`` (default 60 periods). Uses pandas' C-level ``.rolling`` -- no
        Python-level iteration. The first ``window-1`` rows are NaN by construction.

        Parameters
        ----------
        window:
            Look-back length in periods. Default 60.
        spread:
            Series to standardize. Defaults to the ``price_spread`` column.
        ddof:
            Delta degrees of freedom for the rolling std (1 = sample std, default).
        min_periods:
            Minimum observations in a window required to emit a value. Defaults to
            ``window`` (no partial-window estimates).

        Returns
        -------
        pandas.DataFrame
            Columns ``[spread, rolling_mean, rolling_std, zscore]`` on the spread's
            index.
        """
        s = self._spread_series(spread)
        mp = window if min_periods is None else min_periods

        roller = s.rolling(window=window, min_periods=mp)
        rolling_mean = roller.mean()
        rolling_std = roller.std(ddof=ddof)

        # Guard against divide-by-zero on flat windows (std == 0 -> Z undefined).
        safe_std = rolling_std.replace(0.0, np.nan)
        zscore = (s - rolling_mean) / safe_std

        return pd.DataFrame(
            {
                "spread": s,
                "rolling_mean": rolling_mean,
                "rolling_std": rolling_std,
                "zscore": zscore,
            }
        )

    # ------------------------------------------------------------------ #
    # 4) Orchestrator: test, then OU only if cointegrated
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        significance: float = 0.05,
        dependent: str = "kalshi",
        ou_on: str = "price_spread",
    ) -> dict:
        """Run Engle-Granger and, *only if cointegrated*, estimate the OU half-life.

        Parameters
        ----------
        ou_on:
            ``"price_spread"`` (default, honours the pipeline column) or
            ``"signed"`` to fit OU on the signed spread.
        """
        eg = self.engle_granger_test(dependent=dependent, significance=significance)
        ou: Optional[OUParams] = None
        if eg.cointegrated:
            spread = self.signed_spread if ou_on == "signed" else None
            ou = self.estimate_ou(spread=spread)
        return {"cointegration": eg, "ou": ou}


# --------------------------------------------------------------------------- #
# Self-contained demo (synthetic data; runs with pandas/numpy/statsmodels)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.date_range("2026-06-22 09:00", periods=n, freq="1min", tz="UTC")

    # A shared (non-stationary) fair-value path both venues track.
    fair = 0.50 + np.cumsum(rng.normal(0, 0.0015, n))
    fair = np.clip(fair, 0.05, 0.95)

    # A *mean-reverting* (signed) basis between the two venues, simulated as an
    # AR(1) with phi=0.95 -> true half-life = ln(2)/-ln(0.95) ~ 13.51 periods.
    # (This generator loop builds TEST DATA; all StatLab analytics are vectorized.)
    phi_true = 0.95
    basis = np.empty(n)
    basis[0] = 0.0
    shocks = rng.normal(0, 0.004, n)
    for i in range(1, n):
        basis[i] = phi_true * basis[i - 1] + shocks[i]

    kalshi = np.clip(fair + basis / 2.0, 0.01, 0.99)
    poly = np.clip(fair - basis / 2.0, 0.01, 0.99)
    synced = pd.DataFrame(
        {
            _K: kalshi,
            _P: poly,
            _S: np.abs(kalshi - poly),  # price_spread = ABSOLUTE difference
        },
        index=idx,
    )

    lab = StatLab(synced)
    expected_hl = np.log(2) / -np.log(phi_true)

    print("=== Engle-Granger two-step cointegration test ===")
    eg = lab.engle_granger_test()
    print(f"  {eg.dependent} ~ {eg.independent}")
    print(f"  test_stat={eg.test_stat:.4f}  pvalue={eg.pvalue:.4g}")
    print(f"  crit={ {k: round(v,3) for k,v in eg.crit_values.items()} }")
    print(f"  {eg.verdict}")

    print("\n=== OU half-life ===")
    print(f"  true half-life of the simulated basis ~ {expected_hl:.2f} periods")
    ou_abs = lab.estimate_ou()                          # on price_spread (abs)
    ou_signed = lab.estimate_ou(spread=lab.signed_spread)  # recommended
    print(f"  on price_spread (abs):  {ou_abs.verdict}")
    print(f"  on signed_spread:       {ou_signed.verdict}")

    print("\n=== 60-period rolling Z-score (tail) ===")
    z = lab.rolling_zscore(window=60, spread=lab.signed_spread)
    print(z.tail(3).round(4).to_string())
    print(f"  NaNs in first 59 rows (expected 59): {int(z['zscore'].head(59).isna().sum())}")

    print("\n=== analyze() orchestrator ===")
    out = lab.analyze(ou_on="signed")
    print(f"  cointegrated={out['cointegration'].cointegrated}; "
          f"ou={'fitted' if out['ou'] else 'skipped'}")


if __name__ == "__main__":
    _demo()
