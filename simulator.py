"""
Quantitative simulation, portfolio optimization, and sentiment engine for
Stock Future Simulator V2.0.

Architecture Rule (STRICT):
This module is 100% pure Python (ZERO Streamlit or UI framework imports).
All simulation paths, cash-flow math, and institutional risk metrics use
vectorized NumPy/SciPy operations, with defensive try/except fallbacks so
external API hiccups or ill-conditioned covariance matrices never crash the app.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t
import yfinance as yf

# ---------------------------------------------------------
# TUNABLE ENGINE DEFAULTS
# ---------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252
TRADING_DAYS_PER_MONTH: int = 21       # 252 trading days / 12 months
MAX_PORTFOLIO_ASSETS: int = 6
DEFAULT_HISTORY_PERIOD: str = "15y"
DEFAULT_SIMULATIONS: int = 2500        # 1,000-2,500 for fast mobile response; 5,000 for smoother bands
DEFAULT_STUDENT_T_DF: int = 4          # 3 = frequent extreme shocks; 10+ = near-normal bell curve
DEFAULT_BLOCK_SIZE: int = 20           # 20 trading days (~1 month) preserves volatility clustering
DEFAULT_STRESS_FLOOR: float = 0.55     # 0.55 = -45% Year 1 drawdown shock
DEFAULT_STRESS_DAYS: int = 252         # Length of the Year 1 crash phase in trading days
DEFAULT_RISK_FREE_RATE: float = 0.04   # 4.0% annual risk-free benchmark (T-bill proxy)
LONG_TERM_EQUITY_DRIFT: float = 0.095  # ~9.5% historical long-term equity mean-reversion anchor
BASELINE_MARKET_PE: float = 20.0       # Neutral market P/E reference for valuation scaling
DEFAULT_INFLATION_RATE: float = 0.03   # 3.0% default annual inflation rate

ENGINE_STUDENT_T: str = "Standard Student-t Monte Carlo"
ENGINE_BLOCK_BOOTSTRAP: str = "Historical Bootstrapping"
ENGINE_2008_STRESS: str = "2008 Crisis Stress Test"

AVAILABLE_ENGINES: tuple[str, ...] = (
    ENGINE_STUDENT_T,
    ENGINE_BLOCK_BOOTSTRAP,
    ENGINE_2008_STRESS,
)


class StockDataError(RuntimeError):
    """Raised when stock market data cannot be retrieved or validated."""


# ---------------------------------------------------------
# 1. TICKER & WEIGHT NORMALIZATION
# ---------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """Strips whitespace and uppercases a stock ticker symbol."""
    if not isinstance(ticker, str):
        raise ValueError("Ticker symbol must be a text string.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Please enter a valid stock ticker symbol (for example, AAPL).")
    return cleaned


def normalize_portfolio_weights(
    tickers: Sequence[str],
    raw_weights: Sequence[float] | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Cleans up to 6 ticker symbols and auto-normalizes their weights to sum to 100% (1.0).

    Duplicate tickers are automatically merged by summing their assigned weights so
    covariance matrices never contain duplicate collinear columns.
    """
    if not tickers:
        raise ValueError("Please provide at least one ticker symbol.")

    if raw_weights is not None and len(raw_weights) != len(tickers):
        raise ValueError("Tickers and weights must have the same length.")

    merged: dict[str, float] = {}
    for idx, raw_sym in enumerate(tickers):
        if raw_sym is None or not str(raw_sym).strip():
            continue
        sym = normalize_ticker(str(raw_sym))
        w = 1.0 if raw_weights is None else float(raw_weights[idx])
        if not math.isfinite(w) or w < 0.0:
            w = 0.0
        merged[sym] = merged.get(sym, 0.0) + w

    if not merged:
        raise ValueError("Please enter at least one valid stock ticker symbol.")
    if len(merged) > MAX_PORTFOLIO_ASSETS:
        raise ValueError(f"Portfolio mode supports up to {MAX_PORTFOLIO_ASSETS} unique tickers.")

    clean_tickers = list(merged.keys())
    weights_arr = np.array([merged[sym] for sym in clean_tickers], dtype=np.float64)
    total_weight = float(np.sum(weights_arr))

    # Fall back to equal weights if all user-supplied weights are zero
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        weights_arr = np.full(len(clean_tickers), 1.0 / len(clean_tickers), dtype=np.float64)
    else:
        weights_arr = weights_arr / total_weight

    return clean_tickers, weights_arr


# ---------------------------------------------------------
# 2. DATA FETCHER & MULTI-ASSET ALIGNMENT LAYER
# ---------------------------------------------------------
def _extract_close_series(hist: pd.DataFrame, symbol: str) -> pd.Series:
    """Safely extracts a clean, timezone-naive, date-indexed Close price Series."""
    if hist is None or hist.empty:
        raise StockDataError(
            f"No historical price data was found for '{symbol}'. "
            "Check that the ticker is spelled correctly and has not been delisted."
        )

    # Handle both standard single-level and MultiIndex columns returned by yfinance
    close_data: Any = None
    if isinstance(hist.columns, pd.MultiIndex):
        if "Close" in hist.columns.get_level_values(0):
            close_data = hist["Close"]
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
    elif "Close" in hist.columns:
        close_data = hist["Close"]

    if close_data is None:
        raise StockDataError(f"Price history for '{symbol}' is missing a 'Close' column.")

    series = pd.to_numeric(close_data, errors="coerce").dropna()
    if len(series) < 2:
        raise StockDataError(
            f"Price history for '{symbol}' contains fewer than 2 valid closing prices."
        )

    # Normalize timestamps to midnight UTC-naive dates so cross-exchange tickers align cleanly
    idx = pd.to_datetime(series.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    series = series[~series.index.isna()]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[series > 0]

    if len(series) < 2:
        raise StockDataError(
            f"Price history for '{symbol}' contains insufficient positive closing prices."
        )
    return series.rename(symbol)


def fetch_stock_history(
    ticker: str,
    period: str = DEFAULT_HISTORY_PERIOD,
) -> tuple[pd.DataFrame, float | str]:
    """
    Pulls up to 15 years of adjusted daily prices and trailing P/E from Yahoo Finance.

    Wraps all network calls defensively so empty or rate-limited API responses
    raise a clean StockDataError rather than an unhandled crash.
    """
    symbol = normalize_ticker(ticker)

    try:
        stock = yf.Ticker(symbol)
        # auto_adjust=True ensures Close reflects dividends and splits consistently
        hist = stock.history(period=period, auto_adjust=True)
    except Exception as exc:
        raise StockDataError(
            f"Could not download market data for '{symbol}'. "
            "Yahoo Finance may be temporarily rate-limiting requests or your network may be offline."
        ) from exc

    close_series = _extract_close_series(hist, symbol)
    clean_hist = pd.DataFrame({"Close": close_series})

    pe_ratio: float | str = "N/A"
    try:
        info = stock.info or {}
        raw_pe = info.get("trailingPE")
        if raw_pe is not None:
            pe_float = float(raw_pe)
            if math.isfinite(pe_float) and pe_float > 0:
                pe_ratio = pe_float
    except Exception:
        # Valuation metadata is optional; never block simulations if .info fails
        pe_ratio = "N/A"

    return clean_hist, pe_ratio


def extract_return_stats(hist: pd.DataFrame) -> dict[str, Any]:
    """
    Extracts current price, clean daily returns, annualized volatility, and history length
    for a single-asset price DataFrame.
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError("Historical price DataFrame is empty or missing a 'Close' column.")

    close_prices = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close_prices) < 2:
        raise ValueError("At least two valid closing prices are required to compute daily returns.")

    current_price = float(close_prices.iloc[-1])
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("Current stock price must be a positive finite number.")

    daily_returns = close_prices.pct_change().dropna().to_numpy(dtype=np.float64)
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) < 2:
        raise ValueError("Insufficient valid daily returns in price history.")

    annualized_volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    years_of_history = float(len(close_prices) / TRADING_DAYS_PER_YEAR)

    return {
        "current_price": current_price,
        "daily_returns": daily_returns,
        "annualized_volatility": annualized_volatility,
        "years_of_history": years_of_history,
        "num_trading_days": int(len(close_prices)),
    }


def _compute_weighted_harmonic_pe(
    tickers: Sequence[str],
    weights: np.ndarray,
    trailing_pes: Mapping[str, float | None],
) -> float | None:
    """
    Computes institutional portfolio P/E via weighted harmonic mean (inverse of
    weighted earnings yield) across assets with valid positive P/E ratios.
    """
    valid_weight_sum = 0.0
    weighted_earnings_yield = 0.0
    for sym, w in zip(tickers, weights):
        pe = trailing_pes.get(sym)
        if pe is not None and math.isfinite(pe) and pe > 0 and w > 0:
            valid_weight_sum += float(w)
            weighted_earnings_yield += float(w) / float(pe)

    if valid_weight_sum <= 0.0 or weighted_earnings_yield <= 0.0:
        return None
    return float(valid_weight_sum / weighted_earnings_yield)


def fetch_portfolio_data(
    tickers: Sequence[str],
    raw_weights: Sequence[float] | None = None,
    period: str = DEFAULT_HISTORY_PERIOD,
) -> dict[str, Any]:
    """
    Downloads and date-aligns historical price data for 1 to 6 portfolio tickers.

    Every ticker is aligned onto the shared intersection of trading dates so that
    covariance estimation and block bootstrapping reflect exact synchronized co-movements.
    """
    clean_tickers, norm_weights = normalize_portfolio_weights(tickers, raw_weights)

    close_series_list: list[pd.Series] = []
    trailing_pes: dict[str, float | None] = {}

    for sym in clean_tickers:
        hist_df, pe_val = fetch_stock_history(sym, period=period)
        close_series_list.append(hist_df["Close"].rename(sym))
        trailing_pes[sym] = float(pe_val) if isinstance(pe_val, (int, float)) else None

    # Inner-join across tickers so every row represents a shared trading date
    prices_df = pd.concat(close_series_list, axis=1, join="inner").dropna().sort_index()
    if len(prices_df) < 3:
        raise StockDataError(
            "Selected tickers have fewer than 3 overlapping trading days of history. "
            "Remove newly listed tickers or try another combination."
        )

    returns_df = prices_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns_df) < 2:
        raise StockDataError(
            "Insufficient overlapping daily returns across the selected tickers."
        )

    returns_matrix = returns_df.to_numpy(dtype=np.float64)
    n_assets = len(clean_tickers)

    # Build (N, N) daily covariance and correlation matrices even for N == 1
    if n_assets == 1:
        var_1d = float(np.var(returns_matrix[:, 0], ddof=1))
        cov_matrix = np.array([[max(var_1d, 0.0)]], dtype=np.float64)
        corr_matrix = np.array([[1.0]], dtype=np.float64)
    else:
        cov_matrix = np.cov(returns_matrix, rowvar=False, ddof=1)
        cov_matrix = np.atleast_2d(np.asarray(cov_matrix, dtype=np.float64))
        corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
        corr_matrix = np.atleast_2d(np.nan_to_num(corr_matrix, nan=0.0))

    # Portfolio weighted daily returns and annualized volatility from w^T * Sigma * w
    portfolio_daily_returns = returns_matrix @ norm_weights
    port_daily_var = float(norm_weights @ cov_matrix @ norm_weights)
    annualized_volatility = float(np.sqrt(max(port_daily_var, 0.0) * TRADING_DAYS_PER_YEAR))
    historical_annual_return = float(np.mean(portfolio_daily_returns) * TRADING_DAYS_PER_YEAR)

    current_prices = {sym: float(prices_df[sym].iloc[-1]) for sym in clean_tickers}
    asset_vols = {
        sym: float(np.std(returns_df[sym].to_numpy(dtype=np.float64), ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        for sym in clean_tickers
    }
    weighted_pe = _compute_weighted_harmonic_pe(clean_tickers, norm_weights, trailing_pes)

    return {
        "tickers": clean_tickers,
        "weights": norm_weights,
        "prices_df": prices_df,
        "returns_df": returns_df,
        "returns_matrix": returns_matrix,
        "current_prices": current_prices,
        "trailing_pes": trailing_pes,
        "weighted_pe": weighted_pe,
        "asset_vols": asset_vols,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "portfolio_daily_returns": portfolio_daily_returns,
        "annualized_volatility": annualized_volatility,
        "historical_annual_return": historical_annual_return,
        "years_of_history": float(len(prices_df) / TRADING_DAYS_PER_YEAR),
        "num_trading_days": int(len(prices_df)),
    }


# ---------------------------------------------------------
# 3. VALUATION / MEAN-REVERSION DRIFT DAMPENER
# ---------------------------------------------------------
def compute_dampened_daily_drift(
    returns_matrix: np.ndarray,
    tickers: Sequence[str],
    weights: np.ndarray,
    trailing_pes: Mapping[str, float | None],
    baseline_annual_drift: float = LONG_TERM_EQUITY_DRIFT,
    baseline_pe: float = BASELINE_MARKET_PE,
) -> dict[str, Any]:
    """
    Blends extreme historical annual drift toward a realistic ~9.5% long-term equity
    baseline scaled by each asset's trailing P/E ratio.

    Plain-English Intuition:
    - Stocks with unusually high historical returns (e.g., +45%/yr) rarely sustain that
      rate indefinitely. Meanwhile, stocks trading at elevated P/E multiples face
      valuation headwinds, whereas lower-P/E stocks carry higher earnings yields.
    - Each asset's long-term target is anchored to:
          target_drift = 9.5% * (20 / trailing_PE) ^ 0.35   (clamped to [4%, 13%])
      Assets without a valid P/E default cleanly to the 9.5% equity baseline.
    - We then blend historical drift toward that target, applying stronger mean-reversion
      pull when historical drift deviates sharply from the valuation anchor.
    """
    returns_2d = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    if returns_2d.shape[1] != len(tickers):
        returns_2d = returns_2d.reshape(-1, len(tickers))

    raw_daily_mu = np.mean(returns_2d, axis=0)
    raw_ann_mu = raw_daily_mu * TRADING_DAYS_PER_YEAR

    target_ann_mu = np.empty(len(tickers), dtype=np.float64)
    dampened_ann_mu = np.empty(len(tickers), dtype=np.float64)
    blend_weights = np.empty(len(tickers), dtype=np.float64)
    per_asset: list[dict[str, Any]] = []

    for idx, sym in enumerate(tickers):
        pe = trailing_pes.get(sym)
        if pe is not None and math.isfinite(pe) and pe > 0:
            # Higher P/E compresses forward return target; lower P/E modestly raises it
            scaled_target = baseline_annual_drift * ((baseline_pe / float(pe)) ** 0.35)
            target_ann = float(np.clip(scaled_target, 0.04, 0.13))
        else:
            target_ann = float(baseline_annual_drift)

        # Increase mean-reversion pull dynamically when historical drift is extreme
        excess_deviation = abs(float(raw_ann_mu[idx]) - target_ann)
        blend = min(0.80, 0.50 + 0.50 * max(0.0, excess_deviation - 0.10))
        dampened_ann = (1.0 - blend) * float(raw_ann_mu[idx]) + blend * target_ann

        target_ann_mu[idx] = target_ann
        dampened_ann_mu[idx] = dampened_ann
        blend_weights[idx] = blend
        per_asset.append(
            {
                "ticker": sym,
                "trailing_pe": pe,
                "raw_annual_drift": float(raw_ann_mu[idx]),
                "valuation_target_drift": target_ann,
                "dampened_annual_drift": dampened_ann,
                "blend_weight": blend,
            }
        )

    dampened_daily_mu = dampened_ann_mu / TRADING_DAYS_PER_YEAR
    daily_drift_shift = dampened_daily_mu - raw_daily_mu

    return {
        "raw_daily_mu": raw_daily_mu,
        "dampened_daily_mu": dampened_daily_mu,
        "daily_drift_shift": daily_drift_shift,
        "raw_portfolio_annual_drift": float(np.dot(weights, raw_ann_mu)),
        "valuation_target_portfolio_drift": float(np.dot(weights, target_ann_mu)),
        "dampened_portfolio_annual_drift": float(np.dot(weights, dampened_ann_mu)),
        "per_asset": per_asset,
    }


# ---------------------------------------------------------
# 4. CHOLESKY DECOMPOSITION WITH NEAREST-PSD FALLBACK
# ---------------------------------------------------------
def cholesky_with_psd_fallback(cov_matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Computes the lower-triangular Cholesky factor L such that L @ L.T == cov_matrix.

    If finite-sample noise or collinear tickers (e.g., SPY + VOO) make the covariance
    matrix non-positive-definite, falls back to eigenvalue clipping (nearest symmetric
    positive-definite matrix), and finally to diagonal volatilities if needed.

    Returns
    -------
    tuple[np.ndarray, bool]
        (L factor of shape (N, N), True if a PSD fallback was required).
    """
    cov = np.atleast_2d(np.asarray(cov_matrix, dtype=np.float64))
    n = cov.shape[0]
    if n == 1:
        val = float(cov[0, 0])
        if not math.isfinite(val) or val < 0.0:
            return np.array([[1e-6]], dtype=np.float64), True
        return np.array([[math.sqrt(val)]], dtype=np.float64), False

    # Enforce exact symmetry before factorization
    sym = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(sym)):
        sym = np.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        return np.linalg.cholesky(sym), False
    except np.linalg.LinAlgError:
        pass

    # Nearest-PSD projection via spectral decomposition (clip negative/zero eigenvalues)
    try:
        eigvals, eigvecs = np.linalg.eigh(sym)
        max_eig = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
        eps = max(1e-10, 1e-8 * max_eig)
        clipped_eigvals = np.maximum(eigvals, eps)
        nearest_psd = (eigvecs * clipped_eigvals) @ eigvecs.T
        nearest_psd = 0.5 * (nearest_psd + nearest_psd.T)
        return np.linalg.cholesky(nearest_psd), True
    except np.linalg.LinAlgError:
        # Final defensive fallback: uncorrelated diagonal standard deviations
        diag_stds = np.sqrt(np.maximum(np.diag(sym), 1e-12))
        return np.diag(diag_stds), True


# ---------------------------------------------------------
# 5. VECTORIZED MULTI-ASSET SIMULATION ENGINES
# ---------------------------------------------------------
def simulate_portfolio_student_t(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    days: int,
    simulations: int,
    df: int = DEFAULT_STUDENT_T_DF,
    daily_mean_override: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Mode 1 (Multi-Asset): Correlated Multivariate Student-t Monte Carlo.

    Uses the historical covariance matrix and Cholesky decomposition (with nearest-PSD
    fallback) to correlate asset shocks, then scales all assets by a shared chi-square
    volatility factor so heavy-tailed crashes strike correlated assets simultaneously.

    Returns
    -------
    tuple[np.ndarray, bool]
        (Simulated portfolio daily returns of shape (days, simulations), used_psd_fallback).
    """
    if df <= 2:
        raise ValueError("Degrees of freedom (df) must be greater than 2 for finite variance.")
    days = int(days)
    simulations = int(simulations)
    if days < 1 or simulations < 1:
        raise ValueError("days and simulations must be positive integers.")

    returns_2d = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    n_assets = len(w)
    if returns_2d.shape[1] != n_assets:
        returns_2d = returns_2d.reshape(-1, n_assets)

    mu = (
        np.asarray(daily_mean_override, dtype=np.float64)
        if daily_mean_override is not None
        else np.mean(returns_2d, axis=0)
    )

    if n_assets == 1:
        cov = np.array([[float(np.var(returns_2d[:, 0], ddof=1))]], dtype=np.float64)
    else:
        cov = np.atleast_2d(np.cov(returns_2d, rowvar=False, ddof=1))

    cholesky_l, used_psd_fallback = cholesky_with_psd_fallback(cov)

    generator = rng if rng is not None else np.random.default_rng()

    # Multivariate Student-t construction:
    # 1) Draw standard normal matrix G of shape (days, simulations, n_assets)
    # 2) Correlate across assets via G @ L.T so Cov(G @ L.T) == cov
    # 3) Scale by sqrt((df - 2) / chi2(df)) so each asset has Student-t(df) fat tails
    #    with EXACT variance equal to its historical covariance (since E[(df-2)/chi2(df)] = 1)
    normals = generator.standard_normal(size=(days, simulations, n_assets))
    chi2_draws = generator.chisquare(df, size=(days, simulations, 1))
    t_scale = np.sqrt((df - 2.0) / np.maximum(chi2_draws, 1e-12))

    correlated_shocks = np.matmul(normals, cholesky_l.T) * t_scale
    asset_daily_returns = mu[None, None, :] + correlated_shocks

    # Combine asset returns into portfolio daily returns via target weights
    port_daily_returns = np.tensordot(asset_daily_returns, w, axes=([2], [0]))
    # Guard against impossible < -100% single-day wipeouts from synthetic extreme tails
    port_daily_returns = np.maximum(port_daily_returns, -1.0 + 1e-6)
    return port_daily_returns, used_psd_fallback


def simulate_portfolio_block_bootstrap(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    days: int,
    simulations: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    daily_drift_shift: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Mode 2 (Multi-Asset): Synchronized Date-Aligned Historical Block Bootstrapping.

    Resamples contiguous multi-day blocks using the EXACT same historical start date
    across every asset in the portfolio, preserving real-world cross-asset correlations,
    volatility clustering, and multi-week market regimes.
    """
    days = int(days)
    simulations = int(simulations)
    block_size = int(block_size)
    if days < 1 or simulations < 1 or block_size < 1:
        raise ValueError("days, simulations, and block_size must be positive integers.")

    returns_2d = np.atleast_2d(np.asarray(returns_matrix, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    n_assets = len(w)
    if returns_2d.shape[1] != n_assets:
        returns_2d = returns_2d.reshape(-1, n_assets)

    num_history_days = returns_2d.shape[0]
    if num_history_days < block_size + 1:
        raise ValueError(
            f"Not enough overlapping price history ({num_history_days} trading days) for "
            f"{block_size}-day block bootstrapping. At least {block_size + 1} aligned days are needed."
        )

    num_blocks_needed = int(np.ceil(days / block_size))
    max_start_exclusive = num_history_days - block_size + 1

    generator = rng if rng is not None else np.random.default_rng()
    # One shared start index per (block, simulation) guarantees synchronized dates across assets
    random_starts = generator.integers(0, max_start_exclusive, size=(num_blocks_needed, simulations))

    # Build 3D index grid: (block_size, num_blocks_needed, simulations)
    offsets = np.arange(block_size)[:, None, None]
    block_indices = random_starts[None, :, :] + offsets

    # Pull all assets synchronously: shape (block_size, num_blocks_needed, simulations, n_assets)
    sampled_blocks = returns_2d[block_indices]
    sampled_returns = sampled_blocks.transpose(1, 0, 2, 3).reshape(-1, simulations, n_assets)[:days]

    if daily_drift_shift is not None:
        shift = np.asarray(daily_drift_shift, dtype=np.float64)
        sampled_returns = sampled_returns + shift[None, None, :]

    port_daily_returns = np.tensordot(sampled_returns, w, axes=([2], [0]))
    return np.maximum(port_daily_returns, -1.0 + 1e-6)


def simulate_portfolio_2008_stress(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    days: int,
    simulations: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crash_floor: float = DEFAULT_STRESS_FLOOR,
    crash_duration_days: int = DEFAULT_STRESS_DAYS,
    daily_drift_shift: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Mode 3 (Multi-Asset): 2008 Crisis Stress Test on Synchronized Historical Blocks.

    Injects a smooth -45% Year 1 drawdown (crash_floor = 0.55) onto synchronized
    date-aligned portfolio blocks via equivalent daily shock multipliers, then allows
    natural historical returns to drive Year 2+ recovery. Applying the crash as daily
    return multipliers ensures monthly contributions made during the downturn buy
    shares at depressed crisis prices rather than retroactively losing pre-deposit value.
    """
    if not (0.0 < crash_floor <= 1.0):
        raise ValueError("crash_floor must be between 0 (exclusive) and 1 (inclusive).")
    if int(crash_duration_days) < 1:
        raise ValueError("crash_duration_days must be at least 1.")

    days = int(days)
    base_returns = simulate_portfolio_block_bootstrap(
        returns_matrix=returns_matrix,
        weights=weights,
        days=days,
        simulations=simulations,
        block_size=block_size,
        daily_drift_shift=daily_drift_shift,
        rng=rng,
    )

    # Construct cumulative stress trajectory from Day 0 (1.0) down to crash_floor
    crash_days = min(days, int(crash_duration_days))
    drawdown_curve = np.linspace(1.0, crash_floor, crash_days + 1)
    if days > crash_days:
        post_crisis_hold = np.full(days - crash_days, crash_floor, dtype=np.float64)
        stress_curve = np.concatenate([drawdown_curve, post_crisis_hold])
    else:
        stress_curve = drawdown_curve

    # Convert cumulative drawdown curve into exact per-day multiplicative shock factors
    daily_stress_mult = (stress_curve[1:] / stress_curve[:-1])[:, None]
    stressed_growth = np.maximum((1.0 + base_returns) * daily_stress_mult, 1e-6)
    return stressed_growth - 1.0


# ---------------------------------------------------------
# 6. INSTITUTIONAL RISK METRICS, CASH FLOWS & ANALYTICS
# ---------------------------------------------------------
def compute_institutional_risk_metrics(
    daily_returns: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """
    Computes institutional risk statistics cleanly in vectorized NumPy:
      - Annualized Sharpe Ratio (median across simulated paths)
      - Annualized Sortino Ratio (downside semi-deviation below daily risk-free rate)
      - Maximum Drawdown (%) — median, mean, and worst 10th-percentile tail
      - 95% Value at Risk (VaR) — Daily %, 1-Year %, and Full-Horizon %
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] < 1:
        raise ValueError("daily_returns must be a non-empty 1D or 2D array.")

    days, simulations = r.shape
    rf_daily = float(risk_free_rate) / float(trading_days_per_year)
    excess = r - rf_daily
    ann_factor = math.sqrt(float(trading_days_per_year))

    # 1. Annualized Sharpe Ratio per path -> median across paths
    mean_excess = np.mean(excess, axis=0)
    ddof = 1 if days > 1 else 0
    std_excess = np.std(excess, axis=0, ddof=ddof)
    path_sharpe = np.where(std_excess > 1e-12, (mean_excess / std_excess) * ann_factor, 0.0)
    sharpe_ratio = float(np.median(path_sharpe))

    # 2. Annualized Sortino Ratio using downside deviation below daily risk-free threshold
    downside_diff = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2, axis=0))
    path_sortino = np.where(downside_dev > 1e-12, (mean_excess / downside_dev) * ann_factor, 0.0)
    sortino_ratio = float(np.median(path_sortino))

    # 3. Maximum Drawdown (%) on pure market wealth index (starts at 1.0 on Day 0)
    growth = np.maximum(1.0 + r, 1e-6)
    wealth_index = np.vstack([np.ones((1, simulations), dtype=np.float64), np.cumprod(growth, axis=0)])
    running_peaks = np.maximum.accumulate(wealth_index, axis=0)
    drawdowns = (wealth_index - running_peaks) / running_peaks
    path_max_dd = np.min(drawdowns, axis=0)

    median_max_drawdown_pct = float(np.median(path_max_dd) * 100.0)
    worst_10_max_drawdown_pct = float(np.percentile(path_max_dd, 10) * 100.0)
    mean_max_drawdown_pct = float(np.mean(path_max_dd) * 100.0)

    # 4. 95% Value at Risk (5th percentile of return distribution; negative = loss)
    var_95_daily_pct = float(np.percentile(r, 5) * 100.0)
    one_year_idx = min(days, trading_days_per_year)
    one_year_returns = wealth_index[one_year_idx, :] - 1.0
    var_95_annual_pct = float(np.percentile(one_year_returns, 5) * 100.0)
    horizon_returns = wealth_index[-1, :] - 1.0
    var_95_horizon_pct = float(np.percentile(horizon_returns, 5) * 100.0)

    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "median_max_drawdown_pct": median_max_drawdown_pct,
        "worst_10_max_drawdown_pct": worst_10_max_drawdown_pct,
        "mean_max_drawdown_pct": mean_max_drawdown_pct,
        "var_95_daily_pct": var_95_daily_pct,
        "var_95_annual_pct": var_95_annual_pct,
        "var_95_horizon_pct": var_95_horizon_pct,
    }


def compute_portfolio_analytics(
    sim_daily_returns: np.ndarray,
    initial_investment: float = 10_000.0,
    monthly_contribution: float = 0.0,
    inflation_adjusted: bool = False,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """
    Transforms simulated portfolio daily returns into full cash-flow dollar paths,
    optional inflation-adjusted (real purchasing power) trajectories, percentile fan
    bands, recovery analytics, and institutional risk metrics — 100% vectorized.
    """
    r = np.asarray(sim_daily_returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] < 1:
        raise ValueError("sim_daily_returns must be a non-empty 2D array of shape (days, simulations).")
    if not math.isfinite(initial_investment) or initial_investment <= 0:
        raise ValueError("initial_investment must be a positive finite number.")
    if not math.isfinite(monthly_contribution) or monthly_contribution < 0:
        raise ValueError("monthly_contribution must be a non-negative finite number.")

    days, simulations = r.shape

    # Cumulative market wealth index G[t, s] starting at 1.0 on Day 0
    growth = np.maximum(1.0 + r, 1e-6)
    wealth_index = np.vstack([np.ones((1, simulations), dtype=np.float64), np.cumprod(growth, axis=0)])

    # Vectorized Dollar-Cost Averaging (DCA):
    # Deposit initial_investment on Day 0, plus monthly_contribution every 21 trading days.
    # Units bought on day t = contribution[t] / G[t, s]; total nominal value = cumsum(units) * G[t, s].
    contrib_schedule = np.zeros(days + 1, dtype=np.float64)
    contrib_schedule[0] = float(initial_investment)
    if monthly_contribution > 0.0 and days >= TRADING_DAYS_PER_MONTH:
        contrib_schedule[TRADING_DAYS_PER_MONTH::TRADING_DAYS_PER_MONTH] = float(monthly_contribution)

    units_bought = contrib_schedule[:, None] / np.maximum(wealth_index, 1e-12)
    nominal_value_paths = np.cumsum(units_bought, axis=0) * wealth_index

    # Optional inflation adjustment: deflate nominal dollars back into Today's Purchasing Power
    if inflation_adjusted and inflation_rate != 0.0:
        years_elapsed = np.arange(days + 1, dtype=np.float64) / float(trading_days_per_year)
        deflator = (1.0 + float(inflation_rate)) ** years_elapsed
    else:
        deflator = np.ones(days + 1, dtype=np.float64)

    value_paths = nominal_value_paths / deflator[:, None]
    cumulative_contributions = np.cumsum(contrib_schedule / deflator)
    total_nominal_contributed = float(np.sum(contrib_schedule))
    total_basis_contributed = float(cumulative_contributions[-1])

    # Percentile trajectories (10th, 25th, 50th, 75th, 90th)
    percentile_levels = (10, 25, 50, 75, 90)
    raw_val_pcts = np.percentile(value_paths, percentile_levels, axis=1)
    value_percentiles = {level: raw_val_pcts[idx] for idx, level in enumerate(percentile_levels)}
    ending_values = {level: float(value_percentiles[level][-1]) for level in percentile_levels}

    # Probability of ending below total invested capital (in matching nominal or real dollar basis)
    prob_loss_pct = float(np.mean(value_paths[-1, :] < total_basis_contributed) * 100.0)

    # Ending 95% VaR in dollars (5th percentile ending balance and P&L vs invested capital)
    var_95_ending_value = float(np.percentile(value_paths[-1, :], 5))
    var_95_dollar_pnl = var_95_ending_value - total_basis_contributed

    # Institutional risk metrics computed cleanly in NumPy
    risk_metrics = compute_institutional_risk_metrics(
        daily_returns=r,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )

    # Recovery time (months) on the market wealth index after each path's trough
    index_after_start = wealth_index[1:, :]
    trough_idx = np.argmin(index_after_start, axis=0)
    day_indices = np.arange(days)[:, None]
    after_trough = day_indices > trough_idx[None, :]
    recovered_mask = after_trough & (index_after_start >= 1.0)

    has_recovered = np.any(recovered_mask, axis=0)
    first_recovery_day = np.argmax(recovered_mask, axis=0) + 1
    never_dipped = np.min(index_after_start, axis=0) >= 1.0

    recovery_days = np.where(
        never_dipped,
        0.0,
        np.where(has_recovered, first_recovery_day.astype(np.float64), np.inf),
    )
    recovery_months = recovery_days / (float(trading_days_per_year) / 12.0)
    median_rec = float(np.median(recovery_months))
    median_recovery_months: float | None = median_rec if math.isfinite(median_rec) else None
    recovery_rate_pct = float(np.mean(np.isfinite(recovery_months)) * 100.0)

    # Median annualized CAGR of the market wealth index (in displayed real or nominal terms)
    horizon_years_exact = float(days) / float(trading_days_per_year)
    effective_wealth_end = wealth_index[-1, :] / deflator[-1]
    median_cagr_pct = float((np.median(effective_wealth_end) ** (1.0 / max(horizon_years_exact, 1e-6)) - 1.0) * 100.0)

    return {
        "value_percentiles": value_percentiles,
        "ending_values": ending_values,
        "value_paths_with_start": value_paths,
        "cumulative_contributions": cumulative_contributions,
        "total_nominal_contributed": total_nominal_contributed,
        "total_basis_contributed": total_basis_contributed,
        "inflation_adjusted": bool(inflation_adjusted),
        "inflation_rate": float(inflation_rate),
        "prob_loss_pct": prob_loss_pct,
        "var_95_ending_value": var_95_ending_value,
        "var_95_dollar_pnl": var_95_dollar_pnl,
        "median_cagr_pct": median_cagr_pct,
        "median_recovery_months": median_recovery_months,
        "recovery_rate_pct": recovery_rate_pct,
        **risk_metrics,
    }


def run_portfolio_simulation(
    portfolio_data: Mapping[str, Any],
    engine_name: str,
    horizon_years: float,
    initial_investment: float = 10_000.0,
    monthly_contribution: float = 0.0,
    inflation_adjusted: bool = False,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    apply_drift_dampener: bool = False,
    simulations: int = DEFAULT_SIMULATIONS,
    df: int = DEFAULT_STUDENT_T_DF,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crash_floor: float = DEFAULT_STRESS_FLOOR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    End-to-end multi-asset simulation dispatcher combining drift dampening,
    correlated / block-bootstrapped return generation, cash flows, inflation
    deflation, and institutional risk analytics.
    """
    days = max(1, int(round(horizon_years * TRADING_DAYS_PER_YEAR)))
    returns_matrix = portfolio_data["returns_matrix"]
    tickers = portfolio_data["tickers"]
    weights = portfolio_data["weights"]
    trailing_pes = portfolio_data["trailing_pes"]

    drift_info = compute_dampened_daily_drift(
        returns_matrix=returns_matrix,
        tickers=tickers,
        weights=weights,
        trailing_pes=trailing_pes,
    )

    daily_mean_override = drift_info["dampened_daily_mu"] if apply_drift_dampener else None
    daily_drift_shift = drift_info["daily_drift_shift"] if apply_drift_dampener else None
    used_psd_fallback = False

    if engine_name == ENGINE_STUDENT_T:
        sim_returns, used_psd_fallback = simulate_portfolio_student_t(
            returns_matrix=returns_matrix,
            weights=weights,
            days=days,
            simulations=simulations,
            df=df,
            daily_mean_override=daily_mean_override,
            rng=rng,
        )
    elif engine_name == ENGINE_BLOCK_BOOTSTRAP:
        sim_returns = simulate_portfolio_block_bootstrap(
            returns_matrix=returns_matrix,
            weights=weights,
            days=days,
            simulations=simulations,
            block_size=block_size,
            daily_drift_shift=daily_drift_shift,
            rng=rng,
        )
    elif engine_name == ENGINE_2008_STRESS:
        sim_returns = simulate_portfolio_2008_stress(
            returns_matrix=returns_matrix,
            weights=weights,
            days=days,
            simulations=simulations,
            block_size=block_size,
            crash_floor=crash_floor,
            daily_drift_shift=daily_drift_shift,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown simulation engine '{engine_name}'. "
            f"Expected one of: {', '.join(AVAILABLE_ENGINES)}."
        )

    analytics = compute_portfolio_analytics(
        sim_daily_returns=sim_returns,
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        inflation_adjusted=inflation_adjusted,
        inflation_rate=inflation_rate,
        risk_free_rate=risk_free_rate,
    )
    analytics["drift_info"] = drift_info
    analytics["drift_dampener_active"] = bool(apply_drift_dampener)
    analytics["used_psd_fallback"] = used_psd_fallback
    analytics["days"] = days
    return analytics


def simulate_spy_benchmark(
    engine_name: str,
    horizon_years: float,
    initial_investment: float = 10_000.0,
    monthly_contribution: float = 0.0,
    inflation_adjusted: bool = False,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    apply_drift_dampener: bool = False,
    simulations: int = DEFAULT_SIMULATIONS,
    df: int = DEFAULT_STUDENT_T_DF,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crash_floor: float = DEFAULT_STRESS_FLOOR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    spy_portfolio_data: Mapping[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any] | None:
    """
    Runs the exact same simulation configuration on S&P 500 (SPY) to produce
    a median benchmark trajectory overlay. Returns None gracefully if SPY data
    cannot be fetched so the main simulation is never blocked.
    """
    try:
        spy_data = (
            spy_portfolio_data
            if spy_portfolio_data is not None
            else fetch_portfolio_data(["SPY"], [1.0])
        )
        spy_summary = run_portfolio_simulation(
            portfolio_data=spy_data,
            engine_name=engine_name,
            horizon_years=horizon_years,
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            inflation_adjusted=inflation_adjusted,
            inflation_rate=inflation_rate,
            apply_drift_dampener=apply_drift_dampener,
            simulations=simulations,
            df=df,
            block_size=block_size,
            crash_floor=crash_floor,
            risk_free_rate=risk_free_rate,
            rng=rng,
        )
        return {
            "median_path": spy_summary["value_percentiles"][50],
            "ending_median": spy_summary["ending_values"][50],
            "sharpe_ratio": spy_summary["sharpe_ratio"],
            "sortino_ratio": spy_summary["sortino_ratio"],
            "median_max_drawdown_pct": spy_summary["median_max_drawdown_pct"],
            "var_95_annual_pct": spy_summary["var_95_annual_pct"],
            "prob_loss_pct": spy_summary["prob_loss_pct"],
        }
    except Exception:
        return None


# ---------------------------------------------------------
# 7. EFFICIENT FRONTIER & MAX-SHARPE OPTIMIZER (PyPortfolioOpt)
# ---------------------------------------------------------
def compute_efficient_frontier(
    prices_df: pd.DataFrame,
    current_weights: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    n_points: int = 35,
) -> dict[str, Any]:
    """
    Uses PyPortfolioOpt (EfficientFrontier) to compute:
      - Optimal Maximum Sharpe Ratio portfolio weights & performance
      - Optimal Minimum Volatility portfolio weights & performance
      - Current user portfolio performance
      - Interactive Efficient Frontier curve coordinates (volatility vs expected return)
      - Individual asset risk/return coordinates

    Wrapped in defensive try/except blocks so single-asset inputs or convex solver
    edge cases return a clean status payload instead of crashing the app.
    """
    tickers = list(prices_df.columns)
    w = np.asarray(current_weights, dtype=np.float64)

    if len(tickers) < 2:
        # Compute simple single-asset stats so UI can still show informative feedback
        returns_1d = prices_df.iloc[:, 0].pct_change().dropna().to_numpy(dtype=np.float64)
        ann_ret = float((prices_df.iloc[-1, 0] / prices_df.iloc[0, 0]) ** (TRADING_DAYS_PER_YEAR / max(len(returns_1d), 1)) - 1.0)
        ann_vol = float(np.std(returns_1d, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns_1d) > 1 else 0.0
        return {
            "available": False,
            "reason": "single_asset",
            "message": "Add at least 2 tickers in Portfolio Mode to compute the Efficient Frontier and optimize asset weights.",
            "single_asset": {
                "ticker": tickers[0] if tickers else "",
                "expected_return": ann_ret,
                "volatility": ann_vol,
            },
        }

    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models
    except Exception as exc:
        return {
            "available": False,
            "reason": "missing_dependency",
            "message": f"PyPortfolioOpt is unavailable ({exc}).",
        }

    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
        raw_cov = risk_models.sample_cov(prices_df, frequency=TRADING_DAYS_PER_YEAR)
        cov_matrix = risk_models.fix_nonpositive_semidefinite(raw_cov, fix_method="spectral")

        # 1. Current user portfolio metrics
        mu_vec = mu.to_numpy(dtype=np.float64)
        cov_arr = cov_matrix.to_numpy(dtype=np.float64)
        curr_ret = float(np.dot(w, mu_vec))
        curr_vol = float(np.sqrt(max(float(w @ cov_arr @ w), 0.0)))
        curr_sharpe = (curr_ret - risk_free_rate) / curr_vol if curr_vol > 1e-12 else 0.0

        # 2. Individual asset risk/return points
        individual_assets = []
        for idx, sym in enumerate(tickers):
            a_ret = float(mu_vec[idx])
            a_vol = float(np.sqrt(max(float(cov_arr[idx, idx]), 0.0)))
            a_sharpe = (a_ret - risk_free_rate) / a_vol if a_vol > 1e-12 else 0.0
            individual_assets.append(
                {
                    "ticker": sym,
                    "expected_return": a_ret,
                    "volatility": a_vol,
                    "sharpe": a_sharpe,
                    "current_weight": float(w[idx]),
                }
            )

        # 3. Minimum Volatility Portfolio
        ef_min = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
        ef_min.min_volatility()
        min_vol_weights = {sym: float(val) for sym, val in ef_min.clean_weights().items()}
        min_ret, min_vol, min_sharpe = ef_min.portfolio_performance(
            verbose=False, risk_free_rate=risk_free_rate
        )

        # 4. Maximum Sharpe Portfolio (with defensive fallback if all returns <= risk_free_rate)
        solver_note: str | None = None
        ef_max = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
        try:
            ef_max.max_sharpe(risk_free_rate=risk_free_rate)
            max_sharpe_weights = {sym: float(val) for sym, val in ef_max.clean_weights().items()}
            max_ret, max_vol, max_sharpe = ef_max.portfolio_performance(
                verbose=False, risk_free_rate=risk_free_rate
            )
        except Exception:
            # If every asset's historical return is below risk_free_rate, maximize Sharpe vs a lower floor
            fallback_rf = min(0.0, float(np.max(mu_vec)) - 0.01)
            ef_max_fb = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
            try:
                ef_max_fb.max_sharpe(risk_free_rate=fallback_rf)
                max_sharpe_weights = {sym: float(val) for sym, val in ef_max_fb.clean_weights().items()}
                max_ret, max_vol, max_sharpe = ef_max_fb.portfolio_performance(
                    verbose=False, risk_free_rate=risk_free_rate
                )
                solver_note = (
                    "All selected assets had historical returns at or below the risk-free rate; "
                    "tangent weights were solved using an adjusted baseline."
                )
            except Exception:
                max_sharpe_weights = dict(min_vol_weights)
                max_ret, max_vol, max_sharpe = float(min_ret), float(min_vol), float(min_sharpe)
                solver_note = "Max-Sharpe solver fell back to the Minimum Volatility allocation."

        # 5. Trace Efficient Frontier curve from Min-Vol return up to Max-Asset return
        frontier_vols: list[float] = []
        frontier_rets: list[float] = []
        frontier_sharpes: list[float] = []

        ret_low = float(min_ret)
        ret_high = float(np.max(mu_vec))
        if ret_high > ret_low + 1e-4:
            target_grid = np.linspace(ret_low, ret_high * 0.998, max(int(n_points), 10))
            for target_r in target_grid:
                try:
                    ef_pt = EfficientFrontier(mu, cov_matrix, weight_bounds=(0.0, 1.0))
                    ef_pt.efficient_return(target_return=float(target_r))
                    pt_ret, pt_vol, pt_sharpe = ef_pt.portfolio_performance(
                        verbose=False, risk_free_rate=risk_free_rate
                    )
                    if math.isfinite(pt_vol) and math.isfinite(pt_ret):
                        frontier_vols.append(float(pt_vol))
                        frontier_rets.append(float(pt_ret))
                        frontier_sharpes.append(float(pt_sharpe))
                except Exception:
                    continue

        return {
            "available": True,
            "reason": "ok",
            "note": solver_note,
            "tickers": tickers,
            "risk_free_rate": float(risk_free_rate),
            "frontier_vols": frontier_vols,
            "frontier_rets": frontier_rets,
            "frontier_sharpes": frontier_sharpes,
            "individual_assets": individual_assets,
            "current": {
                "weights": {sym: float(val) for sym, val in zip(tickers, w)},
                "expected_return": curr_ret,
                "volatility": curr_vol,
                "sharpe": curr_sharpe,
            },
            "min_vol": {
                "weights": min_vol_weights,
                "expected_return": float(min_ret),
                "volatility": float(min_vol),
                "sharpe": float(min_sharpe),
            },
            "max_sharpe": {
                "weights": max_sharpe_weights,
                "expected_return": float(max_ret),
                "volatility": float(max_vol),
                "sharpe": float(max_sharpe),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": "solver_error",
            "message": f"Efficient Frontier optimization could not converge for this ticker mix: {exc}",
        }


# ---------------------------------------------------------
# 8. LIVE NEWS SENTIMENT PULSE (yfinance + vaderSentiment)
# ---------------------------------------------------------
def _parse_news_item(item: Mapping[str, Any], ticker: str) -> dict[str, Any] | None:
    """
    Defensively extracts title, publisher, link, and timestamp from both
    newer nested yfinance (>=0.2.50 / 1.x `content`) and legacy flat news schemas.
    """
    if not isinstance(item, Mapping):
        return None

    content = item.get("content") if isinstance(item.get("content"), Mapping) else item

    title = str(content.get("title") or item.get("title") or "").strip()
    if not title:
        return None

    # Publisher
    provider = content.get("provider")
    if isinstance(provider, Mapping) and provider.get("displayName"):
        publisher = str(provider["displayName"]).strip()
    else:
        publisher = str(item.get("publisher") or content.get("publisher") or "Yahoo Finance").strip()

    # URL
    link = ""
    for url_key in ("canonicalUrl", "clickThroughUrl"):
        url_obj = content.get(url_key)
        if isinstance(url_obj, Mapping) and url_obj.get("url"):
            link = str(url_obj["url"]).strip()
            break
    if not link:
        link = str(item.get("link") or content.get("link") or "").strip()

    # Timestamp (epoch seconds or ISO 8601 string)
    published_ts = 0.0
    published_str = ""
    raw_date = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
    if isinstance(raw_date, (int, float)) and raw_date > 0:
        published_ts = float(raw_date)
        published_str = datetime.fromtimestamp(published_ts, tz=timezone.utc).strftime("%b %d, %Y")
    elif isinstance(raw_date, str) and raw_date.strip():
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            published_ts = dt.timestamp()
            published_str = dt.strftime("%b %d, %Y")
        except Exception:
            published_str = raw_date[:10]

    return {
        "ticker": ticker,
        "title": title,
        "publisher": publisher,
        "link": link,
        "published": published_str,
        "published_ts": published_ts,
    }


def _classify_vader_score(compound: float) -> str:
    """Maps a VADER compound score in [-1, 1] to Bullish / Neutral / Bearish."""
    if compound >= 0.05:
        return "Bullish"
    if compound <= -0.05:
        return "Bearish"
    return "Neutral"


def fetch_news_sentiment(
    tickers: Sequence[str],
    max_headlines: int = 3,
) -> dict[str, Any]:
    """
    Fetches recent headlines for the portfolio tickers via yfinance (Ticker.news)
    and scores them with VADER sentiment analysis.

    Returns a Bullish / Neutral / Bearish pulse, average compound score, and the
    top headlines. Never raises on empty or unavailable Yahoo News responses.
    """
    fallback = {
        "available": False,
        "label": "Neutral",
        "compound_score": 0.0,
        "headline_count": 0,
        "headlines": [],
    }

    if not tickers:
        return fallback

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        return fallback

    seen_titles: set[str] = set()
    collected: list[dict[str, Any]] = []

    for raw_sym in tickers:
        try:
            sym = normalize_ticker(str(raw_sym))
            news_items = yf.Ticker(sym).news or []
        except Exception:
            continue

        if not isinstance(news_items, list):
            continue

        for item in news_items[:6]:
            parsed = _parse_news_item(item, sym)
            if parsed is None:
                continue
            dedupe_key = parsed["title"].lower()
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)

            try:
                scores = analyzer.polarity_scores(parsed["title"])
                compound = float(scores.get("compound", 0.0))
            except Exception:
                compound = 0.0

            parsed["compound"] = compound
            parsed["sentiment"] = _classify_vader_score(compound)
            collected.append(parsed)

    if not collected:
        return fallback

    # Sort newest first (falls back to insertion order when timestamps are missing)
    collected.sort(key=lambda h: h["published_ts"], reverse=True)
    scored_pool = collected[:15]
    avg_compound = float(np.mean([h["compound"] for h in scored_pool]))

    return {
        "available": True,
        "label": _classify_vader_score(avg_compound),
        "compound_score": avg_compound,
        "headline_count": len(scored_pool),
        "headlines": scored_pool[: max(1, int(max_headlines))],
    }


# ---------------------------------------------------------
# 9. BACKWARD-COMPATIBLE SINGLE-ASSET WRAPPERS (V1 API)
# ---------------------------------------------------------
def _validate_simulation_inputs(
    current_price: float,
    daily_returns: np.ndarray,
    days: int,
    simulations: int,
) -> np.ndarray:
    """Validates common arguments shared across single-asset simulation wrappers."""
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price must be a positive finite number.")
    if int(days) < 1:
        raise ValueError("Simulation horizon must be at least 1 trading day.")
    if int(simulations) < 1:
        raise ValueError("Number of simulations must be at least 1.")

    returns_arr = np.asarray(daily_returns, dtype=np.float64)
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size < 2:
        raise ValueError("daily_returns must contain at least 2 finite observations.")
    return returns_arr


def run_student_t(
    current_price: float,
    daily_returns: np.ndarray,
    days: int,
    simulations: int,
    df: int = DEFAULT_STUDENT_T_DF,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single-asset heavy-tailed Student-t Monte Carlo returning share price paths."""
    returns_arr = _validate_simulation_inputs(current_price, daily_returns, days, simulations)
    if df <= 2:
        raise ValueError("Degrees of freedom (df) must be greater than 2 for finite variance.")

    days = int(days)
    simulations = int(simulations)
    avg_daily_return = float(np.mean(returns_arr))
    daily_volatility = float(np.std(returns_arr))

    raw_shocks = t.rvs(df, size=(days, simulations), random_state=rng)
    shock_std = float(np.std(raw_shocks))
    if shock_std == 0.0 or not math.isfinite(shock_std):
        scaled_shocks = np.full((days, simulations), avg_daily_return, dtype=np.float64)
    else:
        scaled_shocks = (raw_shocks / shock_std) * daily_volatility + avg_daily_return

    growth_factors = np.maximum(1.0 + scaled_shocks, 1e-6)
    return current_price * np.cumprod(growth_factors, axis=0)


def run_block_bootstrap(
    current_price: float,
    daily_returns: np.ndarray,
    days: int,
    simulations: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single-asset Historical Block Bootstrap returning share price paths."""
    returns_arr = _validate_simulation_inputs(current_price, daily_returns, days, simulations)
    sim_r = simulate_portfolio_block_bootstrap(
        returns_matrix=returns_arr[:, None],
        weights=np.array([1.0], dtype=np.float64),
        days=days,
        simulations=simulations,
        block_size=block_size,
        rng=rng,
    )
    return current_price * np.cumprod(np.maximum(1.0 + sim_r, 1e-6), axis=0)


def run_2008_stress_test(
    current_price: float,
    daily_returns: np.ndarray,
    days: int,
    simulations: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crash_floor: float = DEFAULT_STRESS_FLOOR,
    crash_duration_days: int = DEFAULT_STRESS_DAYS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single-asset 2008 Crisis Stress Test returning share price paths."""
    returns_arr = _validate_simulation_inputs(current_price, daily_returns, days, simulations)
    sim_r = simulate_portfolio_2008_stress(
        returns_matrix=returns_arr[:, None],
        weights=np.array([1.0], dtype=np.float64),
        days=days,
        simulations=simulations,
        block_size=block_size,
        crash_floor=crash_floor,
        crash_duration_days=crash_duration_days,
        rng=rng,
    )
    return current_price * np.cumprod(np.maximum(1.0 + sim_r, 1e-6), axis=0)


def run_simulation(
    engine_name: str,
    current_price: float,
    daily_returns: np.ndarray,
    horizon_years: float,
    simulations: int = DEFAULT_SIMULATIONS,
    df: int = DEFAULT_STUDENT_T_DF,
    block_size: int = DEFAULT_BLOCK_SIZE,
    crash_floor: float = DEFAULT_STRESS_FLOOR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single-asset simulation dispatcher preserved for backward compatibility."""
    days = max(1, int(round(horizon_years * TRADING_DAYS_PER_YEAR)))
    if engine_name == ENGINE_STUDENT_T:
        return run_student_t(current_price, daily_returns, days, simulations, df=df, rng=rng)
    if engine_name == ENGINE_BLOCK_BOOTSTRAP:
        return run_block_bootstrap(current_price, daily_returns, days, simulations, block_size=block_size, rng=rng)
    if engine_name == ENGINE_2008_STRESS:
        return run_2008_stress_test(
            current_price, daily_returns, days, simulations, block_size=block_size, crash_floor=crash_floor, rng=rng
        )
    raise ValueError(f"Unknown simulation engine '{engine_name}'.")


def compute_summary_metrics(
    price_paths: np.ndarray,
    current_price: float,
    initial_investment: float = 10_000.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """Single-asset summary wrapper preserved for backward compatibility."""
    paths = np.asarray(price_paths, dtype=np.float64)
    day_zero = np.full((1, paths.shape[1]), current_price, dtype=np.float64)
    full_prices = np.vstack([day_zero, paths])
    sim_returns = (full_prices[1:, :] / np.maximum(full_prices[:-1, :], 1e-12)) - 1.0
    analytics = compute_portfolio_analytics(
        sim_daily_returns=sim_returns,
        initial_investment=initial_investment,
        trading_days_per_year=trading_days_per_year,
    )
    shares = initial_investment / current_price
    analytics["shares"] = shares
    analytics["price_percentiles"] = {k: v / shares for k, v in analytics["value_percentiles"].items()}
    analytics["ending_prices"] = {k: v / shares for k, v in analytics["ending_values"].items()}
    return analytics


# ---------------------------------------------------------
# 10. STANDALONE SELF-TEST (V2.0 Multi-Asset + All Modules)
# ---------------------------------------------------------
if __name__ == "__main__":
    TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]
    TEST_WEIGHTS = [40.0, 35.0, 25.0]
    HORIZON_YEARS = 3
    INITIAL_INVESTMENT = 10_000.0
    MONTHLY_CONTRIBUTION = 250.0
    SIMULATIONS = 2_000

    print(
        f"Running V2.0 standalone self-test for {TEST_TICKERS} "
        f"(weights {TEST_WEIGHTS}, {HORIZON_YEARS}y, {SIMULATIONS:,} paths)..."
    )
    try:
        portfolio = fetch_portfolio_data(TEST_TICKERS, TEST_WEIGHTS)
    except StockDataError as err:
        print(f"Data fetch failed: {err}")
        raise SystemExit(1)

    w_pe = portfolio["weighted_pe"]
    pe_display = f"{w_pe:.2f}" if isinstance(w_pe, (int, float)) else "N/A"
    print(
        f"Aligned {portfolio['num_trading_days']:,} trading days ({portfolio['years_of_history']:.1f} yrs) | "
        f"Portfolio Ann. Vol: {portfolio['annualized_volatility'] * 100:.1f}% | "
        f"Weighted P/E: {pe_display}"
    )

    rng = np.random.default_rng(42)
    for engine in AVAILABLE_ENGINES:
        summary = run_portfolio_simulation(
            portfolio_data=portfolio,
            engine_name=engine,
            horizon_years=HORIZON_YEARS,
            initial_investment=INITIAL_INVESTMENT,
            monthly_contribution=MONTHLY_CONTRIBUTION,
            inflation_adjusted=True,
            apply_drift_dampener=True,
            simulations=SIMULATIONS,
            rng=rng,
        )
        rec_str = (
            f"{summary['median_recovery_months']:.1f} mo"
            if summary["median_recovery_months"] is not None
            else f"> {HORIZON_YEARS * 12} mo ({summary['recovery_rate_pct']:.1f}% recovered)"
        )
        print(
            f"\n[{engine}] (Real $, +${MONTHLY_CONTRIBUTION:.0f}/mo, Drift Dampener ON)\n"
            f"  10th (Unlucky): ${summary['ending_values'][10]:>10,.2f} | "
            f"50th (Median): ${summary['ending_values'][50]:>10,.2f} | "
            f"90th (Lucky): ${summary['ending_values'][90]:>10,.2f}\n"
            f"  Sharpe: {summary['sharpe_ratio']:.2f} | Sortino: {summary['sortino_ratio']:.2f} | "
            f"Median Max DD: {summary['median_max_drawdown_pct']:.1f}% | "
            f"95% VaR (1Y): {summary['var_95_annual_pct']:.1f}%\n"
            f"  Prob Loss: {summary['prob_loss_pct']:.1f}% | Recovery: {rec_str}"
        )

    # Verify nearest-PSD fallback explicitly on a deliberately non-PSD matrix
    bad_cov = np.array([[0.01, 0.02], [0.02, 0.01]], dtype=np.float64)
    l_mat, used_fb = cholesky_with_psd_fallback(bad_cov)
    assert used_fb and l_mat.shape == (2, 2) and np.all(np.isfinite(l_mat))
    print("\n[Cholesky Nearest-PSD Fallback] Passed on non-PSD test matrix.")

    # Verify SPY benchmark overlay
    spy_bench = simulate_spy_benchmark(
        engine_name=ENGINE_STUDENT_T,
        horizon_years=HORIZON_YEARS,
        initial_investment=INITIAL_INVESTMENT,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        inflation_adjusted=True,
        apply_drift_dampener=True,
        simulations=1_000,
        rng=np.random.default_rng(7),
    )
    if spy_bench is not None:
        print(
            f"[SPY Benchmark] Ending Median: ${spy_bench['ending_median']:,.2f} | "
            f"Sharpe: {spy_bench['sharpe_ratio']:.2f}"
        )
    else:
        print("[SPY Benchmark] Gracefully skipped (data unavailable).")

    # Verify Efficient Frontier (PyPortfolioOpt)
    ef_res = compute_efficient_frontier(portfolio["prices_df"], portfolio["weights"])
    if ef_res["available"]:
        print(
            f"[Efficient Frontier] Frontier pts: {len(ef_res['frontier_vols'])} | "
            f"Max-Sharpe ({ef_res['max_sharpe']['sharpe']:.2f}): {ef_res['max_sharpe']['weights']} | "
            f"Min-Vol ({ef_res['min_vol']['volatility'] * 100:.1f}%): {ef_res['min_vol']['weights']}"
        )
    else:
        print(f"[Efficient Frontier] Fallback: {ef_res['message']}")

    # Verify News Sentiment Pulse
    sentiment = fetch_news_sentiment(TEST_TICKERS, max_headlines=3)
    print(
        f"[News Sentiment] Available: {sentiment['available']} | "
        f"Pulse: {sentiment['label']} ({sentiment['compound_score']:+.2f}) | "
        f"Headlines: {len(sentiment['headlines'])}"
    )
    for idx, item in enumerate(sentiment["headlines"], start=1):
        print(f"  {idx}. [{item['ticker']} | {item['sentiment']}] {item['title']}")
