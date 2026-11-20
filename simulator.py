"""
Quantitative simulation engine for the Stock Future Simulator.

This module is strictly decoupled from Streamlit and all UI frameworks so the
data layer, vectorized simulation engines, and summary analytics can be tested,
tuned, or reused independently from the command line or other applications.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t
import yfinance as yf

# ---------------------------------------------------------
# TUNABLE ENGINE DEFAULTS (Safe for non-specialists to edit)
# ---------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_HISTORY_PERIOD: str = "15y"
DEFAULT_SIMULATIONS: int = 2500       # 1,000-2,500 for fast mobile response; 10,000 for smoother bands
DEFAULT_STUDENT_T_DF: int = 4         # 3 = frequent extreme shocks; 10+ = near-normal bell curve
DEFAULT_BLOCK_SIZE: int = 20          # 5 = 1-week blocks; 20 = 1-month blocks; 63 = quarterly blocks
DEFAULT_STRESS_FLOOR: float = 0.55    # 0.55 = -45% Year 1 crash; 0.70 = -30% bear; 0.40 = -60% severe crash
DEFAULT_STRESS_DAYS: int = 252        # Length of the Year 1 drawdown phase in trading days

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
# 1. DATA FETCHER & RETURN EXTRACTION (Pure data layer)
# ---------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """Strips whitespace and uppercases a stock ticker symbol."""
    if not isinstance(ticker, str):
        raise ValueError("Ticker symbol must be a text string.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Please enter a valid stock ticker symbol (for example, AAPL).")
    return cleaned


def fetch_stock_history(
    ticker: str,
    period: str = DEFAULT_HISTORY_PERIOD,
) -> tuple[pd.DataFrame, float | str]:
    """
    Pulls up to 15 years of adjusted daily prices and trailing P/E from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "SPY").
    period : str, optional
        History window passed to yfinance (default: "15y").

    Returns
    -------
    tuple[pd.DataFrame, float | str]
        A tuple of (price history DataFrame, trailing P/E float or "N/A").

    Raises
    ------
    ValueError
        If the ticker string is empty or invalid.
    StockDataError
        If Yahoo Finance is unreachable, rate-limited, or returns no price data.
    """
    symbol = normalize_ticker(ticker)

    try:
        stock = yf.Ticker(symbol)
        # auto_adjust=True ensures Close reflects dividends and stock splits consistently
        hist = stock.history(period=period, auto_adjust=True)
    except Exception as exc:
        raise StockDataError(
            f"Could not download market data for '{symbol}'. "
            "Yahoo Finance may be temporarily rate-limiting requests or your network may be offline."
        ) from exc

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise StockDataError(
            f"No historical price data was found for '{symbol}'. "
            "Check that the ticker is spelled correctly and has not been delisted."
        )

    hist = hist.dropna(subset=["Close"]).copy()
    if hist.empty:
        raise StockDataError(
            f"Price history for '{symbol}' contains no valid closing prices."
        )

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

    return hist, pe_ratio


def extract_return_stats(hist: pd.DataFrame) -> dict[str, Any]:
    """
    Extracts current price, clean daily returns, annualized volatility, and history length.

    Parameters
    ----------
    hist : pd.DataFrame
        Historical price DataFrame containing a 'Close' column.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: 'current_price', 'daily_returns',
        'annualized_volatility', 'years_of_history', and 'num_trading_days'.
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


# ---------------------------------------------------------
# 2. THREE MODULAR SIMULATION ENGINES (100% Vectorized)
# ---------------------------------------------------------
def _validate_simulation_inputs(
    current_price: float,
    daily_returns: np.ndarray,
    days: int,
    simulations: int,
) -> np.ndarray:
    """Validates common arguments shared across all simulation engines."""
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
    """
    Mode 1: Generates synthetic daily returns using heavy-tailed Student-t shocks.

    Parameters
    ----------
    current_price : float
        Latest closing price of the stock.
    daily_returns : np.ndarray
        1D array of historical daily percentage returns.
    days : int
        Number of future trading days to simulate (e.g., 252 per year).
    simulations : int
        Number of independent future price paths to generate.
    df : int, optional
        Student-t degrees of freedom controlling fat tails (default: 4).
        Lower values (e.g., 3) produce sharper shocks; 10+ resembles a normal curve.
    rng : np.random.Generator | None, optional
        Optional NumPy random generator for reproducible results.

    Returns
    -------
    np.ndarray
        Simulated price paths of shape (days, simulations).
    """
    returns_arr = _validate_simulation_inputs(current_price, daily_returns, days, simulations)
    if df <= 2:
        raise ValueError("Degrees of freedom (df) must be greater than 2 for finite variance.")

    days = int(days)
    simulations = int(simulations)
    avg_daily_return = float(np.mean(returns_arr))
    daily_volatility = float(np.std(returns_arr))

    # Draw heavy-tailed Student-t shocks and standardize them to match the stock's volatility
    raw_shocks = t.rvs(df, size=(days, simulations), random_state=rng)
    shock_std = float(np.std(raw_shocks))

    if shock_std == 0.0 or not math.isfinite(shock_std):
        scaled_shocks = np.full((days, simulations), avg_daily_return, dtype=np.float64)
    else:
        scaled_shocks = (raw_shocks / shock_std) * daily_volatility + avg_daily_return

    # Prevent extreme synthetic tail draws from producing negative share prices
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
    """
    Mode 2: Resamples contiguous historical return blocks to preserve crashes and rallies.

    Parameters
    ----------
    current_price : float
        Latest closing price of the stock.
    daily_returns : np.ndarray
        1D array of historical daily percentage returns.
    days : int
        Number of future trading days to simulate.
    simulations : int
        Number of independent future price paths to generate.
    block_size : int, optional
        Consecutive trading days per resampled block (default: 20, ~1 trading month).
        Use 5 for 1-week blocks or 63 for quarterly blocks.
    rng : np.random.Generator | None, optional
        Optional NumPy random generator for reproducible results.

    Returns
    -------
    np.ndarray
        Simulated price paths of shape (days, simulations).
    """
    returns_arr = _validate_simulation_inputs(current_price, daily_returns, days, simulations)
    days = int(days)
    simulations = int(simulations)
    block_size = int(block_size)

    if block_size < 1:
        raise ValueError("block_size must be at least 1 trading day.")

    num_history_days = len(returns_arr)
    if num_history_days < block_size + 1:
        raise ValueError(
            f"Not enough price history ({num_history_days} trading days) for "
            f"{block_size}-day block bootstrapping. At least {block_size + 1} daily returns are required."
        )

    num_blocks_needed = int(np.ceil(days / block_size))
    max_start_idx = num_history_days - block_size

    generator = rng if rng is not None else np.random.default_rng()
    random_starts = generator.integers(0, max_start_idx, size=(num_blocks_needed, simulations))

    # Build a 3D index grid to pull contiguous blocks across all simulations at once
    offsets = np.arange(block_size)[:, None, None]
    block_indices = random_starts[None, :, :] + offsets  # Shape: (block_size, num_blocks, simulations)
    sampled_blocks = returns_arr[block_indices]

    # Reshape into (total_days, simulations) and trim to the exact horizon length
    bootstrapped_returns = sampled_blocks.transpose(1, 0, 2).reshape(-1, simulations)[:days]
    growth_factors = np.maximum(1.0 + bootstrapped_returns, 1e-6)
    return current_price * np.cumprod(growth_factors, axis=0)


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
    """
    Mode 3: Injects a -45% Year 1 drawdown onto block-bootstrapped paths, then tracks recovery.

    Parameters
    ----------
    current_price : float
        Latest closing price of the stock.
    daily_returns : np.ndarray
        1D array of historical daily percentage returns.
    days : int
        Number of future trading days to simulate.
    simulations : int
        Number of independent future price paths to generate.
    block_size : int, optional
        Consecutive trading days per resampled block (default: 20).
    crash_floor : float, optional
        Year 1 crash floor multiplier (default: 0.55, representing a -45% shock).
        Use 0.70 for a -30% bear market or 0.40 for a -60% severe crash.
    crash_duration_days : int, optional
        Trading days over which the initial shock unfolds (default: 252, i.e., 1 year).
    rng : np.random.Generator | None, optional
        Optional NumPy random generator for reproducible results.

    Returns
    -------
    np.ndarray
        Stress-tested price paths of shape (days, simulations).
    """
    if not (0.0 < crash_floor <= 1.0):
        raise ValueError("crash_floor must be between 0 (exclusive) and 1 (inclusive).")
    if int(crash_duration_days) < 1:
        raise ValueError("crash_duration_days must be at least 1.")

    days = int(days)
    baseline_paths = run_block_bootstrap(
        current_price=current_price,
        daily_returns=daily_returns,
        days=days,
        simulations=simulations,
        block_size=block_size,
        rng=rng,
    )

    # Smoothly apply the Year 1 crash multiplier down to crash_floor (e.g., 0.55 = -45%)
    crash_days = min(days, int(crash_duration_days))
    drawdown_curve = np.linspace(1.0, crash_floor, crash_days)

    # Hold the shock floor steady in Years 2+ so natural historical returns drive recovery
    if days > crash_days:
        post_crisis_hold = np.full(days - crash_days, crash_floor, dtype=np.float64)
        stress_multiplier = np.concatenate([drawdown_curve, post_crisis_hold])[:, None]
    else:
        stress_multiplier = drawdown_curve[:, None]

    return baseline_paths * stress_multiplier


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
    """Dispatches the selected simulation engine for the requested time horizon."""
    days = max(1, int(round(horizon_years * TRADING_DAYS_PER_YEAR)))

    if engine_name == ENGINE_STUDENT_T:
        return run_student_t(
            current_price=current_price,
            daily_returns=daily_returns,
            days=days,
            simulations=simulations,
            df=df,
            rng=rng,
        )
    if engine_name == ENGINE_BLOCK_BOOTSTRAP:
        return run_block_bootstrap(
            current_price=current_price,
            daily_returns=daily_returns,
            days=days,
            simulations=simulations,
            block_size=block_size,
            rng=rng,
        )
    if engine_name == ENGINE_2008_STRESS:
        return run_2008_stress_test(
            current_price=current_price,
            daily_returns=daily_returns,
            days=days,
            simulations=simulations,
            block_size=block_size,
            crash_floor=crash_floor,
            rng=rng,
        )
    raise ValueError(
        f"Unknown simulation engine '{engine_name}'. "
        f"Expected one of: {', '.join(AVAILABLE_ENGINES)}."
    )


# ---------------------------------------------------------
# 3. SUMMARY METRICS & RECOVERY ANALYTICS (100% Vectorized)
# ---------------------------------------------------------
def compute_summary_metrics(
    price_paths: np.ndarray,
    current_price: float,
    initial_investment: float = 10_000.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """
    Computes percentile trajectories, loss probability, max drawdowns, and recovery time.

    Parameters
    ----------
    price_paths : np.ndarray
        Simulated share price paths of shape (days, simulations).
    current_price : float
        Starting share price on Day 0.
    initial_investment : float, optional
        Starting dollar investment (default: $10,000.00).
    trading_days_per_year : int, optional
        Trading days per calendar year used to convert days into months (default: 252).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'price_percentiles': dict[int, np.ndarray] for 10, 25, 50, 75, 90 (includes Day 0)
        - 'value_percentiles': dict[int, np.ndarray] for 10, 25, 50, 75, 90 (includes Day 0)
        - 'ending_prices': dict[int, float] for 10, 25, 50, 75, 90
        - 'ending_values': dict[int, float] for 10, 25, 50, 75, 90
        - 'prob_loss_pct': float (% of paths ending below starting capital)
        - 'median_max_drawdown_pct': float (negative percentage, e.g., -32.4)
        - 'worst_10_max_drawdown_pct': float (10th percentile of max drawdowns, severe tail)
        - 'mean_max_drawdown_pct': float
        - 'median_recovery_months': float | None (None if <50% of paths recover within horizon)
        - 'recovery_rate_pct': float (% of paths that regain starting price within horizon)
        - 'shares': float
        - 'value_paths_with_start': np.ndarray of shape (days + 1, simulations)
    """
    paths = np.asarray(price_paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 1:
        raise ValueError("price_paths must be a non-empty 2D array of shape (days, simulations).")
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price must be a positive finite number.")
    if not math.isfinite(initial_investment) or initial_investment <= 0:
        raise ValueError("initial_investment must be a positive finite number.")

    days, simulations = paths.shape
    shares = initial_investment / current_price

    # Prepend Day 0 so percentile bands and drawdown math anchor cleanly at today's price
    day_zero_prices = np.full((1, simulations), current_price, dtype=np.float64)
    full_price_paths = np.vstack([day_zero_prices, paths])
    full_value_paths = full_price_paths * shares

    percentile_levels = (10, 25, 50, 75, 90)
    raw_price_pcts = np.percentile(full_price_paths, percentile_levels, axis=1)
    raw_value_pcts = raw_price_pcts * shares

    price_percentiles = {
        level: raw_price_pcts[idx] for idx, level in enumerate(percentile_levels)
    }
    value_percentiles = {
        level: raw_value_pcts[idx] for idx, level in enumerate(percentile_levels)
    }
    ending_prices = {level: float(price_percentiles[level][-1]) for level in percentile_levels}
    ending_values = {level: float(value_percentiles[level][-1]) for level in percentile_levels}

    # Probability of ending below starting capital
    prob_loss_pct = float(np.mean(paths[-1, :] < current_price) * 100.0)

    # Maximum drawdown across every path (0% = no drop, -50% = cut in half from peak)
    running_peaks = np.maximum.accumulate(full_price_paths, axis=0)
    drawdowns = (full_price_paths - running_peaks) / running_peaks
    path_max_drawdowns = np.min(drawdowns, axis=0)

    median_max_drawdown_pct = float(np.median(path_max_drawdowns) * 100.0)
    worst_10_max_drawdown_pct = float(np.percentile(path_max_drawdowns, 10) * 100.0)
    mean_max_drawdown_pct = float(np.mean(path_max_drawdowns) * 100.0)

    # Median recovery time (in months) to regain pre-crash price levels after each path's trough
    trough_idx = np.argmin(paths, axis=0)  # 0-indexed trading day of lowest price
    day_indices = np.arange(days)[:, None]
    after_trough = day_indices > trough_idx[None, :]
    recovered_mask = after_trough & (paths >= current_price)

    has_recovered = np.any(recovered_mask, axis=0)
    first_recovery_day = np.argmax(recovered_mask, axis=0) + 1  # 1-indexed trading day
    never_dipped = np.min(paths, axis=0) >= current_price

    recovery_days = np.where(
        never_dipped,
        0.0,
        np.where(has_recovered, first_recovery_day.astype(np.float64), np.inf),
    )
    trading_days_per_month = trading_days_per_year / 12.0
    recovery_months = recovery_days / trading_days_per_month

    median_rec = float(np.median(recovery_months))
    median_recovery_months: float | None = median_rec if math.isfinite(median_rec) else None
    recovery_rate_pct = float(np.mean(np.isfinite(recovery_months)) * 100.0)

    return {
        "price_percentiles": price_percentiles,
        "value_percentiles": value_percentiles,
        "ending_prices": ending_prices,
        "ending_values": ending_values,
        "prob_loss_pct": prob_loss_pct,
        "median_max_drawdown_pct": median_max_drawdown_pct,
        "worst_10_max_drawdown_pct": worst_10_max_drawdown_pct,
        "mean_max_drawdown_pct": mean_max_drawdown_pct,
        "median_recovery_months": median_recovery_months,
        "recovery_rate_pct": recovery_rate_pct,
        "shares": shares,
        "value_paths_with_start": full_value_paths,
    }


# ---------------------------------------------------------
# 4. STANDALONE SELF-TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    TEST_TICKER = "AAPL"
    HORIZON_YEARS = 3
    INITIAL_INVESTMENT = 10_000.0
    SIMULATIONS = 2_000

    print(f"Running standalone self-test for {TEST_TICKER} ({HORIZON_YEARS}y, {SIMULATIONS:,} paths)...")
    try:
        hist_df, trailing_pe = fetch_stock_history(TEST_TICKER)
    except StockDataError as err:
        print(f"Data fetch failed: {err}")
        raise SystemExit(1)

    stats = extract_return_stats(hist_df)
    pe_display = f"{trailing_pe:.2f}" if isinstance(trailing_pe, (int, float)) else trailing_pe
    print(
        f"Loaded {stats['num_trading_days']:,} trading days ({stats['years_of_history']:.1f} yrs) | "
        f"Current Price: ${stats['current_price']:,.2f} | "
        f"Annualized Vol: {stats['annualized_volatility'] * 100:.1f}% | "
        f"Trailing P/E: {pe_display}"
    )

    rng = np.random.default_rng(42)
    for engine in AVAILABLE_ENGINES:
        sim_paths = run_simulation(
            engine_name=engine,
            current_price=stats["current_price"],
            daily_returns=stats["daily_returns"],
            horizon_years=HORIZON_YEARS,
            simulations=SIMULATIONS,
            rng=rng,
        )
        summary = compute_summary_metrics(
            price_paths=sim_paths,
            current_price=stats["current_price"],
            initial_investment=INITIAL_INVESTMENT,
        )
        rec_str = (
            f"{summary['median_recovery_months']:.1f} mo"
            if summary["median_recovery_months"] is not None
            else f"> {HORIZON_YEARS * 12} mo ({summary['recovery_rate_pct']:.1f}% recovered)"
        )
        print(
            f"\n[{engine}]\n"
            f"  10th (Unlucky): ${summary['ending_values'][10]:>10,.2f} (${summary['ending_prices'][10]:,.2f}/sh)\n"
            f"  50th (Median):  ${summary['ending_values'][50]:>10,.2f} (${summary['ending_prices'][50]:,.2f}/sh)\n"
            f"  90th (Lucky):   ${summary['ending_values'][90]:>10,.2f} (${summary['ending_prices'][90]:,.2f}/sh)\n"
            f"  Prob of Loss:   {summary['prob_loss_pct']:.1f}% | "
            f"Median Max Drawdown: {summary['median_max_drawdown_pct']:.1f}% | "
            f"Median Recovery: {rec_str}"
        )