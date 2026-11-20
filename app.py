"""
Streamlit web frontend for the Stock Future Simulator.

Designed for responsive desktop and mobile browsing. All quantitative math and
Yahoo Finance data loading live in simulator.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulator import (
    AVAILABLE_ENGINES,
    DEFAULT_BLOCK_SIZE,
    ENGINE_2008_STRESS,
    StockDataError,
    compute_summary_metrics,
    extract_return_stats,
    fetch_stock_history,
    normalize_ticker,
    run_simulation,
)

# ---------------------------------------------------------
# 1. PAGE CONFIG & CACHED DATA LAYER
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Future Simulator",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_stock_data(ticker: str) -> tuple[pd.DataFrame, float | str]:
    """Caches Yahoo Finance history for 1 hour per normalized ticker symbol."""
    return fetch_stock_history(ticker)


# ---------------------------------------------------------
# 2. PLOTLY FAN CHART BUILDER
# ---------------------------------------------------------
def build_fan_chart(
    ticker: str,
    summary: dict,
    initial_investment: float,
    horizon_years: int,
    engine_name: str,
    num_sample_paths: int = 15,
) -> go.Figure:
    """Builds an interactive Plotly fan chart with confidence bands and sample paths."""
    val_pcts = summary["value_percentiles"]
    value_paths = summary["value_paths_with_start"]
    num_points = len(val_pcts[50])

    # Generate future business-day dates anchored to today
    start_date = pd.Timestamp.today().normalize()
    x_dates = pd.bdate_range(start=start_date, periods=num_points)

    fig = go.Figure()

    # 15 faint individual sample paths for visual intuition
    total_paths = value_paths.shape[1]
    sample_count = min(num_sample_paths, total_paths)
    sample_indices = np.linspace(0, total_paths - 1, sample_count, dtype=int)
    for idx in sample_indices:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=value_paths[:, idx],
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.22)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 10th-90th percentile outer confidence band
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[90],
            mode="lines",
            line=dict(color="rgba(37, 99, 235, 0.35)", width=1),
            name="90th Percentile (Lucky)",
            hovertemplate="90th (Lucky): $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[10],
            mode="lines",
            line=dict(color="rgba(37, 99, 235, 0.35)", width=1),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.12)",
            name="10th-90th Range",
            hovertemplate="10th (Unlucky): $%{y:,.2f}<extra></extra>",
        )
    )

    # 25th-75th percentile interquartile band
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[75],
            mode="lines",
            line=dict(color="rgba(37, 99, 235, 0.0)", width=0),
            name="75th Percentile",
            hovertemplate="75th Percentile: $%{y:,.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[25],
            mode="lines",
            line=dict(color="rgba(37, 99, 235, 0.0)", width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.24)",
            name="25th-75th Range",
            hovertemplate="25th Percentile: $%{y:,.2f}<extra></extra>",
        )
    )

    # Bold Median (50th percentile) trajectory line
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[50],
            mode="lines",
            line=dict(color="#1D4ED8", width=3),
            name="50th Percentile (Median)",
            hovertemplate="Median: $%{y:,.2f}<extra></extra>",
        )
    )

    # Starting capital reference line
    fig.add_hline(
        y=initial_investment,
        line_dash="dot",
        line_color="rgba(100, 116, 139, 0.75)",
        annotation_text=f"Starting Capital (${initial_investment:,.0f})",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=f"{ticker} — {horizon_years}-Year Projected Portfolio Value ({engine_name})",
            font=dict(size=18),
        ),
        xaxis_title="Projected Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=16, r=16, t=72, b=24),
        height=520,
    )

    return fig


# ---------------------------------------------------------
# 3. MAIN APPLICATION LAYOUT
# ---------------------------------------------------------
def main() -> None:
    st.title("Stock Future Simulator")
    st.caption(
        "Explore potential multi-year stock outcomes using heavy-tailed Monte Carlo, "
        "historical block bootstrapping, and 2008-style crisis stress testing."
    )

    # Top control bar
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.2, 1.4, 2.2, 1.5, 1.3])
    with ctrl1:
        raw_ticker = st.text_input("Stock Ticker", value="AAPL", max_chars=12)
    with ctrl2:
        initial_investment = st.number_input(
            "Initial Investment ($)",
            min_value=100.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=500.0,
            format="%.2f",
        )
    with ctrl3:
        engine_name = st.selectbox(
            "Simulation Engine",
            options=list(AVAILABLE_ENGINES),
            index=0,
        )
    with ctrl4:
        horizon_years = st.slider(
            "Time Horizon (Years)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )
    with ctrl5:
        simulations = st.selectbox(
            "Simulation Paths",
            options=[1_000, 2_500, 5_000],
            index=1,
            format_func=lambda n: f"{n:,}",
        )

    # Validate ticker input
    try:
        ticker = normalize_ticker(raw_ticker)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    # Load and validate historical data
    with st.spinner(f"Loading 15-year price history for {ticker}..."):
        try:
            hist, pe_ratio = load_cached_stock_data(ticker)
            stats = extract_return_stats(hist)
        except StockDataError as err:
            st.error(str(err))
            st.stop()
        except ValueError as err:
            st.warning(str(err))
            st.stop()

    daily_returns = stats["daily_returns"]
    current_price = stats["current_price"]

    # Guard against insufficient return history for block-based engines
    if engine_name != AVAILABLE_ENGINES[0] and len(daily_returns) < DEFAULT_BLOCK_SIZE + 1:
        st.warning(
            f"{ticker} only has {len(daily_returns)} trading days of history, which is too short "
            f"for {DEFAULT_BLOCK_SIZE}-day block resampling. Try another ticker or switch to "
            f"{AVAILABLE_ENGINES[0]}."
        )
        st.stop()

    # 4-column live snapshot bar
    st.subheader(f"Live Snapshot — {ticker}")
    snap1, snap2, snap3, snap4 = st.columns(4)
    pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
    snap1.metric("Current Price", f"${current_price:,.2f}")
    snap2.metric("Annualized Volatility", f"{stats['annualized_volatility'] * 100:.1f}%")
    snap3.metric("Trailing P/E Ratio", pe_display)
    snap4.metric("Years of History Loaded", f"{stats['years_of_history']:.1f} yrs")

    # Run vectorized simulation and compute summary analytics
    try:
        price_paths = run_simulation(
            engine_name=engine_name,
            current_price=current_price,
            daily_returns=daily_returns,
            horizon_years=horizon_years,
            simulations=simulations,
        )
        summary = compute_summary_metrics(
            price_paths=price_paths,
            current_price=current_price,
            initial_investment=float(initial_investment),
        )
    except ValueError as err:
        st.error(f"Simulation could not run: {err}")
        st.stop()

    # Projected ending outcomes
    st.subheader(f"Projected {horizon_years}-Year Outcomes ({simulations:,} Simulated Paths)")

    end_vals = summary["ending_values"]
    end_prices = summary["ending_prices"]
    horizon_months = horizon_years * 12

    if summary["median_recovery_months"] is not None:
        recovery_display = f"{summary['median_recovery_months']:.1f} mo"
        recovery_help = (
            f"Median time to regain starting capital after a path's trough "
            f"({summary['recovery_rate_pct']:.0f}% of paths recovered within {horizon_months} months)."
        )
    else:
        recovery_display = f"> {horizon_months} mo"
        recovery_help = (
            f"Fewer than half of simulated paths ({summary['recovery_rate_pct']:.0f}%) "
            f"regained starting capital within {horizon_months} months."
        )

    if engine_name == ENGINE_2008_STRESS:
        out1, out2, out3, out4, out5 = st.columns(5)
    else:
        out1, out2, out3, out4 = st.columns(4)

    out1.metric(
        "10th Percentile (Unlucky)",
        f"${end_vals[10]:,.2f}",
        delta=f"${end_prices[10]:,.2f}/sh",
        delta_color="off",
    )
    out2.metric(
        "50th Percentile (Median)",
        f"${end_vals[50]:,.2f}",
        delta=f"${end_prices[50]:,.2f}/sh",
        delta_color="off",
    )
    out3.metric(
        "90th Percentile (Lucky)",
        f"${end_vals[90]:,.2f}",
        delta=f"${end_prices[90]:,.2f}/sh",
        delta_color="off",
    )
    out4.metric(
        "Probability of Loss",
        f"{summary['prob_loss_pct']:.1f}%",
        delta=f"Median Max DD: {summary['median_max_drawdown_pct']:.1f}%",
        delta_color="off",
    )

    if engine_name == ENGINE_2008_STRESS:
        out5.metric(
            "Median Recovery Time",
            recovery_display,
            delta=f"{summary['recovery_rate_pct']:.0f}% recovered",
            delta_color="off",
            help=recovery_help,
        )
        st.info(
            f"**2008 Crisis Stress Test Active:** Injects a -45% Year 1 drawdown onto "
            f"historical 20-day return blocks. Estimated median time to recover pre-crash "
            f"value: **{recovery_display}** ({summary['recovery_rate_pct']:.1f}% of paths "
            f"recovered within {horizon_months} months; 10th-percentile worst drawdown: "
            f"{summary['worst_10_max_drawdown_pct']:.1f}%)."
        )

    # Interactive Plotly fan chart
    fig = build_fan_chart(
        ticker=ticker,
        summary=summary,
        initial_investment=float(initial_investment),
        horizon_years=horizon_years,
        engine_name=engine_name,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Financial disclaimer
    st.caption(
        "**Disclaimer:** This simulator is for educational and illustrative purposes only and "
        "does not constitute financial or investment advice. Historical price behavior, "
        "bootstrapped return sequences, and statistical stress tests do not guarantee future "
        "market performance."
    )


if __name__ == "__main__":
    main()