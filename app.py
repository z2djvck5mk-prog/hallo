"""
Streamlit V2.0 web frontend for the Stock Future Simulator.

Designed as a sleek, mobile-first fintech portfolio simulator using
streamlit-shadcn-ui cards/tabs and touch-safe Plotly charts. All quantitative
simulation math, optimization, and Yahoo Finance data loading live in simulator.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui

from simulator import (
    AVAILABLE_ENGINES,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_INFLATION_RATE,
    DEFAULT_RISK_FREE_RATE,
    ENGINE_2008_STRESS,
    MAX_PORTFOLIO_ASSETS,
    StockDataError,
    TRADING_DAYS_PER_YEAR,
    compute_efficient_frontier,
    fetch_news_sentiment,
    fetch_portfolio_data,
    normalize_portfolio_weights,
    run_portfolio_simulation,
    simulate_spy_benchmark,
)

# ---------------------------------------------------------
# 1. PAGE CONFIG, CONSTANTS & MOBILE CSS POLISH
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Future Simulator V2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enforced on EVERY Plotly chart so mobile touch-scrubbing never zooms or locks scroll
MOBILE_PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "scrollZoom": False,
}

DEFAULT_SLOTS: list[tuple[str, float]] = [
    ("AAPL", 40.0),
    ("MSFT", 35.0),
    ("NVDA", 25.0),
    ("GOOGL", 15.0),
    ("AMZN", 15.0),
    ("SPY", 15.0),
]

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }
        .sentiment-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.01em;
        }
        .sentiment-bullish {
            background-color: rgba(16, 185, 129, 0.14);
            color: #047857;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .sentiment-neutral {
            background-color: rgba(100, 116, 139, 0.14);
            color: #334155;
            border: 1px solid rgba(100, 116, 139, 0.35);
        }
        .sentiment-bearish {
            background-color: rgba(239, 68, 68, 0.14);
            color: #B91C1C;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .headline-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
            background: rgba(248, 250, 252, 0.65);
        }
        .headline-title {
            font-weight: 600;
            font-size: 0.94rem;
            line-height: 1.35;
            margin-bottom: 0.3rem;
        }
        .headline-title a {
            text-decoration: none;
            color: inherit;
        }
        .headline-title a:hover {
            text-decoration: underline;
        }
        .headline-meta {
            font-size: 0.78rem;
            color: #64748B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. SESSION STATE INITIALIZATION & CALLBACKS
# ---------------------------------------------------------
def _init_session_state() -> None:
    if "num_assets" not in st.session_state:
        st.session_state["num_assets"] = 3
    for idx, (sym, wt) in enumerate(DEFAULT_SLOTS):
        st.session_state.setdefault(f"ticker_{idx}", sym)
        st.session_state.setdefault(f"weight_{idx}", wt)


def _apply_preset(preset_pairs: list[tuple[str, float]]) -> None:
    st.session_state["num_assets"] = len(preset_pairs)
    for idx, (sym, wt) in enumerate(preset_pairs):
        st.session_state[f"ticker_{idx}"] = sym
        st.session_state[f"weight_{idx}"] = float(wt)


def _apply_optimized_weights(optimized_weights: dict[str, float]) -> None:
    """Callback that applies PyPortfolioOpt weights back into the portfolio editor."""
    pairs = [(sym, round(val * 100.0, 2)) for sym, val in optimized_weights.items()]
    st.session_state["num_assets"] = len(pairs)
    for idx, (sym, pct) in enumerate(pairs):
        st.session_state[f"ticker_{idx}"] = sym
        st.session_state[f"weight_{idx}"] = float(pct)


# ---------------------------------------------------------
# 3. CACHED DATA ACCESS LAYER
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_base_portfolio(tickers_tuple: tuple[str, ...]) -> dict[str, Any]:
    """Caches aligned Yahoo Finance price history for 1 hour per ticker set."""
    return fetch_portfolio_data(list(tickers_tuple), raw_weights=None)


def apply_weights_to_cached_portfolio(
    base_data: dict[str, Any],
    norm_weights: np.ndarray,
) -> dict[str, Any]:
    """Recomputes weight-dependent portfolio stats instantaneously without re-fetching."""
    tickers = base_data["tickers"]
    returns_matrix = base_data["returns_matrix"]
    cov_matrix = base_data["cov_matrix"]
    trailing_pes = base_data["trailing_pes"]

    port_daily_returns = returns_matrix @ norm_weights
    port_var = float(norm_weights @ cov_matrix @ norm_weights)
    ann_vol = float(np.sqrt(max(port_var, 0.0) * TRADING_DAYS_PER_YEAR))
    hist_ann_ret = float(np.mean(port_daily_returns) * TRADING_DAYS_PER_YEAR)

    # Weighted harmonic P/E
    valid_w = 0.0
    earnings_yield = 0.0
    for sym, w in zip(tickers, norm_weights):
        pe = trailing_pes.get(sym)
        if pe is not None and np.isfinite(pe) and pe > 0 and w > 0:
            valid_w += float(w)
            earnings_yield += float(w) / float(pe)
    weighted_pe = float(valid_w / earnings_yield) if valid_w > 0 and earnings_yield > 0 else None

    updated = dict(base_data)
    updated["weights"] = norm_weights
    updated["portfolio_daily_returns"] = port_daily_returns
    updated["annualized_volatility"] = ann_vol
    updated["historical_annual_return"] = hist_ann_ret
    updated["weighted_pe"] = weighted_pe
    return updated


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_spy_data() -> dict[str, Any] | None:
    """Caches SPY benchmark price history defensively; returns None if unavailable."""
    try:
        return fetch_portfolio_data(["SPY"], [1.0])
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_news_sentiment(tickers_tuple: tuple[str, ...]) -> dict[str, Any]:
    """Caches live news sentiment scores for 30 minutes."""
    return fetch_news_sentiment(list(tickers_tuple), max_headlines=3)


@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_efficient_frontier(
    prices_df: pd.DataFrame,
    weights_tuple: tuple[float, ...],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, Any]:
    """Caches Efficient Frontier optimization for the active price matrix & weights."""
    return compute_efficient_frontier(
        prices_df=prices_df,
        current_weights=np.array(weights_tuple, dtype=np.float64),
        risk_free_rate=risk_free_rate,
    )


# ---------------------------------------------------------
# 4. MOBILE-SAFE PLOTLY CHART BUILDERS
# ---------------------------------------------------------
def _apply_mobile_touch_layout(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 470,
) -> go.Figure:
    """Enforces smooth mobile touch-scrubbing (no accidental drag/zoom) on every chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.0, xanchor="left"),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(fixedrange=True, showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)"),
        yaxis=dict(fixedrange=True, showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
        ),
        margin=dict(l=12, r=12, t=78, b=24),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_fan_chart(
    portfolio_label: str,
    summary: dict[str, Any],
    horizon_years: int,
    engine_name: str,
    spy_benchmark: dict[str, Any] | None = None,
    num_sample_paths: int = 15,
) -> go.Figure:
    """Builds the interactive portfolio simulation fan chart with optional SPY overlay."""
    val_pcts = summary["value_percentiles"]
    value_paths = summary["value_paths_with_start"]
    cum_contrib = summary["cumulative_contributions"]
    is_real = summary["inflation_adjusted"]
    num_points = len(val_pcts[50])

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
                line=dict(color="rgba(148, 163, 184, 0.18)", width=1),
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
            name="90th (Lucky)",
            hovertemplate="90th (Lucky): $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[10],
            mode="lines",
            line=dict(color="rgba(37, 99, 235, 0.35)", width=1),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.11)",
            name="10th-90th Band",
            hovertemplate="10th (Unlucky): $%{y:,.0f}<extra></extra>",
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
            hovertemplate="75th Pct: $%{y:,.0f}<extra></extra>",
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
            fillcolor="rgba(37, 99, 235, 0.22)",
            name="25th-75th Band",
            hovertemplate="25th Pct: $%{y:,.0f}<extra></extra>",
        )
    )

    # Bold 50th Percentile (Median) Portfolio line
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=val_pcts[50],
            mode="lines",
            line=dict(color="#1D4ED8", width=3),
            name="Portfolio Median",
            hovertemplate="Portfolio Median: $%{y:,.0f}<extra></extra>",
        )
    )

    # Optional S&P 500 (SPY) Median Benchmark overlay
    if spy_benchmark is not None and len(spy_benchmark["median_path"]) == num_points:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=spy_benchmark["median_path"],
                mode="lines",
                line=dict(color="#D97706", width=2.4, dash="dash"),
                name="S&P 500 (SPY) Median",
                hovertemplate="SPY Median: $%{y:,.0f}<extra></extra>",
            )
        )

    # Cumulative Capital Contributed reference trajectory
    basis_label = "Real Capital Contributed" if is_real else "Capital Contributed"
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=cum_contrib,
            mode="lines",
            line=dict(color="rgba(100, 116, 139, 0.85)", width=1.6, dash="dot"),
            name=basis_label,
            hovertemplate=f"{basis_label}: $%{{y:,.0f}}<extra></extra>",
        )
    )

    dollar_axis = "Portfolio Value (Today's $)" if is_real else "Portfolio Value ($)"
    title_suffix = " · Real $" if is_real else ""
    _apply_mobile_touch_layout(
        fig,
        title=f"{portfolio_label} — {horizon_years}Y Projection ({engine_name}{title_suffix})",
        xaxis_title="Projected Date",
        yaxis_title=dollar_axis,
        height=490,
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def build_efficient_frontier_chart(ef_data: dict[str, Any]) -> go.Figure:
    """Plots the interactive PyPortfolioOpt Efficient Frontier curve and key portfolios."""
    fig = go.Figure()

    # Smooth Efficient Frontier curve
    if ef_data["frontier_vols"]:
        fig.add_trace(
            go.Scatter(
                x=[v * 100.0 for v in ef_data["frontier_vols"]],
                y=[r * 100.0 for r in ef_data["frontier_rets"]],
                mode="lines",
                line=dict(color="#2563EB", width=3),
                name="Efficient Frontier",
                customdata=ef_data["frontier_sharpes"],
                hovertemplate="Frontier Return: %{y:.1f}% (Sharpe %{customdata:.2f})<extra></extra>",
            )
        )

    # Individual constituent assets
    assets = ef_data["individual_assets"]
    fig.add_trace(
        go.Scatter(
            x=[a["volatility"] * 100.0 for a in assets],
            y=[a["expected_return"] * 100.0 for a in assets],
            mode="markers+text",
            text=[a["ticker"] for a in assets],
            textposition="top center",
            marker=dict(size=9, color="#64748B", line=dict(width=1, color="#FFFFFF")),
            name="Individual Assets",
            customdata=[a["ticker"] for a in assets],
            hovertemplate="%{customdata}: %{y:.1f}% ret<extra></extra>",
        )
    )

    # Current user portfolio
    curr = ef_data["current"]
    fig.add_trace(
        go.Scatter(
            x=[curr["volatility"] * 100.0],
            y=[curr["expected_return"] * 100.0],
            mode="markers",
            marker=dict(size=13, color="#7C3AED", symbol="circle", line=dict(width=2, color="#FFFFFF")),
            name=f"Current Portfolio (Sharpe {curr['sharpe']:.2f})",
            hovertemplate=f"Current: %{{y:.1f}}% ret (Sharpe {curr['sharpe']:.2f})<extra></extra>",
        )
    )

    # Minimum Volatility optimal portfolio
    min_v = ef_data["min_vol"]
    fig.add_trace(
        go.Scatter(
            x=[min_v["volatility"] * 100.0],
            y=[min_v["expected_return"] * 100.0],
            mode="markers",
            marker=dict(size=14, color="#059669", symbol="diamond", line=dict(width=2, color="#FFFFFF")),
            name=f"Min Volatility (Sharpe {min_v['sharpe']:.2f})",
            hovertemplate=f"Min Vol: %{{y:.1f}}% ret (Sharpe {min_v['sharpe']:.2f})<extra></extra>",
        )
    )

    # Maximum Sharpe optimal portfolio
    max_s = ef_data["max_sharpe"]
    fig.add_trace(
        go.Scatter(
            x=[max_s["volatility"] * 100.0],
            y=[max_s["expected_return"] * 100.0],
            mode="markers",
            marker=dict(size=16, color="#D97706", symbol="star", line=dict(width=1.5, color="#FFFFFF")),
            name=f"Max Sharpe (Sharpe {max_s['sharpe']:.2f})",
            hovertemplate=f"Max Sharpe: %{{y:.1f}}% ret (Sharpe {max_s['sharpe']:.2f})<extra></extra>",
        )
    )

    _apply_mobile_touch_layout(
        fig,
        title="Efficient Frontier — Risk vs. Annualized Expected Return",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Expected Return (%)",
        height=460,
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    return fig


def build_weights_comparison_chart(ef_data: dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing Current, Max-Sharpe, and Min-Volatility weights."""
    tickers = ef_data["tickers"]
    curr_w = [ef_data["current"]["weights"].get(t, 0.0) * 100.0 for t in tickers]
    max_w = [ef_data["max_sharpe"]["weights"].get(t, 0.0) * 100.0 for t in tickers]
    min_w = [ef_data["min_vol"]["weights"].get(t, 0.0) * 100.0 for t in tickers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=curr_w,
            name="Current",
            marker_color="#7C3AED",
            hovertemplate="Current: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=max_w,
            name="Max Sharpe",
            marker_color="#D97706",
            hovertemplate="Max Sharpe: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=min_w,
            name="Min Volatility",
            marker_color="#059669",
            hovertemplate="Min Vol: %{y:.1f}%<extra></extra>",
        )
    )

    _apply_mobile_touch_layout(
        fig,
        title="Portfolio Weight Comparison (%)",
        xaxis_title="Ticker",
        yaxis_title="Allocation (%)",
        height=340,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    return fig


def build_correlation_heatmap(tickers: list[str], corr_matrix: np.ndarray) -> go.Figure:
    """Clean annotated cross-asset correlation matrix with mobile-safe touch settings."""
    text_vals = [[f"{val:.2f}" for val in row] for row in corr_matrix]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=tickers,
            y=tickers,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            text=text_vals,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            showscale=True,
        )
    )
    _apply_mobile_touch_layout(
        fig,
        title="Historical Cross-Asset Return Correlations",
        xaxis_title="",
        yaxis_title="",
        height=360,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------
# 5. SENTIMENT PULSE RENDERER
# ---------------------------------------------------------
def render_news_sentiment_section(sentiment: dict[str, Any], tickers: list[str]) -> None:
    """Renders the Live News Sentiment Pulse badge and top 3 scored headlines."""
    st.subheader("Live News Sentiment Pulse")

    if not sentiment.get("available"):
        ui.alert(
            title="Live Headlines Temporarily Unavailable",
            description=(
                f"Yahoo Finance did not return recent news items for {', '.join(tickers)}. "
                "Your simulation and risk analytics remain fully unaffected."
            ),
            variant="default",
            key="news_unavailable_alert",
        )
        return

    label = sentiment["label"]
    score = sentiment["compound_score"]
    count = sentiment["headline_count"]

    if label == "Bullish":
        pill_class = "sentiment-bullish"
        arrow = "▲"
        badge_variant = "default"
    elif label == "Bearish":
        pill_class = "sentiment-bearish"
        arrow = "▼"
        badge_variant = "destructive"
    else:
        pill_class = "sentiment-neutral"
        arrow = "●"
        badge_variant = "secondary"

    pulse_col1, pulse_col2 = st.columns([1.4, 2.6])
    with pulse_col1:
        ui.badge(
            text=f"{label} {arrow}  (VADER {score:+.2f})",
            variant=badge_variant,
            key="sentiment_main_badge",
        )
    with pulse_col2:
        st.markdown(
            f'<span class="sentiment-pill {pill_class}">{arrow} {label} Pulse</span> '
            f'<span style="color:#64748B; font-size:0.85rem; margin-left:0.5rem;">'
            f"Scored across {count} recent headlines via VADER sentiment</span>",
            unsafe_allow_html=True,
        )

    st.write("")
    for idx, item in enumerate(sentiment["headlines"], start=1):
        h_sent = item["sentiment"]
        h_class = (
            "sentiment-bullish"
            if h_sent == "Bullish"
            else "sentiment-bearish"
            if h_sent == "Bearish"
            else "sentiment-neutral"
        )
        title_text = item["title"]
        link = item.get("link") or ""
        title_html = (
            f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title_text}</a>'
            if link
            else title_text
        )
        pub = item.get("publisher", "Yahoo Finance")
        pub_date = f" · {item['published']}" if item.get("published") else ""
        st.markdown(
            f"""
            <div class="headline-card">
                <div class="headline-title">{idx}. {title_html}</div>
                <div class="headline-meta">
                    <strong>{item['ticker']}</strong> · {pub}{pub_date} ·
                    <span class="sentiment-pill {h_class}" style="padding:0.1rem 0.5rem; font-size:0.74rem;">
                        {h_sent} ({item['compound']:+.2f})
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# 6. MAIN APPLICATION
# ---------------------------------------------------------
def main() -> None:
    _init_session_state()

    # Header
    hdr_left, hdr_right = st.columns([4.5, 1.5])
    with hdr_left:
        st.title("Stock Future Simulator")
        st.caption(
            "Version 2.0 · Multi-asset correlated Monte Carlo, historical block bootstrapping, "
            "2008 crisis stress testing, Efficient Frontier optimization, and live VADER sentiment."
        )
    with hdr_right:
        st.write("")
        ui.badges(
            [("V2.0 Pro", "default"), ("Mobile-Ready", "secondary")],
            key="header_version_badges",
        )

    # -----------------------------------------------------
    # PORTFOLIO BUILDER & SIMULATION CONTROLS
    # -----------------------------------------------------
    with st.container(border=True):
        top_bar1, top_bar2 = st.columns([1.8, 3.2])
        with top_bar1:
            num_assets = st.slider(
                "Assets in Portfolio (1–6)",
                min_value=1,
                max_value=MAX_PORTFOLIO_ASSETS,
                key="num_assets",
            )
        with top_bar2:
            st.caption("Quick Presets")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.button(
                    "Single Stock",
                    width="stretch",
                    on_click=_apply_preset,
                    args=([("AAPL", 100.0)],),
                )
            with p_col2:
                st.button(
                    "Tech Trio",
                    width="stretch",
                    on_click=_apply_preset,
                    args=([("AAPL", 40.0), ("MSFT", 35.0), ("NVDA", 25.0)],),
                )
            with p_col3:
                st.button(
                    "Core 4-Asset",
                    width="stretch",
                    on_click=_apply_preset,
                    args=([("SPY", 45.0), ("QQQ", 25.0), ("AAPL", 15.0), ("MSFT", 15.0)],),
                )

        # Ticker + Weight % Grid (responsive 3-up on desktop, wraps cleanly on mobile)
        raw_tickers: list[str] = []
        raw_weights: list[float] = []
        cols_per_row = min(num_assets, 3)
        for row_start in range(0, num_assets, cols_per_row):
            row_count = min(cols_per_row, num_assets - row_start)
            row_cols = st.columns(row_count)
            for offset in range(row_count):
                idx = row_start + offset
                with row_cols[offset]:
                    t_col, w_col = st.columns([1.15, 1.0])
                    with t_col:
                        sym_val = st.text_input(
                            f"Ticker {idx + 1}",
                            key=f"ticker_{idx}",
                            max_chars=12,
                        )
                    with w_col:
                        wt_val = st.number_input(
                            f"Weight %",
                            min_value=0.0,
                            max_value=1000.0,
                            step=5.0,
                            format="%.1f",
                            key=f"weight_{idx}",
                        )
                    raw_tickers.append(sym_val)
                    raw_weights.append(float(wt_val))

        # Validate & auto-normalize weights immediately
        try:
            clean_tickers, norm_weights = normalize_portfolio_weights(raw_tickers, raw_weights)
        except ValueError as err:
            st.warning(str(err))
            st.stop()

        raw_weight_sum = sum(w for w in raw_weights if w > 0)
        alloc_badges = [
            (f"{sym} {w * 100:.1f}%", "outline") for sym, w in zip(clean_tickers, norm_weights)
        ]
        if abs(raw_weight_sum - 100.0) > 0.05 and len(clean_tickers) > 1:
            alloc_badges.append(("Auto-Normalized to 100%", "secondary"))
        ui.badges(alloc_badges, key="portfolio_alloc_badges")

        ui.separator(key="controls_sep")

        # Core Simulation & Cash Flow Parameters
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.9, 1.3, 1.0])
        with c1:
            initial_investment = st.number_input(
                "Initial Investment ($)",
                min_value=100.0,
                max_value=10_000_000.0,
                value=10_000.0,
                step=500.0,
                format="%.0f",
            )
        with c2:
            monthly_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0.0,
                max_value=250_000.0,
                value=0.0,
                step=50.0,
                format="%.0f",
                help="Recurring monthly dollar-cost averaging (DCA) deposit added every 21 trading days.",
            )
        with c3:
            engine_name = st.selectbox(
                "Simulation Engine",
                options=list(AVAILABLE_ENGINES),
                index=0,
            )
        with c4:
            horizon_years = st.slider(
                "Horizon (Years)",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
            )
        with c5:
            simulations = st.selectbox(
                "Paths",
                options=[1_000, 2_500, 5_000],
                index=1,
                format_func=lambda n: f"{n:,}",
            )

        # V2.0 Analytical Toggles
        t1, t2, t3, t4 = st.columns([1.4, 1.4, 1.2, 1.0])
        with t1:
            apply_drift_dampener = st.toggle(
                "Valuation Drift Dampener",
                value=False,
                help=(
                    "Blends extreme historical annual returns toward a ~9.5% long-term "
                    "equity baseline scaled by each stock's trailing P/E ratio."
                ),
            )
        with t2:
            inflation_adjusted = st.toggle(
                "Inflation-Adjusted (Real $)",
                value=False,
                help="Deflates projected balances back into today's purchasing power.",
            )
        with t3:
            show_spy_benchmark = st.toggle(
                "SPY Benchmark Overlay",
                value=True,
                help="Overlays a simulated S&P 500 (SPY) median benchmark line on the fan chart.",
            )
        with t4:
            inflation_rate_pct = st.number_input(
                "Inflation (%)",
                min_value=0.0,
                max_value=15.0,
                value=DEFAULT_INFLATION_RATE * 100.0,
                step=0.5,
                format="%.1f",
                disabled=not inflation_adjusted,
            )

    # -----------------------------------------------------
    # LOAD & VALIDATE MARKET DATA
    # -----------------------------------------------------
    tickers_tuple = tuple(clean_tickers)
    portfolio_label = (
        clean_tickers[0]
        if len(clean_tickers) == 1
        else f"{len(clean_tickers)}-Asset Portfolio ({', '.join(clean_tickers[:3])}{'…' if len(clean_tickers) > 3 else ''})"
    )

    with st.spinner(f"Loading aligned 15-year market history for {', '.join(clean_tickers)}..."):
        try:
            base_portfolio = load_cached_base_portfolio(tickers_tuple)
            portfolio_data = apply_weights_to_cached_portfolio(base_portfolio, norm_weights)
        except StockDataError as err:
            st.error(str(err))
            st.stop()
        except ValueError as err:
            st.warning(str(err))
            st.stop()

    # Guard against short overlapping history for block engines
    if engine_name != AVAILABLE_ENGINES[0] and len(portfolio_data["portfolio_daily_returns"]) < DEFAULT_BLOCK_SIZE + 1:
        st.warning(
            f"Overlapping history across {', '.join(clean_tickers)} is only "
            f"{len(portfolio_data['portfolio_daily_returns'])} trading days, which is too short "
            f"for {DEFAULT_BLOCK_SIZE}-day block resampling. Switch to {AVAILABLE_ENGINES[0]} "
            "or remove newly listed tickers."
        )
        st.stop()

    # Run core vectorized portfolio simulation
    inflation_rate = float(inflation_rate_pct) / 100.0
    try:
        summary = run_portfolio_simulation(
            portfolio_data=portfolio_data,
            engine_name=engine_name,
            horizon_years=horizon_years,
            initial_investment=float(initial_investment),
            monthly_contribution=float(monthly_contribution),
            inflation_adjusted=bool(inflation_adjusted),
            inflation_rate=inflation_rate,
            apply_drift_dampener=bool(apply_drift_dampener),
            simulations=int(simulations),
        )
    except ValueError as err:
        st.error(f"Simulation could not run: {err}")
        st.stop()

    # Optional SPY benchmark simulation (defensive; never blocks app if SPY fails)
    spy_benchmark: dict[str, Any] | None = None
    if show_spy_benchmark:
        spy_base = load_cached_spy_data()
        if spy_base is not None:
            spy_benchmark = simulate_spy_benchmark(
                engine_name=engine_name,
                horizon_years=horizon_years,
                initial_investment=float(initial_investment),
                monthly_contribution=float(monthly_contribution),
                inflation_adjusted=bool(inflation_adjusted),
                inflation_rate=inflation_rate,
                apply_drift_dampener=bool(apply_drift_dampener),
                simulations=int(simulations),
                spy_portfolio_data=spy_base,
            )

    # -----------------------------------------------------
    # SHADCN TABS NAVIGATION
    # -----------------------------------------------------
    st.write("")
    active_tab = ui.tabs(
        options=["Simulation & Risk", "Efficient Frontier", "Portfolio Diagnostics"],
        value="Simulation & Risk",
        key="main_navigation_tabs",
        variant="default",
    )
    st.write("")

    # =====================================================
    # TAB 1: SIMULATION, RISK METRICS & NEWS SENTIMENT
    # =====================================================
    if active_tab == "Simulation & Risk":
        end_vals = summary["ending_values"]
        basis_contributed = summary["total_basis_contributed"]
        horizon_months = horizon_years * 12
        dollar_tag = "Today's $" if inflation_adjusted else "Nominal $"

        if summary["median_recovery_months"] is not None:
            recovery_display = f"{summary['median_recovery_months']:.1f} mo"
        else:
            recovery_display = f"> {horizon_months} mo"

        st.subheader(f"Projected {horizon_years}-Year Outcomes ({simulations:,} Paths · {dollar_tag})")

        if engine_name == ENGINE_2008_STRESS:
            o1, o2, o3, o4, o5 = st.columns(5)
        else:
            o1, o2, o3, o4 = st.columns(4)

        with o1:
            pnl_10 = ((end_vals[10] / max(basis_contributed, 1e-6)) - 1.0) * 100.0
            ui.metric_card(
                label="10th Pct (Unlucky)",
                value=f"${end_vals[10]:,.0f}",
                delta=f"{pnl_10:+.1f}% vs invested",
                description=f"Invested: ${basis_contributed:,.0f}",
                variant="dashboard",
                size="sm",
                key="card_out_10",
            )
        with o2:
            pnl_50 = ((end_vals[50] / max(basis_contributed, 1e-6)) - 1.0) * 100.0
            ui.metric_card(
                label="50th Pct (Median)",
                value=f"${end_vals[50]:,.0f}",
                delta=f"{pnl_50:+.1f}% ({summary['median_cagr_pct']:+.1f}% CAGR)",
                description=(
                    f"vs SPY: ${spy_benchmark['ending_median']:,.0f}"
                    if spy_benchmark is not None
                    else f"Invested: ${basis_contributed:,.0f}"
                ),
                variant="dashboard",
                size="sm",
                key="card_out_50",
            )
        with o3:
            pnl_90 = ((end_vals[90] / max(basis_contributed, 1e-6)) - 1.0) * 100.0
            ui.metric_card(
                label="90th Pct (Lucky)",
                value=f"${end_vals[90]:,.0f}",
                delta=f"{pnl_90:+.1f}% vs invested",
                description="Top decile outcome",
                variant="dashboard",
                size="sm",
                key="card_out_90",
            )
        with o4:
            ui.metric_card(
                label="Probability of Loss",
                value=f"{summary['prob_loss_pct']:.1f}%",
                delta=f"Recovery: {recovery_display}",
                description=f"Ending below ${basis_contributed:,.0f}",
                variant="dashboard",
                size="sm",
                key="card_out_loss",
            )
        if engine_name == ENGINE_2008_STRESS:
            with o5:
                ui.metric_card(
                    label="Median Recovery",
                    value=recovery_display,
                    delta=f"{summary['recovery_rate_pct']:.0f}% recovered",
                    description=f"Within {horizon_months} months",
                    variant="dashboard",
                    size="sm",
                    key="card_out_rec",
                )

        # Active Scenario Alerts
        if apply_drift_dampener:
            d_info = summary["drift_info"]
            pe_str = (
                f"{portfolio_data['weighted_pe']:.1f}x"
                if portfolio_data["weighted_pe"] is not None
                else "N/A (9.5% baseline)"
            )
            ui.alert(
                title="Valuation / Mean-Reversion Drift Dampener Active",
                description=(
                    f"Blended raw historical annual drift ({d_info['raw_portfolio_annual_drift'] * 100:.1f}%) "
                    f"toward valuation-scaled target ({d_info['valuation_target_portfolio_drift'] * 100:.1f}%, "
                    f"Portfolio P/E {pe_str}) -> Simulated drift: "
                    f"{d_info['dampened_portfolio_annual_drift'] * 100:.1f}%/yr."
                ),
                variant="default",
                key="drift_dampener_alert",
            )

        if engine_name == ENGINE_2008_STRESS:
            ui.alert(
                title="2008 Crisis Stress Test Active",
                description=(
                    f"Injects a -45% Year 1 drawdown across synchronized 20-day return blocks. "
                    f"Estimated median recovery: {recovery_display} "
                    f"({summary['recovery_rate_pct']:.1f}% recovered; worst 10th-pct drawdown: "
                    f"{summary['worst_10_max_drawdown_pct']:.1f}%)."
                ),
                variant="destructive",
                key="stress_test_alert",
            )

        if summary.get("used_psd_fallback"):
            st.caption(
                "Note: Historical covariance matrix was near-singular; Cholesky decomposition "
                "automatically applied the nearest positive-semidefinite (PSD) projection."
            )

        # Mobile-Safe Interactive Fan Chart
        fan_fig = build_fan_chart(
            portfolio_label=portfolio_label,
            summary=summary,
            horizon_years=horizon_years,
            engine_name=engine_name,
            spy_benchmark=spy_benchmark,
        )
        st.plotly_chart(fan_fig, width="stretch", config=MOBILE_PLOTLY_CONFIG)

        # Institutional Risk Metrics Row (4 Shadcn Metric Cards)
        st.subheader("Institutional Risk Metrics")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            spy_sharpe_note = (
                f"SPY Sharpe: {spy_benchmark['sharpe_ratio']:.2f}"
                if spy_benchmark is not None
                else f"Rf = {DEFAULT_RISK_FREE_RATE * 100:.1f}%"
            )
            ui.metric_card(
                label="Annualized Sharpe Ratio",
                value=f"{summary['sharpe_ratio']:.2f}",
                delta=spy_sharpe_note,
                description="Return per unit of total risk",
                variant="dashboard",
                size="sm",
                key="risk_card_sharpe",
            )
        with r2:
            spy_sortino_note = (
                f"SPY Sortino: {spy_benchmark['sortino_ratio']:.2f}"
                if spy_benchmark is not None
                else "Downside semi-deviation"
            )
            ui.metric_card(
                label="Annualized Sortino Ratio",
                value=f"{summary['sortino_ratio']:.2f}",
                delta=spy_sortino_note,
                description="Return per unit of downside risk",
                variant="dashboard",
                size="sm",
                key="risk_card_sortino",
            )
        with r3:
            ui.metric_card(
                label="Median Max Drawdown",
                value=f"{summary['median_max_drawdown_pct']:.1f}%",
                delta=f"Worst 10%: {summary['worst_10_max_drawdown_pct']:.1f}%",
                description="Peak-to-trough market decline",
                variant="dashboard",
                size="sm",
                key="risk_card_mdd",
            )
        with r4:
            ui.metric_card(
                label="95% Value at Risk (1-Yr)",
                value=f"{summary['var_95_annual_pct']:.1f}%",
                delta=f"Daily 95% VaR: {summary['var_95_daily_pct']:.2f}%",
                description=f"5th-pct ending: ${summary['var_95_ending_value']:,.0f}",
                variant="dashboard",
                size="sm",
                key="risk_card_var",
            )

        ui.separator(key="sentiment_sep")

        # Live News Sentiment Pulse
        with st.spinner("Scanning recent Yahoo Finance headlines..."):
            sentiment_data = load_cached_news_sentiment(tickers_tuple)
        render_news_sentiment_section(sentiment_data, clean_tickers)

    # =====================================================
    # TAB 2: EFFICIENT FRONTIER & MAX-SHARPE OPTIMIZER
    # =====================================================
    elif active_tab == "Efficient Frontier":
        st.subheader("Efficient Frontier & Portfolio Optimizer (PyPortfolioOpt)")

        with st.spinner("Solving Efficient Frontier & optimal allocations..."):
            ef_data = load_cached_efficient_frontier(
                prices_df=portfolio_data["prices_df"],
                weights_tuple=tuple(float(x) for x in norm_weights),
                risk_free_rate=DEFAULT_RISK_FREE_RATE,
            )

        if not ef_data.get("available"):
            ui.alert(
                title="Multi-Asset Portfolio Required",
                description=ef_data.get(
                    "message",
                    "Add at least 2 tickers above to compute the Efficient Frontier.",
                ),
                variant="default",
                key="ef_unavailable_alert",
            )
        else:
            if ef_data.get("note"):
                ui.alert(
                    title="Optimizer Note",
                    description=ef_data["note"],
                    variant="default",
                    key="ef_solver_note",
                )

            curr = ef_data["current"]
            max_s = ef_data["max_sharpe"]
            min_v = ef_data["min_vol"]

            # Comparison Cards
            ef_c1, ef_c2, ef_c3 = st.columns(3)
            with ef_c1:
                ui.metric_card(
                    label="Current Portfolio",
                    value=f"Sharpe {curr['sharpe']:.2f}",
                    delta=f"Ret {curr['expected_return'] * 100:.1f}% · Vol {curr['volatility'] * 100:.1f}%",
                    description="Your active allocation",
                    variant="dashboard",
                    size="sm",
                    key="ef_card_current",
                )
            with ef_c2:
                ui.metric_card(
                    label="Max-Sharpe Optimal",
                    value=f"Sharpe {max_s['sharpe']:.2f}",
                    delta=f"Ret {max_s['expected_return'] * 100:.1f}% · Vol {max_s['volatility'] * 100:.1f}%",
                    description="Highest risk-adjusted return",
                    variant="dashboard",
                    size="sm",
                    key="ef_card_max_sharpe",
                )
            with ef_c3:
                ui.metric_card(
                    label="Min-Volatility Optimal",
                    value=f"Vol {min_v['volatility'] * 100:.1f}%",
                    delta=f"Ret {min_v['expected_return'] * 100:.1f}% · Sharpe {min_v['sharpe']:.2f}",
                    description="Lowest historical variance",
                    variant="dashboard",
                    size="sm",
                    key="ef_card_min_vol",
                )

            # One-Click Weight Application Buttons
            btn1, btn2, _ = st.columns([1.4, 1.4, 2.2])
            with btn1:
                st.button(
                    "Apply Max-Sharpe Weights",
                    type="primary",
                    width="stretch",
                    on_click=_apply_optimized_weights,
                    args=(max_s["weights"],),
                )
            with btn2:
                st.button(
                    "Apply Min-Volatility Weights",
                    width="stretch",
                    on_click=_apply_optimized_weights,
                    args=(min_v["weights"],),
                )

            # Interactive Efficient Frontier Chart (Mobile-Safe)
            ef_fig = build_efficient_frontier_chart(ef_data)
            st.plotly_chart(ef_fig, width="stretch", config=MOBILE_PLOTLY_CONFIG)

            # Weights Comparison Bar Chart & Table
            w_chart_col, w_table_col = st.columns([1.4, 1.0])
            with w_chart_col:
                w_fig = build_weights_comparison_chart(ef_data)
                st.plotly_chart(w_fig, width="stretch", config=MOBILE_PLOTLY_CONFIG)
            with w_table_col:
                st.markdown("**Optimal Allocation Breakdown (%)**")
                rows = []
                for sym in ef_data["tickers"]:
                    rows.append(
                        {
                            "Ticker": sym,
                            "Current": f"{curr['weights'].get(sym, 0.0) * 100:.1f}%",
                            "Max Sharpe": f"{max_s['weights'].get(sym, 0.0) * 100:.1f}%",
                            "Min Vol": f"{min_v['weights'].get(sym, 0.0) * 100:.1f}%",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # =====================================================
    # TAB 3: PORTFOLIO DIAGNOSTICS & CORRELATIONS
    # =====================================================
    else:
        st.subheader("Portfolio Snapshot & Quantitative Diagnostics")

        w_pe_display = (
            f"{portfolio_data['weighted_pe']:.2f}x"
            if portfolio_data["weighted_pe"] is not None
            else "N/A"
        )
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            ui.metric_card(
                label="Annualized Volatility",
                value=f"{portfolio_data['annualized_volatility'] * 100:.1f}%",
                description="Covariance-adjusted portfolio σ",
                variant="dashboard",
                size="sm",
                key="diag_card_vol",
            )
        with d2:
            ui.metric_card(
                label="Weighted Portfolio P/E",
                value=w_pe_display,
                description="Harmonic mean trailing P/E",
                variant="dashboard",
                size="sm",
                key="diag_card_pe",
            )
        with d3:
            ui.metric_card(
                label="Historical Annual Drift",
                value=f"{portfolio_data['historical_annual_return'] * 100:.1f}%",
                delta=f"Dampened: {summary['drift_info']['dampened_portfolio_annual_drift'] * 100:.1f}%",
                description="Annualized mean daily return",
                variant="dashboard",
                size="sm",
                key="diag_card_drift",
            )
        with d4:
            ui.metric_card(
                label="Aligned History Loaded",
                value=f"{portfolio_data['years_of_history']:.1f} yrs",
                delta=f"{portfolio_data['num_trading_days']:,} trading days",
                description="Synchronized across all tickers",
                variant="dashboard",
                size="sm",
                key="diag_card_hist",
            )

        st.markdown("**Asset-Level Breakdown & Valuation Dampener Targets**")
        asset_rows = []
        per_asset_drift = {item["ticker"]: item for item in summary["drift_info"]["per_asset"]}
        for sym, w in zip(clean_tickers, norm_weights):
            d_item = per_asset_drift[sym]
            pe_val = portfolio_data["trailing_pes"].get(sym)
            asset_rows.append(
                {
                    "Ticker": sym,
                    "Weight": f"{w * 100:.1f}%",
                    "Last Price": f"${portfolio_data['current_prices'][sym]:,.2f}",
                    "Ann. Volatility": f"{portfolio_data['asset_vols'][sym] * 100:.1f}%",
                    "Trailing P/E": f"{pe_val:.1f}x" if pe_val is not None else "N/A",
                    "Raw Hist. Drift": f"{d_item['raw_annual_drift'] * 100:.1f}%",
                    "Valuation Target": f"{d_item['valuation_target_drift'] * 100:.1f}%",
                    "Dampened Drift": f"{d_item['dampened_annual_drift'] * 100:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(asset_rows), width="stretch", hide_index=True)

        if len(clean_tickers) >= 2:
            corr_fig = build_correlation_heatmap(clean_tickers, portfolio_data["corr_matrix"])
            st.plotly_chart(corr_fig, width="stretch", config=MOBILE_PLOTLY_CONFIG)
        else:
            st.info("Add 2 or more tickers to inspect the cross-asset correlation matrix.")

    # Financial Disclaimer
    ui.separator(key="footer_sep")
    st.caption(
        "**Disclaimer:** Stock Future Simulator V2.0 is for educational and illustrative purposes "
        "only and does not constitute financial, tax, or investment advice. Simulated trajectories, "
        "historical covariances, Efficient Frontier allocations, and news sentiment scores do not "
        "guarantee future market results."
    )


if __name__ == "__main__":
    main()
