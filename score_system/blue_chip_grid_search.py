from __future__ import annotations

import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import TradePlanBacktester
from blue_chip_range_reversion import BlueChipRangeReversionResearcher, RangeStrategyConfig


def expand_param_grid(param_grid: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand a dict-of-lists grid into a list of parameter dictionaries."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    value_lists = [list(param_grid[key]) for key in keys]
    if any(len(values) == 0 for values in value_lists):
        raise ValueError("Every parameter in param_grid must contain at least one candidate value.")

    combinations: list[dict[str, Any]] = []
    for values in product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def format_param_label(params: Mapping[str, Any], *, max_items: int | None = None) -> str:
    """Create a compact human-readable label for one parameter combination."""
    items = list(params.items())
    if max_items is not None:
        items = items[:max_items]
    if not items:
        return "base"
    return ", ".join(f"{key}={value}" for key, value in items)


def build_grid_search_figure(
    nav_curves: pd.DataFrame,
    sharpe_curves: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    *,
    benchmark_sharpe_curve: pd.DataFrame | None = None,
    title: str = "Blue Chip Range Reversion Grid Search",
) -> go.Figure:
    """Plot normalized NAV curves and rolling Sharpe curves for each parameter set."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Normalized NAV", "Rolling Sharpe"),
    )

    if not benchmark_curve.empty:
        figure.add_trace(
            go.Scatter(
                x=benchmark_curve["date"],
                y=benchmark_curve["benchmark_nav_norm"],
                mode="lines",
                name="Benchmark NAV",
                line=dict(color="black", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    if not nav_curves.empty:
        for label, group in nav_curves.groupby("label", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["strategy_nav_norm"],
                    mode="lines",
                    name=f"{label} NAV",
                ),
                row=1,
                col=1,
            )

    if benchmark_sharpe_curve is not None and not benchmark_sharpe_curve.empty:
        figure.add_trace(
            go.Scatter(
                x=benchmark_sharpe_curve["date"],
                y=benchmark_sharpe_curve["rolling_sharpe"],
                mode="lines",
                name="Benchmark Sharpe",
                line=dict(color="black", width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    if not sharpe_curves.empty:
        for label, group in sharpe_curves.groupby("label", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["rolling_sharpe"],
                    mode="lines",
                    name=f"{label} Sharpe",
                ),
                row=2,
                col=1,
            )

    figure.update_layout(
        height=900,
        width=1300,
        template="plotly_white",
        hovermode="x unified",
        title=title,
    )
    figure.update_yaxes(title_text="NAV", row=1, col=1)
    figure.update_yaxes(title_text="Sharpe", row=2, col=1)
    return figure


def run_blue_chip_grid_search(
    stock_candle_df: pd.DataFrame,
    *,
    param_grid: Mapping[str, Iterable[Any]],
    base_config: RangeStrategyConfig | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    backtester_kwargs: Mapping[str, Any] | None = None,
    sharpe_window: int = 63,
    sharpe_min_periods: int | None = None,
    skip_failures: bool = True,
) -> dict[str, Any]:
    """
    Run a parameter grid over the blue-chip range strategy and keep full curves.

    Returns summary tables plus long-form NAV and Sharpe curves so callers can
    inspect, rank, and replot the grid search outputs.
    """
    if sharpe_window < 2:
        raise ValueError("sharpe_window must be at least 2.")

    combinations = expand_param_grid(param_grid)
    if sharpe_min_periods is None:
        sharpe_min_periods = min(sharpe_window, max(5, sharpe_window // 3))
    if sharpe_min_periods < 1:
        raise ValueError("sharpe_min_periods must be at least 1.")

    config_template = base_config or RangeStrategyConfig()
    backtester_kwargs = dict(backtester_kwargs or {})

    summary_rows: list[dict[str, Any]] = []
    nav_frames: list[pd.DataFrame] = []
    sharpe_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, Any]] = []
    benchmark_curve = pd.DataFrame(columns=["date", "benchmark_nav_norm"])
    benchmark_sharpe_curve = pd.DataFrame(columns=["date", "rolling_sharpe"])
    benchmark_captured = False

    for combo_id, overrides in enumerate(combinations):
        label = format_param_label(overrides)
        try:
            config = replace(config_template, **overrides)
            researcher = BlueChipRangeReversionResearcher(stock_candle_df, config=config)
            backtester = TradePlanBacktester(
                stock_candle_df,
                researcher=researcher,
                **backtester_kwargs,
            )
            results = backtester.compute_metrics(start_date=start_date, end_date=end_date)
        except Exception as exc:
            error_row = {
                "combo_id": combo_id,
                "label": label,
                "status": "failed",
                "error": str(exc),
            }
            error_row.update(overrides)
            error_rows.append(error_row)
            if skip_failures:
                continue
            raise

        portfolio = results["portfolio"].copy()
        portfolio["rolling_sharpe"] = (
            portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).mean()
            / portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).std(ddof=0)
        ) * np.sqrt(252.0)
        portfolio["benchmark_rolling_sharpe"] = (
            portfolio["benchmark_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).mean()
            / portfolio["benchmark_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).std(ddof=0)
        ) * np.sqrt(252.0)

        nav_frame = portfolio.loc[:, ["date", "strategy_nav_norm"]].copy()
        nav_frame["combo_id"] = combo_id
        nav_frame["label"] = label
        nav_frames.append(nav_frame)

        sharpe_frame = portfolio.loc[:, ["date", "rolling_sharpe"]].copy()
        sharpe_frame["combo_id"] = combo_id
        sharpe_frame["label"] = label
        sharpe_frames.append(sharpe_frame)

        if not benchmark_captured:
            benchmark_curve = portfolio.loc[:, ["date", "benchmark_nav_norm"]].copy()
            benchmark_sharpe_curve = portfolio.loc[:, ["date", "benchmark_rolling_sharpe"]].rename(
                columns={"benchmark_rolling_sharpe": "rolling_sharpe"}
            )
            benchmark_captured = True

        summary_row = {
            "combo_id": combo_id,
            "label": label,
            "status": "ok",
        }
        summary_row.update(overrides)
        summary_row.update(asdict(config))
        summary_row.update(results["summary"])
        summary_row.update(
            {
                "final_nav": float(portfolio["strategy_nav_norm"].iat[-1]),
                "final_benchmark_nav": float(portfolio["benchmark_nav_norm"].iat[-1]),
            }
        )
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["sharpe", "total_return", "combo_id"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )

    errors = pd.DataFrame(error_rows)
    nav_curves = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame(
        columns=["date", "strategy_nav_norm", "combo_id", "label"]
    )
    sharpe_curves = pd.concat(sharpe_frames, ignore_index=True) if sharpe_frames else pd.DataFrame(
        columns=["date", "rolling_sharpe", "combo_id", "label"]
    )

    figure = build_grid_search_figure(
        nav_curves=nav_curves,
        sharpe_curves=sharpe_curves,
        benchmark_curve=benchmark_curve,
        benchmark_sharpe_curve=benchmark_sharpe_curve,
    )
    return {
        "summary": summary,
        "nav_curves": nav_curves,
        "sharpe_curves": sharpe_curves,
        "benchmark_curve": benchmark_curve,
        "benchmark_sharpe_curve": benchmark_sharpe_curve,
        "errors": errors,
        "figure": figure,
    }
