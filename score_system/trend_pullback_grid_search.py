from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import TradePlanBacktester
from score_system.blue_chip_grid_search import (
    build_grid_search_figure,
    expand_param_grid,
    format_param_label,
)
from strategies.trend_pullback_continuation import (
    TrendPullbackContinuationResearcher,
    TrendPullbackStrategyConfig,
)


def run_trend_pullback_grid_search(
    stock_candle_df: pd.DataFrame,
    *,
    param_grid: Mapping[str, list[Any]],
    base_config: TrendPullbackStrategyConfig | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    backtester_kwargs: Mapping[str, Any] | None = None,
    sharpe_window: int = 63,
    sharpe_min_periods: int | None = None,
    skip_failures: bool = True,
) -> dict[str, Any]:
    """
    Run a parameter grid over the trend-pullback continuation strategy.

    The return shape intentionally mirrors ``run_blue_chip_grid_search`` so
    notebook analysis can be reused with minimal changes.
    """
    if sharpe_window < 2:
        raise ValueError("sharpe_window must be at least 2.")

    combinations = expand_param_grid(param_grid)
    if sharpe_min_periods is None:
        sharpe_min_periods = min(sharpe_window, max(5, sharpe_window // 3))
    if sharpe_min_periods < 1:
        raise ValueError("sharpe_min_periods must be at least 1.")

    config_template = base_config or TrendPullbackStrategyConfig()
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
            researcher = TrendPullbackContinuationResearcher(stock_candle_df, config=config)
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
        title="Trend Pullback Continuation Grid Search",
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


__all__ = ["run_trend_pullback_grid_search"]
