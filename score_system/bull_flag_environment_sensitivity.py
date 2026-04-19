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
from strategies.bull_flag_continuation import (
    BullFlagContinuationResearcher,
    BullFlagStrategyConfig,
)


def _consecutive_true_run_length(series: pd.Series) -> pd.Series:
    values = series.fillna(False).to_numpy(dtype=bool)
    output = np.zeros(len(values), dtype=np.int32)
    run_length = 0
    for index, value in enumerate(values):
        if value:
            run_length += 1
        else:
            run_length = 0
        output[index] = run_length
    return pd.Series(output, index=series.index, dtype="int32")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _parse_filter_spec(key: str) -> tuple[str, str]:
    if "__" not in key:
        raise ValueError(
            f"Invalid filter grid key '{key}'. Use suffixes like '__min', '__max', or '__eq'."
        )
    column, operator = key.rsplit("__", 1)
    if operator not in {"min", "max", "eq"}:
        raise ValueError(
            f"Unsupported filter operator '{operator}' in key '{key}'. "
            "Use '__min', '__max', or '__eq'."
        )
    return column, operator


def build_bull_flag_environment_trade_frame(
    stock_candle_df: pd.DataFrame,
    *,
    config: BullFlagStrategyConfig | None = None,
) -> tuple[BullFlagContinuationResearcher, pd.DataFrame]:
    """
    Build one bull-flag researcher instance and enrich its trade_df with
    background-regime diagnostics for post-hoc sensitivity studies.

    The function is intentionally designed so downstream experiments can reuse
    one expensive researcher pass and then backtest many stricter environment
    filters cheaply by subsetting the planned trade_df.
    """
    researcher = BullFlagContinuationResearcher(stock_candle_df, config=config or BullFlagStrategyConfig())
    scored = researcher._sort_for_calculation(researcher.stock_candle_df.copy())
    if scored.empty:
        return researcher, researcher.trade_df.copy()

    ticker_group = scored.groupby("ticker", sort=False)
    scored["bullish_stack_run_length"] = ticker_group["bullish_stack"].transform(_consecutive_true_run_length)
    scored["stack_spread_pct"] = _safe_ratio(scored["sma_20"] - scored["sma_120"], scored["close"])
    scored["close_to_sma20_pct"] = _safe_ratio(scored["close"] - scored["sma_20"], scored["close"])
    scored["close_to_sma60_pct"] = _safe_ratio(scored["close"] - scored["sma_60"], scored["close"])
    scored["close_to_sma120_pct"] = _safe_ratio(scored["close"] - scored["sma_120"], scored["close"])
    scored["sma20_return_5"] = ticker_group["sma_20"].transform(lambda s: s.div(s.shift(5)) - 1.0)
    scored["sma60_return_10"] = ticker_group["sma_60"].transform(lambda s: s.div(s.shift(10)) - 1.0)
    scored["sma120_return_20"] = ticker_group["sma_120"].transform(lambda s: s.div(s.shift(20)) - 1.0)

    trade_df = researcher.trade_df.copy()
    if trade_df.empty:
        return researcher, trade_df

    signal_columns = [
        "ticker",
        "date",
        "bullish_stack_run_length",
        "stack_spread_pct",
        "close_to_sma20_pct",
        "close_to_sma60_pct",
        "close_to_sma120_pct",
        "sma20_return_5",
        "sma60_return_10",
        "sma120_return_20",
        "sma_20",
        "sma_60",
        "sma_120",
        "close",
    ]
    signal_features = scored.loc[:, signal_columns].rename(
        columns={
            "date": "signal_date",
            "bullish_stack_run_length": "signal_bullish_stack_run_length",
            "stack_spread_pct": "signal_stack_spread_pct",
            "close_to_sma20_pct": "signal_close_to_sma20_pct",
            "close_to_sma60_pct": "signal_close_to_sma60_pct",
            "close_to_sma120_pct": "signal_close_to_sma120_pct",
            "sma20_return_5": "signal_sma20_return_5",
            "sma60_return_10": "signal_sma60_return_10",
            "sma120_return_20": "signal_sma120_return_20",
            "sma_20": "signal_sma_20",
            "sma_60": "signal_sma_60",
            "sma_120": "signal_sma_120",
            "close": "signal_close",
        }
    )
    peak_features = scored.loc[:, signal_columns].rename(
        columns={
            "date": "flag_peak_date",
            "bullish_stack_run_length": "peak_bullish_stack_run_length",
            "stack_spread_pct": "peak_stack_spread_pct",
            "close_to_sma20_pct": "peak_close_to_sma20_pct",
            "close_to_sma60_pct": "peak_close_to_sma60_pct",
            "close_to_sma120_pct": "peak_close_to_sma120_pct",
            "sma20_return_5": "peak_sma20_return_5",
            "sma60_return_10": "peak_sma60_return_10",
            "sma120_return_20": "peak_sma120_return_20",
            "sma_20": "peak_sma_20",
            "sma_60": "peak_sma_60",
            "sma_120": "peak_sma_120",
            "close": "peak_close",
        }
    )

    enriched = trade_df.merge(signal_features, on=["ticker", "signal_date"], how="left", validate="one_to_one")
    enriched = enriched.merge(peak_features, on=["ticker", "flag_peak_date"], how="left", validate="many_to_one")
    enriched["peak_high_to_sma20_pct"] = _safe_ratio(enriched["flag_peak_high"] - enriched["peak_sma_20"], enriched["flag_peak_high"])
    enriched["peak_high_to_sma60_pct"] = _safe_ratio(enriched["flag_peak_high"] - enriched["peak_sma_60"], enriched["flag_peak_high"])
    enriched["peak_high_to_sma120_pct"] = _safe_ratio(enriched["flag_peak_high"] - enriched["peak_sma_120"], enriched["flag_peak_high"])
    return researcher, enriched


def _apply_trade_filters(trade_df: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    filtered = trade_df.copy()
    for key, value in filters.items():
        column, operator = _parse_filter_spec(key)
        if column not in filtered.columns:
            raise ValueError(f"Trade frame is missing filter column '{column}'.")
        if operator == "min":
            filtered = filtered.loc[filtered[column].ge(value)]
        elif operator == "max":
            filtered = filtered.loc[filtered[column].le(value)]
        else:
            filtered = filtered.loc[filtered[column].eq(value)]
    return filtered.sort_values(
        ["entry_date", "ticker", "signal_date"],
        kind="mergesort",
        ignore_index=True,
    )


def run_bull_flag_environment_filter_grid(
    stock_candle_df: pd.DataFrame,
    *,
    filter_grid: Mapping[str, list[Any]],
    base_config: BullFlagStrategyConfig | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    backtester_kwargs: Mapping[str, Any] | None = None,
    sharpe_window: int = 63,
    sharpe_min_periods: int | None = None,
) -> dict[str, Any]:
    """
    Reuse one bull-flag researcher pass and grid-search stricter environment
    filters by subsetting the planned trade_df before backtesting.
    """
    if sharpe_window < 2:
        raise ValueError("sharpe_window must be at least 2.")
    if sharpe_min_periods is None:
        sharpe_min_periods = min(sharpe_window, max(5, sharpe_window // 3))
    if sharpe_min_periods < 1:
        raise ValueError("sharpe_min_periods must be at least 1.")

    config = base_config or BullFlagStrategyConfig()
    backtester_kwargs = dict(backtester_kwargs or {})
    researcher, enriched_trade_df = build_bull_flag_environment_trade_frame(stock_candle_df, config=config)
    results = run_bull_flag_environment_filter_grid_from_trade_frame(
        stock_candle_df,
        trade_frame=enriched_trade_df,
        filter_grid=filter_grid,
        base_config=config,
        start_date=start_date,
        end_date=end_date,
        backtester_kwargs=backtester_kwargs,
        sharpe_window=sharpe_window,
        sharpe_min_periods=sharpe_min_periods,
    )
    results["researcher"] = researcher
    return results


def run_bull_flag_environment_filter_grid_from_trade_frame(
    stock_candle_df: pd.DataFrame,
    *,
    trade_frame: pd.DataFrame,
    filter_grid: Mapping[str, list[Any]],
    base_config: BullFlagStrategyConfig | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    backtester_kwargs: Mapping[str, Any] | None = None,
    sharpe_window: int = 63,
    sharpe_min_periods: int | None = None,
) -> dict[str, Any]:
    if sharpe_window < 2:
        raise ValueError("sharpe_window must be at least 2.")
    if sharpe_min_periods is None:
        sharpe_min_periods = min(sharpe_window, max(5, sharpe_window // 3))
    if sharpe_min_periods < 1:
        raise ValueError("sharpe_min_periods must be at least 1.")

    config = base_config or BullFlagStrategyConfig()
    backtester_kwargs = dict(backtester_kwargs or {})
    enriched_trade_df = trade_frame.copy()
    combinations = expand_param_grid(filter_grid)

    summary_rows: list[dict[str, Any]] = []
    nav_frames: list[pd.DataFrame] = []
    sharpe_frames: list[pd.DataFrame] = []
    benchmark_curve = pd.DataFrame(columns=["date", "benchmark_nav_norm"])
    benchmark_sharpe_curve = pd.DataFrame(columns=["date", "rolling_sharpe"])
    benchmark_captured = False

    baseline_backtester = TradePlanBacktester(
        stock_candle_df,
        trade_df=enriched_trade_df,
        **backtester_kwargs,
    )
    baseline_results = baseline_backtester.compute_metrics(start_date=start_date, end_date=end_date)
    baseline_summary = {
        "combo_id": -1,
        "label": "baseline",
        "status": "ok",
        "retained_trade_ratio": 1.0,
        "retained_trade_count": int(len(enriched_trade_df)),
    }
    baseline_summary.update(asdict(config))
    baseline_summary.update(baseline_results["summary"])
    summary_rows.append(baseline_summary)

    baseline_portfolio = baseline_results["portfolio"].copy()
    baseline_portfolio["rolling_sharpe"] = (
        baseline_portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).mean()
        / baseline_portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).std(ddof=0)
    ) * np.sqrt(252.0)
    baseline_portfolio["benchmark_rolling_sharpe"] = (
        baseline_portfolio["benchmark_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).mean()
        / baseline_portfolio["benchmark_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).std(ddof=0)
    ) * np.sqrt(252.0)
    nav_frame = baseline_portfolio.loc[:, ["date", "strategy_nav_norm"]].copy()
    nav_frame["combo_id"] = -1
    nav_frame["label"] = "baseline"
    nav_frames.append(nav_frame)
    sharpe_frame = baseline_portfolio.loc[:, ["date", "rolling_sharpe"]].copy()
    sharpe_frame["combo_id"] = -1
    sharpe_frame["label"] = "baseline"
    sharpe_frames.append(sharpe_frame)
    benchmark_curve = baseline_portfolio.loc[:, ["date", "benchmark_nav_norm"]].copy()
    benchmark_sharpe_curve = baseline_portfolio.loc[:, ["date", "benchmark_rolling_sharpe"]].rename(
        columns={"benchmark_rolling_sharpe": "rolling_sharpe"}
    )
    benchmark_captured = True

    for combo_id, filters in enumerate(combinations):
        label = format_param_label(filters)
        filtered_trade_df = _apply_trade_filters(enriched_trade_df, filters)
        backtester = TradePlanBacktester(
            stock_candle_df,
            trade_df=filtered_trade_df,
            **backtester_kwargs,
        )
        results = backtester.compute_metrics(start_date=start_date, end_date=end_date)
        portfolio = results["portfolio"].copy()
        portfolio["rolling_sharpe"] = (
            portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).mean()
            / portfolio["strategy_return"].rolling(sharpe_window, min_periods=sharpe_min_periods).std(ddof=0)
        ) * np.sqrt(252.0)

        nav_frame = portfolio.loc[:, ["date", "strategy_nav_norm"]].copy()
        nav_frame["combo_id"] = combo_id
        nav_frame["label"] = label
        nav_frames.append(nav_frame)

        sharpe_frame = portfolio.loc[:, ["date", "rolling_sharpe"]].copy()
        sharpe_frame["combo_id"] = combo_id
        sharpe_frame["label"] = label
        sharpe_frames.append(sharpe_frame)

        summary_row = {
            "combo_id": combo_id,
            "label": label,
            "status": "ok",
            "retained_trade_ratio": float(len(filtered_trade_df) / len(enriched_trade_df)) if len(enriched_trade_df) else np.nan,
            "retained_trade_count": int(len(filtered_trade_df)),
        }
        summary_row.update(filters)
        summary_row.update(asdict(config))
        summary_row.update(results["summary"])
        summary_rows.append(summary_row)

        if not benchmark_captured:
            benchmark_curve = portfolio.loc[:, ["date", "benchmark_nav_norm"]].copy()
            benchmark_sharpe_curve = portfolio.loc[:, ["date", "benchmark_rolling_sharpe"]].rename(
                columns={"benchmark_rolling_sharpe": "rolling_sharpe"}
            )
            benchmark_captured = True

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["sharpe", "total_return", "combo_id"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )

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
        title="Bull Flag Environment Sensitivity",
    )
    return {
        "trade_frame": enriched_trade_df,
        "summary": summary,
        "nav_curves": nav_curves,
        "sharpe_curves": sharpe_curves,
        "benchmark_curve": benchmark_curve,
        "benchmark_sharpe_curve": benchmark_sharpe_curve,
        "figure": figure,
    }


__all__ = [
    "build_bull_flag_environment_trade_frame",
    "run_bull_flag_environment_filter_grid",
    "run_bull_flag_environment_filter_grid_from_trade_frame",
]
