import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from score_system.bull_flag_grid_search import run_bull_flag_grid_search
from strategies.bull_flag_continuation import (
    BullFlagContinuationResearcher,
    BullFlagStrategyConfig,
)


def make_stock_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    dates: pd.DatetimeIndex,
    open_values: list[float] | np.ndarray | None = None,
    high_values: list[float] | np.ndarray | None = None,
    low_values: list[float] | np.ndarray | None = None,
    weight: float = 1.0,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if open_values is None:
        open_values = closes - 0.2
    if high_values is None:
        high_values = np.maximum(open_values, closes) + 0.5
    if low_values is None:
        low_values = np.minimum(open_values, closes) - 0.5

    open_values = np.asarray(open_values, dtype=float)
    high_values = np.asarray(high_values, dtype=float)
    low_values = np.asarray(low_values, dtype=float)
    volume_values = 1_000 + np.arange(len(closes)) * 10
    pre_close = np.concatenate(([closes[0]], closes[:-1]))
    turnover = volume_values * closes
    safe_pre_close = np.where(pre_close > 0, pre_close, np.nan)
    amplitude_pct = (high_values - low_values) / safe_pre_close * 100.0
    change_amount = closes - pre_close
    change_pct = change_amount / safe_pre_close * 100.0

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "ts_code": f"{ticker}.SZ",
            "name": f"{ticker} Corp",
            "weight": weight,
            "constituent_trade_date": dates[-1],
            "open": open_values,
            "close": closes,
            "high": high_values,
            "low": low_values,
            "pre_close": pre_close,
            "volume": volume_values,
            "turnover": turnover,
            "amplitude_pct": amplitude_pct,
            "change_pct": change_pct,
            "change_amount": change_amount,
        }
    )


def make_annotation_researcher(
    config: BullFlagStrategyConfig | None = None,
) -> BullFlagContinuationResearcher:
    dates = pd.date_range("2025-01-01", periods=14, freq="B")
    panel = make_stock_frame("AAA", np.linspace(10.0, 24.0, len(dates)), dates=dates)
    return BullFlagContinuationResearcher(panel, config=config or BullFlagStrategyConfig())


def make_annotation_frame(
    values: list[float],
    *,
    pivot_low_positions: list[int],
    pivot_high_positions: list[int],
    bullish_stack_false_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
) -> pd.DataFrame:
    bullish_stack_false_positions = bullish_stack_false_positions or []
    signal_quality_positions = signal_quality_positions or []
    close_gt_prev_high_positions = close_gt_prev_high_positions or []
    dates = pd.date_range("2025-01-01", periods=len(values), freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.asarray(values, dtype=float) - 0.1,
            "high": values,
            "low": values,
            "close": values,
            "bullish_stack": True,
            "bullish_stack_run_length": np.arange(1, len(values) + 1, dtype=float),
            "stack_spread_pct": 0.08,
            "sma20_return_5": 0.03,
            "sma60_return_10": 0.03,
            "pivot_high": False,
            "pivot_low": False,
            "signal_quality_ok": False,
            "close_gt_prev_high": False,
        }
    )
    for position in bullish_stack_false_positions:
        frame.loc[position, "bullish_stack"] = False
    for position in pivot_low_positions:
        frame.loc[position, "pivot_low"] = True
    for position in pivot_high_positions:
        frame.loc[position, "pivot_high"] = True
    for position in signal_quality_positions:
        frame.loc[position, "signal_quality_ok"] = True
    for position in close_gt_prev_high_positions:
        frame.loc[position, "close_gt_prev_high"] = True
    return frame


def annotate_case(
    values: list[float],
    *,
    pivot_low_positions: list[int],
    pivot_high_positions: list[int],
    bullish_stack_false_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
    config: BullFlagStrategyConfig | None = None,
) -> pd.DataFrame:
    researcher = make_annotation_researcher(config=config)
    frame = make_annotation_frame(
        values,
        pivot_low_positions=pivot_low_positions,
        pivot_high_positions=pivot_high_positions,
        bullish_stack_false_positions=bullish_stack_false_positions,
        signal_quality_positions=signal_quality_positions,
        close_gt_prev_high_positions=close_gt_prev_high_positions,
    )
    annotations = researcher._annotate_ticker_context(frame)
    annotated = frame.copy()
    for column, values_array in annotations.items():
        annotated[column] = values_array
    return annotated


def make_manual_signal_researcher(
    *,
    config: BullFlagStrategyConfig | None = None,
    follow_through_high: float = 105.0,
) -> tuple[BullFlagContinuationResearcher, pd.DatetimeIndex]:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    panel = make_stock_frame(
        "AAA",
        [102.0, 104.0, 106.0, 107.0, 108.0, 109.0],
        dates=dates,
        open_values=[101.0, 103.0, 102.0, 107.0, 108.0, 109.0],
        high_values=[103.0, follow_through_high, 107.0, 108.0, 109.0, 110.0],
        low_values=[100.0, 102.5, 101.5, 106.0, 107.0, 108.0],
    )
    researcher = BullFlagContinuationResearcher(
        panel,
        config=config
        or BullFlagStrategyConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=2,
            enable_time_stop=True,
        ),
    )
    prepared = researcher.stock_candle_df.copy()
    removable_columns = (
        BullFlagContinuationResearcher.SIGNAL_COLUMNS
        + BullFlagContinuationResearcher.OUTCOME_COLUMNS
        + ["entry_date_next", "entry_open_next"]
    )
    prepared = prepared.drop(columns=[column for column in removable_columns if column in prepared.columns])
    prepared["sma_20"] = 101.0
    prepared["sma_60"] = 100.0
    prepared["sma_120"] = 99.0
    prepared["bullish_stack"] = True
    prepared["pivot_high"] = False
    prepared["pivot_low"] = False
    prepared["close_gt_open"] = True
    prepared["close_gt_prev_high"] = True
    prepared["bullish_stack_run_length"] = np.arange(1, len(prepared) + 1, dtype=float)
    prepared["stack_spread_pct"] = 0.08
    prepared["sma20_return_5"] = 0.03
    prepared["sma60_return_10"] = 0.03
    prepared["signal_body_pct"] = 0.70
    prepared["signal_upper_shadow_pct"] = 0.10
    prepared["signal_lower_shadow_pct"] = 0.10
    prepared["signal_quality_ok"] = True
    prepared["signal_bullish_stack_run_length"] = np.nan
    prepared["signal_stack_spread_pct"] = np.nan
    prepared["signal_sma20_return_5"] = np.nan
    prepared["peak_bullish_stack_run_length"] = np.nan
    prepared["peak_sma60_return_10"] = np.nan
    prepared["flagpole_start_date"] = dates[0]
    prepared["flagpole_start_low"] = 80.0
    prepared["flagpole_length"] = 20.0
    prepared["flag_peak_date"] = dates[0]
    prepared["flag_peak_high"] = 100.0
    prepared["flagpole_bars"] = 5
    prepared["flagpole_return"] = 0.25
    prepared["flag_start_date"] = dates[0]
    prepared["flag_end_date"] = dates[0]
    prepared["flag_bars"] = 4
    prepared["flag_low_date"] = dates[0]
    prepared["flag_low"] = 95.0
    prepared["flag_retrace_ratio"] = 0.25
    prepared["flag_width_pct"] = 0.05
    prepared["flag_upper_slope"] = -0.10
    prepared["flag_lower_slope"] = -0.08
    prepared["flag_upper_line_value"] = 99.0
    prepared["flag_lower_line_value"] = 96.0
    prepared["flag_shape_ok"] = False
    prepared["flag_retrace_ok"] = False
    prepared["flag_channel_ok"] = False
    prepared["bull_flag_candidate"] = False
    prepared["breakout_candle"] = False
    prepared["signal_candle"] = False
    prepared["signal_stack_spread_ok"] = True
    prepared["signal_sma20_return_ok"] = True
    prepared["peak_sma60_return_ok"] = True
    prepared["trend_environment_ok"] = True

    first_day_mask = prepared["date"].eq(dates[0])
    prepared.loc[first_day_mask, "signal_bullish_stack_run_length"] = 10.0
    prepared.loc[first_day_mask, "signal_stack_spread_pct"] = 0.08
    prepared.loc[first_day_mask, "signal_sma20_return_5"] = 0.03
    prepared.loc[first_day_mask, "peak_bullish_stack_run_length"] = 8.0
    prepared.loc[first_day_mask, "peak_sma60_return_10"] = 0.03
    prepared.loc[first_day_mask, "flag_shape_ok"] = True
    prepared.loc[first_day_mask, "flag_retrace_ok"] = True
    prepared.loc[first_day_mask, "flag_channel_ok"] = True
    prepared.loc[first_day_mask, "bull_flag_candidate"] = True
    prepared.loc[first_day_mask, "breakout_candle"] = True
    prepared.loc[first_day_mask, "signal_candle"] = True

    researcher.stock_candle_df = prepared
    return researcher, dates


def make_live_candidate_researcher(
    *,
    config: BullFlagStrategyConfig | None = None,
    follow_through_high: float = 105.0,
) -> tuple[BullFlagContinuationResearcher, pd.DatetimeIndex]:
    researcher, dates = make_manual_signal_researcher(
        config=config,
        follow_through_high=follow_through_high,
    )
    researcher.stock_candle_df = researcher.stock_candle_df.iloc[:2].copy().reset_index(drop=True)
    return researcher, dates[:2]


class BullFlagContinuationResearcherTests(unittest.TestCase):
    def test_short_flag_and_breakout_generate_signal_candle(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.2, 19.1, 19.0, 18.9, 20.6],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["flag_shape_ok"]))
        self.assertTrue(bool(signal_row["flag_retrace_ok"]))
        self.assertTrue(bool(signal_row["flag_channel_ok"]))
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertTrue(bool(signal_row["breakout_candle"]))
        self.assertAlmostEqual(float(signal_row["flagpole_return"]), 1.0)
        self.assertAlmostEqual(float(signal_row["flag_retrace_ratio"]), 0.11, places=2)

    def test_deep_flag_retrace_invalidates_setup(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 18.0, 17.0, 16.0, 15.0, 20.6],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertFalse(bool(signal_row["flag_retrace_ok"]))
        self.assertFalse(bool(signal_row["bull_flag_candidate"]))
        self.assertFalse(bool(signal_row["breakout_candle"]))

    def test_steep_downward_channel_is_rejected(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.8, 19.4, 19.0, 18.6, 20.5],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["flag_retrace_ok"]))
        self.assertFalse(bool(signal_row["flag_channel_ok"]))
        self.assertFalse(bool(signal_row["bull_flag_candidate"]))

    def test_horizontal_flag_is_allowed(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.4, 19.4, 19.3, 19.4, 20.2],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertTrue(bool(signal_row["signal_candle"]))

    def test_mild_positive_slope_is_allowed_under_default_symmetric_bounds(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.20, 19.25, 19.30, 19.35, 20.50],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["flag_channel_ok"]))
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))

    def test_positive_slope_is_rejected_when_channel_requires_non_positive_slopes(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.20, 19.25, 19.30, 19.35, 20.50],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
            config=BullFlagStrategyConfig(
                min_flag_channel_slope_pct_per_bar=-0.008,
                max_flag_channel_slope_pct_per_bar=0.0,
            ),
        )

        signal_row = annotated.iloc[-1]
        self.assertFalse(bool(signal_row["flag_channel_ok"]))
        self.assertFalse(bool(signal_row["bull_flag_candidate"]))

    def test_setup_does_not_revive_after_bullish_stack_breaks_inside_flag(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.4, 19.4, 19.3, 19.4, 20.2],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            bullish_stack_false_positions=[7],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertFalse(bool(signal_row["bull_flag_candidate"]))
        self.assertFalse(bool(signal_row["breakout_candle"]))
        self.assertFalse(bool(signal_row["signal_candle"]))

    def test_breakout_must_clear_projected_upper_line(self) -> None:
        annotated = annotate_case(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 19.20, 19.22, 19.24, 19.26, 19.265],
            pivot_low_positions=[0],
            pivot_high_positions=[5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertFalse(bool(signal_row["breakout_candle"]))

    def test_signal_without_follow_through_does_not_enter(self) -> None:
        researcher, dates = make_manual_signal_researcher(follow_through_high=103.0)
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(signal_row["signal_candle"]))
        self.assertFalse(bool(signal_row["follow_through_confirmed"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_optional_follow_through_close_filter_blocks_weak_close(self) -> None:
        researcher, dates = make_manual_signal_researcher(
            config=BullFlagStrategyConfig(
                ma_windows=(2, 3, 4),
                pivot_window=1,
                max_holding_days=2,
                enable_time_stop=True,
                require_follow_through_close_gt_signal_close=True,
            )
        )
        researcher.stock_candle_df.loc[researcher.stock_candle_df["date"].eq(dates[1]), "close"] = 101.0
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(signal_row["follow_through_confirmed"]))
        self.assertFalse(bool(signal_row["follow_through_close_gt_signal_close"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_optional_signal_stack_spread_filter_blocks_overextended_stack(self) -> None:
        researcher, dates = make_manual_signal_researcher(
            config=BullFlagStrategyConfig(
                ma_windows=(2, 3, 4),
                pivot_window=1,
                max_holding_days=2,
                enable_time_stop=True,
                max_signal_stack_spread_pct=0.05,
            )
        )
        researcher.stock_candle_df.loc[researcher.stock_candle_df["date"].eq(dates[0]), "signal_stack_spread_pct"] = 0.08
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertFalse(bool(signal_row["signal_stack_spread_ok"]))
        self.assertFalse(bool(signal_row["trend_environment_ok"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_optional_peak_sma60_environment_filter_blocks_overheated_peak(self) -> None:
        researcher, dates = make_manual_signal_researcher(
            config=BullFlagStrategyConfig(
                ma_windows=(2, 3, 4),
                pivot_window=1,
                max_holding_days=2,
                enable_time_stop=True,
                max_peak_sma60_return_10=0.02,
            )
        )
        researcher.stock_candle_df.loc[researcher.stock_candle_df["date"].eq(dates[0]), "peak_sma60_return_10"] = 0.03
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertFalse(bool(signal_row["peak_sma60_return_ok"]))
        self.assertFalse(bool(signal_row["trend_environment_ok"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_stop_target_and_reward_to_risk_use_flag_low_and_flagpole_length(self) -> None:
        researcher, dates = make_manual_signal_researcher()
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertAlmostEqual(float(signal_row["signal_hard_stop_price"]), 94.05, places=2)
        self.assertAlmostEqual(float(signal_row["signal_take_profit_price"]), 119.0, places=2)
        self.assertAlmostEqual(float(signal_row["reward_to_risk"]), 15.0 / 9.95, places=6)
        self.assertTrue(bool(signal_row["reward_to_risk_ok"]))
        self.assertTrue(bool(signal_row["entry_signal"]))

    def test_reward_to_risk_filter_blocks_marginal_setup(self) -> None:
        researcher, dates = make_manual_signal_researcher(
            config=BullFlagStrategyConfig(
                ma_windows=(2, 3, 4),
                pivot_window=1,
                max_holding_days=2,
                enable_time_stop=True,
                min_reward_r=1.6,
            )
        )
        signal_frame = researcher.add_signals()
        signal_row = signal_frame[signal_frame["date"] == dates[0]].iloc[0]

        self.assertFalse(bool(signal_row["reward_to_risk_ok"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_plot_signal_context_draws_flag_annotations(self) -> None:
        researcher, dates = make_manual_signal_researcher()
        researcher.add_research_outcomes()

        inspection = researcher.inspect_signal("AAA", dates[0], lookback=2, lookahead=3)
        figure = researcher.plot_signal_context("AAA", dates[0], lookback=2, lookahead=3)
        trace_names = {trace.name for trace in figure.data}

        self.assertAlmostEqual(float(inspection["summary"]["flagpole_return"]), 0.25)
        self.assertNotIn("left_trend_mode", inspection["summary"])
        self.assertNotIn("left_state_start_date", inspection["summary"])
        self.assertIn("Flagpole Start", trace_names)
        self.assertIn("Flag Low", trace_names)
        self.assertIn("Flag Upper", trace_names)
        self.assertIn("Flag Lower", trace_names)

    def test_classic_candidates_do_not_expose_narrow_trend_columns(self) -> None:
        researcher, dates = make_live_candidate_researcher()
        candidates = researcher.get_next_session_candidates(as_of_date=dates[1])

        self.assertIn("signal_date", candidates.columns)
        self.assertNotIn("left_state_start_date", candidates.columns)
        self.assertNotIn("narrow_uptrend_run_length", candidates.columns)

    def test_inspect_signal_supports_live_candidate_with_planned_values(self) -> None:
        researcher, dates = make_live_candidate_researcher()
        inspection = researcher.inspect_signal("AAA", dates[0], lookback=1, lookahead=2)

        summary = inspection["summary"]
        signal_row = inspection["signal_row"].iloc[0]

        self.assertEqual(summary["review_mode"], "live_candidate")
        self.assertFalse(bool(summary["executed_signal"]))
        self.assertEqual(pd.Timestamp(summary["planned_entry_date"]), dates[1] + pd.offsets.BDay(1))
        self.assertAlmostEqual(float(summary["planned_entry_price"]), 104.0, places=2)
        self.assertAlmostEqual(float(summary["planned_hard_stop_price"]), 94.05, places=2)
        self.assertAlmostEqual(float(summary["planned_take_profit_price"]), 119.0, places=2)
        self.assertTrue(pd.isna(summary["entry_date_next"]))
        self.assertTrue(pd.isna(summary["exit_date_next"]))
        self.assertIsNone(summary["exit_reason"])
        self.assertEqual(signal_row["review_mode"], "live_candidate")
        self.assertAlmostEqual(float(signal_row["entry_reference_price"]), 104.0, places=2)
        self.assertTrue(bool(signal_row["trend_environment_ok"]))

    def test_plot_signal_context_draws_planned_lines_for_live_candidate(self) -> None:
        researcher, dates = make_live_candidate_researcher()
        figure = researcher.plot_signal_context("AAA", dates[0], lookback=1, lookahead=2)
        trace_names = {trace.name for trace in figure.data}

        self.assertIn("Planned Entry", trace_names)
        self.assertIn("Planned Hard Stop", trace_names)
        self.assertIn("Planned Take Profit", trace_names)
        self.assertNotIn("TP1", trace_names)

    def test_run_bull_flag_grid_search_returns_curves_summary_and_figure(self) -> None:
        dates = pd.date_range("2025-01-01", periods=220, freq="B")
        panel = pd.concat(
            [
                make_stock_frame("AAA", np.linspace(20.0, 80.0, len(dates)), dates=dates),
                make_stock_frame("BBB", np.linspace(30.0, 90.0, len(dates)), dates=dates),
            ],
            ignore_index=True,
        )

        results = run_bull_flag_grid_search(
            panel,
            param_grid={"min_reward_r": [1.0, 1.5]},
            base_config=BullFlagStrategyConfig(
                max_holding_days=5,
                enable_time_stop=True,
            ),
            start_date=dates[120],
            end_date=dates[-1],
            backtester_kwargs={
                "initial_capital": 100_000.0,
                "board_lot_size": 1,
                "fixed_entry_notional": 20_000.0,
            },
            sharpe_window=20,
            sharpe_min_periods=5,
        )

        self.assertEqual(len(results["summary"]), 2)
        self.assertTrue(results["errors"].empty)
        self.assertEqual(
            set(results["nav_curves"]["label"].unique().tolist()),
            {"min_reward_r=1.0", "min_reward_r=1.5"},
        )
        self.assertFalse(results["benchmark_curve"].empty)
        self.assertFalse(results["benchmark_sharpe_curve"].empty)
        self.assertIsInstance(results["figure"], go.Figure)


if __name__ == "__main__":
    unittest.main()
