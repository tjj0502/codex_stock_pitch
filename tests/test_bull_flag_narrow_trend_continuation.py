import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategies.bull_flag_narrow_trend_continuation import (
    BullFlagNarrowTrendContinuationResearcher,
    BullFlagNarrowTrendStrategyConfig,
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
    config: BullFlagNarrowTrendStrategyConfig | None = None,
) -> BullFlagNarrowTrendContinuationResearcher:
    dates = pd.date_range("2025-01-01", periods=14, freq="B")
    panel = make_stock_frame("AAA", np.linspace(10.0, 24.0, len(dates)), dates=dates)
    return BullFlagNarrowTrendContinuationResearcher(
        panel,
        config=config or BullFlagNarrowTrendStrategyConfig(),
    )


def make_annotation_frame(
    values: list[float],
    *,
    pivot_low_positions: list[int],
    pivot_high_positions: list[int],
    narrow_state_positions: list[int],
    bullish_stack_false_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
    low_values: list[float] | None = None,
) -> pd.DataFrame:
    bullish_stack_false_positions = bullish_stack_false_positions or []
    signal_quality_positions = signal_quality_positions or []
    close_gt_prev_high_positions = close_gt_prev_high_positions or []
    dates = pd.date_range("2025-01-01", periods=len(values), freq="B")
    low_series = np.asarray(low_values if low_values is not None else values, dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.asarray(values, dtype=float) - 0.1,
            "high": np.asarray(values, dtype=float),
            "low": low_series,
            "close": np.asarray(values, dtype=float),
            "ema_20": np.asarray(values, dtype=float) - 0.5,
            "bullish_stack": True,
            "bullish_stack_run_length": np.arange(1, len(values) + 1, dtype=float),
            "stack_spread_pct": 0.08,
            "sma20_return_5": 0.03,
            "sma60_return_10": 0.03,
            "pivot_high": False,
            "pivot_low": False,
            "signal_quality_ok": False,
            "close_gt_prev_high": False,
            "narrow_uptrend_state": False,
            "narrow_uptrend_run_length": 0,
            "narrow_state_bear_ratio": np.nan,
            "narrow_state_ema20_above_ratio": np.nan,
            "narrow_state_max_consecutive_bear_bars": 0,
            "narrow_state_peak_upper_shadow_pct": np.nan,
        }
    )
    for position in bullish_stack_false_positions:
        frame.loc[position, "bullish_stack"] = False
    for position in pivot_low_positions:
        frame.loc[position, "pivot_low"] = True
    for position in pivot_high_positions:
        frame.loc[position, "pivot_high"] = True
    for position in narrow_state_positions:
        frame.loc[position, "narrow_uptrend_state"] = True
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
    narrow_state_positions: list[int],
    bullish_stack_false_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
    low_values: list[float] | None = None,
    config: BullFlagNarrowTrendStrategyConfig | None = None,
) -> pd.DataFrame:
    researcher = make_annotation_researcher(config=config)
    frame = make_annotation_frame(
        values,
        pivot_low_positions=pivot_low_positions,
        pivot_high_positions=pivot_high_positions,
        narrow_state_positions=narrow_state_positions,
        bullish_stack_false_positions=bullish_stack_false_positions,
        signal_quality_positions=signal_quality_positions,
        close_gt_prev_high_positions=close_gt_prev_high_positions,
        low_values=low_values,
    )
    annotations = researcher._annotate_ticker_context(frame)
    annotated = frame.copy()
    for column, values_array in annotations.items():
        annotated[column] = values_array
    return annotated


def make_narrow_state_frame(
    *,
    open_values: list[float],
    high_values: list[float],
    low_values: list[float],
    close_values: list[float],
    ema20_values: list[float],
    upper_shadow_values: list[float],
) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(close_values), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.asarray(open_values, dtype=float),
            "high": np.asarray(high_values, dtype=float),
            "low": np.asarray(low_values, dtype=float),
            "close": np.asarray(close_values, dtype=float),
            "ema_20": np.asarray(ema20_values, dtype=float),
            "signal_upper_shadow_pct": np.asarray(upper_shadow_values, dtype=float),
        }
    )


def make_manual_signal_researcher(
    *,
    config: BullFlagNarrowTrendStrategyConfig | None = None,
    follow_through_high: float = 105.0,
) -> tuple[BullFlagNarrowTrendContinuationResearcher, pd.DatetimeIndex]:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    panel = make_stock_frame(
        "AAA",
        [102.0, 104.0, 106.0, 107.0, 108.0, 109.0],
        dates=dates,
        open_values=[101.0, 103.0, 102.0, 107.0, 108.0, 109.0],
        high_values=[103.0, follow_through_high, 107.0, 108.0, 109.0, 110.0],
        low_values=[100.0, 102.5, 101.5, 106.0, 107.0, 108.0],
    )
    researcher = BullFlagNarrowTrendContinuationResearcher(
        panel,
        config=config
        or BullFlagNarrowTrendStrategyConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=2,
            enable_time_stop=True,
        ),
    )
    prepared = researcher.stock_candle_df.copy()
    removable_columns = (
        BullFlagNarrowTrendContinuationResearcher.SIGNAL_COLUMNS
        + BullFlagNarrowTrendContinuationResearcher.OUTCOME_COLUMNS
        + ["entry_date_next", "entry_open_next"]
    )
    prepared = prepared.drop(columns=[column for column in removable_columns if column in prepared.columns])
    prepared["sma_20"] = 101.0
    prepared["ema_20"] = 101.5
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
    prepared["narrow_uptrend_state"] = False
    prepared["narrow_uptrend_run_length"] = 0
    prepared["narrow_state_bear_ratio"] = np.nan
    prepared["narrow_state_ema20_above_ratio"] = np.nan
    prepared["narrow_state_max_consecutive_bear_bars"] = 0
    prepared["narrow_state_peak_upper_shadow_pct"] = np.nan
    prepared["left_state_start_date"] = pd.NaT
    prepared["left_state_end_date"] = pd.NaT
    prepared["left_state_bars"] = np.nan
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
    prepared.loc[first_day_mask, "left_state_start_date"] = dates[0]
    prepared.loc[first_day_mask, "left_state_end_date"] = dates[0]
    prepared.loc[first_day_mask, "left_state_bars"] = 1
    prepared.loc[first_day_mask, "narrow_uptrend_run_length"] = 1
    prepared.loc[first_day_mask, "narrow_state_bear_ratio"] = 0.0
    prepared.loc[first_day_mask, "narrow_state_ema20_above_ratio"] = 1.0

    researcher.stock_candle_df = prepared
    return researcher, dates


class BullFlagNarrowTrendContinuationResearcherTests(unittest.TestCase):
    def test_narrow_uptrend_state_true_when_window_meets_rules(self) -> None:
        researcher = make_annotation_researcher(
            config=BullFlagNarrowTrendStrategyConfig(
                narrow_trend_lookback_bars=5,
                narrow_trend_max_bear_ratio=0.20,
                narrow_trend_max_consecutive_bear_bars=2,
                narrow_trend_min_ema20_above_ratio=0.90,
                narrow_trend_max_upper_shadow_pct=0.25,
            )
        )
        frame = make_narrow_state_frame(
            open_values=[10.0, 10.4, 10.8, 11.1, 11.2, 11.5],
            high_values=[10.7, 11.0, 11.2, 11.15, 11.8, 12.2],
            low_values=[9.8, 10.2, 10.6, 10.8, 11.0, 11.3],
            close_values=[10.5, 10.9, 11.1, 10.95, 11.6, 11.9],
            ema20_values=[9.8, 10.0, 10.2, 10.4, 10.8, 11.0],
            upper_shadow_values=[0.10, 0.10, 0.08, 0.10, 0.08, 0.10],
        )

        annotations = researcher._compute_narrow_state_features(frame)

        self.assertTrue(bool(annotations["narrow_uptrend_state"][-1]))
        self.assertEqual(int(annotations["narrow_uptrend_run_length"][-1]), 2)
        self.assertAlmostEqual(float(annotations["narrow_state_bear_ratio"][-1]), 0.20, places=6)
        self.assertAlmostEqual(float(annotations["narrow_state_ema20_above_ratio"][-1]), 1.0, places=6)
        self.assertEqual(int(annotations["narrow_state_max_consecutive_bear_bars"][-1]), 1)

    def test_narrow_uptrend_state_false_when_bear_ratio_exceeds_limit(self) -> None:
        researcher = make_annotation_researcher(
            config=BullFlagNarrowTrendStrategyConfig(narrow_trend_lookback_bars=5)
        )
        frame = make_narrow_state_frame(
            open_values=[10.0, 10.4, 10.8, 11.1, 11.5, 11.8],
            high_values=[10.7, 11.0, 11.2, 11.3, 11.9, 12.1],
            low_values=[9.8, 10.2, 10.6, 10.8, 11.0, 11.4],
            close_values=[10.5, 10.2, 11.1, 10.9, 11.2, 11.9],
            ema20_values=[9.8, 9.9, 10.0, 10.1, 10.2, 10.3],
            upper_shadow_values=[0.10] * 6,
        )

        annotations = researcher._compute_narrow_state_features(frame)

        self.assertFalse(bool(annotations["narrow_uptrend_state"][-1]))
        self.assertAlmostEqual(float(annotations["narrow_state_bear_ratio"][-1]), 0.60, places=6)

    def test_run_end_is_flag_peak_and_next_bar_starts_flag(self) -> None:
        dates = pd.date_range("2025-01-01", periods=11, freq="B")
        annotated = annotate_case(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.984, 14.976, 14.968, 14.960, 15.6],
            pivot_low_positions=[],
            pivot_high_positions=[],
            narrow_state_positions=[0, 1, 2, 3, 4, 5],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertEqual(pd.Timestamp(signal_row["flag_peak_date"]), dates[5])
        self.assertEqual(pd.Timestamp(signal_row["flag_start_date"]), dates[6])
        self.assertEqual(pd.Timestamp(signal_row["left_state_start_date"]), dates[0])
        self.assertEqual(pd.Timestamp(signal_row["left_state_end_date"]), dates[5])
        self.assertEqual(int(signal_row["left_state_bars"]), 6)

    def test_flagpole_start_comes_from_lowest_pivot_low_before_run_end(self) -> None:
        dates = pd.date_range("2025-01-01", periods=13, freq="B")
        annotated = annotate_case(
            [10.0, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.984, 15.976, 15.968, 15.960, 16.6],
            pivot_low_positions=[0, 2],
            pivot_high_positions=[],
            narrow_state_positions=[5, 6, 7],
            signal_quality_positions=[12],
            close_gt_prev_high_positions=[12],
            low_values=[10.0, 10.4, 8.5, 11.8, 12.8, 13.8, 14.8, 15.8, 15.8, 15.8, 15.8, 15.8, 16.5],
        )

        signal_row = annotated.iloc[12]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertEqual(pd.Timestamp(signal_row["flagpole_start_date"]), dates[2])
        self.assertEqual(pd.Timestamp(signal_row["left_state_start_date"]), dates[5])

    def test_flagpole_start_falls_back_to_window_low_when_no_pivot_low_exists(self) -> None:
        dates = pd.date_range("2025-01-01", periods=13, freq="B")
        annotated = annotate_case(
            [10.0, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.984, 15.976, 15.968, 15.960, 16.6],
            pivot_low_positions=[],
            pivot_high_positions=[],
            narrow_state_positions=[5, 6, 7],
            signal_quality_positions=[12],
            close_gt_prev_high_positions=[12],
            low_values=[10.0, 8.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 15.8, 15.8, 15.8, 15.8, 16.5],
        )

        signal_row = annotated.iloc[12]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertEqual(pd.Timestamp(signal_row["flagpole_start_date"]), dates[1])

    def test_narrow_strategy_does_not_require_bullish_stack(self) -> None:
        annotated = annotate_case(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.984, 14.976, 14.968, 14.960, 15.6],
            pivot_low_positions=[],
            pivot_high_positions=[],
            narrow_state_positions=[0, 1, 2, 3, 4, 5],
            bullish_stack_false_positions=[5, 7, 10],
            signal_quality_positions=[10],
            close_gt_prev_high_positions=[10],
        )

        signal_row = annotated.iloc[-1]
        self.assertTrue(bool(signal_row["bull_flag_candidate"]))
        self.assertTrue(bool(signal_row["signal_candle"]))

    def test_inspect_signal_and_plot_show_left_state_fields(self) -> None:
        researcher, dates = make_manual_signal_researcher()
        researcher.add_research_outcomes()

        inspection = researcher.inspect_signal("AAA", dates[0], lookback=2, lookahead=3)
        figure = researcher.plot_signal_context("AAA", dates[0], lookback=2, lookahead=3)
        trace_names = {trace.name for trace in figure.data}

        self.assertEqual(inspection["summary"]["left_trend_mode"], "narrow_state")
        self.assertEqual(pd.Timestamp(inspection["summary"]["left_state_start_date"]), dates[0])
        self.assertEqual(pd.Timestamp(inspection["summary"]["left_state_end_date"]), dates[0])
        self.assertIn("Flagpole Start", trace_names)
        self.assertIn("Left State Start", trace_names)
        self.assertIn("Left State End / Flag Peak", trace_names)
        self.assertIn("EMA 20", trace_names)
        self.assertIsInstance(figure, go.Figure)


if __name__ == "__main__":
    unittest.main()
