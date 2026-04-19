import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategies.bull_flag_exit_variants import (
    BullFlagBreakevenAfterTp1Researcher,
    BullFlagCloseRetraceAfterTp1Researcher,
    BullFlagDynamicExitConfig,
    BullFlagMaTrailAfterTp1Researcher,
    BullFlagMaTrailVolumeFailureResearcher,
    BullFlagStructureTrailAfterTp1Researcher,
    BullFlagTrailingAfterTp1Researcher,
    BullFlagTrailingStopOnlyAfterTp1Researcher,
    BullFlagTrailingVolumeFailureResearcher,
    BullFlagVolumeFailureAfterTp1Researcher,
)


def make_stock_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    dates: pd.DatetimeIndex,
    open_values: list[float] | np.ndarray,
    high_values: list[float] | np.ndarray,
    low_values: list[float] | np.ndarray,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
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
            "weight": 1.0,
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


def make_dynamic_exit_researcher(
    researcher_cls,
    *,
    open_values: list[float],
    high_values: list[float],
    low_values: list[float],
    close_values: list[float],
    config: BullFlagDynamicExitConfig | None = None,
) -> tuple[object, pd.DatetimeIndex]:
    dates = pd.date_range("2025-01-01", periods=len(close_values), freq="B")
    panel = make_stock_frame(
        "AAA",
        close_values,
        dates=dates,
        open_values=open_values,
        high_values=high_values,
        low_values=low_values,
    )
    researcher = researcher_cls(
        panel,
        config=config
        or BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=10,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            breakeven_buffer_pct=0.0,
            trailing_stop_fraction_of_flagpole=0.25,
        ),
    )
    prepared = researcher.stock_candle_df.copy()
    removable_columns = (
        researcher.SIGNAL_COLUMNS
        + researcher.OUTCOME_COLUMNS
        + getattr(researcher, "DYNAMIC_FEATURE_COLUMNS", [])
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


class BullFlagExitVariantTests(unittest.TestCase):
    def test_dynamic_exit_variants_keep_entry_signal_definition(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=10,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
        )
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
            config=config,
        )
        baseline_like, _ = make_dynamic_exit_researcher(
            BullFlagBreakevenAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
            config=config,
        )

        trailing_signals = researcher.add_signals()
        breakeven_signals = baseline_like.add_signals()

        self.assertEqual(
            trailing_signals["entry_signal"].fillna(False).tolist(),
            breakeven_signals["entry_signal"].fillna(False).tolist(),
        )
        self.assertTrue(bool(trailing_signals.loc[trailing_signals["date"] == dates[0], "entry_signal"].iat[0]))

    def test_breakeven_variant_raises_stop_after_tp1(self) -> None:
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagBreakevenAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 109.0, 103.0, 105.0, 106.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 110.0, 106.0, 107.0, 108.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 103.5, 102.0, 104.0, 105.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 109.0, 104.0, 106.0, 107.0],
        )

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(signal_row["entry_signal"]))
        self.assertTrue(bool(signal_row["tp1_reached"]))
        self.assertAlmostEqual(float(signal_row["tp1_price"]), 111.5, places=2)
        self.assertEqual(pd.Timestamp(signal_row["tp1_hit_date"]), dates[3])
        self.assertAlmostEqual(float(signal_row["post_tp1_stop_price"]), 104.0, places=2)
        self.assertEqual(str(signal_row["exit_reason"]), "breakeven_stop")
        self.assertEqual(pd.Timestamp(signal_row["exit_signal_date"]), dates[4])
        self.assertEqual(pd.Timestamp(signal_row["exit_date_next"]), dates[5])

    def test_trailing_variant_uses_high_water_mark_after_tp1(self) -> None:
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
        )

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(signal_row["tp1_reached"]))
        self.assertEqual(pd.Timestamp(signal_row["tp1_hit_date"]), dates[3])
        self.assertAlmostEqual(float(signal_row["post_tp1_stop_price"]), 109.0, places=2)
        self.assertEqual(str(signal_row["exit_reason"]), "trailing_stop")
        self.assertEqual(pd.Timestamp(signal_row["exit_signal_date"]), dates[5])
        self.assertEqual(pd.Timestamp(signal_row["exit_date_next"]), dates[6])

    def test_trailing_monitor_positions_uses_dynamic_exit_schedule(self) -> None:
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
        )

        positions = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "entry_date": dates[2],
                    "entry_price": 104.0,
                    "signal_date": dates[0],
                    "shares": 100,
                }
            ]
        )

        same_day_monitor = researcher.monitor_positions(positions, as_of_date=dates[5], next_trade_date=dates[6])
        same_day_row = same_day_monitor.iloc[0]
        self.assertEqual(str(same_day_row["action"]), "exit_next_open")
        self.assertEqual(pd.Timestamp(same_day_row["planned_exit_date"]), dates[6])
        self.assertEqual(str(same_day_row["exit_reason"]), "trailing_stop")
        self.assertAlmostEqual(float(same_day_row["active_protective_stop"]), 109.0, places=2)

        overdue_monitor = researcher.monitor_positions(positions, as_of_date=dates[6], next_trade_date=dates[7])
        overdue_row = overdue_monitor.iloc[0]
        self.assertEqual(str(overdue_row["action"]), "exit_overdue")
        self.assertEqual(pd.Timestamp(overdue_row["planned_exit_date"]), dates[6])
        self.assertEqual(str(overdue_row["exit_reason"]), "trailing_stop")

    def test_trailing_stop_only_matches_trailing_before_tp1(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=10,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
        )
        trailing, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 104.5, 104.0, 103.5, 103.0, 102.5],
            high_values=[103.0, 105.0, 108.0, 109.0, 105.0, 104.0, 103.5, 103.0],
            low_values=[100.0, 102.5, 103.0, 103.2, 102.0, 101.5, 101.0, 100.5],
            close_values=[102.0, 104.0, 106.0, 104.0, 103.0, 102.5, 102.0, 101.5],
            config=config,
        )
        stop_only, _ = make_dynamic_exit_researcher(
            BullFlagTrailingStopOnlyAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 104.5, 104.0, 103.5, 103.0, 102.5],
            high_values=[103.0, 105.0, 108.0, 109.0, 105.0, 104.0, 103.5, 103.0],
            low_values=[100.0, 102.5, 103.0, 103.2, 102.0, 101.5, 101.0, 100.5],
            close_values=[102.0, 104.0, 106.0, 104.0, 103.0, 102.5, 102.0, 101.5],
            config=config,
        )

        trailing_row = trailing.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        stop_only_row = stop_only.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]

        self.assertFalse(bool(trailing_row["tp1_reached"]))
        self.assertFalse(bool(stop_only_row["tp1_reached"]))
        self.assertEqual(str(trailing_row["exit_reason"]), str(stop_only_row["exit_reason"]))
        self.assertTrue(pd.isna(trailing_row["exit_date_next"]))
        self.assertTrue(pd.isna(stop_only_row["exit_date_next"]))

    def test_trailing_stop_only_disables_take_profit_after_tp1(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
            measured_move_fraction=0.75,
        )
        open_values = [101.0, 103.0, 104.0, 108.0, 110.0, 111.0, 112.0, 111.0, 109.0]
        high_values = [103.0, 105.0, 108.0, 112.0, 113.0, 121.0, 124.0, 118.0, 111.0]
        low_values = [100.0, 102.5, 103.0, 106.0, 109.0, 110.0, 112.0, 108.0, 104.0]
        close_values = [102.0, 104.0, 106.0, 110.0, 112.0, 118.0, 114.0, 109.0, 106.0]

        trailing, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )
        stop_only, _ = make_dynamic_exit_researcher(
            BullFlagTrailingStopOnlyAfterTp1Researcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )

        trailing_row = trailing.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        stop_only_row = stop_only.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(trailing_row["tp1_reached"]))
        self.assertTrue(bool(stop_only_row["tp1_reached"]))
        self.assertEqual(str(trailing_row["exit_reason"]), "take_profit")
        self.assertEqual(str(stop_only_row["exit_reason"]), "trailing_stop")
        self.assertEqual(pd.Timestamp(trailing_row["exit_signal_date"]), dates[5])
        self.assertEqual(pd.Timestamp(stop_only_row["exit_signal_date"]), dates[6])

    def test_trailing_stop_only_disables_time_stop_after_tp1(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=4,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
            measured_move_fraction=1.50,
        )
        open_values = [101.0, 103.0, 104.0, 108.0, 112.0, 116.0, 117.0, 118.0, 119.0]
        high_values = [103.0, 105.0, 108.0, 112.0, 120.0, 121.0, 122.0, 123.0, 124.0]
        low_values = [100.0, 102.5, 103.0, 106.0, 111.0, 116.0, 117.0, 118.0, 119.0]
        close_values = [102.0, 104.0, 106.0, 110.0, 118.0, 119.0, 120.0, 121.0, 122.0]

        trailing, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )
        stop_only, _ = make_dynamic_exit_researcher(
            BullFlagTrailingStopOnlyAfterTp1Researcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )

        trailing_row = trailing.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        stop_only_row = stop_only.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(trailing_row["tp1_reached"]))
        self.assertTrue(bool(stop_only_row["tp1_reached"]))
        self.assertEqual(str(trailing_row["exit_reason"]), "time_stop")
        self.assertEqual(str(stop_only_row["exit_reason"]), "open_position")
        self.assertTrue(pd.isna(stop_only_row["exit_date_next"]))

    def test_trailing_stop_only_same_day_tp1_and_target_still_take_profit(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
        )
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagTrailingStopOnlyAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 120.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
            config=config,
        )

        signal_row = researcher.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        self.assertTrue(bool(signal_row["tp1_reached"]))
        self.assertEqual(pd.Timestamp(signal_row["tp1_hit_date"]), dates[3])
        self.assertEqual(str(signal_row["exit_reason"]), "take_profit")
        self.assertEqual(pd.Timestamp(signal_row["exit_signal_date"]), dates[3])

    def test_trailing_stop_only_keeps_entry_trade_timing(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
        )
        trailing, dates = make_dynamic_exit_researcher(
            BullFlagTrailingAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
            config=config,
        )
        stop_only, _ = make_dynamic_exit_researcher(
            BullFlagTrailingStopOnlyAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 110.0, 109.0, 108.5, 109.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 114.0, 113.0, 110.0, 111.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 109.0, 108.0, 107.5, 108.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 113.0, 109.0, 108.5, 109.5],
            config=config,
        )

        trailing_row = trailing.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        stop_only_row = stop_only.add_research_outcomes().loc[lambda df: df["date"] == dates[0]].iloc[0]
        self.assertEqual(pd.Timestamp(trailing_row["date"]), pd.Timestamp(stop_only_row["date"]))
        self.assertEqual(pd.Timestamp(trailing_row["entry_date_next"]), pd.Timestamp(stop_only_row["entry_date_next"]))
        self.assertAlmostEqual(float(trailing_row["entry_open_next"]), float(stop_only_row["entry_open_next"]), places=8)

    def test_ma_trail_only_activates_after_tp1(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            ma_trail_window=10,
            ma_exit_buffer_pct=0.0,
        )
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagMaTrailAfterTp1Researcher,
            open_values=[101, 103, 104, 108, 109, 110, 111, 112, 113, 114, 108, 107],
            high_values=[103, 105, 108, 112, 113, 114, 115, 116, 117, 118, 109, 108],
            low_values=[100, 102.5, 103, 106, 108, 109, 110, 111, 112, 113, 107, 106],
            close_values=[102, 104, 106, 110, 111, 112, 113, 114, 115, 116, 107, 106],
            config=config,
        )

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertTrue(bool(signal_row["tp1_reached"]))
        self.assertEqual(str(signal_row["exit_reason"]), "ma_trail_exit")
        self.assertEqual(pd.Timestamp(signal_row["exit_signal_date"]), dates[10])
        self.assertTrue(pd.notna(signal_row["ma_trail_value"]))

    def test_structure_trail_uses_completed_bars_only(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            structure_trail_lookback=3,
            structure_trail_buffer_pct=0.0,
        )
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagStructureTrailAfterTp1Researcher,
            open_values=[101, 103, 104, 108, 110, 111, 112, 113],
            high_values=[103, 105, 108, 112, 114, 115, 116, 117],
            low_values=[100, 102.5, 103, 106, 109, 110, 108.5, 106.0],
            close_values=[102, 104, 106, 110, 113, 114, 109.0, 107.0],
            config=config,
        )

        inspection = researcher.inspect_signal("AAA", dates[0], lookback=2, lookahead=6)
        exit_path = inspection["exit_path"]
        row = exit_path[exit_path["date"] == dates[6]].iloc[0]

        self.assertAlmostEqual(float(row["structure_trail_value"]), 106.0, places=2)
        self.assertEqual(str(inspection["summary"]["exit_reason"]), "structure_trail_stop")

    def test_volume_failure_requires_high_relative_volume_and_failure_bar(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=40,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            vol_failure_threshold=1.8,
            measured_move_fraction=1.50,
        )
        open_values = [100.0 + index for index in range(22)]
        high_values = [value + 2.0 for value in open_values]
        low_values = [value - 1.0 for value in open_values]
        close_values = [value + 1.0 for value in open_values]
        open_values[-2:] = [130.0, 121.0]
        high_values[-2:] = [131.0, 122.0]
        low_values[-2:] = [116.0, 118.0]
        close_values[-2:] = [117.0, 120.0]
        volume_values = [1000] * 20 + [10000, 1200]
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagVolumeFailureAfterTp1Researcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )
        researcher.stock_candle_df["volume"] = volume_values
        researcher.stock_candle_df["turnover"] = researcher.stock_candle_df["volume"] * researcher.stock_candle_df["close"]

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertEqual(str(signal_row["exit_reason"]), "volume_failure_exit")
        self.assertTrue(float(signal_row["relative_volume_20_signal"]) >= 1.8)
        self.assertEqual(pd.Timestamp(signal_row["exit_signal_date"]), dates[20])

    def test_close_retrace_uses_highest_close_not_intraday_high(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            close_retrace_pct=0.05,
            measured_move_fraction=1.50,
        )
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagCloseRetraceAfterTp1Researcher,
            open_values=[101, 103, 104, 108, 110, 111, 112, 113],
            high_values=[103, 105, 108, 112, 120, 118, 117, 116],
            low_values=[100, 102.5, 103, 106, 109, 110, 106, 105],
            close_values=[102, 104, 106, 110, 113, 112, 107, 106],
            config=config,
        )

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertEqual(str(signal_row["exit_reason"]), "close_retrace_exit")
        self.assertAlmostEqual(float(signal_row["highest_close_since_tp1"]), 113.0, places=2)
        self.assertAlmostEqual(float(signal_row["close_retrace_threshold"]), 113.0 * 0.95, places=2)

    def test_overlay_prioritizes_protective_stop_over_close_trigger(self) -> None:
        config = BullFlagDynamicExitConfig(
            ma_windows=(2, 3, 4),
            pivot_window=1,
            max_holding_days=20,
            enable_time_stop=True,
            tp1_fraction_of_target=0.5,
            trailing_stop_fraction_of_flagpole=0.25,
            vol_failure_threshold=1.5,
            measured_move_fraction=1.50,
        )
        open_values = [100.0 + index for index in range(22)]
        high_values = [value + 2.0 for value in open_values]
        low_values = [value - 1.0 for value in open_values]
        close_values = [value + 1.0 for value in open_values]
        open_values[-2:] = [130.0, 121.0]
        high_values[-2:] = [132.0, 122.0]
        low_values[-2:] = [105.0, 104.0]
        close_values[-2:] = [106.0, 105.0]
        volume_values = [1000] * 20 + [5000, 1200]
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagTrailingVolumeFailureResearcher,
            open_values=open_values,
            high_values=high_values,
            low_values=low_values,
            close_values=close_values,
            config=config,
        )
        researcher.stock_candle_df["volume"] = volume_values
        researcher.stock_candle_df["turnover"] = researcher.stock_candle_df["volume"] * researcher.stock_candle_df["close"]

        outcome_frame = researcher.add_research_outcomes()
        signal_row = outcome_frame[outcome_frame["date"] == dates[0]].iloc[0]

        self.assertEqual(str(signal_row["exit_reason"]), "trailing_stop")

    def test_dynamic_trade_df_keeps_variant_metadata(self) -> None:
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagBreakevenAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 109.0, 103.0, 105.0, 106.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 110.0, 106.0, 107.0, 108.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 103.5, 102.0, 104.0, 105.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 109.0, 104.0, 106.0, 107.0],
        )

        trade_df = researcher.add_trade_df()
        trade_row = trade_df.iloc[0]

        self.assertEqual(trade_df.attrs["strategy_name"], "bull_flag_breakeven_after_tp1")
        self.assertIn("tp1_price", trade_df.columns)
        self.assertIn("post_tp1_stop_price", trade_df.columns)
        self.assertEqual(pd.Timestamp(trade_row["signal_date"]), dates[0])
        self.assertEqual(str(trade_row["exit_reason"]), "breakeven_stop")

    def test_plot_signal_context_adds_tp1_trace(self) -> None:
        researcher, dates = make_dynamic_exit_researcher(
            BullFlagBreakevenAfterTp1Researcher,
            open_values=[101.0, 103.0, 104.0, 108.0, 109.0, 103.0, 105.0, 106.0],
            high_values=[103.0, 105.0, 108.0, 112.0, 110.0, 106.0, 107.0, 108.0],
            low_values=[100.0, 102.5, 103.0, 106.0, 103.5, 102.0, 104.0, 105.0],
            close_values=[102.0, 104.0, 106.0, 110.0, 109.0, 104.0, 106.0, 107.0],
        )

        researcher.add_research_outcomes()
        inspection = researcher.inspect_signal("AAA", dates[0], lookback=2, lookahead=4)
        figure = researcher.plot_signal_context("AAA", dates[0], lookback=2, lookahead=4)
        trace_names = {trace.name for trace in figure.data}

        self.assertAlmostEqual(float(inspection["summary"]["tp1_price"]), 111.5, places=2)
        self.assertIn("TP1", trace_names)
        self.assertIn("Active Stop", trace_names)
        self.assertIsInstance(figure, go.Figure)


if __name__ == "__main__":
    unittest.main()
