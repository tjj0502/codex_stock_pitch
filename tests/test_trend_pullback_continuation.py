import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backtester import ExecutionCostModel, TradePlanBacktester
from score_system.trend_pullback_grid_search import run_trend_pullback_grid_search
from strategies.trend_pullback_continuation import (
    TrendPullbackContinuationResearcher,
    TrendPullbackStrategyConfig,
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


def make_manual_outcome_researcher(
    config: TrendPullbackStrategyConfig | None = None,
) -> tuple[TrendPullbackContinuationResearcher, pd.DatetimeIndex]:
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    panel = pd.concat(
        [
            make_stock_frame(
                "AAA",
                [99.0, 100.0, 89.0, 88.0, 87.0, 86.0],
                dates=dates,
                open_values=[99.0, 100.0, 100.0, 88.0, 87.0, 86.0],
                high_values=[100.0, 101.0, 101.0, 89.0, 88.0, 87.0],
                low_values=[98.0, 99.0, 89.0, 87.0, 86.0, 85.0],
            ),
            make_stock_frame(
                "BBB",
                [99.0, 100.0, 110.0, 112.0, 111.0, 110.0],
                dates=dates,
                open_values=[99.0, 100.0, 100.0, 112.0, 111.0, 110.0],
                high_values=[100.0, 101.0, 121.0, 113.0, 112.0, 111.0],
                low_values=[98.0, 99.0, 99.0, 111.0, 110.0, 109.0],
            ),
            make_stock_frame(
                "CCC",
                [99.0, 100.0, 104.0, 105.0, 106.0, 107.0],
                dates=dates,
                open_values=[99.0, 100.0, 100.0, 102.0, 103.0, 104.0],
                high_values=[100.0, 101.0, 105.0, 106.0, 107.0, 108.0],
                low_values=[98.0, 99.0, 99.0, 101.0, 102.0, 103.0],
            ),
        ],
        ignore_index=True,
    )
    researcher = TrendPullbackContinuationResearcher(
        panel,
        config=config
        or TrendPullbackStrategyConfig(
            min_trend_bars=3,
            max_holding_days=2,
            enable_time_stop=True,
        ),
    )
    prepared = researcher.stock_candle_df.copy()
    prepared["signal_candle"] = False
    prepared["follow_through_confirmed"] = False
    prepared["signal_before_trend_end"] = False
    prepared["reward_to_risk_ok"] = False
    prepared["reward_to_risk"] = np.nan
    prepared["entry_signal"] = False
    prepared["signal_quality_ok"] = True
    prepared["close_gt_prev_high"] = True
    prepared["post_trend_phase"] = True
    prepared["three_push_pullback"] = True
    prepared["lower_lows_confirmed"] = True
    prepared["lower_highs_confirmed"] = True
    prepared["downswing_contraction"] = True
    prepared["trend_start"] = prepared["date"].eq(dates[0])
    prepared["trend_end"] = prepared["date"].eq(dates[0])
    prepared["trend_start_date"] = dates[0]
    prepared["trend_end_date"] = dates[0]
    prepared["trend_high_date"] = dates[0]
    prepared["trend_high"] = 120.0
    prepared["trend_low"] = 80.0
    prepared["trend_midpoint_50"] = 100.0
    prepared["trend_length"] = 20
    prepared["pullback_bars"] = 6
    prepared["pullback_low"] = 90.0
    prepared["pullback_depth_pct"] = 0.10
    prepared["expected_upside_to_target"] = 0.20
    prepared["drop1_pct"] = 0.15
    prepared["drop2_pct"] = 0.10
    prepared["drop3_pct"] = 0.06
    prepared["signal_body_pct"] = 0.70
    prepared["signal_upper_shadow_pct"] = 0.10
    prepared["signal_lower_shadow_pct"] = 0.10
    prepared["push1_low_date"] = dates[0]
    prepared["push2_low_date"] = dates[1]
    prepared["push3_low_date"] = dates[2]
    prepared["push1_low"] = 95.0
    prepared["push2_low"] = 92.0
    prepared["push3_low"] = 90.0
    prepared["push1_rebound_high_date"] = dates[1]
    prepared["push2_rebound_high_date"] = dates[3]
    prepared["push1_rebound_high"] = 110.0
    prepared["push2_rebound_high"] = 105.0
    prepared["wedge_pivot_low_1"] = False
    prepared["wedge_pivot_low_2"] = False
    prepared["wedge_pivot_low_3"] = False
    prepared["wedge_pivot_high_1"] = False
    prepared["wedge_pivot_high_2"] = False
    prepared["follow_through_date"] = prepared.groupby("ticker", sort=False)["date"].shift(-1)
    prepared["follow_through_high"] = prepared.groupby("ticker", sort=False)["high"].shift(-1)
    prepared["follow_through_close"] = prepared.groupby("ticker", sort=False)["close"].shift(-1)
    prepared["entry_date_next"] = prepared.groupby("ticker", sort=False)["date"].shift(-2)
    prepared["entry_open_next"] = prepared.groupby("ticker", sort=False)["open"].shift(-2)
    prepared["signal_take_profit_price"] = 120.0
    prepared["signal_hard_stop_price"] = 90.0

    first_day_mask = prepared["date"].eq(dates[0])
    prepared.loc[first_day_mask, "signal_candle"] = True
    prepared.loc[first_day_mask, "follow_through_confirmed"] = True
    prepared.loc[first_day_mask, "reward_to_risk_ok"] = True
    prepared.loc[first_day_mask, "entry_signal"] = True
    prepared.loc[(prepared["ticker"] == "AAA") & first_day_mask, "reward_to_risk"] = 1.6
    prepared.loc[(prepared["ticker"] == "BBB") & first_day_mask, "reward_to_risk"] = 2.4
    prepared.loc[(prepared["ticker"] == "CCC") & first_day_mask, "reward_to_risk"] = 1.8
    prepared.loc[(prepared["ticker"] == "AAA") & prepared["date"].eq(dates[1]), "entry_signal"] = True

    researcher.stock_candle_df = prepared
    researcher.add_signals = lambda: researcher.stock_candle_df
    return researcher, dates


def make_annotation_researcher() -> TrendPullbackContinuationResearcher:
    dates = pd.date_range("2025-01-01", periods=12, freq="B")
    panel = make_stock_frame("AAA", np.linspace(10.0, 21.0, len(dates)), dates=dates)
    return TrendPullbackContinuationResearcher(
        panel,
        config=TrendPullbackStrategyConfig(min_trend_bars=3),
    )


def make_annotation_frame(
    trend_values: list[float],
    pullback_values: list[float],
    *,
    pivot_low_positions: list[int],
    pivot_high_positions: list[int],
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
) -> pd.DataFrame:
    signal_quality_positions = signal_quality_positions or []
    close_gt_prev_high_positions = close_gt_prev_high_positions or []
    total_values = np.asarray(trend_values + pullback_values, dtype=float)
    dates = pd.date_range("2025-01-01", periods=len(total_values), freq="B")
    trend_length = len(trend_values)

    frame = pd.DataFrame(
        {
            "date": dates,
            "high": total_values,
            "low": total_values,
            "close": total_values,
            "bullish_stack": [True] * trend_length + [False] * len(pullback_values),
            "pivot_high": False,
            "pivot_low": False,
            "signal_quality_ok": False,
            "close_gt_prev_high": False,
        }
    )
    for position in pivot_low_positions:
        frame.loc[trend_length + position, "pivot_low"] = True
    for position in pivot_high_positions:
        frame.loc[trend_length + position, "pivot_high"] = True
    for position in signal_quality_positions:
        frame.loc[trend_length + position, "signal_quality_ok"] = True
    for position in close_gt_prev_high_positions:
        frame.loc[trend_length + position, "close_gt_prev_high"] = True
    return frame


def annotate_case(
    trend_values: list[float],
    pullback_values: list[float],
    *,
    pivot_low_positions: list[int],
    pivot_high_positions: list[int],
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
) -> pd.DataFrame:
    researcher = make_annotation_researcher()
    frame = make_annotation_frame(
        trend_values,
        pullback_values,
        pivot_low_positions=pivot_low_positions,
        pivot_high_positions=pivot_high_positions,
        signal_quality_positions=signal_quality_positions,
        close_gt_prev_high_positions=close_gt_prev_high_positions,
    )
    annotations = researcher._annotate_ticker_context(frame)
    annotated = frame.copy()
    for column, values in annotations.items():
        annotated[column] = values
    return annotated


def make_absolute_annotation_frame(
    values: list[float],
    *,
    bullish_stack_until: int,
    pivot_low_positions: list[int] | None = None,
    pivot_high_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
) -> pd.DataFrame:
    pivot_low_positions = pivot_low_positions or []
    pivot_high_positions = pivot_high_positions or []
    signal_quality_positions = signal_quality_positions or []
    close_gt_prev_high_positions = close_gt_prev_high_positions or []

    total_values = np.asarray(values, dtype=float)
    dates = pd.date_range("2025-01-01", periods=len(total_values), freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "high": total_values,
            "low": total_values,
            "close": total_values,
            "bullish_stack": [idx <= bullish_stack_until for idx in range(len(total_values))],
            "pivot_high": False,
            "pivot_low": False,
            "signal_quality_ok": False,
            "close_gt_prev_high": False,
        }
    )
    for position in pivot_low_positions:
        frame.loc[position, "pivot_low"] = True
    for position in pivot_high_positions:
        frame.loc[position, "pivot_high"] = True
    for position in signal_quality_positions:
        frame.loc[position, "signal_quality_ok"] = True
    for position in close_gt_prev_high_positions:
        frame.loc[position, "close_gt_prev_high"] = True
    return frame


def annotate_absolute_case(
    values: list[float],
    *,
    bullish_stack_until: int,
    pivot_low_positions: list[int] | None = None,
    pivot_high_positions: list[int] | None = None,
    signal_quality_positions: list[int] | None = None,
    close_gt_prev_high_positions: list[int] | None = None,
) -> pd.DataFrame:
    researcher = make_annotation_researcher()
    frame = make_absolute_annotation_frame(
        values,
        bullish_stack_until=bullish_stack_until,
        pivot_low_positions=pivot_low_positions,
        pivot_high_positions=pivot_high_positions,
        signal_quality_positions=signal_quality_positions,
        close_gt_prev_high_positions=close_gt_prev_high_positions,
    )
    annotations = researcher._annotate_ticker_context(frame)
    annotated = frame.copy()
    for column, values in annotations.items():
        annotated[column] = values
    return annotated


def make_signal_generation_researcher(
    *,
    trend_end_offset: int = 0,
    take_profit_fraction_of_trend_move: float = 0.50,
) -> tuple[TrendPullbackContinuationResearcher, pd.DatetimeIndex]:
    dates = pd.date_range("2025-02-03", periods=4, freq="B")
    panel = make_stock_frame(
        "AAA",
        [11.0, 11.5, 11.4, 11.3],
        dates=dates,
        open_values=[10.9, 11.1, 11.4, 11.3],
        high_values=[11.2, 12.0, 11.6, 11.5],
        low_values=[10.8, 11.0, 11.2, 11.1],
    )
    researcher = TrendPullbackContinuationResearcher(
        panel,
        config=TrendPullbackStrategyConfig(
            min_trend_bars=3,
            min_reward_r=1.5,
            stop_buffer_pct=0.0,
            take_profit_fraction_of_trend_move=take_profit_fraction_of_trend_move,
        ),
    )
    prepared = researcher.stock_candle_df.drop(
        columns=[
            column
            for column in list(researcher.SIGNAL_COLUMNS) + list(researcher.OUTCOME_COLUMNS)
            if column in researcher.stock_candle_df.columns
        ]
    ).copy()
    prepared["signal_candle"] = prepared["date"].eq(dates[0])
    prepared["trend_end_date"] = dates[trend_end_offset]
    prepared["trend_high"] = 14.5
    prepared["pullback_low"] = 10.5

    researcher.stock_candle_df = prepared
    researcher.add_features = lambda: researcher.stock_candle_df
    return researcher, dates


class TrendPullbackContinuationResearcherTest(unittest.TestCase):
    ZERO_COSTS = ExecutionCostModel(
        commission_rate=0.0,
        min_commission=0.0,
        stamp_duty_rate=0.0,
        transfer_fee_rate=0.0,
        half_spread_bps=0.0,
    )

    def test_add_features_freezes_trend_high_inside_completed_ma_stack(self) -> None:
        dates = pd.date_range("2025-01-01", periods=220, freq="B")
        closes = np.concatenate(
            [
                np.linspace(20, 100, 170),
                [101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0],
                np.linspace(91.0, 88.0, 40),
            ]
        )
        panel = make_stock_frame("AAA", closes, dates=dates, open_values=closes - 0.3)

        researcher = TrendPullbackContinuationResearcher(
            panel,
            config=TrendPullbackStrategyConfig(min_trend_bars=5),
        )
        output = researcher.stock_candle_df
        post_trend = output[output["post_trend_phase"]].reset_index(drop=True)

        self.assertFalse(post_trend.empty)
        first_post_trend = post_trend.iloc[0]
        self.assertLess(first_post_trend["trend_high_date"], first_post_trend["trend_end_date"])
        self.assertGreater(first_post_trend["trend_high"], first_post_trend["close"])

    def test_pullback_phase_starts_from_trend_high_not_trend_end(self) -> None:
        annotated = annotate_absolute_case(
            [10.0, 12.0, 18.0, 15.0, 14.0, 13.0, 12.0],
            bullish_stack_until=4,
        )

        first_pullback = annotated.iloc[3]
        self.assertTrue(bool(first_pullback["post_trend_phase"]))
        self.assertEqual(first_pullback["pullback_bars"], 1)
        self.assertEqual(first_pullback["trend_high_date"], annotated.loc[2, "date"])
        self.assertEqual(first_pullback["trend_end_date"], annotated.loc[4, "date"])

    def test_trend_anchor_prefers_lowest_pivot_low_within_lookback_window(self) -> None:
        annotated = annotate_absolute_case(
            [12.0, 9.0, 14.0, 11.0, 18.0, 17.0, 16.0],
            bullish_stack_until=5,
            pivot_low_positions=[1, 3],
        )

        anchor_row = annotated.iloc[5]
        self.assertTrue(bool(annotated.loc[1, "trend_start"]))
        self.assertEqual(anchor_row["trend_start_date"], annotated.loc[1, "date"])
        self.assertEqual(anchor_row["trend_low"], 9.0)
        self.assertEqual(anchor_row["trend_length"], 4)

    def test_trend_anchor_falls_back_to_window_low_when_no_pivot_low_exists(self) -> None:
        annotated = annotate_absolute_case(
            [9.0, 11.0, 13.0, 10.0, 18.0, 17.0, 16.0],
            bullish_stack_until=5,
        )

        anchor_row = annotated.iloc[5]
        self.assertEqual(anchor_row["trend_start_date"], annotated.loc[0, "date"])
        self.assertEqual(anchor_row["trend_low"], 9.0)
        self.assertEqual(anchor_row["trend_midpoint_50"], (18.0 + 9.0) / 2.0)

    def test_add_research_outcomes_and_trade_df_use_t_plus_2_entry_timing(self) -> None:
        researcher, dates = make_manual_outcome_researcher()
        researcher.add_research_outcomes()
        trade_df = researcher.add_trade_df()

        first_rows = researcher.stock_candle_df.groupby("ticker", sort=False).head(1).set_index("ticker")
        self.assertEqual(first_rows.loc["AAA", "exit_reason"], "hard_stop")
        self.assertAlmostEqual(first_rows.loc["AAA", "realized_open_to_open_return"], -0.12)
        self.assertEqual(first_rows.loc["BBB", "exit_reason"], "take_profit")
        self.assertAlmostEqual(first_rows.loc["BBB", "realized_open_to_open_return"], 0.12)
        self.assertEqual(first_rows.loc["CCC", "exit_reason"], "time_stop")
        self.assertAlmostEqual(first_rows.loc["CCC", "realized_open_to_open_return"], 0.03)

        aaa_rows = researcher.stock_candle_df[researcher.stock_candle_df["ticker"] == "AAA"].reset_index(drop=True)
        self.assertTrue(bool(aaa_rows.loc[0, "entry_signal_executed"]))
        self.assertTrue(bool(aaa_rows.loc[1, "entry_signal_suppressed"]))
        self.assertEqual(trade_df["entry_date"].tolist(), [dates[2], dates[2], dates[2]])
        self.assertEqual(trade_df["ticker"].tolist(), ["AAA", "BBB", "CCC"])

    def test_get_next_session_candidates_uses_follow_through_bar(self) -> None:
        researcher, dates = make_manual_outcome_researcher()
        candidates = researcher.get_next_session_candidates(
            as_of_date=dates[1],
            next_trade_date=dates[2],
        )

        self.assertEqual(candidates["ticker"].tolist(), ["BBB", "CCC", "AAA"])
        self.assertTrue(candidates["follow_through_date"].eq(dates[1]).all())
        self.assertTrue(candidates["planned_entry_date"].eq(dates[2]).all())
        self.assertTrue(candidates["entry_reference_price"].eq(100.0).all())
        self.assertTrue(candidates["planned_hard_stop_price"].eq(90.0).all())
        self.assertTrue(candidates["planned_take_profit_price"].eq(120.0).all())
        self.assertTrue(candidates["signal_before_trend_end"].eq(False).all())

    def test_add_signals_blocks_entries_before_trend_end(self) -> None:
        researcher, dates = make_signal_generation_researcher(trend_end_offset=1)
        output = researcher.add_signals()

        signal_row = output.loc[output["date"].eq(dates[0])].iloc[0]
        self.assertTrue(bool(signal_row["signal_candle"]))
        self.assertTrue(bool(signal_row["follow_through_confirmed"]))
        self.assertTrue(bool(signal_row["signal_before_trend_end"]))
        self.assertTrue(bool(signal_row["reward_to_risk_ok"]))
        self.assertFalse(bool(signal_row["entry_signal"]))

    def test_add_signals_allows_entries_on_or_after_trend_end(self) -> None:
        researcher, dates = make_signal_generation_researcher(trend_end_offset=0)
        output = researcher.add_signals()

        signal_row = output.loc[output["date"].eq(dates[0])].iloc[0]
        self.assertFalse(bool(signal_row["signal_before_trend_end"]))
        self.assertTrue(bool(signal_row["entry_signal"]))

        candidates = researcher.get_next_session_candidates(
            as_of_date=dates[1],
            next_trade_date=dates[2],
        )
        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])
        self.assertFalse(bool(candidates["signal_before_trend_end"].iat[0]))

    def test_take_profit_fraction_half_targets_halfway_to_trend_high(self) -> None:
        researcher, dates = make_signal_generation_researcher(
            trend_end_offset=0,
            take_profit_fraction_of_trend_move=0.50,
        )
        output = researcher.add_signals()

        signal_row = output.loc[output["date"].eq(dates[0])].iloc[0]
        self.assertAlmostEqual(signal_row["entry_reference_price"], 11.5)
        self.assertAlmostEqual(signal_row["signal_take_profit_price"], 13.0)

    def test_take_profit_fraction_one_matches_prior_trend_high_target(self) -> None:
        researcher, dates = make_signal_generation_researcher(
            trend_end_offset=0,
            take_profit_fraction_of_trend_move=1.0,
        )
        output = researcher.add_signals()

        signal_row = output.loc[output["date"].eq(dates[0])].iloc[0]
        self.assertAlmostEqual(signal_row["signal_take_profit_price"], 14.5)

    def test_trade_plan_backtester_accepts_researcher_trade_df(self) -> None:
        researcher, dates = make_manual_outcome_researcher()
        backtester = TradePlanBacktester(
            researcher.stock_candle_df,
            researcher=researcher,
            initial_capital=300.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()
        strategy_trades = results["strategy_trades"].set_index("ticker")

        self.assertEqual(results["summary"]["planned_trade_count"], 3)
        self.assertEqual(results["summary"]["entered_trade_count"], 3)
        self.assertEqual(results["summary"]["closed_trade_count"], 3)
        self.assertEqual(strategy_trades.loc["AAA", "actual_entry_date"], dates[2])
        self.assertEqual(strategy_trades.loc["BBB", "actual_exit_date"], dates[3])
        self.assertAlmostEqual(float(results["summary"]["total_return"]), 0.01)

    def test_run_trend_pullback_grid_search_returns_curves_summary_and_figure(self) -> None:
        dates = pd.date_range("2025-01-01", periods=220, freq="B")
        panel = pd.concat(
            [
                make_stock_frame("AAA", np.linspace(20.0, 80.0, len(dates)), dates=dates),
                make_stock_frame("BBB", np.linspace(30.0, 90.0, len(dates)), dates=dates),
            ],
            ignore_index=True,
        )

        results = run_trend_pullback_grid_search(
            panel,
            param_grid={"min_reward_r": [1.0, 1.5]},
            base_config=TrendPullbackStrategyConfig(min_trend_bars=5, max_holding_days=5, enable_time_stop=True),
            start_date=dates[120],
            end_date=dates[-1],
            backtester_kwargs={
                "initial_capital": 100_000.0,
                "board_lot_size": 1,
                "costs": self.ZERO_COSTS,
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

    def test_plot_signal_context_draws_trend_and_wedge_annotations(self) -> None:
        researcher, dates = make_manual_outcome_researcher()
        researcher.add_research_outcomes()
        inspection = researcher.inspect_signal("AAA", dates[0], lookback=2, lookahead=3)

        figure = researcher.plot_signal_context("AAA", dates[0], lookback=2, lookahead=3)
        trace_names = {trace.name for trace in figure.data}

        self.assertAlmostEqual(float(inspection["summary"]["trend_midpoint_50"]), 100.0)
        self.assertIn("Trend Start", trace_names)
        self.assertIn("Trend End", trace_names)
        self.assertIn("Trend Low", trace_names)
        self.assertIn("Trend Mid 50%", trace_names)
        self.assertIn("Setup Lows", trace_names)
        self.assertIn("Setup Highs", trace_names)
        self.assertIn("Wedge Lower", trace_names)
        self.assertIn("Wedge Upper", trace_names)

    def test_leg_compression_ignores_higher_intermediate_pivot_lows(self) -> None:
        annotated = annotate_case(
            [18.0, 19.0, 20.0],
            [16.0, 12.0, 15.0, 13.0, 14.0, 10.0, 14.0, 12.0, 13.0, 9.0, 14.0],
            pivot_low_positions=[1, 3, 5, 7, 9],
            pivot_high_positions=[2, 4, 6, 8],
        )

        last_row = annotated.iloc[-1]
        self.assertEqual(last_row["push1_low"], 12.0)
        self.assertEqual(last_row["push2_low"], 10.0)
        self.assertEqual(last_row["push3_low"], 9.0)
        self.assertEqual(last_row["push1_rebound_high"], 15.0)
        self.assertEqual(last_row["push2_rebound_high"], 14.0)
        self.assertTrue(bool(last_row["lower_lows_confirmed"]))

    def test_lower_low_without_intermediate_pivot_high_updates_same_leg(self) -> None:
        annotated = annotate_case(
            [18.0, 19.0, 20.0],
            [16.0, 12.0, 11.0, 15.0, 8.0, 14.0, 6.0, 13.0],
            pivot_low_positions=[1, 2, 4, 6],
            pivot_high_positions=[3, 5],
        )

        last_row = annotated.iloc[-1]
        self.assertEqual(last_row["push1_low"], 11.0)
        self.assertEqual(last_row["push2_low"], 8.0)
        self.assertEqual(last_row["push3_low"], 6.0)
        self.assertEqual(last_row["push1_rebound_high"], 15.0)
        self.assertEqual(last_row["push2_rebound_high"], 14.0)

    def test_rebound_highs_take_interval_maximum_confirmed_pivot_high(self) -> None:
        annotated = annotate_case(
            [18.0, 19.0, 20.0],
            [16.0, 12.0, 15.0, 14.0, 17.0, 10.0, 14.0, 13.0, 16.0, 9.0, 15.0],
            pivot_low_positions=[1, 5, 9],
            pivot_high_positions=[2, 4, 6, 8],
        )

        last_row = annotated.iloc[-1]
        self.assertEqual(last_row["push1_rebound_high"], 17.0)
        self.assertEqual(last_row["push2_rebound_high"], 16.0)

    def test_fourth_lower_leg_invalidates_setup_instead_of_rolling(self) -> None:
        annotated = annotate_case(
            [28.0, 29.0, 30.0],
            [24.0, 15.0, 22.0, 14.0, 18.0, 13.0, 17.0, 12.0, 26.0],
            pivot_low_positions=[1, 3, 5, 7],
            pivot_high_positions=[2, 4, 6],
            signal_quality_positions=[8],
            close_gt_prev_high_positions=[8],
        )

        setup_row = annotated.iloc[-2]
        invalidated_row = annotated.iloc[-1]
        self.assertTrue(bool(setup_row["three_push_pullback"]))
        self.assertTrue(pd.isna(invalidated_row["push1_low"]))
        self.assertTrue(pd.isna(invalidated_row["push2_low"]))
        self.assertTrue(pd.isna(invalidated_row["push3_low"]))
        self.assertFalse(bool(invalidated_row["three_push_pullback"]))
        self.assertFalse(bool(invalidated_row["signal_candle"]))

    def test_downswing_contraction_uses_trend_high_then_rebound_high_anchors(self) -> None:
        annotated = annotate_case(
            [28.0, 29.0, 30.0],
            [24.0, 15.0, 22.0, 14.0, 18.0, 13.0, 17.0],
            pivot_low_positions=[1, 3, 5],
            pivot_high_positions=[2, 4],
        )

        last_row = annotated.iloc[-1]
        self.assertAlmostEqual(last_row["drop1_pct"], 1.0)
        self.assertAlmostEqual(last_row["drop2_pct"], 22.0 / 14.0 - 1.0)
        self.assertAlmostEqual(last_row["drop3_pct"], 18.0 / 13.0 - 1.0)
        self.assertTrue(bool(last_row["downswing_contraction"]))
        self.assertTrue(bool(last_row["three_push_pullback"]))


if __name__ == "__main__":
    unittest.main()
