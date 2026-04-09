import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from blue_chip_range_reversion import BlueChipRangeReversionResearcher, RangeStrategyConfig


def make_stock_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    dates: pd.DatetimeIndex | None = None,
    open_values: list[float] | np.ndarray | None = None,
    high_values: list[float] | np.ndarray | None = None,
    low_values: list[float] | np.ndarray | None = None,
    volume_values: list[float] | np.ndarray | None = None,
    weight: float = 1.0,
    constituent_trade_date: str = "20250131",
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if dates is None:
        dates = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    if open_values is None:
        open_values = closes - 0.2
    if high_values is None:
        high_values = np.maximum(open_values, closes) + 0.5
    if low_values is None:
        low_values = np.minimum(open_values, closes) - 0.5
    if volume_values is None:
        volume_values = 1_000 + np.arange(len(closes)) * 10

    open_values = np.asarray(open_values, dtype=float)
    high_values = np.asarray(high_values, dtype=float)
    low_values = np.asarray(low_values, dtype=float)
    volume_values = np.asarray(volume_values, dtype=float)
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
            "constituent_trade_date": constituent_trade_date,
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


def make_signal_research_panel() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = pd.date_range("2025-01-01", periods=130, freq="B")
    x = np.linspace(0, 14 * np.pi, len(dates))

    aaa_closes = 100 + 18 * np.sin(x)
    aaa_closes[-5:] = [88.0, 84.0, 82.0, 84.0, 86.0]
    aaa_opens = aaa_closes - 1.0
    aaa_opens[-5:] = [86.5, 82.5, 81.0, 82.5, 84.0]

    bbb_closes = 100 + 10 * np.sin(x)
    bbb_closes[-5:] = [92.0, 90.0, 89.0, 90.0, 91.0]
    bbb_opens = bbb_closes - 1.0
    bbb_opens[-5:] = [90.5, 88.5, 88.0, 88.5, 89.5]

    flat_closes = np.full(len(dates), 100.0)
    flat_opens = np.full(len(dates), 100.0)
    flat_highs = np.full(len(dates), 100.0)
    flat_lows = np.full(len(dates), 100.0)

    panel = pd.concat(
        [
            make_stock_frame("AAA", aaa_closes, dates=dates, open_values=aaa_opens),
            make_stock_frame("BBB", bbb_closes, dates=dates, open_values=bbb_opens),
            make_stock_frame(
                "FLAT",
                flat_closes,
                dates=dates,
                open_values=flat_opens,
                high_values=flat_highs,
                low_values=flat_lows,
            ),
        ],
        ignore_index=True,
    )
    return panel, dates[125], dates[126]


def make_manual_outcome_researcher(
    config: RangeStrategyConfig | None = None,
) -> BlueChipRangeReversionResearcher:
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    panel = pd.concat(
        [
            make_stock_frame("AAA", [99.0, 96.0, 89.0, 88.0, 87.0], dates=dates, open_values=[99.0, 100.0, 100.0, 88.0, 87.0]),
            make_stock_frame("BBB", [99.0, 96.0, 95.0, 94.0, 93.0], dates=dates, open_values=[99.0, 100.0, 96.0, 94.0, 93.0]),
            make_stock_frame("CCC", [99.0, 105.0, 111.0, 112.0, 113.0], dates=dates, open_values=[99.0, 100.0, 105.0, 112.0, 113.0]),
            make_stock_frame("DDD", [99.0, 100.0, 101.0, 102.0, 103.0], dates=dates, open_values=[99.0, 100.0, 101.0, 102.0, 103.0]),
        ],
        ignore_index=True,
    )
    researcher = BlueChipRangeReversionResearcher(
        panel,
        config=config or RangeStrategyConfig(range_window=5, max_holding_days=2),
    )
    prepared = researcher.stock_candle_df.copy()
    prepared["range_lower"] = 80.0
    prepared["signal_take_profit_price"] = 120.0
    prepared["signal_hard_stop_price"] = 90.0
    prepared["entry_signal"] = False

    for ticker in ["AAA", "BBB", "CCC", "DDD"]:
        prepared.loc[
            (prepared["ticker"] == ticker) & (prepared["date"] == dates[0]),
            "entry_signal",
        ] = True

    prepared.loc[prepared["ticker"] == "AAA", "signal_take_profit_price"] = 130.0
    prepared.loc[prepared["ticker"] == "AAA", "signal_hard_stop_price"] = 90.0
    prepared.loc[
        (prepared["ticker"] == "AAA") & (prepared["date"].isin([dates[1], dates[2]])),
        "entry_signal",
    ] = True

    prepared.loc[prepared["ticker"] == "BBB", "range_lower"] = 100.0
    prepared.loc[prepared["ticker"] == "BBB", "signal_take_profit_price"] = 130.0
    prepared.loc[prepared["ticker"] == "BBB", "signal_hard_stop_price"] = 80.0

    prepared.loc[prepared["ticker"] == "CCC", "range_lower"] = 80.0
    prepared.loc[prepared["ticker"] == "CCC", "signal_take_profit_price"] = 110.0
    prepared.loc[prepared["ticker"] == "CCC", "signal_hard_stop_price"] = 90.0

    prepared.loc[prepared["ticker"] == "DDD", "range_lower"] = 80.0
    prepared.loc[prepared["ticker"] == "DDD", "signal_take_profit_price"] = 120.0
    prepared.loc[prepared["ticker"] == "DDD", "signal_hard_stop_price"] = 90.0

    researcher.stock_candle_df = prepared
    researcher.add_signals = lambda: researcher.stock_candle_df
    researcher.add_research_outcomes()
    return researcher


class BlueChipRangeReversionResearcherTest(unittest.TestCase):
    def test_add_features_computes_range_metrics_and_keeps_zero_width_zone_stable(self) -> None:
        panel, signal_date, _ = make_signal_research_panel()
        config = RangeStrategyConfig(range_window=20, max_ma_dispersion=0.2)
        researcher = BlueChipRangeReversionResearcher(panel, config=config)
        researcher.add_features()

        output = researcher.stock_candle_df
        aaa = output[output["ticker"] == "AAA"].reset_index(drop=True)
        target_loc = int(aaa.index[aaa["date"] == signal_date][0])
        window_slice = aaa.iloc[target_loc - config.range_window + 1 : target_loc + 1]
        expected_upper = window_slice["high"].quantile(config.upper_quantile)
        expected_lower = window_slice["low"].quantile(config.lower_quantile)
        expected_amplitude = (expected_upper - expected_lower) / aaa.loc[target_loc, "close"]
        expected_zone = (aaa.loc[target_loc, "close"] - expected_lower) / (expected_upper - expected_lower)

        self.assertAlmostEqual(aaa.loc[target_loc, "range_upper"], expected_upper)
        self.assertAlmostEqual(aaa.loc[target_loc, "range_lower"], expected_lower)
        self.assertAlmostEqual(aaa.loc[target_loc, "range_mid"], (expected_upper + expected_lower) / 2.0)
        self.assertAlmostEqual(aaa.loc[target_loc, "range_amplitude"], expected_amplitude)
        self.assertAlmostEqual(aaa.loc[target_loc, "zone_position"], expected_zone)
        self.assertEqual(
            int(aaa.loc[target_loc, "lower_touch_count"]),
            int(window_slice["zone_position"].le(config.touch_zone_pct).sum()),
        )

        flat_last = output[(output["ticker"] == "FLAT") & (output["date"] == output["date"].max())].iloc[0]
        self.assertAlmostEqual(flat_last["range_width"], 0.0)
        self.assertAlmostEqual(flat_last["zone_position"], 0.5)

    def test_add_signals_requires_rebound_confirmation_and_sufficient_upside(self) -> None:
        panel, signal_date, no_rebound_date = make_signal_research_panel()
        researcher = BlueChipRangeReversionResearcher(
            panel,
            config=RangeStrategyConfig(range_window=20, max_ma_dispersion=0.2),
        )
        researcher.add_signals()

        output = researcher.stock_candle_df
        aaa_signal = output[(output["ticker"] == "AAA") & (output["date"] == signal_date)].iloc[0]
        bbb_signal = output[(output["ticker"] == "BBB") & (output["date"] == signal_date)].iloc[0]
        aaa_no_rebound = output[(output["ticker"] == "AAA") & (output["date"] == no_rebound_date)].iloc[0]

        self.assertTrue(bool(aaa_signal["range_candidate"]))
        self.assertTrue(bool(aaa_signal["rebound_confirmed"]))
        self.assertTrue(bool(aaa_signal["expected_upside_ok"]))
        self.assertTrue(bool(aaa_signal["entry_signal"]))
        self.assertEqual(aaa_signal["entry_date_next"], signal_date + pd.offsets.BDay(1))

        self.assertTrue(bool(bbb_signal["range_candidate"]))
        self.assertFalse(bool(bbb_signal["expected_upside_ok"]))
        self.assertFalse(bool(bbb_signal["entry_signal"]))

        self.assertTrue(bool(aaa_no_rebound["range_candidate"]))
        self.assertEqual(int(aaa_no_rebound["rebound_confirm_count"]), 1)
        self.assertFalse(bool(aaa_no_rebound["rebound_confirmed"]))
        self.assertFalse(bool(aaa_no_rebound["entry_signal"]))

        candidates = researcher.get_candidates(as_of_date=signal_date)
        self.assertEqual(candidates["ticker"].tolist(), ["AAA"])

    def test_add_research_outcomes_handles_exit_paths_and_suppresses_reentries(self) -> None:
        researcher = make_manual_outcome_researcher()
        researcher.add_research_outcomes()

        output = researcher.stock_candle_df
        first_rows = output.groupby("ticker", sort=False).head(1).set_index("ticker")

        self.assertEqual(first_rows.loc["AAA", "exit_reason"], "hard_stop")
        self.assertAlmostEqual(first_rows.loc["AAA", "realized_open_to_open_return"], -0.12)
        self.assertEqual(int(first_rows.loc["AAA", "holding_days"]), 2)

        self.assertEqual(first_rows.loc["BBB", "exit_reason"], "breakdown_stop")
        self.assertAlmostEqual(first_rows.loc["BBB", "realized_open_to_open_return"], -0.06)

        self.assertEqual(first_rows.loc["CCC", "exit_reason"], "take_profit")
        self.assertAlmostEqual(first_rows.loc["CCC", "realized_open_to_open_return"], 0.12)

        self.assertEqual(first_rows.loc["DDD", "exit_reason"], "time_stop")
        self.assertAlmostEqual(first_rows.loc["DDD", "realized_open_to_open_return"], 0.02)

        aaa_rows = output[output["ticker"] == "AAA"].reset_index(drop=True)
        self.assertTrue(bool(aaa_rows.loc[0, "entry_signal_executed"]))
        self.assertTrue(bool(aaa_rows.loc[1, "entry_signal_suppressed"]))
        self.assertTrue(bool(aaa_rows.loc[2, "entry_signal_suppressed"]))
        self.assertFalse(bool(aaa_rows.loc[1, "entry_signal_executed"]))

    def test_add_research_outcomes_respects_enabled_exit_rule_booleans(self) -> None:
        no_hard_stop = make_manual_outcome_researcher(
            config=RangeStrategyConfig(
                range_window=5,
                max_holding_days=2,
                enable_hard_stop=False,
            )
        )
        no_hard_stop.add_research_outcomes()
        no_hard_stop_rows = no_hard_stop.stock_candle_df.groupby("ticker", sort=False).head(1).set_index("ticker")
        self.assertEqual(no_hard_stop_rows.loc["AAA", "exit_reason"], "time_stop")

        no_breakdown_stop = make_manual_outcome_researcher(
            config=RangeStrategyConfig(
                range_window=5,
                max_holding_days=2,
                enable_breakdown_stop=False,
            )
        )
        no_breakdown_stop.add_research_outcomes()
        no_breakdown_rows = (
            no_breakdown_stop.stock_candle_df.groupby("ticker", sort=False).head(1).set_index("ticker")
        )
        self.assertEqual(no_breakdown_rows.loc["BBB", "exit_reason"], "time_stop")

        no_take_profit = make_manual_outcome_researcher(
            config=RangeStrategyConfig(
                range_window=5,
                max_holding_days=2,
                enable_take_profit=False,
            )
        )
        no_take_profit.add_research_outcomes()
        no_take_profit_rows = (
            no_take_profit.stock_candle_df.groupby("ticker", sort=False).head(1).set_index("ticker")
        )
        self.assertEqual(no_take_profit_rows.loc["CCC", "exit_reason"], "time_stop")

        no_time_stop = make_manual_outcome_researcher(
            config=RangeStrategyConfig(
                range_window=5,
                max_holding_days=2,
                enable_hard_stop=True,
                enable_breakdown_stop=False,
                enable_take_profit=False,
                enable_time_stop=False,
            )
        )
        no_time_stop.add_research_outcomes()
        no_time_stop_rows = (
            no_time_stop.stock_candle_df.groupby("ticker", sort=False).head(1).set_index("ticker")
        )
        self.assertEqual(no_time_stop_rows.loc["AAA", "exit_reason"], "hard_stop")
        self.assertEqual(no_time_stop_rows.loc["DDD", "exit_reason"], "open_position")
        self.assertTrue(pd.isna(no_time_stop_rows.loc["DDD", "exit_date_next"]))

    def test_range_strategy_config_requires_at_least_one_exit_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one exit rule must be enabled"):
            RangeStrategyConfig(
                range_window=5,
                max_holding_days=2,
                enable_hard_stop=False,
                enable_breakdown_stop=False,
                enable_take_profit=False,
                enable_time_stop=False,
            )

    def test_add_trade_df_builds_trade_level_output_for_closed_trades(self) -> None:
        researcher = make_manual_outcome_researcher()
        trade_df = researcher.add_trade_df()

        self.assertIs(researcher.trade_df, trade_df)
        self.assertEqual(trade_df["ticker"].tolist(), ["AAA", "BBB", "CCC", "DDD"])
        self.assertTrue(
            {
                "signal_date",
                "ticker",
                "name",
                "entry_date",
                "exit_date",
                "exit_reason",
                "pnl",
                "pnl_pct",
                "entry_open",
                "exit_open",
                "trade_status",
            }.issubset(trade_df.columns)
        )
        self.assertTrue(trade_df["trade_status"].eq("closed").all())
        self.assertTrue(trade_df.attrs["all_trades_closed"])
        self.assertEqual(trade_df.attrs["open_trade_count"], 0)

        aaa_trade = trade_df.loc[trade_df["ticker"] == "AAA"].iloc[0]
        self.assertEqual(aaa_trade["entry_date"], pd.Timestamp("2025-01-02"))
        self.assertEqual(aaa_trade["exit_date"], pd.Timestamp("2025-01-06"))
        self.assertEqual(aaa_trade["exit_reason"], "hard_stop")
        self.assertAlmostEqual(float(aaa_trade["entry_open"]), 100.0)
        self.assertAlmostEqual(float(aaa_trade["exit_open"]), 88.0)
        self.assertAlmostEqual(float(aaa_trade["pnl"]), -12.0)
        self.assertAlmostEqual(float(aaa_trade["pnl_pct"]), -0.12)

    def test_add_trade_df_flags_open_trades_in_sanity_check(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        panel = make_stock_frame(
            "OPEN",
            [100.0, 101.0, 102.0],
            dates=dates,
            open_values=[100.0, 101.0, 102.0],
        )
        researcher = BlueChipRangeReversionResearcher(
            panel,
            config=RangeStrategyConfig(range_window=3, max_holding_days=10),
        )
        prepared = researcher.stock_candle_df.copy()
        prepared["range_lower"] = 80.0
        prepared["signal_take_profit_price"] = 120.0
        prepared["signal_hard_stop_price"] = 90.0
        prepared["entry_signal"] = False
        prepared.loc[prepared["date"] == dates[0], "entry_signal"] = True
        researcher.stock_candle_df = prepared
        researcher.add_signals = lambda: researcher.stock_candle_df

        trade_df = researcher.add_trade_df()

        self.assertEqual(len(trade_df), 1)
        self.assertEqual(trade_df["trade_status"].iat[0], "open")
        self.assertEqual(trade_df["exit_reason"].iat[0], "open_position")
        self.assertTrue(pd.isna(trade_df["exit_date"].iat[0]))
        self.assertTrue(pd.isna(trade_df["pnl"].iat[0]))
        self.assertFalse(trade_df.attrs["all_trades_closed"])
        self.assertEqual(trade_df.attrs["open_trade_count"], 1)
        self.assertEqual(trade_df.attrs["open_trade_tickers"], ["OPEN"])

    def test_analyze_feature_win_rates_identifies_best_and_worst_feature_ranges(self) -> None:
        researcher = make_manual_outcome_researcher()
        researcher.trade_df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F", "G"],
                "trade_status": ["closed", "closed", "closed", "closed", "closed", "closed", "open"],
                "pnl_pct": [0.12, 0.08, -0.03, -0.05, 0.04, -0.01, np.nan],
                "zone_position": [0.05, 0.10, 0.15, 0.40, 0.45, 0.50, 0.30],
                "rebound_confirm_count": [3, 3, 2, 2, 1, 1, 2],
            }
        )

        analysis = researcher.analyze_feature_win_rates(
            feature_columns=["zone_position", "rebound_confirm_count", "missing_feature"],
            n_buckets=2,
            min_bucket_size=2,
        )

        bucket_summary = analysis["bucket_summary"]
        feature_summary = analysis["feature_summary"].set_index("feature")
        best_buckets = analysis["best_buckets"].set_index("feature")
        worst_buckets = analysis["worst_buckets"].set_index("feature")
        skipped = analysis["skipped_features"].set_index("feature")

        self.assertFalse(bucket_summary.empty)
        self.assertEqual(set(feature_summary.index.tolist()), {"zone_position", "rebound_confirm_count"})
        self.assertAlmostEqual(feature_summary.loc["zone_position", "best_win_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(feature_summary.loc["zone_position", "worst_win_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(float(best_buckets.loc["zone_position", "feature_min"]), 0.05)
        self.assertAlmostEqual(float(best_buckets.loc["zone_position", "feature_max"]), 0.15)
        self.assertEqual(str(best_buckets.loc["rebound_confirm_count", "feature_bucket"]), "3.0")
        self.assertEqual(str(worst_buckets.loc["rebound_confirm_count", "feature_bucket"]), "2.0")
        self.assertEqual(skipped.loc["missing_feature", "reason"], "missing_column")

    def test_inspect_signal_and_plot_signal_context_return_explainable_context(self) -> None:
        panel, signal_date, _ = make_signal_research_panel()
        researcher = BlueChipRangeReversionResearcher(
            panel,
            config=RangeStrategyConfig(range_window=20, max_ma_dispersion=0.2),
        )

        self.assertIn("exit_reason", researcher.stock_candle_df.columns)
        inspection = researcher.inspect_signal("AAA", signal_date, lookback=5, lookahead=3)

        self.assertEqual(inspection["summary"]["ticker"], "AAA")
        self.assertEqual(inspection["summary"]["signal_date"], signal_date)
        self.assertTrue(inspection["summary"]["raw_signal"])
        self.assertTrue(inspection["summary"]["executed_signal"])
        self.assertEqual(inspection["signal_row"]["ticker"].iat[0], "AAA")
        self.assertIn("entry_signal", inspection["condition_checklist"]["condition"].tolist())
        self.assertFalse(inspection["price_window"].empty)
        self.assertEqual(
            inspection["price_window"]["date"].max(),
            signal_date + pd.offsets.BDay(3),
        )

        figure = researcher.plot_signal_context("AAA", signal_date, lookback=5, lookahead=3)
        self.assertIsInstance(figure, go.Figure)
        self.assertGreaterEqual(len(figure.data), 4)
        self.assertTrue(any("open_position" in getattr(annotation, "text", "") for annotation in figure.layout.annotations))

        closed_researcher = make_manual_outcome_researcher()
        closed_figure = closed_researcher.plot_signal_context("AAA", pd.Timestamp("2025-01-01"), lookback=1, lookahead=3)
        self.assertTrue(any("Exit (" in (trace.name or "") for trace in closed_figure.data))
        self.assertTrue(any("hard_stop" in getattr(annotation, "text", "") for annotation in closed_figure.layout.annotations))

        with self.assertRaisesRegex(ValueError, "suppressed signal"):
            closed_researcher.inspect_signal("AAA", pd.Timestamp("2025-01-02"), lookback=1, lookahead=2)


if __name__ == "__main__":
    unittest.main()
