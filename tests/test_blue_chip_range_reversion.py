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


def make_manual_outcome_researcher() -> BlueChipRangeReversionResearcher:
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
        config=RangeStrategyConfig(range_window=5, max_holding_days=2),
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

    def test_inspect_signal_and_plot_signal_context_return_explainable_context(self) -> None:
        panel, signal_date, _ = make_signal_research_panel()
        researcher = BlueChipRangeReversionResearcher(
            panel,
            config=RangeStrategyConfig(range_window=20, max_ma_dispersion=0.2),
        )

        inspection = researcher.inspect_signal("AAA", signal_date, lookback=5, lookahead=3)

        self.assertEqual(inspection["summary"]["ticker"], "AAA")
        self.assertEqual(inspection["summary"]["signal_date"], signal_date)
        self.assertTrue(inspection["summary"]["raw_signal"])
        self.assertTrue(inspection["summary"]["executed_signal"])
        self.assertEqual(inspection["signal_row"]["ticker"].iat[0], "AAA")
        self.assertIn("entry_signal", inspection["condition_checklist"]["condition"].tolist())
        self.assertFalse(inspection["price_window"].empty)

        figure = researcher.plot_signal_context("AAA", signal_date, lookback=5, lookahead=3)
        self.assertIsInstance(figure, go.Figure)
        self.assertGreaterEqual(len(figure.data), 4)


if __name__ == "__main__":
    unittest.main()
