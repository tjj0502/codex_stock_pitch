import unittest

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from strategies.china_stock_data import DailyTechnicalScorer


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
        volume_values = 1_000 + np.arange(len(closes)) * 25

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


class DailyTechnicalScorerTest(unittest.TestCase):
    def test_init_validates_schema_coerces_types_and_deduplicates(self) -> None:
        dates = pd.to_datetime(["2025-01-03", "2025-01-02"])
        aaa = make_stock_frame("AAA", [11.0, 10.0], dates=dates)
        bbb = make_stock_frame("BBB", [21.0, 20.0], dates=dates)
        duplicate = aaa.iloc[[1]].copy()
        duplicate["open"] = "10.5"
        duplicate["close"] = "10.5"
        duplicate["high"] = "11.0"
        duplicate["low"] = "10.0"
        duplicate["pre_close"] = "10.0"
        duplicate["volume"] = "1100"
        duplicate["turnover"] = "11550"
        duplicate["amplitude_pct"] = "10.0"
        duplicate["change_pct"] = "5.0"
        duplicate["change_amount"] = "0.5"

        raw = pd.concat([bbb, aaa, duplicate], ignore_index=True)
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        raw["constituent_trade_date"] = raw["constituent_trade_date"].astype(str)
        for column in DailyTechnicalScorer.NUMERIC_COLUMNS:
            raw[column] = raw[column].astype(str)

        scorer = DailyTechnicalScorer(raw)

        self.assertEqual(len(scorer.stock_candle_df), 4)
        output_pairs = list(
            zip(scorer.stock_candle_df["date"].tolist(), scorer.stock_candle_df["ticker"].tolist())
        )
        expected_pairs = sorted(output_pairs)
        self.assertEqual(output_pairs, expected_pairs)
        aaa_day_two = scorer.stock_candle_df[
            (scorer.stock_candle_df["ticker"] == "AAA")
            & (scorer.stock_candle_df["date"] == pd.Timestamp("2025-01-02"))
        ]
        self.assertAlmostEqual(aaa_day_two["close"].iat[0], 10.5)
        self.assertTrue(is_datetime64_any_dtype(scorer.stock_candle_df["date"]))
        self.assertTrue(is_datetime64_any_dtype(scorer.stock_candle_df["constituent_trade_date"]))
        self.assertTrue(is_string_dtype(scorer.stock_candle_df["ticker"]))
        self.assertTrue(is_numeric_dtype(scorer.stock_candle_df["close"]))

        with self.assertRaises(ValueError):
            DailyTechnicalScorer(raw.drop(columns=["close"]))

    def test_features_and_targets_use_expected_time_alignment(self) -> None:
        dates = pd.date_range("2025-01-01", periods=7, freq="B")
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 200.0]
        opens = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 30.0]
        frame = make_stock_frame("AAA", closes, dates=dates, open_values=opens)

        scorer = DailyTechnicalScorer(frame, min_history=1)
        scorer.add_technical_features()
        scorer.add_research_targets()

        features = scorer.stock_candle_df
        day_six = features.loc[features["date"] == dates[5]].iloc[0]
        self.assertAlmostEqual(day_six["ret_5d"], 1.0)

        day_five = features.loc[features["date"] == dates[4]].iloc[0]
        self.assertEqual(day_five["next_execution_date"], dates[5])
        self.assertAlmostEqual(day_five["entry_open_next"], 15.0)
        self.assertAlmostEqual(day_five["exit_open_next"], 30.0)
        self.assertAlmostEqual(day_five["target_next_open_to_open"], 1.0)

    def test_feature_formulas_cover_breakout_atr_and_zero_range_candles(self) -> None:
        dates = pd.date_range("2025-01-01", periods=25, freq="B")
        closes = np.arange(10.0, 35.0)
        opens = closes - 0.5
        highs = closes + 0.5
        lows = closes - 1.5
        zero_range_index = 5
        opens[zero_range_index] = closes[zero_range_index]
        highs[zero_range_index] = closes[zero_range_index]
        lows[zero_range_index] = closes[zero_range_index]
        frame = make_stock_frame(
            "AAA",
            closes,
            dates=dates,
            open_values=opens,
            high_values=highs,
            low_values=lows,
        )

        scorer = DailyTechnicalScorer(frame, min_history=1)
        scorer.add_technical_features()
        features = scorer.stock_candle_df

        latest = features.loc[features["date"] == dates[-1]].iloc[0]
        expected_breakout = closes[-1] / highs[-2] - 1.0
        expected_atr_pct = 2.0 / closes[-1] * 100.0
        self.assertAlmostEqual(latest["breakout_20d"], expected_breakout, places=8)
        self.assertAlmostEqual(latest["drawdown_20d"], 0.0, places=8)
        self.assertAlmostEqual(latest["atr_pct_14"], expected_atr_pct, places=8)

        zero_range_row = features.loc[features["date"] == dates[zero_range_index]].iloc[0]
        self.assertAlmostEqual(zero_range_row["close_location"], 0.5)
        self.assertAlmostEqual(zero_range_row["body_to_range"], 0.0)
        self.assertAlmostEqual(zero_range_row["upper_shadow_pct"], 0.0)

    def test_invalid_denominators_are_sanitized_before_scoring(self) -> None:
        dates = pd.date_range("2025-01-01", periods=25, freq="B")
        closes = np.linspace(10.0, 20.0, len(dates))
        closes[4] = 0.0
        opens = np.maximum(closes - 0.2, 0.0)
        highs = np.maximum(opens, closes) + 0.5
        lows = np.maximum(np.minimum(opens, closes) - 0.5, 0.0)
        frame = make_stock_frame(
            "AAA",
            closes,
            dates=dates,
            open_values=opens,
            high_values=highs,
            low_values=lows,
        )

        scorer = DailyTechnicalScorer(frame, min_history=20)
        scorer.add_technical_score()
        latest = scorer.stock_candle_df.loc[scorer.stock_candle_df["date"] == dates[-1]].iloc[0]

        self.assertTrue(pd.isna(latest["ret_20d"]))
        self.assertFalse(bool(latest["technical_score_eligible"]))
        self.assertTrue(pd.isna(latest["technical_score"]))

    def test_scores_rank_cross_section_and_break_ties_deterministically(self) -> None:
        dates = pd.date_range("2025-01-01", periods=70, freq="B")
        strong_closes = np.linspace(20.0, 40.0, len(dates))
        weak_closes = 28.0 + np.sin(np.arange(len(dates)) * 0.8) * 3.0 + np.linspace(0.0, -7.0, len(dates))

        aaa = make_stock_frame(
            "AAA",
            strong_closes,
            dates=dates,
            open_values=strong_closes - 0.2,
            high_values=strong_closes + 0.4,
            low_values=strong_closes - 0.6,
            volume_values=1_500 + np.arange(len(dates)) * 15,
        )
        aac = make_stock_frame(
            "AAC",
            strong_closes,
            dates=dates,
            open_values=strong_closes - 0.2,
            high_values=strong_closes + 0.4,
            low_values=strong_closes - 0.6,
            volume_values=1_500 + np.arange(len(dates)) * 15,
        )
        bbb = make_stock_frame(
            "BBB",
            weak_closes,
            dates=dates,
            open_values=weak_closes - 0.4,
            high_values=weak_closes + 1.6,
            low_values=weak_closes - 1.8,
            volume_values=900 + np.arange(len(dates)) * 8,
        )

        scorer = DailyTechnicalScorer(pd.concat([bbb, aaa, aac], ignore_index=True))
        scorer.add_technical_score(top_n=2)
        latest_date = scorer.stock_candle_df["date"].max()
        latest = scorer.stock_candle_df[scorer.stock_candle_df["date"] == latest_date].copy()
        latest = latest.sort_values("technical_rank", ignore_index=True)

        self.assertEqual(latest["ticker"].tolist(), ["AAA", "AAC", "BBB"])
        self.assertEqual(latest["technical_rank"].tolist(), [1, 2, 3])
        self.assertTrue(latest.loc[latest["ticker"] == "AAA", "selected_top_n"].iat[0])
        self.assertTrue(latest.loc[latest["ticker"] == "AAC", "selected_top_n"].iat[0])
        self.assertFalse(latest.loc[latest["ticker"] == "BBB", "selected_top_n"].iat[0])
        aaa_risk = latest.loc[latest["ticker"] == "AAA", "risk_score"].iat[0]
        bbb_risk = latest.loc[latest["ticker"] == "BBB", "risk_score"].iat[0]
        self.assertGreater(aaa_risk, bbb_risk)

    def test_get_top_candidates_defaults_to_latest_scored_date(self) -> None:
        dates = pd.date_range("2025-01-01", periods=70, freq="B")
        strong_closes = np.linspace(20.0, 40.0, len(dates))
        weak_closes = 28.0 + np.sin(np.arange(len(dates)) * 0.8) * 3.0 + np.linspace(0.0, -7.0, len(dates))

        scorer = DailyTechnicalScorer(
            pd.concat(
                [
                    make_stock_frame("BBB", weak_closes, dates=dates),
                    make_stock_frame("AAA", strong_closes, dates=dates),
                    make_stock_frame("AAC", strong_closes, dates=dates),
                ],
                ignore_index=True,
            )
        )
        scorer.add_research_targets()
        scorer.add_technical_score(top_n=2)

        latest_top = scorer.get_top_candidates(2)
        self.assertEqual(latest_top["ticker"].tolist(), ["AAA", "AAC"])
        self.assertEqual(latest_top["date"].nunique(), 1)
        self.assertEqual(latest_top["date"].iat[0], scorer.stock_candle_df["date"].max())
        self.assertIn("target_next_open_to_open", latest_top.columns)

        prior_date = scorer.stock_candle_df["date"].drop_duplicates().sort_values().iloc[-2]
        prior_top = scorer.get_top_candidates(2, as_of_date=prior_date)
        self.assertEqual(prior_top["date"].iat[0], prior_date)

    def test_get_top_candidates_can_skip_the_top_quantile_before_selection(self) -> None:
        dates = pd.date_range("2025-01-01", periods=70, freq="B")
        strong_closes = np.linspace(20.0, 40.0, len(dates))
        weak_closes = 28.0 + np.sin(np.arange(len(dates)) * 0.8) * 3.0 + np.linspace(0.0, -7.0, len(dates))

        scorer = DailyTechnicalScorer(
            pd.concat(
                [
                    make_stock_frame("BBB", weak_closes, dates=dates),
                    make_stock_frame("AAA", strong_closes, dates=dates),
                    make_stock_frame("AAC", strong_closes, dates=dates),
                ],
                ignore_index=True,
            )
        )
        scorer.add_technical_score(top_n=2)

        filtered_top = scorer.get_top_candidates(1, exclude_top_quantile=0.2)
        self.assertEqual(filtered_top["ticker"].tolist(), ["AAC"])


if __name__ == "__main__":
    unittest.main()
