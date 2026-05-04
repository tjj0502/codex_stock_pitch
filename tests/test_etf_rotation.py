import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategies.etf_rotation import (
    ETFHeatRotationConfig,
    ETFHeatRotationScorer,
    build_rotation_membership_frame,
    fetch_single_etf_price_history,
)


def make_etf_price_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    name: str | None = None,
    dates: pd.DatetimeIndex | None = None,
    volume_start: float = 1_000_000.0,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if dates is None:
        dates = pd.date_range("2025-01-01", periods=len(closes), freq="B")

    open_values = closes * 0.995
    high_values = np.maximum(open_values, closes) * 1.01
    low_values = np.minimum(open_values, closes) * 0.99
    volume_values = volume_start + np.arange(len(closes)) * 20_000.0
    turnover_values = volume_values * closes
    pre_close = np.concatenate(([closes[0]], closes[:-1]))
    change_amount = closes - pre_close
    safe_pre_close = np.where(pre_close > 0, pre_close, np.nan)
    change_pct = change_amount / safe_pre_close * 100.0
    amplitude_pct = (high_values - low_values) / safe_pre_close * 100.0

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "ts_code": ticker,
            "name": name or ticker,
            "weight": 1.0,
            "constituent_trade_date": dates[-1],
            "open": open_values,
            "close": closes,
            "high": high_values,
            "low": low_values,
            "pre_close": pre_close,
            "volume": volume_values,
            "turnover": turnover_values,
            "amplitude_pct": amplitude_pct,
            "change_pct": change_pct,
            "change_amount": change_amount,
        }
    )


def make_etf_panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    strong = np.linspace(1.0, 2.0, len(dates))
    medium = np.linspace(1.0, 1.45, len(dates))
    weak = 1.1 + np.sin(np.arange(len(dates)) * 0.6) * 0.02 - np.linspace(0.0, 0.08, len(dates))
    return pd.concat(
        [
            make_etf_price_frame("AAA", strong, name="Strong ETF", dates=dates, volume_start=1_500_000.0),
            make_etf_price_frame("BBB", medium, name="Medium ETF", dates=dates, volume_start=1_250_000.0),
            make_etf_price_frame("CCC", weak, name="Weak ETF", dates=dates, volume_start=900_000.0),
        ],
        ignore_index=True,
    )


class ETFRotationTests(unittest.TestCase):
    @patch("strategies.etf_rotation._get_tushare_client")
    def test_fetch_single_etf_price_history_normalizes_schema(self, mock_get_client) -> None:
        class FakeClient:
            @staticmethod
            def fund_daily(ts_code, start_date, end_date):
                return pd.DataFrame(
                    {
                        "ts_code": ["510300.SH", "510300.SH"],
                        "trade_date": ["20250103", "20250102"],
                        "pre_close": [1.02, 1.00],
                        "open": [1.05, 1.00],
                        "high": [1.09, 1.03],
                        "low": [1.04, 0.99],
                        "close": [1.08, 1.02],
                        "change": [0.06, 0.02],
                        "pct_chg": [5.88, 2.00],
                        "vol": [1200, 1000],
                        "amount": [1296, 1020],
                    }
                )

        mock_get_client.return_value = FakeClient()

        result = fetch_single_etf_price_history("510300", name="CSI300 ETF", sd="2025-01-02", ed="2025-01-03")

        self.assertEqual(
            result.columns.tolist(),
            [
                "date",
                "ticker",
                "ts_code",
                "name",
                "weight",
                "constituent_trade_date",
                "open",
                "close",
                "high",
                "low",
                "pre_close",
                "volume",
                "turnover",
                "amplitude_pct",
                "change_pct",
                "change_amount",
            ],
        )
        self.assertEqual(result["ticker"].tolist(), ["510300", "510300"])
        self.assertEqual(result["ts_code"].tolist(), ["510300.SH", "510300.SH"])
        self.assertEqual(result["name"].tolist(), ["CSI300 ETF", "CSI300 ETF"])
        self.assertAlmostEqual(float(result["pre_close"].iat[0]), 1.0)
        self.assertAlmostEqual(float(result["pre_close"].iat[1]), 1.02)

    def test_etf_heat_rotation_scorer_ranks_stronger_etfs_first(self) -> None:
        scorer = ETFHeatRotationScorer(
            make_etf_panel(),
            config=ETFHeatRotationConfig(min_history=20),
        )
        scored = scorer.add_technical_score(top_n=2)
        latest_date = scored["date"].max()
        latest = scored[scored["date"] == latest_date].sort_values("technical_rank").reset_index(drop=True)

        self.assertEqual(latest["ticker"].tolist(), ["AAA", "BBB", "CCC"])
        self.assertEqual(latest["technical_rank"].tolist(), [1, 2, pd.NA])
        self.assertTrue(bool(latest.loc[latest["ticker"] == "AAA", "selected_top_n"].iat[0]))
        self.assertTrue(bool(latest.loc[latest["ticker"] == "BBB", "selected_top_n"].iat[0]))
        self.assertTrue(pd.isna(latest.loc[latest["ticker"] == "CCC", "technical_score"].iat[0]))

        top = scorer.get_top_candidates(2)
        self.assertEqual(top["ticker"].tolist(), ["AAA", "BBB"])

    def test_build_rotation_membership_frame_keeps_daily_selected_names(self) -> None:
        scorer = ETFHeatRotationScorer(make_etf_panel(), config=ETFHeatRotationConfig(min_history=20))
        scored = scorer.add_technical_score(top_n=1)

        membership = build_rotation_membership_frame(scored, top_n=1)

        self.assertTrue((membership["technical_rank"] == 1).all())
        self.assertEqual(membership["ticker"].nunique(), 1)
        self.assertEqual(membership["ticker"].iat[0], "AAA")


if __name__ == "__main__":
    unittest.main()
