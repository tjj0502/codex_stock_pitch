import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from score_system.etf_rotation_backtest import (
    run_etf_rotation_backtest,
    update_etf_rotation_cache,
)
from strategies.etf_rotation import ETFUniverseMember
from tests.test_etf_rotation import make_etf_panel


def make_filtered_panel(
    universe,
    *,
    sd,
    ed,
    adjust="",
    pause_seconds=0.0,
    token=None,
) -> pd.DataFrame:
    panel = make_etf_panel().copy()
    tickers = {
        item.ticker if isinstance(item, ETFUniverseMember) else str(item)
        for item in universe
    }
    panel = panel[panel["ticker"].astype(str).isin(tickers)].copy()
    panel = panel[panel["date"].between(pd.Timestamp(sd), pd.Timestamp(ed))].copy()
    return panel.reset_index(drop=True)


class ETFRotationBacktestTests(unittest.TestCase):
    @patch("score_system.etf_rotation_backtest.fetch_etf_price_panel")
    def test_update_etf_rotation_cache_fetches_and_reuses_cached_rows(self, mock_fetch_panel) -> None:
        mock_fetch_panel.side_effect = make_filtered_panel
        universe = [
            ETFUniverseMember("AAA", "Strong ETF", "growth"),
            ETFUniverseMember("BBB", "Medium ETF", "growth"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "etf_rotation_price.csv"

            first_df, first_meta = update_etf_rotation_cache(
                universe=universe,
                end_date="2025-02-25",
                lookback_calendar_days=80,
                cache_path=cache_path,
            )
            second_df, second_meta = update_etf_rotation_cache(
                universe=universe,
                end_date="2025-02-25",
                lookback_calendar_days=80,
                cache_path=cache_path,
            )

            self.assertTrue(cache_path.exists())
            self.assertGreater(len(first_df), 0)
            self.assertEqual(first_meta["fetched_rows"], len(first_df))
            self.assertEqual(second_meta["fetched_rows"], 0)
            self.assertEqual(len(second_df), len(first_df))

    @patch("score_system.etf_rotation_backtest.fetch_etf_price_panel")
    def test_run_etf_rotation_backtest_writes_outputs_and_returns_latest_selection(self, mock_fetch_panel) -> None:
        mock_fetch_panel.side_effect = make_filtered_panel
        universe = [
            ETFUniverseMember("AAA", "Strong ETF", "growth"),
            ETFUniverseMember("BBB", "Medium ETF", "growth"),
            ETFUniverseMember("CCC", "Weak ETF", "defensive"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_path = temp_path / "etf_rotation_price.csv"
            output_root = temp_path / "outputs"

            result = run_etf_rotation_backtest(
                universe=universe,
                start_date="2025-02-03",
                end_date="2025-02-25",
                top_n=1,
                lookback_calendar_days=120,
                cache_path=cache_path,
                output_root=output_root,
                initial_capital=1_000_000.0,
            )

            self.assertTrue((result["output_dir"] / "backtest_summary.csv").exists())
            self.assertTrue((result["output_dir"] / "portfolio.csv").exists())
            self.assertTrue((result["output_dir"] / "trades.csv").exists())
            self.assertTrue((result["output_dir"] / "holdings.csv").exists())
            self.assertTrue((result["output_dir"] / "rotation_membership.csv").exists())
            self.assertTrue((result["output_dir"] / "latest_selection.csv").exists())
            self.assertEqual(result["latest_selection"]["ticker"].tolist(), ["AAA"])
            self.assertGreaterEqual(result["summary"]["total_trades"], 1)
            self.assertGreater(result["summary"]["total_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
