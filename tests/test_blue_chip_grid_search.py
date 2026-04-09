import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from blue_chip_range_reversion import RangeStrategyConfig
from score_system.blue_chip_grid_search import run_blue_chip_grid_search


def make_stock_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    dates: pd.DatetimeIndex,
    open_values: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if open_values is None:
        open_values = closes - 0.5
    open_values = np.asarray(open_values, dtype=float)
    high_values = np.maximum(open_values, closes) + 0.4
    low_values = np.minimum(open_values, closes) - 0.4
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


class BlueChipGridSearchTest(unittest.TestCase):
    def test_run_blue_chip_grid_search_returns_curves_summary_and_figure(self) -> None:
        dates = pd.date_range("2025-01-01", periods=140, freq="B")
        x = np.linspace(0, 12 * np.pi, len(dates))
        panel = pd.concat(
            [
                make_stock_frame("AAA", 100 + 10 * np.sin(x), dates=dates),
                make_stock_frame("BBB", 100 + 8 * np.sin(x + 0.8), dates=dates),
                make_stock_frame("CCC", 100 + 6 * np.sin(x + 1.6), dates=dates),
            ],
            ignore_index=True,
        )

        results = run_blue_chip_grid_search(
            panel,
            param_grid={"entry_zone_threshold": [0.15, 0.25]},
            base_config=RangeStrategyConfig(
                range_window=20,
                ma_dispersion_window=(5, 10, 20),
                max_abs_return_60=0.30,
                min_amplitude=0.05,
                max_amplitude=0.50,
                max_ma_dispersion=0.30,
                min_lower_touches=1,
                min_upper_touches=1,
                max_holding_days=10,
            ),
            start_date=dates[60],
            end_date=dates[-1],
            backtester_kwargs={
                "initial_capital": 100_000.0,
                "fixed_entry_notional": 20_000.0,
            },
            sharpe_window=20,
            sharpe_min_periods=5,
        )

        self.assertEqual(len(results["summary"]), 2)
        self.assertTrue(results["errors"].empty)
        self.assertEqual(set(results["nav_curves"]["label"].unique().tolist()), {"entry_zone_threshold=0.15", "entry_zone_threshold=0.25"})
        self.assertEqual(set(results["sharpe_curves"]["label"].unique().tolist()), {"entry_zone_threshold=0.15", "entry_zone_threshold=0.25"})
        self.assertFalse(results["benchmark_curve"].empty)
        self.assertFalse(results["benchmark_sharpe_curve"].empty)
        self.assertIsInstance(results["figure"], go.Figure)
        trace_names = {trace.name for trace in results["figure"].data}
        self.assertIn("Benchmark NAV", trace_names)
        self.assertIn("Benchmark Sharpe", trace_names)


if __name__ == "__main__":
    unittest.main()
