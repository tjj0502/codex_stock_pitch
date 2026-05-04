import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from score_system.daily_blue_chip_scan import format_scan_report, run_daily_scan
from score_system.daily_narrow_trend_scan import UniverseScanSpec


def make_price_frame(
    ticker: str,
    dates: list[str],
    *,
    name: str | None = None,
) -> pd.DataFrame:
    dates_index = pd.to_datetime(dates)
    opens = [10.0 + idx for idx in range(len(dates_index))]
    closes = [value + 0.5 for value in opens]
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in opens]
    volume = [1_000 + 10 * idx for idx in range(len(dates_index))]
    turnover = [close * vol for close, vol in zip(closes, volume)]
    pre_close = [closes[0], *closes[:-1]]
    change_amount = [close - prev for close, prev in zip(closes, pre_close)]
    change_pct = [((close - prev) / prev) * 100 if prev else 0.0 for close, prev in zip(closes, pre_close)]
    amplitude_pct = [((high - low) / prev) * 100 if prev else 0.0 for high, low, prev in zip(highs, lows, pre_close)]
    return pd.DataFrame(
        {
            "date": dates_index,
            "ticker": ticker,
            "ts_code": f"{ticker}.SZ",
            "name": name or f"{ticker} Corp",
            "weight": 1.0,
            "constituent_trade_date": dates_index[-1],
            "open": opens,
            "close": closes,
            "high": highs,
            "low": lows,
            "pre_close": pre_close,
            "volume": volume,
            "turnover": turnover,
            "amplitude_pct": amplitude_pct,
            "change_pct": change_pct,
            "change_amount": change_amount,
        }
    )


class FakeBlueChipResearcher:
    def __init__(self, price_df: pd.DataFrame, config) -> None:
        self.config = config

    def get_next_session_candidates(
        self,
        *,
        as_of_date,
        next_trade_date,
        entry_price_basis: str = "close",
    ) -> pd.DataFrame:
        prefix = self.config.universe.upper()
        return pd.DataFrame(
            {
                "signal_date": [pd.Timestamp(as_of_date)],
                "planned_entry_date": [pd.Timestamp(next_trade_date)],
                "ticker": [f"{prefix}C"],
                "ts_code": [f"{prefix}C.SZ"],
                "name": [f"{prefix} Candidate"],
                "entry_price_basis": [entry_price_basis],
                "entry_reference_price": [12.5],
                "planned_hard_stop_price": [11.25],
                "planned_take_profit_price": [13.5],
                "range_upper": [13.8],
                "range_lower": [11.7],
                "range_mid": [12.75],
                "range_amplitude": [0.24],
                "zone_position": [0.18],
                "expected_upside_to_upper": [0.10],
                "expected_upside_ok": [True],
                "close_gt_open": [True],
                "close_gt_prev_high": [True],
                "close_gt_sma_5": [True],
                "inside_bar": [False],
                "outside_bar": [True],
                "not_inside_bar": [True],
                "rebound_confirm_count": [1],
                "rebound_confirmed": [True],
                "entry_zone_ok": [True],
                "entry_signal_live": [True],
            }
        )

    def monitor_positions(
        self,
        positions_df: pd.DataFrame,
        as_of_date=None,
        *,
        next_trade_date=None,
    ) -> pd.DataFrame:
        if positions_df.empty:
            return pd.DataFrame()
        monitored = positions_df.copy()
        monitored["as_of_date"] = pd.Timestamp(as_of_date)
        monitored["latest_bar_date"] = pd.Timestamp(as_of_date)
        monitored["signal_date_resolved"] = pd.Timestamp(as_of_date) - pd.offsets.BDay(1)
        monitored["signal_range_upper"] = 13.8
        monitored["latest_range_lower"] = 11.7
        monitored["latest_range_upper"] = 13.8
        monitored["latest_zone_position"] = 0.32
        monitored["current_close"] = 11.0
        monitored["pnl_pct"] = -0.05
        monitored["pnl_amount"] = -50.0
        monitored["holding_days"] = 4
        monitored["trading_days_in_trade"] = 4
        monitored["days_until_time_stop"] = 16
        monitored["hard_stop_price"] = 11.25
        monitored["take_profit_price"] = 13.5
        monitored["breakdown_streak"] = 0
        monitored["exit_signal"] = True
        monitored["exit_signal_date"] = pd.Timestamp(as_of_date)
        monitored["planned_exit_date"] = pd.Timestamp(next_trade_date)
        monitored["exit_reason"] = "hard_stop"
        monitored["action"] = "exit_next_open"
        monitored["issue"] = pd.NA
        return monitored


class DailyBlueChipScanTests(unittest.TestCase):
    def test_run_daily_scan_writes_universe_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "outputs"
            positions_path = temp_path / "open_positions.csv"
            pd.DataFrame(
                {
                    "universe": ["hs300", "csi500"],
                    "ticker": ["HS300A", "CSI500A"],
                    "entry_date": pd.to_datetime(["2026-04-20", "2026-04-20"]),
                    "entry_price": [10.0, 10.0],
                    "shares": [100, 100],
                    "signal_date": pd.to_datetime(["2026-04-18", "2026-04-18"]),
                    "note": ["", ""],
                }
            ).to_csv(positions_path, index=False)

            def fake_fetcher(*, sd, ed, token=None, pause_seconds=1.3, max_calls_per_minute=195):
                return make_price_frame("AAA", ["2026-04-23", "2026-04-24"])

            specs = {
                universe: UniverseScanSpec(
                    universe=universe,
                    fetcher=fake_fetcher,
                    cache_path=temp_path / f"{universe}_stock_price.csv",
                )
                for universe in ("hs300", "csi500")
            }

            with patch("score_system.daily_blue_chip_scan.get_next_trading_day", return_value=pd.Timestamp("2026-04-27")):
                results = run_daily_scan(
                    universes=("hs300", "csi500"),
                    end_date="2026-04-24",
                    universe_specs=specs,
                    output_root=output_root,
                    positions_path=positions_path,
                    researcher_cls=FakeBlueChipResearcher,
                )

            daily_output_dir = output_root / "2026-04-24"
            self.assertEqual(len(results), 2)
            self.assertTrue((daily_output_dir / "daily_scan_summary_20260424.csv").exists())
            for universe in ("hs300", "csi500"):
                self.assertTrue((daily_output_dir / f"{universe}_candidates_20260424.csv").exists())
                self.assertTrue((daily_output_dir / f"{universe}_exits_20260424.csv").exists())

            summary_df = pd.read_csv(daily_output_dir / "daily_scan_summary_20260424.csv")
            self.assertEqual(sorted(summary_df["universe"].tolist()), ["csi500", "hs300"])
            self.assertTrue((summary_df["candidate_count"] == 1).all())
            self.assertTrue((summary_df["exit_count"] == 1).all())

            report = format_scan_report(results)
            self.assertIn("[hs300]", report)
            self.assertIn("- candidate_count: 1", report)
            self.assertIn("- exit_count: 1", report)
            self.assertIn("- exits: HS300A", report)


if __name__ == "__main__":
    unittest.main()
