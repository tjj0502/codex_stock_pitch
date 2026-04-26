import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from score_system.daily_narrow_trend_scan import (
    JUST_ENDED_COLUMNS,
    UniverseScanSpec,
    compute_narrow_trend_just_ended,
    format_scan_report,
    run_daily_scan,
    update_universe_cache,
)


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


class FakeNarrowTrendResearcher:
    def __init__(self, price_df: pd.DataFrame, config) -> None:
        self.config = config
        last_date = pd.Timestamp(price_df["date"].max()).normalize()
        prev_date = last_date - pd.offsets.BDay(1)
        prefix = config.universe.upper()
        self.stock_candle_df = pd.DataFrame(
            {
                "date": [prev_date, last_date, prev_date, last_date],
                "ticker": [f"{prefix}A", f"{prefix}A", f"{prefix}B", f"{prefix}B"],
                "ts_code": [f"{prefix}A.SZ", f"{prefix}A.SZ", f"{prefix}B.SZ", f"{prefix}B.SZ"],
                "name": [f"{prefix} Alpha", f"{prefix} Alpha", f"{prefix} Beta", f"{prefix} Beta"],
                "close": [10.0, 10.3, 11.0, 11.2],
                "narrow_uptrend_state": [True, False, False, False],
                "narrow_uptrend_run_length": [4, 0, 0, 0],
                "narrow_state_bear_ratio": [0.10, 0.30, 0.0, 0.0],
                "narrow_state_ema20_above_ratio": [1.0, 0.8, 1.0, 1.0],
                "narrow_state_peak_upper_shadow_pct": [0.15, 0.35, 0.10, 0.10],
            }
        )

    def add_signals(self) -> pd.DataFrame:
        return self.stock_candle_df

    def get_next_session_candidates(
        self,
        *,
        as_of_date,
        next_trade_date,
        entry_price_basis: str = "follow_through_close",
    ) -> pd.DataFrame:
        prefix = self.config.universe.upper()
        return pd.DataFrame(
            {
                "signal_date": [pd.Timestamp(as_of_date)],
                "follow_through_date": [pd.Timestamp(as_of_date)],
                "planned_entry_date": [pd.Timestamp(next_trade_date)],
                "ticker": [f"{prefix}C"],
                "ts_code": [f"{prefix}C.SZ"],
                "name": [f"{prefix} Candidate"],
                "entry_price_basis": [entry_price_basis],
                "entry_reference_price": [12.5],
                "planned_hard_stop_price": [11.8],
                "planned_take_profit_price": [14.2],
                "flag_peak_high": [13.0],
                "flag_low": [12.0],
                "flagpole_return": [0.18],
                "flag_retrace_ratio": [0.30],
                "flag_bars": [6],
                "narrow_uptrend_run_length": [4],
                "narrow_state_bear_ratio": [0.10],
                "narrow_state_ema20_above_ratio": [1.0],
                "narrow_state_max_consecutive_bear_bars": [1],
                "signal_bullish_stack_run_length": [7.0],
                "signal_stack_spread_pct": [0.08],
                "signal_sma20_return_5": [0.03],
                "peak_bullish_stack_run_length": [5.0],
                "peak_sma60_return_10": [0.02],
                "reward_to_risk": [1.8],
                "follow_through_confirmed": [True],
                "follow_through_close_gt_signal_close": [True],
                "trend_environment_ok": [True],
                "reward_to_risk_ok": [True],
                "entry_signal_live": [True],
            }
        )


class DailyNarrowTrendScanTests(unittest.TestCase):
    def test_compute_narrow_trend_just_ended_detects_transitions(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-04-23", "2026-04-24", "2026-04-23", "2026-04-24", "2026-04-23", "2026-04-24"]
                ),
                "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                "ts_code": ["AAA.SZ", "AAA.SZ", "BBB.SZ", "BBB.SZ", "CCC.SZ", "CCC.SZ"],
                "name": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                "close": [10.0, 10.2, 8.0, 8.1, 9.0, 9.1],
                "narrow_uptrend_state": [True, False, True, True, False, False],
                "narrow_uptrend_run_length": [3, 0, 2, 3, 0, 0],
                "narrow_state_bear_ratio": [0.10, 0.30, 0.10, 0.10, 0.0, 0.0],
                "narrow_state_ema20_above_ratio": [1.0, 0.7, 1.0, 1.0, 1.0, 1.0],
                "narrow_state_peak_upper_shadow_pct": [0.12, 0.40, 0.10, 0.10, 0.0, 0.0],
            }
        )

        ended = compute_narrow_trend_just_ended(frame, "2026-04-24")

        self.assertEqual(list(ended.columns), JUST_ENDED_COLUMNS)
        self.assertEqual(ended["ticker"].tolist(), ["AAA"])
        self.assertEqual(pd.Timestamp(ended["previous_state_date"].iat[0]), pd.Timestamp("2026-04-23"))
        self.assertEqual(int(ended["previous_run_length"].iat[0]), 3)

    def test_update_universe_cache_seeds_from_legacy_and_fetches_incremental_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_path = temp_path / "csi1000_stock_price.csv"
            legacy_path = temp_path / "legacy_seed.csv"

            legacy_df = pd.concat(
                [
                    make_price_frame("AAA", ["2026-04-10", "2026-04-21"]),
                    make_price_frame("BBB", ["2026-04-21"]),
                ],
                ignore_index=True,
            )
            legacy_df.to_csv(legacy_path, index=False)

            def fake_fetcher(*, sd, ed, token=None, pause_seconds=1.3, max_calls_per_minute=195):
                self.assertEqual(pd.Timestamp(sd), pd.Timestamp("2026-04-22"))
                self.assertEqual(pd.Timestamp(ed), pd.Timestamp("2026-04-24"))
                return pd.concat(
                    [
                        make_price_frame("AAA", ["2026-04-22", "2026-04-24"]),
                        make_price_frame("BBB", ["2026-04-22", "2026-04-24"]),
                    ],
                    ignore_index=True,
                )

            spec = UniverseScanSpec(
                universe="csi1000",
                fetcher=fake_fetcher,
                cache_path=cache_path,
                legacy_seed_path=legacy_path,
            )

            updated_df, metadata = update_universe_cache(
                spec,
                end_date="2026-04-24",
                lookback_calendar_days=10,
            )

            self.assertTrue(cache_path.exists())
            self.assertTrue(metadata["seeded_from_legacy"])
            self.assertEqual(metadata["fetched_rows"], 4)
            self.assertEqual(pd.Timestamp(metadata["as_of_date"]), pd.Timestamp("2026-04-24"))
            self.assertTrue((updated_df["date"] >= pd.Timestamp("2026-04-14")).all())
            self.assertEqual(
                updated_df.loc[updated_df["ticker"].eq("AAA"), "date"].dt.strftime("%Y-%m-%d").tolist(),
                ["2026-04-21", "2026-04-22", "2026-04-24"],
            )

    def test_run_daily_scan_writes_universe_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "outputs"

            def fake_fetcher(*, sd, ed, token=None, pause_seconds=1.3, max_calls_per_minute=195):
                return make_price_frame("AAA", ["2026-04-23", "2026-04-24"])

            specs = {
                universe: UniverseScanSpec(
                    universe=universe,
                    fetcher=fake_fetcher,
                    cache_path=temp_path / f"{universe}_stock_price.csv",
                )
                for universe in ("hs300", "csi500", "csi1000")
            }

            with patch("score_system.daily_narrow_trend_scan.get_next_trading_day", return_value=pd.Timestamp("2026-04-27")):
                results = run_daily_scan(
                    universes=("hs300", "csi500", "csi1000"),
                    end_date="2026-04-24",
                    universe_specs=specs,
                    output_root=output_root,
                    researcher_cls=FakeNarrowTrendResearcher,
                )

            daily_output_dir = output_root / "2026-04-24"
            self.assertEqual(len(results), 3)
            self.assertTrue((daily_output_dir / "daily_scan_summary_20260424.csv").exists())
            for universe in ("hs300", "csi500", "csi1000"):
                self.assertTrue((daily_output_dir / f"{universe}_candidates_20260424.csv").exists())
                self.assertTrue((daily_output_dir / f"{universe}_narrow_trend_just_ended_20260424.csv").exists())

            summary_df = pd.read_csv(daily_output_dir / "daily_scan_summary_20260424.csv")
            self.assertEqual(sorted(summary_df["universe"].tolist()), ["csi1000", "csi500", "hs300"])
            self.assertTrue((summary_df["candidate_count"] == 1).all())
            self.assertTrue((summary_df["just_ended_count"] == 1).all())

            report = format_scan_report(results)
            self.assertIn("[hs300]", report)
            self.assertIn("- candidate_count: 1", report)
            self.assertIn("- just_ended_count: 1", report)


if __name__ == "__main__":
    unittest.main()
