import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from score_system.gap_breakout_grid_search import run_gap_breakout_grid_search
from strategies.gap_breakout_continuation import (
    GapBreakoutContinuationResearcher,
    GapBreakoutStrategyConfig,
)


def make_stock_frame(
    ticker: str,
    closes: list[float] | np.ndarray,
    *,
    dates: pd.DatetimeIndex,
    open_values: list[float] | np.ndarray | None = None,
    high_values: list[float] | np.ndarray | None = None,
    low_values: list[float] | np.ndarray | None = None,
    volume_values: list[float] | np.ndarray | None = None,
    weight: float = 1.0,
    name: str = "Gap Corp",
) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    if open_values is None:
        open_values = closes - 0.1
    if high_values is None:
        high_values = np.maximum(open_values, closes) + 0.35
    if low_values is None:
        low_values = np.minimum(open_values, closes) - 0.35
    if volume_values is None:
        volume_values = np.full(len(closes), 6_000_000.0)

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
            "ts_code": f"{ticker}.SH",
            "name": name,
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


def make_gap_case_frame(
    *,
    confirm_fill: bool = False,
    exit_mode: str = "ma10",
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = pd.date_range("2025-01-01", periods=150, freq="B")
    closes = [10.0]
    for idx in range(1, len(dates)):
        closes.append(closes[-1] + (-0.05 if idx % 5 == 0 else 0.20))

    closes = np.asarray(closes, dtype=float)
    open_values = closes - 0.10
    high_values = closes + 0.35
    low_values = closes - 0.35
    volume_values = np.full(len(dates), 6_000_000.0)

    gap_idx = 130
    prev_high = high_values[gap_idx - 1]
    low_values[gap_idx] = prev_high * 1.02
    open_values[gap_idx] = low_values[gap_idx] + 0.10
    close_values = closes.copy()
    close_values[gap_idx] = open_values[gap_idx] + 0.55
    high_values[gap_idx] = close_values[gap_idx] + 0.15
    volume_values[gap_idx] = 12_000_000.0

    if confirm_fill:
        low_values[gap_idx + 1] = prev_high - 0.05
        open_values[gap_idx + 1] = low_values[gap_idx + 1] + 0.20
        close_values[gap_idx + 1] = open_values[gap_idx + 1] + 0.10
        high_values[gap_idx + 1] = close_values[gap_idx + 1] + 0.15
    else:
        low_values[gap_idx + 1] = low_values[gap_idx] + 0.05
        open_values[gap_idx + 1] = close_values[gap_idx] + 0.05
        close_values[gap_idx + 1] = close_values[gap_idx] + 0.35
        high_values[gap_idx + 1] = close_values[gap_idx + 1] + 0.15
        volume_values[gap_idx + 1] = 8_000_000.0

    if exit_mode == "ma10":
        open_values[132] = close_values[131] + 0.05
        close_values[132] = open_values[132] + 0.25
        high_values[132] = close_values[132] + 0.20
        low_values[132] = open_values[132] - 0.20

        open_values[133] = close_values[132] - 0.05
        close_values[133] = open_values[133] + 0.20
        high_values[133] = close_values[133] + 0.15
        low_values[133] = open_values[133] - 0.15

        open_values[134] = close_values[133] - 0.05
        close_values[134] = open_values[134] + 0.15
        high_values[134] = close_values[134] + 0.15
        low_values[134] = open_values[134] - 0.15

        open_values[135] = close_values[134] - 0.05
        close_values[135] = open_values[135] + 0.10
        high_values[135] = close_values[135] + 0.15
        low_values[135] = open_values[135] - 0.15

        open_values[136] = close_values[135] - 0.20
        close_values[136] = low_values[gap_idx] + 0.05
        high_values[136] = open_values[136] + 0.10
        low_values[136] = close_values[136] - 0.10
        open_values[137] = close_values[136] - 0.05
    elif exit_mode == "gap_stop":
        open_values[132] = close_values[131] + 0.05
        close_values[132] = open_values[132] + 0.10
        high_values[132] = close_values[132] + 0.20
        low_values[132] = open_values[132] - 0.20

        open_values[133] = close_values[132] - 0.30
        close_values[133] = low_values[gap_idx] - 0.10
        high_values[133] = open_values[133] + 0.15
        low_values[133] = close_values[133] - 0.15
        open_values[134] = close_values[133] - 0.05
    else:
        raise ValueError(f"Unsupported exit_mode: {exit_mode}")

    frame = make_stock_frame(
        "600001",
        close_values,
        dates=dates,
        open_values=open_values,
        high_values=high_values,
        low_values=low_values,
        volume_values=volume_values,
    )
    return frame, dates[gap_idx], dates[gap_idx + 1]


class GapBreakoutContinuationResearcherTests(unittest.TestCase):
    def test_confirm_mode_requires_unfilled_gap_and_reclaim(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(frame)

        signal_row = researcher.stock_candle_df[
            (researcher.stock_candle_df["ticker"] == "600001") & (researcher.stock_candle_df["date"] == signal_date)
        ].iloc[0]

        self.assertTrue(bool(signal_row["entry_signal"]))
        self.assertEqual(int(signal_row["signal_gap_age"]), 1)
        self.assertTrue(bool(signal_row["gap_unfilled"]))
        self.assertTrue(bool(signal_row["confirm_level_ok"]))

    def test_confirm_mode_rejects_filled_gap(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=True, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(frame)
        signal_rows = researcher.stock_candle_df[
            (researcher.stock_candle_df["ticker"] == "600001") & (researcher.stock_candle_df["date"] == signal_date)
        ]
        self.assertFalse(bool(signal_rows["entry_signal"].fillna(False).any()))

    def test_gap_day_mode_emits_signal_on_gap_day(self) -> None:
        frame, gap_day, _ = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(
            frame,
            config=GapBreakoutStrategyConfig(entry_mode="gap_day"),
        )
        gap_row = researcher.stock_candle_df[
            (researcher.stock_candle_df["ticker"] == "600001") & (researcher.stock_candle_df["date"] == gap_day)
        ].iloc[0]

        self.assertTrue(bool(gap_row["entry_signal"]))
        self.assertEqual(int(gap_row["signal_gap_age"]), 0)

    def test_gap_stop_exit_triggers_after_entry(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=False, exit_mode="gap_stop")
        researcher = GapBreakoutContinuationResearcher(frame)
        trade_df = researcher.trade_df
        self.assertEqual(len(trade_df), 1)
        trade = trade_df.iloc[0]
        self.assertEqual(pd.Timestamp(trade["signal_date"]), signal_date)
        self.assertEqual(trade["exit_reason"], "gap_stop")

    def test_ma10_exit_triggers_after_entry(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(frame)
        trade_df = researcher.trade_df
        self.assertEqual(len(trade_df), 1)
        trade = trade_df.iloc[0]
        self.assertEqual(pd.Timestamp(trade["signal_date"]), signal_date)
        self.assertEqual(trade["exit_reason"], "ma10_exit")

    def test_get_next_session_candidates_returns_live_entry_rows(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(frame)
        candidates = researcher.get_next_session_candidates(as_of_date=signal_date)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(pd.Timestamp(candidates["signal_date"].iat[0]), signal_date)
        self.assertTrue(bool(candidates["entry_signal_live"].iat[0]))

    def test_monitor_positions_flags_exit_next_open(self) -> None:
        frame, _, _ = make_gap_case_frame(confirm_fill=False, exit_mode="gap_stop")
        researcher = GapBreakoutContinuationResearcher(frame)
        trade = researcher.trade_df.iloc[0]
        positions = pd.DataFrame(
            {
                "ticker": [trade["ticker"]],
                "entry_date": [trade["entry_date"]],
                "entry_price": [trade["entry_open"]],
            }
        )
        monitored = researcher.monitor_positions(positions, as_of_date=trade["exit_signal_date"])
        self.assertEqual(len(monitored), 1)
        self.assertEqual(monitored["action"].iat[0], "exit_next_open")
        self.assertEqual(monitored["exit_reason"].iat[0], "gap_stop")

    def test_inspect_and_plot_work_for_executed_signal(self) -> None:
        frame, _, signal_date = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        researcher = GapBreakoutContinuationResearcher(frame)
        inspection = researcher.inspect_signal("600001", signal_date, lookback=20, lookahead=10)
        self.assertEqual(pd.Timestamp(inspection["summary"]["signal_date"]), signal_date)
        figure = researcher.plot_signal_context("600001", signal_date, lookback=20, lookahead=10)
        self.assertIsInstance(figure, go.Figure)

    def test_grid_search_runs(self) -> None:
        frame, _, _ = make_gap_case_frame(confirm_fill=False, exit_mode="ma10")
        results = run_gap_breakout_grid_search(
            frame,
            param_grid={"min_gap_pct": [0.01, 0.015], "confirm_window": [2, 3]},
            backtester_kwargs={"initial_capital": 100_000.0, "fixed_entry_notional": 10_000.0, "board_lot_size": 100},
        )
        self.assertFalse(results["summary"].empty)
        self.assertIn("figure", results)


if __name__ == "__main__":
    unittest.main()
