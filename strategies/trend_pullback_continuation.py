from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.blue_chip_range_reversion import (
    BlueChipRangeReversionResearcher,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    STRING_COLUMNS,
)


@dataclass(frozen=True)
class TrendPullbackStrategyConfig:
    """
    Parameter bundle for a trend-pullback continuation study.

    Core idea:
    1. wait for a completed bullish moving-average stack
    2. after that trend ends, look for a three-push pullback / bull-flag shape
    3. require a strong signal candle plus a follow-through bar
    4. enter on the next open only when the reward-to-risk versus the prior
       trend high is still acceptable
    """

    universe: str = "csi500"
    ma_windows: tuple[int, int, int] = (20, 60, 120)
    min_trend_bars: int = 10
    pivot_window: int = 1
    max_pullback_bars: int = 40
    max_signal_delay_after_third_low: int = 5

    min_signal_body_pct: float = 0.50
    max_signal_upper_shadow_pct: float = 0.25
    max_signal_lower_shadow_pct: float = 0.35

    stop_buffer_pct: float = 0.01
    min_reward_r: float = 1.50
    take_profit_fraction_of_trend_move: float = 0.50
    max_holding_days: int | None = None

    enable_hard_stop: bool = True
    enable_take_profit: bool = True
    enable_time_stop: bool = False

    def __post_init__(self) -> None:
        if self.universe not in {"csi500", "hs300"}:
            raise ValueError("universe must be either 'csi500' or 'hs300'.")
        if len(self.ma_windows) != 3 or any(window < 2 for window in self.ma_windows):
            raise ValueError("ma_windows must contain exactly three integers >= 2.")
        if sorted(self.ma_windows) != list(self.ma_windows):
            raise ValueError("ma_windows must be sorted from fast to slow.")
        if self.min_trend_bars < 1:
            raise ValueError("min_trend_bars must be at least 1.")
        if self.pivot_window < 1:
            raise ValueError("pivot_window must be at least 1.")
        if self.max_pullback_bars < 3:
            raise ValueError("max_pullback_bars must be at least 3.")
        if self.max_signal_delay_after_third_low < 1:
            raise ValueError("max_signal_delay_after_third_low must be at least 1.")
        if not 0 < self.min_signal_body_pct <= 1:
            raise ValueError("min_signal_body_pct must be in (0, 1].")
        if not 0 <= self.max_signal_upper_shadow_pct <= 1:
            raise ValueError("max_signal_upper_shadow_pct must be in [0, 1].")
        if not 0 <= self.max_signal_lower_shadow_pct <= 1:
            raise ValueError("max_signal_lower_shadow_pct must be in [0, 1].")
        if self.stop_buffer_pct < 0:
            raise ValueError("stop_buffer_pct must be non-negative.")
        if self.min_reward_r <= 0:
            raise ValueError("min_reward_r must be positive.")
        if not 0 < self.take_profit_fraction_of_trend_move <= 1:
            raise ValueError("take_profit_fraction_of_trend_move must be in (0, 1].")
        if self.max_holding_days is not None and self.max_holding_days < 1:
            raise ValueError("max_holding_days must be at least 1 when provided.")
        if not any([self.enable_hard_stop, self.enable_take_profit, self.enable_time_stop]):
            raise ValueError("At least one exit rule must be enabled.")


class TrendPullbackContinuationResearcher(BlueChipRangeReversionResearcher):
    """
    Research helper for a second-leg trend continuation setup.

    Definitions used here:
    - trend: a completed bullish moving-average stack ``MA20 > MA60 > MA120``
    - pullback: three confirmed local lows with lower highs/lower lows and
      contracting downswing size
    - signal candle: bullish close above the previous high with controlled
      wick proportions
    - follow-through: the next bar trades above the signal-candle high

    Implementation note:
    the reference trend high is frozen to the highest high achieved during the
    completed MA-stack segment itself. This is deliberate: the moving-average
    definition of "trend end" is lagging, so the actual price peak can easily
    occur before the stack formally breaks.
    """

    REQUIRED_COLUMNS = REQUIRED_COLUMNS
    NUMERIC_COLUMNS = NUMERIC_COLUMNS
    STRING_COLUMNS = STRING_COLUMNS
    TREND_LOW_LOOKBACK = 60
    FEATURE_COLUMNS = [
        "sma_20",
        "sma_60",
        "sma_120",
        "bullish_stack",
        "pivot_high",
        "pivot_low",
        "trend_start",
        "trend_end",
        "trend_segment_id",
        "trend_start_date",
        "trend_end_date",
        "trend_length",
        "trend_low",
        "trend_high",
        "trend_midpoint_50",
        "trend_high_date",
        "post_trend_phase",
        "bars_since_trend_end",
        "pullback_bars",
        "pullback_low_so_far",
        "pullback_low",
        "pullback_depth_pct",
        "expected_upside_to_target",
        "push1_low_date",
        "push1_low",
        "push2_low_date",
        "push2_low",
        "push3_low_date",
        "push3_low",
        "push1_rebound_high_date",
        "push1_rebound_high",
        "push2_rebound_high_date",
        "push2_rebound_high",
        "wedge_pivot_low_1",
        "wedge_pivot_low_2",
        "wedge_pivot_low_3",
        "wedge_pivot_high_1",
        "wedge_pivot_high_2",
        "lower_lows_confirmed",
        "lower_highs_confirmed",
        "drop1_pct",
        "drop2_pct",
        "drop3_pct",
        "downswing_contraction",
        "three_push_pullback",
        "close_gt_open",
        "close_gt_prev_high",
        "signal_body_pct",
        "signal_upper_shadow_pct",
        "signal_lower_shadow_pct",
        "signal_quality_ok",
        "signal_candle",
    ]
    SIGNAL_COLUMNS = [
        "follow_through_date",
        "follow_through_high",
        "follow_through_close",
        "follow_through_confirmed",
        "signal_before_trend_end",
        "entry_reference_price",
        "reward_to_risk",
        "reward_to_risk_ok",
        "signal_take_profit_price",
        "signal_hard_stop_price",
        "entry_signal",
    ]
    OUTCOME_COLUMNS = [
        "entry_signal_executed",
        "entry_signal_suppressed",
        "entry_date_next",
        "entry_open_next",
        "exit_signal_date",
        "exit_date_next",
        "exit_open_next",
        "exit_reason",
        "holding_days",
        "realized_open_to_open_return",
        "max_favorable_excursion",
        "max_adverse_excursion",
    ]
    FEATURE_ANALYSIS_COLUMNS = [
        "trend_length",
        "bars_since_trend_end",
        "pullback_bars",
        "pullback_depth_pct",
        "expected_upside_to_target",
        "reward_to_risk",
        "drop1_pct",
        "drop2_pct",
        "drop3_pct",
        "signal_body_pct",
        "signal_upper_shadow_pct",
        "signal_lower_shadow_pct",
        "holding_days",
        "max_favorable_excursion",
        "max_adverse_excursion",
    ]
    TRADE_COLUMNS = [
        "signal_date",
        "ticker",
        "ts_code",
        "name",
        "weight",
        "constituent_trade_date",
        "signal_open",
        "signal_high",
        "signal_low",
        "signal_close",
        "trend_start_date",
        "trend_end_date",
        "trend_length",
        "trend_low",
        "trend_high",
        "trend_midpoint_50",
        "trend_high_date",
        "pullback_bars",
        "pullback_low",
        "pullback_depth_pct",
        "expected_upside_to_target",
        "push1_low_date",
        "push1_low",
        "push2_low_date",
        "push2_low",
        "push3_low_date",
        "push3_low",
        "push1_rebound_high_date",
        "push1_rebound_high",
        "push2_rebound_high_date",
        "push2_rebound_high",
        "drop1_pct",
        "drop2_pct",
        "drop3_pct",
        "signal_body_pct",
        "signal_upper_shadow_pct",
        "signal_lower_shadow_pct",
        "follow_through_date",
        "follow_through_high",
        "follow_through_close",
        "signal_before_trend_end",
        "reward_to_risk",
        "entry_date",
        "entry_open",
        "signal_take_profit_price",
        "signal_hard_stop_price",
        "exit_signal_date",
        "exit_date",
        "exit_open",
        "exit_reason",
        "trade_status",
        "holding_days",
        "pnl",
        "pnl_pct",
        "max_favorable_excursion",
        "max_adverse_excursion",
    ]

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        config: TrendPullbackStrategyConfig | None = None,
        *,
        copy: bool = True,
    ) -> None:
        super().__init__(stock_candle_df, config=config or TrendPullbackStrategyConfig(), copy=copy)
        self.stock_candle_df.attrs["strategy_name"] = "trend_pullback_continuation"
        self.stock_candle_df.attrs["strategy_universe"] = self.config.universe
        self.trade_df.attrs["strategy_name"] = "trend_pullback_continuation"
        self.trade_df.attrs["strategy_universe"] = self.config.universe

    @staticmethod
    def _select_rebound_high_index(
        pivot_high_indices: list[int],
        high_values: np.ndarray,
        left_low_idx: int,
        right_low_idx: int,
    ) -> int | None:
        candidates = [idx for idx in pivot_high_indices if left_low_idx < idx < right_low_idx]
        if not candidates:
            return None
        return max(candidates, key=lambda idx: (high_values[idx], idx))

    @staticmethod
    def _nat_array(length: int) -> np.ndarray:
        return np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]")

    @staticmethod
    def _int_array(length: int) -> np.ndarray:
        return np.full(length, -1, dtype=np.int32)

    @staticmethod
    def _finalize_nullable_int(values: np.ndarray) -> pd.Series:
        finalized = pd.Series(values.astype("int64", copy=False))
        return finalized.where(finalized.ge(0), pd.NA).astype("Int64")

    @classmethod
    def _select_trend_anchor_low_index(
        cls,
        pivot_low_indices: np.ndarray,
        low_values: np.ndarray,
        *,
        trend_high_idx: int,
    ) -> int | None:
        window_start = max(0, trend_high_idx - cls.TREND_LOW_LOOKBACK + 1)
        window_end = trend_high_idx
        pivot_start = int(np.searchsorted(pivot_low_indices, window_start))
        pivot_end = int(np.searchsorted(pivot_low_indices, window_end, side="right"))
        candidate_indices = [
            int(idx)
            for idx in pivot_low_indices[pivot_start:pivot_end]
            if pd.notna(low_values[int(idx)])
        ]
        if candidate_indices:
            return min(candidate_indices, key=lambda idx: (low_values[idx], idx))

        window_values = low_values[window_start : window_end + 1]
        if window_values.size == 0:
            return None
        valid_mask = ~np.isnan(window_values)
        if not valid_mask.any():
            return None
        valid_positions = np.flatnonzero(valid_mask)
        relative_idx = int(valid_positions[np.argmin(window_values[valid_mask])])
        return window_start + relative_idx

    @staticmethod
    def _build_extended_line(
        start_date: pd.Timestamp | np.datetime64 | None,
        start_price: float | int | None,
        anchor_date: pd.Timestamp | np.datetime64 | None,
        anchor_price: float | int | None,
        *,
        extend_to: pd.Timestamp | np.datetime64 | None,
    ) -> tuple[list[pd.Timestamp], list[float]] | None:
        if any(pd.isna(value) for value in [start_date, start_price, anchor_date, anchor_price, extend_to]):
            return None

        start_ts = pd.Timestamp(start_date)
        anchor_ts = pd.Timestamp(anchor_date)
        extend_ts = pd.Timestamp(extend_to)
        if start_ts == anchor_ts:
            return None
        if extend_ts < anchor_ts:
            extend_ts = anchor_ts

        start_price_value = float(start_price)
        anchor_price_value = float(anchor_price)
        span = anchor_ts.value - start_ts.value
        slope = (anchor_price_value - start_price_value) / span
        projected_price = float(start_price_value + slope * (extend_ts.value - start_ts.value))
        return (
            [start_ts, anchor_ts, extend_ts],
            [start_price_value, anchor_price_value, projected_price],
        )

    def _ensure_research_outcomes(self) -> None:
        required_columns = self.SIGNAL_COLUMNS + self.OUTCOME_COLUMNS + ["signal_candle"]
        if not self._has_columns(required_columns):
            self.add_research_outcomes()

    def _annotate_ticker_context(self, ticker_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        cfg = self.config
        row_count = len(ticker_frame)
        if row_count == 0:
            return {}

        date_values = ticker_frame["date"].to_numpy(dtype="datetime64[ns]")
        high_values = pd.to_numeric(ticker_frame["high"], errors="coerce").to_numpy(dtype=float)
        low_values = pd.to_numeric(ticker_frame["low"], errors="coerce").to_numpy(dtype=float)
        close_values = pd.to_numeric(ticker_frame["close"], errors="coerce").to_numpy(dtype=float)
        bullish_stack = ticker_frame["bullish_stack"].fillna(False).to_numpy(dtype=bool)
        pivot_high = ticker_frame["pivot_high"].fillna(False).to_numpy(dtype=bool)
        pivot_low = ticker_frame["pivot_low"].fillna(False).to_numpy(dtype=bool)
        signal_quality_ok = ticker_frame["signal_quality_ok"].fillna(False).to_numpy(dtype=bool)
        close_gt_prev_high = ticker_frame["close_gt_prev_high"].fillna(False).to_numpy(dtype=bool)
        pivot_high_indices = np.flatnonzero(pivot_high)
        pivot_low_indices = np.flatnonzero(pivot_low)

        annotations: dict[str, np.ndarray] = {
            "trend_start": np.zeros(row_count, dtype=bool),
            "trend_end": np.zeros(row_count, dtype=bool),
            "trend_segment_id": self._int_array(row_count),
            "trend_start_date": self._nat_array(row_count),
            "trend_end_date": self._nat_array(row_count),
            "trend_length": self._int_array(row_count),
            "trend_low": np.full(row_count, np.nan, dtype=float),
            "trend_high": np.full(row_count, np.nan, dtype=float),
            "trend_midpoint_50": np.full(row_count, np.nan, dtype=float),
            "trend_high_date": self._nat_array(row_count),
            "post_trend_phase": np.zeros(row_count, dtype=bool),
            "bars_since_trend_end": self._int_array(row_count),
            "pullback_bars": self._int_array(row_count),
            "pullback_low_so_far": np.full(row_count, np.nan, dtype=float),
            "pullback_low": np.full(row_count, np.nan, dtype=float),
            "pullback_depth_pct": np.full(row_count, np.nan, dtype=float),
            "expected_upside_to_target": np.full(row_count, np.nan, dtype=float),
            "push1_low_date": self._nat_array(row_count),
            "push1_low": np.full(row_count, np.nan, dtype=float),
            "push2_low_date": self._nat_array(row_count),
            "push2_low": np.full(row_count, np.nan, dtype=float),
            "push3_low_date": self._nat_array(row_count),
            "push3_low": np.full(row_count, np.nan, dtype=float),
            "push1_rebound_high_date": self._nat_array(row_count),
            "push1_rebound_high": np.full(row_count, np.nan, dtype=float),
            "push2_rebound_high_date": self._nat_array(row_count),
            "push2_rebound_high": np.full(row_count, np.nan, dtype=float),
            "wedge_pivot_low_1": np.zeros(row_count, dtype=bool),
            "wedge_pivot_low_2": np.zeros(row_count, dtype=bool),
            "wedge_pivot_low_3": np.zeros(row_count, dtype=bool),
            "wedge_pivot_high_1": np.zeros(row_count, dtype=bool),
            "wedge_pivot_high_2": np.zeros(row_count, dtype=bool),
            "lower_lows_confirmed": np.zeros(row_count, dtype=bool),
            "lower_highs_confirmed": np.zeros(row_count, dtype=bool),
            "drop1_pct": np.full(row_count, np.nan, dtype=float),
            "drop2_pct": np.full(row_count, np.nan, dtype=float),
            "drop3_pct": np.full(row_count, np.nan, dtype=float),
            "downswing_contraction": np.zeros(row_count, dtype=bool),
            "three_push_pullback": np.zeros(row_count, dtype=bool),
            "signal_candle": np.zeros(row_count, dtype=bool),
        }

        segments: list[dict[str, int | float]] = []
        cursor = 0
        segment_id = 0
        while cursor < row_count:
            if not bullish_stack[cursor]:
                cursor += 1
                continue

            start = cursor
            trend_low = np.nan
            trend_high = np.nan
            trend_high_idx = cursor
            while cursor < row_count and bullish_stack[cursor]:
                current_low = low_values[cursor]
                current_high = high_values[cursor]
                if pd.notna(current_low) and (pd.isna(trend_low) or current_low < trend_low):
                    trend_low = float(current_low)
                if pd.notna(current_high) and (pd.isna(trend_high) or current_high >= trend_high):
                    trend_high = float(current_high)
                    trend_high_idx = cursor
                cursor += 1

            end = cursor - 1
            segment_id += 1
            segments.append(
                {
                    "segment_id": segment_id,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "trend_low": trend_low,
                    "trend_high": trend_high,
                    "trend_high_idx": trend_high_idx,
                }
            )

        for idx, segment in enumerate(segments):
            segment_length = int(segment["length"])
            start = int(segment["start"])
            end = int(segment["end"])
            next_start = int(segments[idx + 1]["start"]) if idx + 1 < len(segments) else row_count
            if segment_length < cfg.min_trend_bars:
                continue

            trend_high = float(segment["trend_high"]) if pd.notna(segment["trend_high"]) else np.nan
            trend_high_idx = int(segment["trend_high_idx"])
            anchor_low_idx = self._select_trend_anchor_low_index(
                pivot_low_indices,
                low_values,
                trend_high_idx=trend_high_idx,
            )
            if anchor_low_idx is None:
                continue

            trend_low = low_values[anchor_low_idx]
            if pd.isna(trend_high) or pd.isna(trend_low):
                continue

            trend_start_idx = int(anchor_low_idx)
            trend_length = trend_high_idx - trend_start_idx + 1
            if trend_length < cfg.min_trend_bars:
                continue

            trend_start_date = date_values[trend_start_idx]
            trend_end_date = date_values[end]
            trend_high_date = date_values[trend_high_idx]
            trend_midpoint_50 = float((float(trend_high) + float(trend_low)) / 2.0)

            annotations["trend_start"][trend_start_idx] = True
            annotations["trend_end"][end] = True

            pullback_start = trend_high_idx + 1
            if pullback_start >= next_start:
                continue

            trend_low = float(trend_low)

            running_pullback_low = np.nan
            next_pivot_low_ptr = int(np.searchsorted(pivot_low_indices, pullback_start))
            next_pivot_high_ptr = int(np.searchsorted(pivot_high_indices, pullback_start))
            selected_low_indices: list[int] = []
            selected_high_indices: list[int] = []
            pending_rebound_high_idx: int | None = None
            setup_invalidated = False

            for current_loc in range(pullback_start, next_start):
                previous_loc = current_loc - 1
                if previous_loc >= pullback_start:
                    while True:
                        next_low_idx = (
                            int(pivot_low_indices[next_pivot_low_ptr])
                            if next_pivot_low_ptr < len(pivot_low_indices)
                            else row_count
                        )
                        next_high_idx = (
                            int(pivot_high_indices[next_pivot_high_ptr])
                            if next_pivot_high_ptr < len(pivot_high_indices)
                            else row_count
                        )
                        next_event_idx = min(next_low_idx, next_high_idx)
                        if next_event_idx > previous_loc:
                            break

                        if next_low_idx <= next_high_idx:
                            pivot_low_idx = next_low_idx
                            next_pivot_low_ptr += 1
                            if setup_invalidated:
                                continue

                            pivot_low_value = low_values[pivot_low_idx]
                            if pd.isna(pivot_low_value):
                                continue

                            if not selected_low_indices:
                                selected_low_indices.append(pivot_low_idx)
                                pending_rebound_high_idx = None
                                continue

                            last_low_idx = selected_low_indices[-1]
                            last_low_value = low_values[last_low_idx]
                            if pd.isna(last_low_value) or pivot_low_value >= last_low_value:
                                continue

                            if pending_rebound_high_idx is None:
                                # Still the same downswing leg. Keep the latest
                                # lower pivot as the leg low until a rebound
                                # high is confirmed.
                                selected_low_indices[-1] = pivot_low_idx
                                continue

                            if len(selected_low_indices) >= 3:
                                setup_invalidated = True
                                selected_low_indices = []
                                selected_high_indices = []
                                pending_rebound_high_idx = None
                                continue

                            selected_high_indices.append(pending_rebound_high_idx)
                            selected_low_indices.append(pivot_low_idx)
                            pending_rebound_high_idx = None
                            continue

                        pivot_high_idx = next_high_idx
                        next_pivot_high_ptr += 1
                        if setup_invalidated or not selected_low_indices:
                            continue
                        if pivot_high_idx <= selected_low_indices[-1]:
                            continue
                        if (
                            pending_rebound_high_idx is None
                            or high_values[pivot_high_idx] >= high_values[pending_rebound_high_idx]
                        ):
                            pending_rebound_high_idx = pivot_high_idx

                current_low = low_values[current_loc]
                if pd.notna(current_low):
                    if pd.isna(running_pullback_low):
                        running_pullback_low = float(current_low)
                    else:
                        running_pullback_low = float(min(running_pullback_low, current_low))

                annotations["trend_segment_id"][current_loc] = int(segment["segment_id"])
                annotations["trend_start_date"][current_loc] = trend_start_date
                annotations["trend_end_date"][current_loc] = trend_end_date
                annotations["trend_length"][current_loc] = trend_length
                annotations["trend_low"][current_loc] = trend_low
                annotations["trend_high"][current_loc] = trend_high
                annotations["trend_midpoint_50"][current_loc] = trend_midpoint_50
                annotations["trend_high_date"][current_loc] = trend_high_date
                annotations["post_trend_phase"][current_loc] = True
                if current_loc > end:
                    annotations["bars_since_trend_end"][current_loc] = int(current_loc - end)
                annotations["pullback_bars"][current_loc] = int(current_loc - pullback_start + 1)
                annotations["pullback_low_so_far"][current_loc] = running_pullback_low
                if pd.notna(trend_high) and trend_high > 0 and pd.notna(running_pullback_low):
                    annotations["pullback_depth_pct"][current_loc] = float((trend_high - running_pullback_low) / trend_high)
                if pd.notna(trend_high) and pd.notna(close_values[current_loc]) and close_values[current_loc] > 0:
                    annotations["expected_upside_to_target"][current_loc] = float(
                        trend_high / close_values[current_loc] - 1.0
                    )

                if annotations["pullback_bars"][current_loc] > cfg.max_pullback_bars:
                    continue
                if setup_invalidated or len(selected_low_indices) != 3 or len(selected_high_indices) != 2:
                    continue

                push1_low_idx, push2_low_idx, push3_low_idx = selected_low_indices
                rebound1_idx, rebound2_idx = selected_high_indices
                push1_low = low_values[push1_low_idx]
                push2_low = low_values[push2_low_idx]
                push3_low = low_values[push3_low_idx]
                rebound1_high = high_values[rebound1_idx]
                rebound2_high = high_values[rebound2_idx]
                if any(
                    pd.isna(value)
                    for value in [trend_high, push1_low, push2_low, push3_low, rebound1_high, rebound2_high]
                ):
                    continue

                push1_low = float(push1_low)
                push2_low = float(push2_low)
                push3_low = float(push3_low)
                rebound1_high = float(rebound1_high)
                rebound2_high = float(rebound2_high)
                lower_lows_confirmed = bool(push1_low > push2_low > push3_low)
                lower_highs_confirmed = bool(trend_high > rebound1_high > rebound2_high)
                drop1_pct = float(trend_high / push1_low - 1.0) if push1_low > 0 else np.nan
                drop2_pct = float(rebound1_high / push2_low - 1.0) if push2_low > 0 else np.nan
                drop3_pct = float(rebound2_high / push3_low - 1.0) if push3_low > 0 else np.nan
                downswing_contraction = bool(
                    pd.notna(drop1_pct)
                    and pd.notna(drop2_pct)
                    and pd.notna(drop3_pct)
                    and drop1_pct > drop2_pct > drop3_pct > 0
                )
                three_push_pullback = lower_lows_confirmed and lower_highs_confirmed and downswing_contraction

                annotations["push1_low_date"][current_loc] = date_values[push1_low_idx]
                annotations["push1_low"][current_loc] = push1_low
                annotations["push2_low_date"][current_loc] = date_values[push2_low_idx]
                annotations["push2_low"][current_loc] = push2_low
                annotations["push3_low_date"][current_loc] = date_values[push3_low_idx]
                annotations["push3_low"][current_loc] = push3_low
                annotations["push1_rebound_high_date"][current_loc] = date_values[rebound1_idx]
                annotations["push1_rebound_high"][current_loc] = rebound1_high
                annotations["push2_rebound_high_date"][current_loc] = date_values[rebound2_idx]
                annotations["push2_rebound_high"][current_loc] = rebound2_high
                annotations["lower_lows_confirmed"][current_loc] = lower_lows_confirmed
                annotations["lower_highs_confirmed"][current_loc] = lower_highs_confirmed
                annotations["drop1_pct"][current_loc] = drop1_pct
                annotations["drop2_pct"][current_loc] = drop2_pct
                annotations["drop3_pct"][current_loc] = drop3_pct
                annotations["downswing_contraction"][current_loc] = downswing_contraction
                annotations["three_push_pullback"][current_loc] = three_push_pullback
                annotations["pullback_low"][current_loc] = float(min(push1_low, push2_low, push3_low))

                signal_delay = current_loc - push3_low_idx
                if signal_delay < 1 or signal_delay > cfg.max_signal_delay_after_third_low:
                    continue

                signal_candle = bool(
                    three_push_pullback
                    and signal_quality_ok[current_loc]
                    and close_gt_prev_high[current_loc]
                )
                annotations["signal_candle"][current_loc] = signal_candle
                if signal_candle:
                    annotations["wedge_pivot_low_1"][push1_low_idx] = True
                    annotations["wedge_pivot_low_2"][push2_low_idx] = True
                    annotations["wedge_pivot_low_3"][push3_low_idx] = True
                    annotations["wedge_pivot_high_1"][rebound1_idx] = True
                    annotations["wedge_pivot_high_2"][rebound2_idx] = True

        return annotations

    def add_features(self) -> pd.DataFrame:
        if self._has_columns(self.FEATURE_COLUMNS):
            return self.stock_candle_df

        cfg = self.config
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        for window in cfg.ma_windows:
            df[f"sma_{window}"] = ticker_group["close"].transform(
                lambda series, current_window=window: self._rolling_sma(series, current_window)
            )

        fast_window, mid_window, slow_window = cfg.ma_windows
        df["bullish_stack"] = (
            df[f"sma_{fast_window}"].gt(df[f"sma_{mid_window}"])
            & df[f"sma_{mid_window}"].gt(df[f"sma_{slow_window}"])
        )

        previous_high = ticker_group["high"].shift(1)
        df["close_gt_open"] = df["close"].gt(df["open"])
        df["close_gt_prev_high"] = df["close"].gt(previous_high)

        candle_range = df["high"] - df["low"]
        body = (df["close"] - df["open"]).abs()
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        valid_range = candle_range.gt(0)

        df["signal_body_pct"] = np.where(valid_range, body.div(candle_range), 0.0)
        df["signal_upper_shadow_pct"] = np.where(valid_range, upper_shadow.div(candle_range), 0.0)
        df["signal_lower_shadow_pct"] = np.where(valid_range, lower_shadow.div(candle_range), 0.0)
        df["signal_quality_ok"] = (
            df["close_gt_open"]
            & df["signal_body_pct"].ge(cfg.min_signal_body_pct)
            & df["signal_upper_shadow_pct"].le(cfg.max_signal_upper_shadow_pct)
            & df["signal_lower_shadow_pct"].le(cfg.max_signal_lower_shadow_pct)
        )

        pivot_high = pd.Series(True, index=df.index)
        pivot_low = pd.Series(True, index=df.index)
        for offset in range(1, cfg.pivot_window + 1):
            prev_high = ticker_group["high"].shift(offset)
            next_high = ticker_group["high"].shift(-offset)
            prev_low = ticker_group["low"].shift(offset)
            next_low = ticker_group["low"].shift(-offset)
            pivot_high &= df["high"].gt(prev_high) & df["high"].ge(next_high)
            pivot_low &= df["low"].lt(prev_low) & df["low"].le(next_low)
        df["pivot_high"] = pivot_high.fillna(False)
        df["pivot_low"] = pivot_low.fillna(False)

        row_count = len(df)
        date_columns = [
            "trend_start_date",
            "trend_end_date",
            "trend_high_date",
            "push1_low_date",
            "push2_low_date",
            "push3_low_date",
            "push1_rebound_high_date",
            "push2_rebound_high_date",
        ]
        float_columns = [
            "trend_low",
            "trend_high",
            "trend_midpoint_50",
            "pullback_low_so_far",
            "pullback_low",
            "pullback_depth_pct",
            "expected_upside_to_target",
            "push1_low",
            "push2_low",
            "push3_low",
            "push1_rebound_high",
            "push2_rebound_high",
            "drop1_pct",
            "drop2_pct",
            "drop3_pct",
        ]
        int_columns = ["trend_segment_id", "trend_length", "bars_since_trend_end", "pullback_bars"]
        bool_columns = [
            "trend_start",
            "trend_end",
            "post_trend_phase",
            "wedge_pivot_low_1",
            "wedge_pivot_low_2",
            "wedge_pivot_low_3",
            "wedge_pivot_high_1",
            "wedge_pivot_high_2",
            "lower_lows_confirmed",
            "lower_highs_confirmed",
            "downswing_contraction",
            "three_push_pullback",
            "signal_candle",
        ]

        annotations: dict[str, np.ndarray] = {}
        for column in date_columns:
            annotations[column] = self._nat_array(row_count)
        for column in float_columns:
            annotations[column] = np.full(row_count, np.nan, dtype=float)
        for column in int_columns:
            annotations[column] = self._int_array(row_count)
        for column in bool_columns:
            annotations[column] = np.zeros(row_count, dtype=bool)

        for _, group_index in df.groupby("ticker", sort=False).groups.items():
            positions = np.asarray(group_index, dtype=int)
            group_frame = df.loc[positions].reset_index(drop=True)
            group_annotations = self._annotate_ticker_context(group_frame)
            if not group_annotations:
                continue
            for column, values in group_annotations.items():
                annotations[column][positions] = values

        for column in date_columns:
            df[column] = pd.to_datetime(annotations[column])
        for column in float_columns:
            df[column] = annotations[column]
        for column in int_columns:
            df[column] = self._finalize_nullable_int(annotations[column])
        for column in bool_columns:
            df[column] = annotations[column]

        return self._store_output(df)

    def add_signals(self) -> pd.DataFrame:
        if self._has_columns(self.SIGNAL_COLUMNS):
            return self.stock_candle_df

        cfg = self.config
        self.add_features()
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        df["follow_through_date"] = ticker_group["date"].shift(-1)
        df["follow_through_high"] = ticker_group["high"].shift(-1)
        df["follow_through_close"] = ticker_group["close"].shift(-1)
        df["entry_date_next"] = ticker_group["date"].shift(-2)
        df["entry_open_next"] = ticker_group["open"].shift(-2)

        df["follow_through_confirmed"] = df["signal_candle"] & df["follow_through_high"].gt(df["high"])
        df["signal_before_trend_end"] = df["date"].lt(df["trend_end_date"]).fillna(False)
        df["signal_hard_stop_price"] = df["pullback_low"] * (1.0 - cfg.stop_buffer_pct)
        df["entry_reference_price"] = df["follow_through_close"]
        trend_move_to_high = df["trend_high"] - df["entry_reference_price"]
        df["signal_take_profit_price"] = np.where(
            df["entry_reference_price"].notna() & df["trend_high"].notna(),
            df["entry_reference_price"] + cfg.take_profit_fraction_of_trend_move * trend_move_to_high,
            np.nan,
        )

        risk_amount = df["entry_reference_price"] - df["signal_hard_stop_price"]
        reward_amount = df["signal_take_profit_price"] - df["entry_reference_price"]
        df["reward_to_risk"] = np.where(
            risk_amount.gt(0) & reward_amount.gt(0),
            reward_amount.div(risk_amount),
            np.nan,
        )
        df["reward_to_risk_ok"] = df["reward_to_risk"].ge(cfg.min_reward_r)
        df["entry_signal"] = (
            df["signal_candle"]
            & df["follow_through_confirmed"]
            & ~df["signal_before_trend_end"]
            & df["reward_to_risk_ok"]
            & df["entry_date_next"].notna()
            & df["entry_open_next"].gt(0)
        )

        return self._store_output(df)

    def add_research_outcomes(self) -> pd.DataFrame:
        cfg = self.config
        self.add_signals()
        df = self._sort_for_calculation(self.stock_candle_df)

        df["entry_signal_executed"] = False
        df["entry_signal_suppressed"] = False
        df["exit_signal_date"] = pd.NaT
        df["exit_date_next"] = pd.NaT
        df["exit_open_next"] = np.nan
        df["exit_reason"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["holding_days"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["realized_open_to_open_return"] = np.nan
        df["max_favorable_excursion"] = np.nan
        df["max_adverse_excursion"] = np.nan

        for _, group_index in df.groupby("ticker", sort=False).groups.items():
            group = df.loc[group_index].copy().reset_index()
            next_search_loc = 0

            while next_search_loc < len(group):
                signal_row = group.iloc[next_search_loc]
                if not bool(signal_row["entry_signal"]):
                    next_search_loc += 1
                    continue

                signal_loc = next_search_loc
                signal_idx = int(signal_row["index"])
                entry_loc = signal_loc + 2
                if entry_loc >= len(group):
                    break

                entry_row = group.iloc[entry_loc]
                entry_price = entry_row["open"]
                if pd.isna(entry_price) or entry_price <= 0:
                    next_search_loc += 1
                    continue

                take_profit_price = signal_row["signal_take_profit_price"]
                stop_price = signal_row["signal_hard_stop_price"]
                exit_signal_loc: int | None = None
                exit_reason: str | None = None

                for eval_loc in range(entry_loc, len(group)):
                    eval_row = group.iloc[eval_loc]
                    eval_low = eval_row["low"]
                    eval_high = eval_row["high"]

                    # Conservative same-day ambiguity handling: if both target
                    # and stop are touched inside a daily bar, assume the stop
                    # was hit first.
                    if (
                        cfg.enable_hard_stop
                        and pd.notna(eval_low)
                        and pd.notna(stop_price)
                        and eval_low <= stop_price
                    ):
                        exit_signal_loc = eval_loc
                        exit_reason = "hard_stop"
                        break

                    if (
                        cfg.enable_take_profit
                        and pd.notna(eval_high)
                        and pd.notna(take_profit_price)
                        and eval_high >= take_profit_price
                    ):
                        exit_signal_loc = eval_loc
                        exit_reason = "take_profit"
                        break

                    if (
                        cfg.enable_time_stop
                        and cfg.max_holding_days is not None
                        and eval_loc - entry_loc + 1 >= cfg.max_holding_days
                    ):
                        exit_signal_loc = eval_loc
                        exit_reason = "time_stop"
                        break

                executed_exit_loc: int | None = None
                if exit_signal_loc is not None and exit_signal_loc + 1 < len(group):
                    executed_exit_loc = exit_signal_loc + 1
                else:
                    exit_reason = "open_position"

                path_end_loc = exit_signal_loc if exit_signal_loc is not None else len(group) - 1
                path_slice = group.iloc[entry_loc : path_end_loc + 1]
                path_high = path_slice["high"].where(path_slice["high"].gt(0), path_slice["close"])
                path_low = path_slice["low"].where(path_slice["low"].gt(0), path_slice["close"])
                if entry_price > 0 and not path_slice.empty:
                    df.at[signal_idx, "max_favorable_excursion"] = float(path_high.max() / entry_price - 1.0)
                    df.at[signal_idx, "max_adverse_excursion"] = float(path_low.min() / entry_price - 1.0)

                df.at[signal_idx, "entry_signal_executed"] = True
                df.at[signal_idx, "entry_date_next"] = entry_row["date"]
                df.at[signal_idx, "entry_open_next"] = float(entry_price)
                df.at[signal_idx, "exit_reason"] = exit_reason

                if exit_signal_loc is not None:
                    df.at[signal_idx, "exit_signal_date"] = group.iloc[exit_signal_loc]["date"]

                if executed_exit_loc is not None:
                    exit_row = group.iloc[executed_exit_loc]
                    exit_open = exit_row["open"]
                    if pd.notna(exit_open) and exit_open > 0:
                        df.at[signal_idx, "exit_date_next"] = exit_row["date"]
                        df.at[signal_idx, "exit_open_next"] = float(exit_open)
                        df.at[signal_idx, "realized_open_to_open_return"] = float(exit_open / entry_price - 1.0)
                        df.at[signal_idx, "holding_days"] = int(executed_exit_loc - entry_loc)
                    suppressed_rows = group.iloc[signal_loc + 1 : executed_exit_loc]
                    next_search_loc = executed_exit_loc
                else:
                    df.at[signal_idx, "holding_days"] = int(len(group) - 1 - entry_loc)
                    suppressed_rows = group.iloc[signal_loc + 1 :]
                    next_search_loc = len(group)

                suppressed_indices = suppressed_rows.loc[suppressed_rows["entry_signal"], "index"]
                if not suppressed_indices.empty:
                    df.loc[suppressed_indices.astype(int), "entry_signal_suppressed"] = True

        return self._store_output(df)

    def get_candidates(self, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if "signal_candle" not in self.stock_candle_df.columns:
            self.add_signals()

        base_columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "trend_end_date",
            "trend_high",
            "pullback_low",
            "pullback_depth_pct",
            "expected_upside_to_target",
            "signal_before_trend_end",
            "drop1_pct",
            "drop2_pct",
            "drop3_pct",
            "signal_body_pct",
            "signal_upper_shadow_pct",
            "signal_lower_shadow_pct",
            "close_gt_prev_high",
            "signal_quality_ok",
            "three_push_pullback",
            "signal_candle",
        ]
        df = self.stock_candle_df.copy()
        candidates = df[df["signal_candle"]].copy()
        if candidates.empty:
            optional_columns = [column for column in self.OUTCOME_COLUMNS if column in df.columns]
            selected_columns = [column for column in base_columns + optional_columns if column in df.columns]
            return candidates.reindex(columns=selected_columns).reset_index(drop=True)

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else candidates["date"].max()
        candidates = candidates[candidates["date"] == target_date].copy()
        candidates = candidates.sort_values(
            ["expected_upside_to_target", "signal_body_pct", "ticker"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )
        optional_columns = [column for column in self.OUTCOME_COLUMNS if column in candidates.columns]
        selected_columns = [column for column in base_columns + optional_columns if column in candidates.columns]
        return candidates.loc[:, selected_columns].reset_index(drop=True)

    def get_next_session_candidates(
        self,
        as_of_date: str | pd.Timestamp | None = None,
        *,
        next_trade_date: str | pd.Timestamp | None = None,
        entry_price_basis: str = "follow_through_close",
    ) -> pd.DataFrame:
        if entry_price_basis not in {"follow_through_close", "signal_close"}:
            raise ValueError("entry_price_basis must be either 'follow_through_close' or 'signal_close'.")

        self.add_signals()
        df = self._sort_for_calculation(self.stock_candle_df.copy())
        output_columns = [
            "signal_date",
            "follow_through_date",
            "planned_entry_date",
            "ticker",
            "ts_code",
            "name",
            "entry_price_basis",
            "entry_reference_price",
            "planned_hard_stop_price",
            "planned_take_profit_price",
            "trend_end_date",
            "trend_high",
            "pullback_low",
            "pullback_depth_pct",
            "expected_upside_to_target",
            "signal_before_trend_end",
            "reward_to_risk",
            "follow_through_confirmed",
            "reward_to_risk_ok",
            "entry_signal_live",
        ]
        if df.empty:
            return pd.DataFrame(columns=output_columns)

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else df["date"].max()
        planned_entry_date = pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)

        candidates = df[
            df["follow_through_date"].eq(target_date)
            & df["signal_candle"]
            & df["follow_through_confirmed"]
            & ~df["signal_before_trend_end"]
            & df["reward_to_risk_ok"]
        ].copy()
        if candidates.empty:
            return pd.DataFrame(columns=output_columns)

        candidates["signal_date"] = candidates["date"]
        candidates["planned_entry_date"] = planned_entry_date
        candidates["entry_price_basis"] = entry_price_basis
        if entry_price_basis == "follow_through_close":
            candidates["entry_reference_price"] = candidates["follow_through_close"]
        else:
            candidates["entry_reference_price"] = candidates["close"]
        candidates["planned_hard_stop_price"] = candidates["signal_hard_stop_price"]
        candidates["planned_take_profit_price"] = candidates["signal_take_profit_price"]
        candidates["entry_signal_live"] = True
        candidates = candidates.sort_values(
            ["reward_to_risk", "expected_upside_to_target", "ticker"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )
        return candidates.loc[:, output_columns].reset_index(drop=True)

    def monitor_positions(
        self,
        positions_df: pd.DataFrame,
        as_of_date: str | pd.Timestamp | None = None,
        *,
        next_trade_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if not isinstance(positions_df, pd.DataFrame):
            raise TypeError("positions_df must be a pandas DataFrame.")

        required_columns = ["ticker", "entry_date", "entry_price"]
        missing = [column for column in required_columns if column not in positions_df.columns]
        if missing:
            raise ValueError(f"positions_df is missing required columns: {missing}")

        computed_columns = [
            "name",
            "as_of_date",
            "latest_bar_date",
            "signal_date_resolved",
            "trend_end_date",
            "trend_high",
            "pullback_low",
            "current_close",
            "pnl_pct",
            "pnl_amount",
            "holding_days",
            "trading_days_in_trade",
            "days_until_time_stop",
            "hard_stop_price",
            "take_profit_price",
            "reward_to_risk",
            "exit_signal",
            "exit_signal_date",
            "planned_exit_date",
            "exit_reason",
            "action",
            "issue",
        ]
        output_columns = list(dict.fromkeys(list(positions_df.columns) + computed_columns))

        self.add_signals()
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        if scored.empty:
            return pd.DataFrame(columns=output_columns)

        cfg = self.config
        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else scored["date"].max()
        resolved_next_trade_date = (
            pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)
        )

        positions = positions_df.copy()
        positions["ticker"] = positions["ticker"].astype("string")
        positions["entry_date"] = pd.to_datetime(positions["entry_date"])
        positions["entry_price"] = pd.to_numeric(positions["entry_price"], errors="coerce")
        if "signal_date" in positions.columns:
            positions["signal_date"] = pd.to_datetime(positions["signal_date"], errors="coerce")
        if "shares" in positions.columns:
            positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce")

        records: list[dict[str, object]] = []
        scored["ticker"] = scored["ticker"].astype("string")

        for _, position in positions.iterrows():
            ticker = str(position["ticker"])
            entry_date = pd.to_datetime(position["entry_date"])
            entry_price = float(position["entry_price"]) if pd.notna(position["entry_price"]) else np.nan
            ticker_frame = scored[
                scored["ticker"].eq(ticker) & scored["date"].le(target_date)
            ].reset_index(drop=True)

            record = {column: position[column] if column in position.index else pd.NA for column in positions.columns}
            record.update(
                {
                    "name": pd.NA,
                    "as_of_date": target_date,
                    "latest_bar_date": pd.NaT,
                    "signal_date_resolved": pd.NaT,
                    "trend_end_date": pd.NaT,
                    "trend_high": np.nan,
                    "pullback_low": np.nan,
                    "current_close": np.nan,
                    "pnl_pct": np.nan,
                    "pnl_amount": np.nan,
                    "holding_days": pd.NA,
                    "trading_days_in_trade": pd.NA,
                    "days_until_time_stop": pd.NA,
                    "hard_stop_price": np.nan,
                    "take_profit_price": np.nan,
                    "reward_to_risk": np.nan,
                    "exit_signal": False,
                    "exit_signal_date": pd.NaT,
                    "planned_exit_date": pd.NaT,
                    "exit_reason": pd.NA,
                    "action": "data_issue",
                    "issue": pd.NA,
                }
            )

            if ticker_frame.empty:
                record["issue"] = "ticker is not present in the price data on or before as_of_date"
                records.append(record)
                continue

            latest_row = ticker_frame.iloc[-1]
            record["name"] = latest_row["name"]
            record["latest_bar_date"] = latest_row["date"]
            record["current_close"] = float(latest_row["close"]) if pd.notna(latest_row["close"]) else np.nan

            if pd.isna(entry_price) or entry_price <= 0:
                record["issue"] = "entry_price must be a positive number"
                records.append(record)
                continue

            if entry_date > latest_row["date"]:
                record["issue"] = "entry_date is after the latest available bar"
                records.append(record)
                continue

            entry_matches = ticker_frame.index[ticker_frame["date"].eq(entry_date)]
            if len(entry_matches) == 0:
                record["issue"] = "entry_date does not match a trading day for this ticker"
                records.append(record)
                continue

            entry_loc = int(entry_matches[0])
            signal_date = position["signal_date"] if "signal_date" in position.index else pd.NaT
            if pd.notna(signal_date):
                signal_matches = ticker_frame.index[ticker_frame["date"].eq(pd.to_datetime(signal_date))]
                if len(signal_matches) == 0:
                    record["issue"] = "signal_date is not available in the price data"
                    records.append(record)
                    continue
                signal_loc = int(signal_matches[0])
            else:
                signal_loc = entry_loc - 2

            if signal_loc < 0 or signal_loc >= entry_loc:
                record["issue"] = "unable to resolve a valid signal row before entry_date"
                records.append(record)
                continue

            signal_row = ticker_frame.iloc[signal_loc]
            current_loc = len(ticker_frame) - 1
            record["signal_date_resolved"] = signal_row["date"]
            record["trend_end_date"] = signal_row["trend_end_date"]
            record["trend_high"] = float(signal_row["trend_high"]) if pd.notna(signal_row["trend_high"]) else np.nan
            record["pullback_low"] = float(signal_row["pullback_low"]) if pd.notna(signal_row["pullback_low"]) else np.nan
            record["reward_to_risk"] = float(signal_row["reward_to_risk"]) if pd.notna(signal_row["reward_to_risk"]) else np.nan
            record["holding_days"] = int(current_loc - entry_loc)
            record["trading_days_in_trade"] = int(current_loc - entry_loc + 1)
            if cfg.max_holding_days is not None:
                record["days_until_time_stop"] = max(cfg.max_holding_days - int(record["trading_days_in_trade"]), 0)

            if pd.notna(record["current_close"]) and entry_price > 0:
                record["pnl_pct"] = float(record["current_close"] / entry_price - 1.0)
                if "shares" in position.index and pd.notna(position["shares"]):
                    record["pnl_amount"] = float(float(position["shares"]) * (record["current_close"] - entry_price))

            hard_stop_price = signal_row["signal_hard_stop_price"]
            take_profit_price = signal_row["signal_take_profit_price"]
            record["hard_stop_price"] = float(hard_stop_price) if pd.notna(hard_stop_price) and cfg.enable_hard_stop else np.nan
            record["take_profit_price"] = (
                float(take_profit_price) if pd.notna(take_profit_price) and cfg.enable_take_profit else np.nan
            )

            exit_signal_loc: int | None = None
            exit_reason: str | None = None
            for eval_loc in range(entry_loc, current_loc + 1):
                eval_row = ticker_frame.iloc[eval_loc]
                eval_low = eval_row["low"]
                eval_high = eval_row["high"]

                if (
                    cfg.enable_hard_stop
                    and pd.notna(eval_low)
                    and pd.notna(hard_stop_price)
                    and eval_low <= hard_stop_price
                ):
                    exit_signal_loc = eval_loc
                    exit_reason = "hard_stop"
                    break

                if (
                    cfg.enable_take_profit
                    and pd.notna(eval_high)
                    and pd.notna(take_profit_price)
                    and eval_high >= take_profit_price
                ):
                    exit_signal_loc = eval_loc
                    exit_reason = "take_profit"
                    break

                if (
                    cfg.enable_time_stop
                    and cfg.max_holding_days is not None
                    and eval_loc - entry_loc + 1 >= cfg.max_holding_days
                ):
                    exit_signal_loc = eval_loc
                    exit_reason = "time_stop"
                    break

            if exit_reason is None:
                record["action"] = "hold"
                records.append(record)
                continue

            record["exit_signal"] = True
            record["exit_signal_date"] = ticker_frame.iloc[exit_signal_loc]["date"] if exit_signal_loc is not None else pd.NaT
            record["exit_reason"] = exit_reason
            if exit_signal_loc is not None and exit_signal_loc < current_loc:
                record["planned_exit_date"] = ticker_frame.iloc[exit_signal_loc + 1]["date"]
                record["action"] = "exit_overdue"
            else:
                record["planned_exit_date"] = resolved_next_trade_date
                record["action"] = "exit_next_open"

            records.append(record)

        monitored = pd.DataFrame(records)
        if monitored.empty:
            return pd.DataFrame(columns=output_columns)

        action_priority = {"exit_overdue": 0, "exit_next_open": 1, "hold": 2, "data_issue": 3}
        monitored["_action_priority"] = monitored["action"].map(action_priority).fillna(99)
        monitored = monitored.sort_values(
            ["_action_priority", "ticker"],
            ascending=[True, True],
            kind="mergesort",
            ignore_index=True,
        ).drop(columns="_action_priority")
        return monitored.reindex(columns=output_columns)

    def inspect_signal(
        self,
        ticker: str,
        signal_date: str | pd.Timestamp,
        *,
        lookback: int = 60,
        lookahead: int = 10,
    ) -> dict[str, pd.DataFrame | dict[str, object]]:
        if lookback < 0 or lookahead < 0:
            raise ValueError("lookback and lookahead must be non-negative.")

        self._ensure_research_outcomes()
        target_date = pd.to_datetime(signal_date)
        ticker = str(ticker)
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        ticker_frame = scored[scored["ticker"].astype(str).eq(ticker)].reset_index(drop=True)
        if ticker_frame.empty:
            raise ValueError(f"Ticker '{ticker}' is not present in stock_candle_df.")

        signal_rows = ticker_frame[ticker_frame["date"] == target_date]
        if signal_rows.empty:
            raise ValueError(f"Ticker '{ticker}' does not have data on {target_date.date()}.")

        executed_rows = signal_rows[signal_rows["entry_signal_executed"]]
        if executed_rows.empty:
            if bool(signal_rows["entry_signal_suppressed"].fillna(False).any()):
                raise ValueError(
                    f"Ticker '{ticker}' on {target_date.date()} was a suppressed signal, not an executed entry."
                )
            raise ValueError(
                f"Ticker '{ticker}' does not have an executed signal on {target_date.date()}."
            )

        signal_row = executed_rows.iloc[[0]].copy().reset_index(drop=True)
        signal_loc = int(ticker_frame.index[ticker_frame["date"] == target_date][0])
        exit_date = signal_row["exit_date_next"].iat[0] if "exit_date_next" in signal_row.columns else pd.NaT
        if pd.notna(exit_date):
            exit_locs = ticker_frame.index[ticker_frame["date"] == exit_date]
            event_end_loc = int(exit_locs[0]) if len(exit_locs) else signal_loc
        else:
            event_end_loc = min(len(ticker_frame) - 1, signal_loc + lookahead)

        start_loc = max(0, signal_loc - lookback)
        end_loc = min(len(ticker_frame), event_end_loc + lookahead + 1)
        price_window = ticker_frame.iloc[start_loc:end_loc].copy().reset_index(drop=True)
        price_window["signal_marker"] = price_window["date"].eq(target_date)
        price_window["follow_through_marker"] = price_window["date"].eq(signal_row["follow_through_date"].iat[0])
        price_window["entry_marker"] = price_window["date"].eq(signal_row["entry_date_next"].iat[0])
        price_window["exit_marker"] = price_window["date"].eq(signal_row["exit_date_next"].iat[0])

        checklist = pd.DataFrame(
            {
                "condition": [
                    "post_trend_phase",
                    "three_push_pullback",
                    "lower_lows_confirmed",
                    "lower_highs_confirmed",
                    "downswing_contraction",
                    "close_gt_prev_high",
                    "signal_quality_ok",
                    "signal_candle",
                    "follow_through_confirmed",
                    "signal_before_trend_end",
                    "reward_to_risk_ok",
                    "entry_signal",
                    "entry_signal_executed",
                ],
                "value": [
                    bool(signal_row["post_trend_phase"].iat[0]),
                    bool(signal_row["three_push_pullback"].iat[0]),
                    bool(signal_row["lower_lows_confirmed"].iat[0]),
                    bool(signal_row["lower_highs_confirmed"].iat[0]),
                    bool(signal_row["downswing_contraction"].iat[0]),
                    bool(signal_row["close_gt_prev_high"].iat[0]),
                    bool(signal_row["signal_quality_ok"].iat[0]),
                    bool(signal_row["signal_candle"].iat[0]),
                    bool(signal_row["follow_through_confirmed"].iat[0]),
                    not bool(signal_row["signal_before_trend_end"].iat[0]),
                    bool(signal_row["reward_to_risk_ok"].iat[0]),
                    bool(signal_row["entry_signal"].iat[0]),
                    bool(signal_row["entry_signal_executed"].iat[0]),
                ],
            }
        )

        summary = {
            "ticker": ticker,
            "signal_date": target_date,
            "follow_through_date": signal_row["follow_through_date"].iat[0],
            "raw_signal": bool(signal_row["signal_candle"].iat[0]),
            "executed_signal": bool(signal_row["entry_signal_executed"].iat[0]),
            "signal_before_trend_end": bool(signal_row["signal_before_trend_end"].iat[0]),
            "trend_start_date": signal_row["trend_start_date"].iat[0],
            "trend_end_date": signal_row["trend_end_date"].iat[0],
            "trend_low": float(signal_row["trend_low"].iat[0]) if pd.notna(signal_row["trend_low"].iat[0]) else np.nan,
            "trend_high": float(signal_row["trend_high"].iat[0]) if pd.notna(signal_row["trend_high"].iat[0]) else np.nan,
            "trend_midpoint_50": float(signal_row["trend_midpoint_50"].iat[0])
            if "trend_midpoint_50" in signal_row.columns and pd.notna(signal_row["trend_midpoint_50"].iat[0])
            else np.nan,
            "pullback_low": float(signal_row["pullback_low"].iat[0]) if pd.notna(signal_row["pullback_low"].iat[0]) else np.nan,
            "reward_to_risk": float(signal_row["reward_to_risk"].iat[0]) if pd.notna(signal_row["reward_to_risk"].iat[0]) else np.nan,
            "entry_date_next": signal_row["entry_date_next"].iat[0],
            "entry_open_next": float(signal_row["entry_open_next"].iat[0])
            if pd.notna(signal_row["entry_open_next"].iat[0])
            else np.nan,
            "exit_date_next": signal_row["exit_date_next"].iat[0],
            "exit_reason": None if pd.isna(signal_row["exit_reason"].iat[0]) else str(signal_row["exit_reason"].iat[0]),
            "holding_days": None if pd.isna(signal_row["holding_days"].iat[0]) else int(signal_row["holding_days"].iat[0]),
            "realized_open_to_open_return": float(signal_row["realized_open_to_open_return"].iat[0])
            if pd.notna(signal_row["realized_open_to_open_return"].iat[0])
            else np.nan,
            "max_favorable_excursion": float(signal_row["max_favorable_excursion"].iat[0])
            if pd.notna(signal_row["max_favorable_excursion"].iat[0])
            else np.nan,
            "max_adverse_excursion": float(signal_row["max_adverse_excursion"].iat[0])
            if pd.notna(signal_row["max_adverse_excursion"].iat[0])
            else np.nan,
        }

        signal_columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "open",
            "high",
            "low",
            "close",
            "sma_20",
            "sma_60",
            "sma_120",
            "trend_start",
            "trend_end",
            "trend_start_date",
            "trend_end_date",
            "trend_length",
            "trend_low",
            "trend_high",
            "trend_midpoint_50",
            "trend_high_date",
            "pullback_bars",
            "pullback_low",
            "pullback_depth_pct",
            "expected_upside_to_target",
            "push1_low_date",
            "push1_low",
            "push2_low_date",
            "push2_low",
            "push3_low_date",
            "push3_low",
            "push1_rebound_high_date",
            "push1_rebound_high",
            "push2_rebound_high_date",
            "push2_rebound_high",
            "wedge_pivot_low_1",
            "wedge_pivot_low_2",
            "wedge_pivot_low_3",
            "wedge_pivot_high_1",
            "wedge_pivot_high_2",
            "drop1_pct",
            "drop2_pct",
            "drop3_pct",
            "signal_body_pct",
            "signal_upper_shadow_pct",
            "signal_lower_shadow_pct",
            "close_gt_prev_high",
            "signal_quality_ok",
            "signal_candle",
            "follow_through_date",
            "follow_through_high",
            "follow_through_close",
            "signal_before_trend_end",
            "reward_to_risk",
            "signal_take_profit_price",
            "signal_hard_stop_price",
            "entry_signal",
            "entry_signal_executed",
            "entry_signal_suppressed",
            "entry_date_next",
            "entry_open_next",
            "exit_signal_date",
            "exit_date_next",
            "exit_open_next",
            "exit_reason",
            "realized_open_to_open_return",
            "max_favorable_excursion",
            "max_adverse_excursion",
        ]
        selected_signal_columns = [column for column in signal_columns if column in signal_row.columns]
        return {
            "summary": summary,
            "signal_row": signal_row.loc[:, selected_signal_columns].reset_index(drop=True),
            "condition_checklist": checklist,
            "price_window": price_window,
        }

    def plot_signal_context(
        self,
        ticker: str,
        signal_date: str | pd.Timestamp,
        *,
        lookback: int = 60,
        lookahead: int = 10,
    ) -> go.Figure:
        inspection = self.inspect_signal(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        signal_row = inspection["signal_row"]
        price_window = inspection["price_window"]
        checklist = inspection["condition_checklist"]
        if signal_row.empty or price_window.empty:
            return go.Figure()

        def _resolve_window_price(date_value: object, price_column: str) -> float | None:
            if pd.isna(date_value):
                return None
            matched = price_window[price_window["date"].eq(pd.Timestamp(date_value))]
            if matched.empty:
                return None
            value = matched[price_column].iat[0]
            if pd.isna(value):
                return None
            return float(value)

        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.62, 0.16, 0.22],
            subplot_titles=("Price Context", "Pullback Metrics", "Signal Conditions"),
        )
        figure.add_trace(
            go.Candlestick(
                x=price_window["date"],
                open=price_window["open"],
                high=price_window["high"],
                low=price_window["low"],
                close=price_window["close"],
                name=str(ticker),
            ),
            row=1,
            col=1,
        )
        for column, name, color, dash in [
            ("sma_20", "SMA 20", "steelblue", "solid"),
            ("sma_60", "SMA 60", "mediumpurple", "dash"),
            ("sma_120", "SMA 120", "slategray", "dot"),
        ]:
            if column in price_window.columns:
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.2, dash=dash),
                    ),
                    row=1,
                    col=1,
                )

        for column, name, color, dash in [
            ("trend_high", "Trend High", "firebrick", "dash"),
            ("trend_low", "Trend Low", "teal", "dot"),
            ("trend_midpoint_50", "Trend Mid 50%", "darkgoldenrod", "dashdot"),
        ]:
            if column in price_window.columns:
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.6, dash=dash),
                    ),
                    row=1,
                    col=1,
                )

        pivot_low_rows = price_window[price_window["pivot_low"].fillna(False)]
        if not pivot_low_rows.empty:
            figure.add_trace(
                go.Scatter(
                    x=pivot_low_rows["date"],
                    y=pivot_low_rows["low"],
                    mode="markers",
                    marker=dict(size=6, symbol="triangle-down", color="darkorange", opacity=0.35),
                    name="All Pivot Lows",
                ),
                row=1,
                col=1,
            )
        pivot_high_rows = price_window[price_window["pivot_high"].fillna(False)]
        if not pivot_high_rows.empty:
            figure.add_trace(
                go.Scatter(
                    x=pivot_high_rows["date"],
                    y=pivot_high_rows["high"],
                    mode="markers",
                    marker=dict(size=6, symbol="triangle-up", color="gray", opacity=0.35),
                    name="All Pivot Highs",
                ),
                row=1,
                col=1,
            )

        signal_date_value = signal_row["date"].iat[0]
        trend_start_date = signal_row["trend_start_date"].iat[0] if "trend_start_date" in signal_row.columns else pd.NaT
        trend_end_date = signal_row["trend_end_date"].iat[0] if "trend_end_date" in signal_row.columns else pd.NaT
        follow_through_date = signal_row["follow_through_date"].iat[0]
        entry_date = signal_row["entry_date_next"].iat[0]
        exit_date = signal_row["exit_date_next"].iat[0]
        exit_reason = None if pd.isna(signal_row["exit_reason"].iat[0]) else str(signal_row["exit_reason"].iat[0])
        stop_price = signal_row["signal_hard_stop_price"].iat[0]
        target_price = signal_row["signal_take_profit_price"].iat[0]
        push1_low_date = signal_row["push1_low_date"].iat[0] if "push1_low_date" in signal_row.columns else pd.NaT
        push2_low_date = signal_row["push2_low_date"].iat[0] if "push2_low_date" in signal_row.columns else pd.NaT
        push3_low_date = signal_row["push3_low_date"].iat[0] if "push3_low_date" in signal_row.columns else pd.NaT
        push1_low = signal_row["push1_low"].iat[0] if "push1_low" in signal_row.columns else np.nan
        push2_low = signal_row["push2_low"].iat[0] if "push2_low" in signal_row.columns else np.nan
        push3_low = signal_row["push3_low"].iat[0] if "push3_low" in signal_row.columns else np.nan
        push1_high_date = (
            signal_row["push1_rebound_high_date"].iat[0] if "push1_rebound_high_date" in signal_row.columns else pd.NaT
        )
        push2_high_date = (
            signal_row["push2_rebound_high_date"].iat[0] if "push2_rebound_high_date" in signal_row.columns else pd.NaT
        )
        push1_high = signal_row["push1_rebound_high"].iat[0] if "push1_rebound_high" in signal_row.columns else np.nan
        push2_high = signal_row["push2_rebound_high"].iat[0] if "push2_rebound_high" in signal_row.columns else np.nan

        figure.add_vline(x=signal_date_value, line_dash="dash", line_color="royalblue", row=1, col=1)
        if pd.notna(follow_through_date):
            figure.add_vline(x=follow_through_date, line_dash="dot", line_color="darkgreen", row=1, col=1)
        if pd.notna(stop_price):
            figure.add_hline(y=stop_price, line_dash="dot", line_color="indianred", row=1, col=1)
        if pd.notna(target_price):
            figure.add_hline(y=target_price, line_dash="dot", line_color="seagreen", row=1, col=1)

        trend_start_price = _resolve_window_price(trend_start_date, "low")
        if trend_start_price is not None:
            figure.add_trace(
                go.Scatter(
                    x=[trend_start_date],
                    y=[trend_start_price],
                    mode="markers+text",
                    marker=dict(size=14, symbol="diamond", color="deepskyblue"),
                    text=["Trend Start"],
                    textposition="bottom left",
                    name="Trend Start",
                ),
                row=1,
                col=1,
            )
        trend_end_price = _resolve_window_price(trend_end_date, "high")
        if trend_end_price is not None:
            figure.add_trace(
                go.Scatter(
                    x=[trend_end_date],
                    y=[trend_end_price],
                    mode="markers+text",
                    marker=dict(size=14, symbol="diamond-wide", color="firebrick"),
                    text=["Trend End"],
                    textposition="top left",
                    name="Trend End",
                ),
                row=1,
                col=1,
            )

        setup_low_points = [
            (push1_low_date, push1_low, "L1"),
            (push2_low_date, push2_low, "L2"),
            (push3_low_date, push3_low, "L3"),
        ]
        valid_low_points = [
            (pd.Timestamp(date_value), float(price_value), label)
            for date_value, price_value, label in setup_low_points
            if pd.notna(date_value) and pd.notna(price_value)
        ]
        low_x = [date_value for date_value, _, _ in valid_low_points]
        low_y = [price_value for _, price_value, _ in valid_low_points]
        low_text = [label for _, _, label in valid_low_points]
        if low_x and len(low_x) == len(low_y) == len(low_text):
            figure.add_trace(
                go.Scatter(
                    x=low_x,
                    y=low_y,
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-down", color="darkorange"),
                    text=low_text,
                    textposition="bottom center",
                    name="Setup Lows",
                ),
                row=1,
                col=1,
            )

        setup_high_points = [
            (push1_high_date, push1_high, "H1"),
            (push2_high_date, push2_high, "H2"),
        ]
        valid_high_points = [
            (pd.Timestamp(date_value), float(price_value), label)
            for date_value, price_value, label in setup_high_points
            if pd.notna(date_value) and pd.notna(price_value)
        ]
        high_x = [date_value for date_value, _, _ in valid_high_points]
        high_y = [price_value for _, price_value, _ in valid_high_points]
        high_text = [label for _, _, label in valid_high_points]
        if high_x and len(high_x) == len(high_y) == len(high_text):
            figure.add_trace(
                go.Scatter(
                    x=high_x,
                    y=high_y,
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-up", color="dimgray"),
                    text=high_text,
                    textposition="top center",
                    name="Setup Highs",
                ),
                row=1,
                col=1,
            )

        line_end_candidates = [signal_date_value, follow_through_date, entry_date, exit_date]
        line_end_candidates = [pd.Timestamp(value) for value in line_end_candidates if pd.notna(value)]
        if line_end_candidates:
            line_end_date = max(line_end_candidates)
            lower_line = self._build_extended_line(
                push1_low_date,
                push1_low,
                push3_low_date,
                push3_low,
                extend_to=line_end_date,
            )
            if lower_line is not None:
                figure.add_trace(
                    go.Scatter(
                        x=lower_line[0],
                        y=lower_line[1],
                        mode="lines",
                        line=dict(color="darkorange", width=1.8, dash="dash"),
                        name="Wedge Lower",
                    ),
                    row=1,
                    col=1,
                )

            upper_line = self._build_extended_line(
                push1_high_date,
                push1_high,
                push2_high_date,
                push2_high,
                extend_to=line_end_date,
            )
            if upper_line is not None:
                figure.add_trace(
                    go.Scatter(
                        x=upper_line[0],
                        y=upper_line[1],
                        mode="lines",
                        line=dict(color="slategray", width=1.8, dash="dash"),
                        name="Wedge Upper",
                    ),
                    row=1,
                    col=1,
                )

        if pd.notna(entry_date):
            entry_price = signal_row["entry_open_next"].iat[0]
            figure.add_trace(
                go.Scatter(
                    x=[entry_date],
                    y=[entry_price],
                    mode="markers+text",
                    marker=dict(size=14, symbol="triangle-up", color="green"),
                    text=["Entry"],
                    textposition="bottom center",
                    name="Entry",
                ),
                row=1,
                col=1,
            )
        if pd.notna(exit_date):
            exit_price = signal_row["exit_open_next"].iat[0]
            figure.add_trace(
                go.Scatter(
                    x=[exit_date],
                    y=[exit_price],
                    mode="markers+text",
                    marker=dict(size=16, symbol="x", color="red", line=dict(width=2, color="darkred")),
                    text=[f"Exit ({exit_reason})" if exit_reason else "Exit"],
                    textposition="top center",
                    name=f"Exit ({exit_reason})" if exit_reason else "Exit",
                ),
                row=1,
                col=1,
            )

        for column, name, color, dash in [
            ("expected_upside_to_target", "Upside To Target", "teal", "solid"),
            ("pullback_depth_pct", "Pullback Depth", "darkgoldenrod", "dash"),
        ]:
            if column in price_window.columns:
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.8, dash=dash),
                    ),
                    row=2,
                    col=1,
                )

        figure.add_trace(
            go.Bar(
                x=checklist["condition"],
                y=checklist["value"].astype(int),
                text=checklist["value"].astype(int).astype(str),
                textposition="outside",
                name="Conditions",
            ),
            row=3,
            col=1,
        )

        summary = inspection["summary"]
        reward_to_risk = summary["reward_to_risk"]
        return_text = summary["realized_open_to_open_return"]
        title_bits = [f"{ticker}", f"signal {pd.Timestamp(summary['signal_date']).date()}"]
        if pd.notna(reward_to_risk):
            title_bits.append(f"rr {reward_to_risk:.2f}")
        if summary["exit_reason"]:
            title_bits.append(f"exit {summary['exit_reason']}")
        if pd.notna(return_text):
            title_bits.append(f"return {return_text:.2%}")
        figure.update_layout(
            height=1020,
            width=1240,
            template="plotly_white",
            hovermode="x unified",
            title=" | ".join(title_bits),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Metric", tickformat=".0%", row=2, col=1)
        figure.update_yaxes(title_text="Met", row=3, col=1, range=[0, 1.2])
        return figure


__all__ = ["TrendPullbackContinuationResearcher", "TrendPullbackStrategyConfig"]
