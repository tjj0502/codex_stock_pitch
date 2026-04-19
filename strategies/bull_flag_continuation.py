from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.trend_pullback_continuation import TrendPullbackContinuationResearcher


@dataclass(frozen=True)
class BullFlagStrategyConfig:
    """
    Parameter bundle for a classic bull-flag continuation study.

    Workflow:
    1. require a bullish moving-average stack as the regime filter
    2. identify a short, strong flagpole into a confirmed pivot high
    3. require a shallow / orderly flag before the breakout bar
    4. enter after breakout + follow-through when reward-to-risk is sufficient
    """

    universe: str = "csi500"
    ma_windows: tuple[int, int, int] = (20, 60, 120)
    pivot_window: int = 1

    flagpole_lookback_bars: int = 20
    min_flagpole_bars: int = 5
    max_flagpole_bars: int = 20
    min_flagpole_return: float = 0.12

    min_flag_bars: int = 4
    max_flag_bars: int = 15
    max_flag_retrace_ratio: float = 0.40
    max_flag_channel_slope_pct_per_bar: float = 0.008
    max_flag_width_pct: float = 0.12

    min_breakout_body_pct: float = 0.60
    max_breakout_upper_shadow_pct: float = 0.25
    max_breakout_lower_shadow_pct: float = 0.35

    max_signal_stack_spread_pct: float | None = None
    min_signal_sma20_return_5: float | None = None
    max_peak_sma60_return_10: float | None = None

    stop_buffer_pct: float = 0.01
    measured_move_fraction: float = 0.75
    min_reward_r: float = 1.50
    require_follow_through_close_gt_signal_close: bool = False
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
        if self.pivot_window < 1:
            raise ValueError("pivot_window must be at least 1.")
        if self.flagpole_lookback_bars < 2:
            raise ValueError("flagpole_lookback_bars must be at least 2.")
        if self.min_flagpole_bars < 1:
            raise ValueError("min_flagpole_bars must be at least 1.")
        if self.max_flagpole_bars < self.min_flagpole_bars:
            raise ValueError("max_flagpole_bars must be >= min_flagpole_bars.")
        if self.min_flagpole_return <= 0:
            raise ValueError("min_flagpole_return must be positive.")
        if self.min_flag_bars < 2:
            raise ValueError("min_flag_bars must be at least 2.")
        if self.max_flag_bars < self.min_flag_bars:
            raise ValueError("max_flag_bars must be >= min_flag_bars.")
        if not 0 < self.max_flag_retrace_ratio < 1:
            raise ValueError("max_flag_retrace_ratio must be in (0, 1).")
        if self.max_flag_channel_slope_pct_per_bar <= 0:
            raise ValueError("max_flag_channel_slope_pct_per_bar must be positive.")
        if self.max_flag_width_pct <= 0:
            raise ValueError("max_flag_width_pct must be positive.")
        if not 0 < self.min_breakout_body_pct <= 1:
            raise ValueError("min_breakout_body_pct must be in (0, 1].")
        if not 0 <= self.max_breakout_upper_shadow_pct <= 1:
            raise ValueError("max_breakout_upper_shadow_pct must be in [0, 1].")
        if not 0 <= self.max_breakout_lower_shadow_pct <= 1:
            raise ValueError("max_breakout_lower_shadow_pct must be in [0, 1].")
        if self.max_signal_stack_spread_pct is not None and self.max_signal_stack_spread_pct <= 0:
            raise ValueError("max_signal_stack_spread_pct must be positive when provided.")
        if self.max_peak_sma60_return_10 is not None and self.max_peak_sma60_return_10 <= 0:
            raise ValueError("max_peak_sma60_return_10 must be positive when provided.")
        if self.stop_buffer_pct < 0:
            raise ValueError("stop_buffer_pct must be non-negative.")
        if self.measured_move_fraction <= 0:
            raise ValueError("measured_move_fraction must be positive.")
        if self.min_reward_r <= 0:
            raise ValueError("min_reward_r must be positive.")
        if self.max_holding_days is not None and self.max_holding_days < 1:
            raise ValueError("max_holding_days must be at least 1 when provided.")
        if not any([self.enable_hard_stop, self.enable_take_profit, self.enable_time_stop]):
            raise ValueError("At least one exit rule must be enabled.")


class BullFlagContinuationResearcher(TrendPullbackContinuationResearcher):
    """
    Research helper for a classic bull-flag continuation setup.

    Definitions used here:
    - regime filter: ``MA20 > MA60 > MA120``
    - flagpole: a short, strong upswing ending in a confirmed pivot high
    - flag: a shallow / orderly consolidation before the breakout bar
    - breakout: close above the projected upper flag line plus previous high
    - follow-through: next bar trades above the breakout-bar high
    """

    FEATURE_COLUMNS = [
        "sma_20",
        "sma_60",
        "sma_120",
        "bullish_stack",
        "pivot_high",
        "pivot_low",
        "close_gt_open",
        "close_gt_prev_high",
        "bullish_stack_run_length",
        "stack_spread_pct",
        "sma20_return_5",
        "sma60_return_10",
        "signal_body_pct",
        "signal_upper_shadow_pct",
        "signal_lower_shadow_pct",
        "signal_quality_ok",
        "signal_bullish_stack_run_length",
        "signal_stack_spread_pct",
        "signal_sma20_return_5",
        "peak_bullish_stack_run_length",
        "peak_sma60_return_10",
        "flagpole_start_date",
        "flagpole_start_low",
        "flagpole_length",
        "flag_peak_date",
        "flag_peak_high",
        "flagpole_bars",
        "flagpole_return",
        "flag_start_date",
        "flag_end_date",
        "flag_bars",
        "flag_low_date",
        "flag_low",
        "flag_retrace_ratio",
        "flag_width_pct",
        "flag_upper_slope",
        "flag_lower_slope",
        "flag_upper_line_value",
        "flag_lower_line_value",
        "flag_shape_ok",
        "flag_retrace_ok",
        "flag_channel_ok",
        "bull_flag_candidate",
        "breakout_candle",
        "signal_candle",
    ]
    SIGNAL_COLUMNS = [
        "follow_through_date",
        "follow_through_high",
        "follow_through_close",
        "follow_through_confirmed",
        "follow_through_close_gt_signal_close",
        "signal_stack_spread_ok",
        "signal_sma20_return_ok",
        "peak_sma60_return_ok",
        "trend_environment_ok",
        "entry_reference_price",
        "reward_to_risk",
        "reward_to_risk_ok",
        "signal_take_profit_price",
        "signal_hard_stop_price",
        "entry_signal",
    ]
    OUTCOME_COLUMNS = TrendPullbackContinuationResearcher.OUTCOME_COLUMNS
    FEATURE_ANALYSIS_COLUMNS = [
        "flagpole_bars",
        "flagpole_return",
        "flagpole_length",
        "flag_bars",
        "flag_retrace_ratio",
        "flag_width_pct",
        "signal_bullish_stack_run_length",
        "signal_stack_spread_pct",
        "signal_sma20_return_5",
        "peak_bullish_stack_run_length",
        "peak_sma60_return_10",
        "flag_upper_slope",
        "flag_lower_slope",
        "reward_to_risk",
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
        "flagpole_start_date",
        "flagpole_start_low",
        "flagpole_length",
        "flag_peak_date",
        "flag_peak_high",
        "flagpole_bars",
        "flagpole_return",
        "flag_start_date",
        "flag_end_date",
        "flag_bars",
        "flag_low_date",
        "flag_low",
        "flag_retrace_ratio",
        "flag_width_pct",
        "flag_upper_slope",
        "flag_lower_slope",
        "flag_upper_line_value",
        "flag_lower_line_value",
        "signal_bullish_stack_run_length",
        "signal_stack_spread_pct",
        "signal_sma20_return_5",
        "peak_bullish_stack_run_length",
        "peak_sma60_return_10",
        "signal_body_pct",
        "signal_upper_shadow_pct",
        "signal_lower_shadow_pct",
        "follow_through_date",
        "follow_through_high",
        "follow_through_close",
        "follow_through_close_gt_signal_close",
        "trend_environment_ok",
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
        config: BullFlagStrategyConfig | None = None,
        *,
        copy: bool = True,
    ) -> None:
        super().__init__(stock_candle_df, config=config or BullFlagStrategyConfig(), copy=copy)
        self.stock_candle_df.attrs["strategy_name"] = "bull_flag_continuation"
        self.stock_candle_df.attrs["strategy_universe"] = self.config.universe
        self.trade_df.attrs["strategy_name"] = "bull_flag_continuation"
        self.trade_df.attrs["strategy_universe"] = self.config.universe

    @staticmethod
    def _fit_line(values: np.ndarray) -> tuple[float, float]:
        clean = np.asarray(values, dtype=float)
        if clean.size == 0:
            return np.nan, np.nan
        if clean.size == 1:
            return float(clean[0]), 0.0
        x = np.arange(clean.size, dtype=float)
        x_mean = float(x.mean())
        y_mean = float(clean.mean())
        denominator = float(np.square(x - x_mean).sum())
        if denominator <= 0:
            return y_mean, 0.0
        slope = float(((x - x_mean) * (clean - y_mean)).sum() / denominator)
        intercept = float(y_mean - slope * x_mean)
        return intercept, slope

    @staticmethod
    def _consecutive_true_run_length(series: pd.Series) -> pd.Series:
        values = series.fillna(False).to_numpy(dtype=bool)
        output = np.zeros(len(values), dtype=np.int32)
        run_length = 0
        for index, value in enumerate(values):
            if value:
                run_length += 1
            else:
                run_length = 0
            output[index] = run_length
        return pd.Series(output, index=series.index, dtype="int32")

    @classmethod
    def _select_flagpole_start_index(
        cls,
        pivot_low_indices: np.ndarray,
        low_values: np.ndarray,
        *,
        peak_idx: int,
        lookback_bars: int,
    ) -> int | None:
        window_start = max(0, peak_idx - lookback_bars + 1)
        pivot_start = int(np.searchsorted(pivot_low_indices, window_start))
        pivot_end = int(np.searchsorted(pivot_low_indices, peak_idx, side="right"))
        candidate_indices = [
            int(idx)
            for idx in pivot_low_indices[pivot_start:pivot_end]
            if int(idx) < peak_idx and pd.notna(low_values[int(idx)])
        ]
        if candidate_indices:
            return min(candidate_indices, key=lambda idx: (low_values[idx], idx))

        window_values = low_values[window_start : peak_idx + 1]
        if window_values.size == 0:
            return None
        valid_mask = ~np.isnan(window_values)
        if not valid_mask.any():
            return None
        valid_positions = np.flatnonzero(valid_mask)
        relative_idx = int(valid_positions[np.argmin(window_values[valid_mask])])
        return window_start + relative_idx

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
        bullish_stack_run_length = pd.to_numeric(
            ticker_frame["bullish_stack_run_length"], errors="coerce"
        ).to_numpy(dtype=float)
        stack_spread_pct = pd.to_numeric(ticker_frame["stack_spread_pct"], errors="coerce").to_numpy(dtype=float)
        sma20_return_5 = pd.to_numeric(ticker_frame["sma20_return_5"], errors="coerce").to_numpy(dtype=float)
        sma60_return_10 = pd.to_numeric(ticker_frame["sma60_return_10"], errors="coerce").to_numpy(dtype=float)
        pivot_high_indices = np.flatnonzero(pivot_high)
        pivot_low_indices = np.flatnonzero(pivot_low)

        annotations: dict[str, np.ndarray] = {
            "signal_bullish_stack_run_length": np.full(row_count, np.nan, dtype=float),
            "signal_stack_spread_pct": np.full(row_count, np.nan, dtype=float),
            "signal_sma20_return_5": np.full(row_count, np.nan, dtype=float),
            "peak_bullish_stack_run_length": np.full(row_count, np.nan, dtype=float),
            "peak_sma60_return_10": np.full(row_count, np.nan, dtype=float),
            "flagpole_start_date": self._nat_array(row_count),
            "flagpole_start_low": np.full(row_count, np.nan, dtype=float),
            "flagpole_length": np.full(row_count, np.nan, dtype=float),
            "flag_peak_date": self._nat_array(row_count),
            "flag_peak_high": np.full(row_count, np.nan, dtype=float),
            "flagpole_bars": self._int_array(row_count),
            "flagpole_return": np.full(row_count, np.nan, dtype=float),
            "flag_start_date": self._nat_array(row_count),
            "flag_end_date": self._nat_array(row_count),
            "flag_bars": self._int_array(row_count),
            "flag_low_date": self._nat_array(row_count),
            "flag_low": np.full(row_count, np.nan, dtype=float),
            "flag_retrace_ratio": np.full(row_count, np.nan, dtype=float),
            "flag_width_pct": np.full(row_count, np.nan, dtype=float),
            "flag_upper_slope": np.full(row_count, np.nan, dtype=float),
            "flag_lower_slope": np.full(row_count, np.nan, dtype=float),
            "flag_upper_line_value": np.full(row_count, np.nan, dtype=float),
            "flag_lower_line_value": np.full(row_count, np.nan, dtype=float),
            "flag_shape_ok": np.zeros(row_count, dtype=bool),
            "flag_retrace_ok": np.zeros(row_count, dtype=bool),
            "flag_channel_ok": np.zeros(row_count, dtype=bool),
            "bull_flag_candidate": np.zeros(row_count, dtype=bool),
            "breakout_candle": np.zeros(row_count, dtype=bool),
            "signal_candle": np.zeros(row_count, dtype=bool),
        }

        for peak_idx in pivot_high_indices:
            peak_idx = int(peak_idx)
            if not bullish_stack[peak_idx]:
                continue

            flagpole_start_idx = self._select_flagpole_start_index(
                pivot_low_indices,
                low_values,
                peak_idx=peak_idx,
                lookback_bars=cfg.flagpole_lookback_bars,
            )
            if flagpole_start_idx is None:
                continue

            flagpole_start_low = low_values[flagpole_start_idx]
            flag_peak_high = high_values[peak_idx]
            if (
                pd.isna(flagpole_start_low)
                or pd.isna(flag_peak_high)
                or flagpole_start_low <= 0
                or flag_peak_high <= flagpole_start_low
            ):
                continue

            flagpole_bars = int(peak_idx - flagpole_start_idx)
            if flagpole_bars < cfg.min_flagpole_bars or flagpole_bars > cfg.max_flagpole_bars:
                continue

            flagpole_return = float(flag_peak_high / flagpole_start_low - 1.0)
            if flagpole_return < cfg.min_flagpole_return:
                continue

            flagpole_length = float(flag_peak_high - flagpole_start_low)
            flag_start_idx = peak_idx + 1
            breakout_start_idx = flag_start_idx + cfg.min_flag_bars
            breakout_end_idx = min(row_count - 1, peak_idx + cfg.max_flag_bars + 1)
            if breakout_start_idx > breakout_end_idx:
                continue

            for current_loc in range(breakout_start_idx, breakout_end_idx + 1):
                if not bullish_stack[current_loc]:
                    continue

                flag_end_idx = current_loc - 1
                flag_bars = int(flag_end_idx - flag_start_idx + 1)
                if flag_bars < cfg.min_flag_bars or flag_bars > cfg.max_flag_bars:
                    continue
                # A bull flag is only valid while the bullish-stack regime
                # remains intact from the peak into the breakout attempt. Once
                # that background breaks, the old setup is considered stale
                # even if the moving-average stack later recovers.
                if not bullish_stack[flag_start_idx : current_loc + 1].all():
                    continue

                flag_high_window = high_values[flag_start_idx : flag_end_idx + 1]
                flag_low_window = low_values[flag_start_idx : flag_end_idx + 1]
                if (
                    flag_high_window.size != flag_bars
                    or flag_low_window.size != flag_bars
                    or np.isnan(flag_high_window).any()
                    or np.isnan(flag_low_window).any()
                ):
                    continue

                flag_low_relative_idx = int(np.argmin(flag_low_window))
                flag_low_idx = flag_start_idx + flag_low_relative_idx
                flag_low = float(flag_low_window[flag_low_relative_idx])
                denominator = float(flag_peak_high - flagpole_start_low)
                if denominator <= 0:
                    continue

                flag_retrace_ratio = float((flag_peak_high - flag_low) / denominator)
                flag_width_pct = float((flag_high_window.max() - flag_low_window.min()) / flag_peak_high)
                upper_intercept, upper_slope = self._fit_line(flag_high_window)
                lower_intercept, lower_slope = self._fit_line(flag_low_window)
                upper_slope_pct = upper_slope / flag_peak_high
                lower_slope_pct = lower_slope / flag_peak_high
                projected_upper_line = float(upper_intercept + upper_slope * flag_bars)
                projected_lower_line = float(lower_intercept + lower_slope * flag_bars)

                flag_shape_ok = bool(flag_width_pct <= cfg.max_flag_width_pct)
                flag_retrace_ok = bool(flag_retrace_ratio <= cfg.max_flag_retrace_ratio)
                flag_channel_ok = bool(
                    abs(upper_slope_pct) <= cfg.max_flag_channel_slope_pct_per_bar
                    and abs(lower_slope_pct) <= cfg.max_flag_channel_slope_pct_per_bar
                )
                bull_flag_candidate = flag_shape_ok and flag_retrace_ok and flag_channel_ok
                breakout_candle = bool(
                    bull_flag_candidate
                    and signal_quality_ok[current_loc]
                    and close_gt_prev_high[current_loc]
                    and pd.notna(close_values[current_loc])
                    and close_values[current_loc] > projected_upper_line
                )

                annotations["signal_bullish_stack_run_length"][current_loc] = bullish_stack_run_length[current_loc]
                annotations["signal_stack_spread_pct"][current_loc] = stack_spread_pct[current_loc]
                annotations["signal_sma20_return_5"][current_loc] = sma20_return_5[current_loc]
                annotations["peak_bullish_stack_run_length"][current_loc] = bullish_stack_run_length[peak_idx]
                annotations["peak_sma60_return_10"][current_loc] = sma60_return_10[peak_idx]
                annotations["flagpole_start_date"][current_loc] = date_values[flagpole_start_idx]
                annotations["flagpole_start_low"][current_loc] = float(flagpole_start_low)
                annotations["flagpole_length"][current_loc] = flagpole_length
                annotations["flag_peak_date"][current_loc] = date_values[peak_idx]
                annotations["flag_peak_high"][current_loc] = float(flag_peak_high)
                annotations["flagpole_bars"][current_loc] = flagpole_bars
                annotations["flagpole_return"][current_loc] = flagpole_return
                annotations["flag_start_date"][current_loc] = date_values[flag_start_idx]
                annotations["flag_end_date"][current_loc] = date_values[flag_end_idx]
                annotations["flag_bars"][current_loc] = flag_bars
                annotations["flag_low_date"][current_loc] = date_values[flag_low_idx]
                annotations["flag_low"][current_loc] = flag_low
                annotations["flag_retrace_ratio"][current_loc] = flag_retrace_ratio
                annotations["flag_width_pct"][current_loc] = flag_width_pct
                annotations["flag_upper_slope"][current_loc] = float(upper_slope)
                annotations["flag_lower_slope"][current_loc] = float(lower_slope)
                annotations["flag_upper_line_value"][current_loc] = projected_upper_line
                annotations["flag_lower_line_value"][current_loc] = projected_lower_line
                annotations["flag_shape_ok"][current_loc] = flag_shape_ok
                annotations["flag_retrace_ok"][current_loc] = flag_retrace_ok
                annotations["flag_channel_ok"][current_loc] = flag_channel_ok
                annotations["bull_flag_candidate"][current_loc] = bull_flag_candidate
                annotations["breakout_candle"][current_loc] = breakout_candle
                annotations["signal_candle"][current_loc] = breakout_candle

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
        df["bullish_stack_run_length"] = ticker_group["bullish_stack"].transform(self._consecutive_true_run_length)
        df["stack_spread_pct"] = self._safe_ratio(df[f"sma_{fast_window}"] - df[f"sma_{slow_window}"], df["close"]).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        df["sma20_return_5"] = ticker_group[f"sma_{fast_window}"].transform(lambda s: s.div(s.shift(5)) - 1.0)
        df["sma60_return_10"] = ticker_group[f"sma_{mid_window}"].transform(lambda s: s.div(s.shift(10)) - 1.0)

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
            & df["signal_body_pct"].ge(cfg.min_breakout_body_pct)
            & df["signal_upper_shadow_pct"].le(cfg.max_breakout_upper_shadow_pct)
            & df["signal_lower_shadow_pct"].le(cfg.max_breakout_lower_shadow_pct)
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
            "flagpole_start_date",
            "flag_peak_date",
            "flag_start_date",
            "flag_end_date",
            "flag_low_date",
        ]
        float_columns = [
            "signal_bullish_stack_run_length",
            "signal_stack_spread_pct",
            "signal_sma20_return_5",
            "peak_bullish_stack_run_length",
            "peak_sma60_return_10",
            "flagpole_start_low",
            "flagpole_length",
            "flag_peak_high",
            "flagpole_return",
            "flag_low",
            "flag_retrace_ratio",
            "flag_width_pct",
            "flag_upper_slope",
            "flag_lower_slope",
            "flag_upper_line_value",
            "flag_lower_line_value",
        ]
        int_columns = ["flagpole_bars", "flag_bars"]
        bool_columns = [
            "flag_shape_ok",
            "flag_retrace_ok",
            "flag_channel_ok",
            "bull_flag_candidate",
            "breakout_candle",
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
        df["follow_through_close_gt_signal_close"] = df["follow_through_close"].gt(df["close"])
        df["signal_stack_spread_ok"] = (
            df["signal_stack_spread_pct"].le(cfg.max_signal_stack_spread_pct)
            if cfg.max_signal_stack_spread_pct is not None
            else True
        )
        df["signal_sma20_return_ok"] = (
            df["signal_sma20_return_5"].ge(cfg.min_signal_sma20_return_5)
            if cfg.min_signal_sma20_return_5 is not None
            else True
        )
        df["peak_sma60_return_ok"] = (
            df["peak_sma60_return_10"].le(cfg.max_peak_sma60_return_10)
            if cfg.max_peak_sma60_return_10 is not None
            else True
        )
        df["trend_environment_ok"] = (
            df["signal_stack_spread_ok"] & df["signal_sma20_return_ok"] & df["peak_sma60_return_ok"]
        )
        df["signal_hard_stop_price"] = df["flag_low"] * (1.0 - cfg.stop_buffer_pct)
        df["entry_reference_price"] = df["follow_through_close"]
        df["signal_take_profit_price"] = np.where(
            df["entry_reference_price"].notna() & df["flagpole_length"].notna(),
            df["entry_reference_price"] + cfg.measured_move_fraction * df["flagpole_length"],
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
            & df["trend_environment_ok"]
            & (
                df["follow_through_close_gt_signal_close"]
                if cfg.require_follow_through_close_gt_signal_close
                else True
            )
            & df["reward_to_risk_ok"]
            & df["entry_date_next"].notna()
            & df["entry_open_next"].gt(0)
        )

        return self._store_output(df)

    def get_candidates(self, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if "signal_candle" not in self.stock_candle_df.columns:
            self.add_signals()

        base_columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "flagpole_start_date",
            "flag_peak_date",
            "flag_peak_high",
            "flagpole_return",
            "flag_bars",
            "flag_low",
            "flag_retrace_ratio",
            "flag_width_pct",
            "signal_bullish_stack_run_length",
            "signal_stack_spread_pct",
            "signal_sma20_return_5",
            "peak_bullish_stack_run_length",
            "peak_sma60_return_10",
            "flag_upper_line_value",
            "signal_body_pct",
            "signal_upper_shadow_pct",
            "signal_lower_shadow_pct",
            "close_gt_prev_high",
            "signal_quality_ok",
            "bull_flag_candidate",
            "trend_environment_ok",
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
            ["flagpole_return", "reward_to_risk", "ticker"],
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
            "flag_peak_high",
            "flag_low",
            "flagpole_return",
            "flag_retrace_ratio",
            "flag_bars",
            "signal_bullish_stack_run_length",
            "signal_stack_spread_pct",
            "signal_sma20_return_5",
            "peak_bullish_stack_run_length",
            "peak_sma60_return_10",
            "reward_to_risk",
            "follow_through_confirmed",
            "follow_through_close_gt_signal_close",
            "trend_environment_ok",
            "reward_to_risk_ok",
            "entry_signal_live",
        ]
        if df.empty:
            return pd.DataFrame(columns=output_columns)

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else df["date"].max()
        planned_entry_date = (
            pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)
        )

        candidates = df[
            df["follow_through_date"].eq(target_date)
            & df["signal_candle"]
            & df["follow_through_confirmed"]
            & (
                df["follow_through_close_gt_signal_close"]
                if self.config.require_follow_through_close_gt_signal_close
                else True
            )
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
            ["reward_to_risk", "flagpole_return", "ticker"],
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
            "flag_peak_date",
            "flag_peak_high",
            "flag_low",
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

        executed_signals = self.add_trade_df().copy()
        records: list[dict[str, object]] = []
        for _, position in positions.iterrows():
            record = {column: position[column] for column in positions.columns}
            ticker = str(position["ticker"])
            entry_date = pd.Timestamp(position["entry_date"])
            entry_price = float(position["entry_price"]) if pd.notna(position["entry_price"]) else np.nan
            shares = float(position["shares"]) if "shares" in positions.columns and pd.notna(position["shares"]) else np.nan

            ticker_rows = scored[
                scored["ticker"].astype(str).eq(ticker) & scored["date"].le(target_date)
            ].copy()
            if ticker_rows.empty:
                record.update(
                    {
                        "as_of_date": target_date,
                        "action": "review",
                        "issue": "ticker_not_found_on_or_before_as_of_date",
                    }
                )
                records.append(record)
                continue

            latest_row = ticker_rows.iloc[-1]
            if "signal_date" in positions.columns and pd.notna(position.get("signal_date", pd.NaT)):
                matched_signal = executed_signals[
                    executed_signals["ticker"].astype(str).eq(ticker)
                    & executed_signals["signal_date"].eq(pd.Timestamp(position["signal_date"]))
                ]
            else:
                matched_signal = executed_signals[
                    executed_signals["ticker"].astype(str).eq(ticker)
                    & executed_signals["entry_date"].eq(entry_date)
                ]
            signal_row = matched_signal.iloc[0] if not matched_signal.empty else pd.Series(dtype="object")

            current_close = float(latest_row["close"]) if pd.notna(latest_row["close"]) else np.nan
            pnl_pct = float(current_close / entry_price - 1.0) if entry_price > 0 and pd.notna(current_close) else np.nan
            pnl_amount = (
                float((current_close - entry_price) * shares)
                if pd.notna(shares) and entry_price > 0 and pd.notna(current_close)
                else np.nan
            )

            trade_rows = ticker_rows[ticker_rows["date"].ge(entry_date)].copy()
            trading_days_in_trade = int(len(trade_rows))
            holding_days = int((target_date.normalize() - entry_date.normalize()).days)
            days_until_time_stop = (
                int(cfg.max_holding_days - trading_days_in_trade)
                if cfg.enable_time_stop and cfg.max_holding_days is not None
                else pd.NA
            )

            hard_stop_price = (
                float(signal_row["signal_hard_stop_price"])
                if "signal_hard_stop_price" in signal_row and pd.notna(signal_row["signal_hard_stop_price"])
                else np.nan
            )
            take_profit_price = (
                float(signal_row["signal_take_profit_price"])
                if "signal_take_profit_price" in signal_row and pd.notna(signal_row["signal_take_profit_price"])
                else np.nan
            )
            reward_to_risk = (
                float(signal_row["reward_to_risk"])
                if "reward_to_risk" in signal_row and pd.notna(signal_row["reward_to_risk"])
                else np.nan
            )

            exit_signal = False
            exit_reason = pd.NA
            if cfg.enable_hard_stop and pd.notna(hard_stop_price) and pd.notna(latest_row["low"]) and latest_row["low"] <= hard_stop_price:
                exit_signal = True
                exit_reason = "hard_stop"
            elif cfg.enable_take_profit and pd.notna(take_profit_price) and pd.notna(latest_row["high"]) and latest_row["high"] >= take_profit_price:
                exit_signal = True
                exit_reason = "take_profit"
            elif (
                cfg.enable_time_stop
                and cfg.max_holding_days is not None
                and trading_days_in_trade >= cfg.max_holding_days
            ):
                exit_signal = True
                exit_reason = "time_stop"

            action = "hold"
            if exit_signal:
                action = "prepare_exit"
            if signal_row.empty:
                action = "review"

            record.update(
                {
                    "name": latest_row["name"],
                    "as_of_date": target_date,
                    "latest_bar_date": latest_row["date"],
                    "signal_date_resolved": signal_row.get("signal_date", pd.NaT) if not signal_row.empty else pd.NaT,
                    "flag_peak_date": signal_row.get("flag_peak_date", pd.NaT) if not signal_row.empty else pd.NaT,
                    "flag_peak_high": signal_row.get("flag_peak_high", np.nan) if not signal_row.empty else np.nan,
                    "flag_low": signal_row.get("flag_low", np.nan) if not signal_row.empty else np.nan,
                    "current_close": current_close,
                    "pnl_pct": pnl_pct,
                    "pnl_amount": pnl_amount,
                    "holding_days": holding_days,
                    "trading_days_in_trade": trading_days_in_trade,
                    "days_until_time_stop": days_until_time_stop,
                    "hard_stop_price": hard_stop_price,
                    "take_profit_price": take_profit_price,
                    "reward_to_risk": reward_to_risk,
                    "exit_signal": exit_signal,
                    "exit_signal_date": latest_row["date"] if exit_signal else pd.NaT,
                    "planned_exit_date": resolved_next_trade_date if exit_signal else pd.NaT,
                    "exit_reason": exit_reason,
                    "action": action,
                    "issue": pd.NA if not signal_row.empty else "matching_signal_not_found",
                }
            )
            records.append(record)

        monitored = pd.DataFrame(records)
        if monitored.empty:
            return pd.DataFrame(columns=output_columns)
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
            raise ValueError(f"Ticker '{ticker}' does not have an executed signal on {target_date.date()}.")

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
                    "bullish_stack",
                    "flag_shape_ok",
                    "flag_retrace_ok",
                    "flag_channel_ok",
                    "bull_flag_candidate",
                    "signal_stack_spread_ok",
                    "signal_sma20_return_ok",
                    "peak_sma60_return_ok",
                    "trend_environment_ok",
                    "close_gt_prev_high",
                    "signal_quality_ok",
                    "breakout_candle",
                    "follow_through_confirmed",
                    "follow_through_close_gt_signal_close",
                    "reward_to_risk_ok",
                    "entry_signal",
                    "entry_signal_executed",
                ],
                "value": [
                    bool(signal_row["bullish_stack"].iat[0]),
                    bool(signal_row["flag_shape_ok"].iat[0]),
                    bool(signal_row["flag_retrace_ok"].iat[0]),
                    bool(signal_row["flag_channel_ok"].iat[0]),
                    bool(signal_row["bull_flag_candidate"].iat[0]),
                    bool(signal_row["signal_stack_spread_ok"].iat[0]),
                    bool(signal_row["signal_sma20_return_ok"].iat[0]),
                    bool(signal_row["peak_sma60_return_ok"].iat[0]),
                    bool(signal_row["trend_environment_ok"].iat[0]),
                    bool(signal_row["close_gt_prev_high"].iat[0]),
                    bool(signal_row["signal_quality_ok"].iat[0]),
                    bool(signal_row["breakout_candle"].iat[0]),
                    bool(signal_row["follow_through_confirmed"].iat[0]),
                    bool(signal_row["follow_through_close_gt_signal_close"].iat[0]),
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
            "flagpole_start_date": signal_row["flagpole_start_date"].iat[0],
            "flag_peak_date": signal_row["flag_peak_date"].iat[0],
            "flag_start_date": signal_row["flag_start_date"].iat[0],
            "flag_end_date": signal_row["flag_end_date"].iat[0],
            "flag_low_date": signal_row["flag_low_date"].iat[0],
            "flagpole_return": float(signal_row["flagpole_return"].iat[0]) if pd.notna(signal_row["flagpole_return"].iat[0]) else np.nan,
            "flag_retrace_ratio": float(signal_row["flag_retrace_ratio"].iat[0]) if pd.notna(signal_row["flag_retrace_ratio"].iat[0]) else np.nan,
            "flag_bars": None if pd.isna(signal_row["flag_bars"].iat[0]) else int(signal_row["flag_bars"].iat[0]),
            "signal_bullish_stack_run_length": None if pd.isna(signal_row["signal_bullish_stack_run_length"].iat[0]) else int(signal_row["signal_bullish_stack_run_length"].iat[0]),
            "signal_stack_spread_pct": float(signal_row["signal_stack_spread_pct"].iat[0]) if pd.notna(signal_row["signal_stack_spread_pct"].iat[0]) else np.nan,
            "signal_sma20_return_5": float(signal_row["signal_sma20_return_5"].iat[0]) if pd.notna(signal_row["signal_sma20_return_5"].iat[0]) else np.nan,
            "peak_bullish_stack_run_length": None if pd.isna(signal_row["peak_bullish_stack_run_length"].iat[0]) else int(signal_row["peak_bullish_stack_run_length"].iat[0]),
            "peak_sma60_return_10": float(signal_row["peak_sma60_return_10"].iat[0]) if pd.notna(signal_row["peak_sma60_return_10"].iat[0]) else np.nan,
            "reward_to_risk": float(signal_row["reward_to_risk"].iat[0]) if pd.notna(signal_row["reward_to_risk"].iat[0]) else np.nan,
            "entry_date_next": signal_row["entry_date_next"].iat[0],
            "entry_open_next": float(signal_row["entry_open_next"].iat[0]) if pd.notna(signal_row["entry_open_next"].iat[0]) else np.nan,
            "exit_date_next": signal_row["exit_date_next"].iat[0],
            "exit_reason": None if pd.isna(signal_row["exit_reason"].iat[0]) else str(signal_row["exit_reason"].iat[0]),
            "holding_days": None if pd.isna(signal_row["holding_days"].iat[0]) else int(signal_row["holding_days"].iat[0]),
            "realized_open_to_open_return": float(signal_row["realized_open_to_open_return"].iat[0]) if pd.notna(signal_row["realized_open_to_open_return"].iat[0]) else np.nan,
            "max_favorable_excursion": float(signal_row["max_favorable_excursion"].iat[0]) if pd.notna(signal_row["max_favorable_excursion"].iat[0]) else np.nan,
            "max_adverse_excursion": float(signal_row["max_adverse_excursion"].iat[0]) if pd.notna(signal_row["max_adverse_excursion"].iat[0]) else np.nan,
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
            "bullish_stack",
            "flagpole_start_date",
            "flagpole_start_low",
            "flagpole_length",
            "flag_peak_date",
            "flag_peak_high",
            "flagpole_bars",
            "flagpole_return",
            "flag_start_date",
            "flag_end_date",
            "flag_bars",
            "flag_low_date",
            "flag_low",
            "flag_retrace_ratio",
            "flag_width_pct",
            "flag_upper_slope",
            "flag_lower_slope",
            "flag_upper_line_value",
            "flag_lower_line_value",
            "signal_bullish_stack_run_length",
            "signal_stack_spread_pct",
            "signal_sma20_return_5",
            "peak_bullish_stack_run_length",
            "peak_sma60_return_10",
            "flag_shape_ok",
            "flag_retrace_ok",
            "flag_channel_ok",
            "bull_flag_candidate",
            "signal_body_pct",
            "signal_upper_shadow_pct",
            "signal_lower_shadow_pct",
            "close_gt_prev_high",
            "signal_quality_ok",
            "breakout_candle",
            "signal_candle",
            "follow_through_date",
            "follow_through_high",
            "follow_through_close",
            "follow_through_close_gt_signal_close",
            "signal_stack_spread_ok",
            "signal_sma20_return_ok",
            "peak_sma60_return_ok",
            "trend_environment_ok",
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

        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.62, 0.16, 0.22],
            subplot_titles=("Price Context", "Flag Metrics", "Signal Conditions"),
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

        signal_date_value = pd.Timestamp(signal_row["date"].iat[0])
        flagpole_start_date = signal_row["flagpole_start_date"].iat[0]
        flag_peak_date = signal_row["flag_peak_date"].iat[0]
        flag_start_date = signal_row["flag_start_date"].iat[0]
        flag_end_date = signal_row["flag_end_date"].iat[0]
        flag_low_date = signal_row["flag_low_date"].iat[0]
        follow_through_date = signal_row["follow_through_date"].iat[0]
        entry_date = signal_row["entry_date_next"].iat[0]
        exit_date = signal_row["exit_date_next"].iat[0]
        stop_price = signal_row["signal_hard_stop_price"].iat[0]
        target_price = signal_row["signal_take_profit_price"].iat[0]

        if pd.notna(flagpole_start_date) and pd.notna(signal_row["flagpole_start_low"].iat[0]):
            figure.add_trace(
                go.Scatter(
                    x=[flagpole_start_date],
                    y=[signal_row["flagpole_start_low"].iat[0]],
                    mode="markers+text",
                    marker=dict(size=13, symbol="diamond", color="deepskyblue"),
                    text=["Flagpole Start"],
                    textposition="bottom left",
                    name="Flagpole Start",
                ),
                row=1,
                col=1,
            )
        if pd.notna(flag_peak_date) and pd.notna(signal_row["flag_peak_high"].iat[0]):
            figure.add_trace(
                go.Scatter(
                    x=[flag_peak_date],
                    y=[signal_row["flag_peak_high"].iat[0]],
                    mode="markers+text",
                    marker=dict(size=13, symbol="diamond-wide", color="firebrick"),
                    text=["Flag Peak"],
                    textposition="top left",
                    name="Flag Peak",
                ),
                row=1,
                col=1,
            )
        if pd.notna(flag_low_date) and pd.notna(signal_row["flag_low"].iat[0]):
            figure.add_trace(
                go.Scatter(
                    x=[flag_low_date],
                    y=[signal_row["flag_low"].iat[0]],
                    mode="markers+text",
                    marker=dict(size=12, symbol="triangle-down", color="darkorange"),
                    text=["Flag Low"],
                    textposition="bottom right",
                    name="Flag Low",
                ),
                row=1,
                col=1,
            )

        flag_bars_value = signal_row["flag_bars"].iat[0]
        if (
            pd.notna(flag_start_date)
            and pd.notna(flag_end_date)
            and pd.notna(flag_bars_value)
            and pd.notna(signal_row["flag_upper_line_value"].iat[0])
            and pd.notna(signal_row["flag_lower_line_value"].iat[0])
            and pd.notna(signal_row["flag_upper_slope"].iat[0])
            and pd.notna(signal_row["flag_lower_slope"].iat[0])
        ):
            flag_bars = int(flag_bars_value)
            upper_slope = float(signal_row["flag_upper_slope"].iat[0])
            lower_slope = float(signal_row["flag_lower_slope"].iat[0])
            projected_upper = float(signal_row["flag_upper_line_value"].iat[0])
            projected_lower = float(signal_row["flag_lower_line_value"].iat[0])
            upper_start = projected_upper - upper_slope * flag_bars
            lower_start = projected_lower - lower_slope * flag_bars
            upper_end = upper_start + upper_slope * (flag_bars - 1)
            lower_end = lower_start + lower_slope * (flag_bars - 1)

            figure.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(flag_start_date), pd.Timestamp(flag_end_date), signal_date_value],
                    y=[upper_start, upper_end, projected_upper],
                    mode="lines",
                    line=dict(color="firebrick", width=2),
                    name="Flag Upper",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(flag_start_date), pd.Timestamp(flag_end_date), signal_date_value],
                    y=[lower_start, lower_end, projected_lower],
                    mode="lines",
                    line=dict(color="teal", width=2),
                    name="Flag Lower",
                ),
                row=1,
                col=1,
            )

        figure.add_vline(x=signal_date_value, line_dash="dash", line_color="royalblue", row=1, col=1)
        if pd.notna(follow_through_date):
            figure.add_vline(x=follow_through_date, line_dash="dot", line_color="darkgreen", row=1, col=1)
        if pd.notna(entry_date):
            figure.add_vline(x=entry_date, line_dash="dot", line_color="mediumpurple", row=1, col=1)
        if pd.notna(exit_date):
            figure.add_vline(x=exit_date, line_dash="dot", line_color="gray", row=1, col=1)
        if pd.notna(stop_price):
            figure.add_hline(y=stop_price, line_dash="dot", line_color="indianred", row=1, col=1)
        if pd.notna(target_price):
            figure.add_hline(y=target_price, line_dash="dot", line_color="seagreen", row=1, col=1)

        for column, name, color in [
            ("flagpole_return", "Flagpole Return", "firebrick"),
            ("flag_retrace_ratio", "Flag Retrace", "darkorange"),
            ("reward_to_risk", "Reward/Risk", "seagreen"),
        ]:
            if column in price_window.columns:
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        line=dict(color=color, width=1.6),
                        name=name,
                    ),
                    row=2,
                    col=1,
                )

        checklist_plot = checklist.copy()
        checklist_plot["value_numeric"] = checklist_plot["value"].astype(int)
        figure.add_trace(
            go.Bar(
                x=checklist_plot["condition"],
                y=checklist_plot["value_numeric"],
                marker_color=np.where(checklist_plot["value_numeric"].eq(1), "seagreen", "indianred"),
                name="Conditions",
            ),
            row=3,
            col=1,
        )

        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Metric", row=2, col=1)
        figure.update_yaxes(title_text="Pass", row=3, col=1, range=[0, 1.2], tickvals=[0, 1])
        figure.update_layout(
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            height=900,
            width=1400,
            title=f"Bull Flag Context | {ticker} | {signal_date_value.date()}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        return figure


__all__ = ["BullFlagContinuationResearcher", "BullFlagStrategyConfig"]
