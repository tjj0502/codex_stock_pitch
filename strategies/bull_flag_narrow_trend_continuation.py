from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategies.bull_flag_continuation import (
    BullFlagContinuationResearcher,
    BullFlagStrategyConfig,
)


@dataclass(frozen=True)
class BullFlagNarrowTrendStrategyConfig(BullFlagStrategyConfig):
    narrow_trend_lookback_bars: int = 10
    narrow_trend_max_bear_ratio: float = 0.25
    narrow_trend_max_consecutive_bear_bars: int = 2
    narrow_trend_min_ema20_above_ratio: float = 0.90
    narrow_trend_max_upper_shadow_pct: float = 0.25
    narrow_trend_min_run_bars: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.narrow_trend_lookback_bars < 3:
            raise ValueError("narrow_trend_lookback_bars must be at least 3.")
        if not 0 <= self.narrow_trend_max_bear_ratio <= 1:
            raise ValueError("narrow_trend_max_bear_ratio must be in [0, 1].")
        if self.narrow_trend_max_consecutive_bear_bars < 0:
            raise ValueError("narrow_trend_max_consecutive_bear_bars must be non-negative.")
        if not 0 <= self.narrow_trend_min_ema20_above_ratio <= 1:
            raise ValueError("narrow_trend_min_ema20_above_ratio must be in [0, 1].")
        if not 0 <= self.narrow_trend_max_upper_shadow_pct <= 1:
            raise ValueError("narrow_trend_max_upper_shadow_pct must be in [0, 1].")
        if self.narrow_trend_min_run_bars < 1:
            raise ValueError("narrow_trend_min_run_bars must be at least 1.")


class BullFlagNarrowTrendContinuationResearcher(BullFlagContinuationResearcher):
    FEATURE_COLUMNS = BullFlagContinuationResearcher.FEATURE_COLUMNS + [
        "ema_20",
        "narrow_uptrend_state",
        "narrow_uptrend_run_length",
        "narrow_state_bear_ratio",
        "narrow_state_ema20_above_ratio",
        "narrow_state_max_consecutive_bear_bars",
        "narrow_state_peak_upper_shadow_pct",
        "left_state_start_date",
        "left_state_end_date",
        "left_state_bars",
    ]
    FEATURE_ANALYSIS_COLUMNS = BullFlagContinuationResearcher.FEATURE_ANALYSIS_COLUMNS + [
        "narrow_uptrend_run_length",
        "narrow_state_bear_ratio",
        "narrow_state_ema20_above_ratio",
        "narrow_state_max_consecutive_bear_bars",
        "left_state_bars",
    ]
    TRADE_COLUMNS = [
        *BullFlagContinuationResearcher.TRADE_COLUMNS[:31],
        "narrow_uptrend_run_length",
        "narrow_state_bear_ratio",
        "narrow_state_ema20_above_ratio",
        "narrow_state_max_consecutive_bear_bars",
        "left_state_start_date",
        "left_state_end_date",
        "left_state_bars",
        *BullFlagContinuationResearcher.TRADE_COLUMNS[31:],
    ]
    STRATEGY_NAME = "bull_flag_narrow_trend_continuation"

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        config: BullFlagNarrowTrendStrategyConfig | None = None,
    ) -> None:
        super().__init__(stock_candle_df, config=config or BullFlagNarrowTrendStrategyConfig())
        self.stock_candle_df.attrs["strategy_name"] = self.STRATEGY_NAME
        self.trade_df.attrs["strategy_name"] = self.STRATEGY_NAME

    def _annotate_ticker_context(self, ticker_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        cfg = self.config
        slope_lower_bound, slope_upper_bound = self._resolve_flag_channel_slope_bounds()
        row_count = len(ticker_frame)
        if row_count == 0:
            return {}

        date_values = ticker_frame["date"].to_numpy(dtype="datetime64[ns]")
        high_values = pd.to_numeric(ticker_frame["high"], errors="coerce").to_numpy(dtype=float)
        low_values = pd.to_numeric(ticker_frame["low"], errors="coerce").to_numpy(dtype=float)
        close_values = pd.to_numeric(ticker_frame["close"], errors="coerce").to_numpy(dtype=float)
        bullish_stack = ticker_frame["bullish_stack"].fillna(False).to_numpy(dtype=bool)
        pivot_low = ticker_frame["pivot_low"].fillna(False).to_numpy(dtype=bool)
        signal_quality_ok = ticker_frame["signal_quality_ok"].fillna(False).to_numpy(dtype=bool)
        close_gt_prev_high = ticker_frame["close_gt_prev_high"].fillna(False).to_numpy(dtype=bool)
        bullish_stack_run_length = pd.to_numeric(
            ticker_frame["bullish_stack_run_length"], errors="coerce"
        ).to_numpy(dtype=float)
        stack_spread_pct = pd.to_numeric(ticker_frame["stack_spread_pct"], errors="coerce").to_numpy(dtype=float)
        sma20_return_5 = pd.to_numeric(ticker_frame["sma20_return_5"], errors="coerce").to_numpy(dtype=float)
        sma60_return_10 = pd.to_numeric(ticker_frame["sma60_return_10"], errors="coerce").to_numpy(dtype=float)
        narrow_uptrend_state = ticker_frame["narrow_uptrend_state"].fillna(False).to_numpy(dtype=bool)
        pivot_low_indices = np.flatnonzero(pivot_low)

        def make_empty_annotations() -> dict[str, np.ndarray]:
            return {
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
                "left_state_start_date": self._nat_array(row_count),
                "left_state_end_date": self._nat_array(row_count),
                "left_state_bars": self._int_array(row_count),
                "flag_shape_ok": np.zeros(row_count, dtype=bool),
                "flag_retrace_ok": np.zeros(row_count, dtype=bool),
                "flag_channel_ok": np.zeros(row_count, dtype=bool),
                "bull_flag_candidate": np.zeros(row_count, dtype=bool),
                "breakout_candle": np.zeros(row_count, dtype=bool),
                "signal_candle": np.zeros(row_count, dtype=bool),
            }

        def record_setup(
            target: dict[str, np.ndarray],
            *,
            current_loc: int,
            flagpole_start_idx: int,
            peak_idx: int,
            flag_start_idx: int,
            left_state_start_idx: int,
            left_state_end_idx: int,
        ) -> None:
            flag_end_idx = current_loc - 1
            flag_bars = int(flag_end_idx - flag_start_idx + 1)
            if flag_bars < cfg.min_flag_bars or flag_bars > cfg.max_flag_bars:
                return

            flagpole_start_low = low_values[flagpole_start_idx]
            flag_peak_high = high_values[peak_idx]
            if (
                pd.isna(flagpole_start_low)
                or pd.isna(flag_peak_high)
                or flagpole_start_low <= 0
                or flag_peak_high <= flagpole_start_low
            ):
                return

            flagpole_bars = int(peak_idx - flagpole_start_idx)
            if flagpole_bars < cfg.min_flagpole_bars or flagpole_bars > cfg.max_flagpole_bars:
                return

            flagpole_return = float(flag_peak_high / flagpole_start_low - 1.0)
            if flagpole_return < cfg.min_flagpole_return:
                return

            flag_high_window = high_values[flag_start_idx : flag_end_idx + 1]
            flag_low_window = low_values[flag_start_idx : flag_end_idx + 1]
            if (
                flag_high_window.size != flag_bars
                or flag_low_window.size != flag_bars
                or np.isnan(flag_high_window).any()
                or np.isnan(flag_low_window).any()
            ):
                return

            flag_low_relative_idx = int(np.argmin(flag_low_window))
            flag_low_idx = flag_start_idx + flag_low_relative_idx
            flag_low = float(flag_low_window[flag_low_relative_idx])
            denominator = float(flag_peak_high - flagpole_start_low)
            if denominator <= 0:
                return

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
                slope_lower_bound <= upper_slope_pct <= slope_upper_bound
                and slope_lower_bound <= lower_slope_pct <= slope_upper_bound
            )
            bull_flag_candidate = flag_shape_ok and flag_retrace_ok and flag_channel_ok
            breakout_candle = bool(
                bull_flag_candidate
                and signal_quality_ok[current_loc]
                and close_gt_prev_high[current_loc]
                and pd.notna(close_values[current_loc])
                and close_values[current_loc] > projected_upper_line
            )

            target["signal_bullish_stack_run_length"][current_loc] = bullish_stack_run_length[current_loc]
            target["signal_stack_spread_pct"][current_loc] = stack_spread_pct[current_loc]
            target["signal_sma20_return_5"][current_loc] = sma20_return_5[current_loc]
            target["peak_bullish_stack_run_length"][current_loc] = bullish_stack_run_length[peak_idx]
            target["peak_sma60_return_10"][current_loc] = sma60_return_10[peak_idx]
            target["flagpole_start_date"][current_loc] = date_values[flagpole_start_idx]
            target["flagpole_start_low"][current_loc] = float(flagpole_start_low)
            target["flagpole_length"][current_loc] = float(flag_peak_high - flagpole_start_low)
            target["flag_peak_date"][current_loc] = date_values[peak_idx]
            target["flag_peak_high"][current_loc] = float(flag_peak_high)
            target["flagpole_bars"][current_loc] = flagpole_bars
            target["flagpole_return"][current_loc] = flagpole_return
            target["flag_start_date"][current_loc] = date_values[flag_start_idx]
            target["flag_end_date"][current_loc] = date_values[flag_end_idx]
            target["flag_bars"][current_loc] = flag_bars
            target["flag_low_date"][current_loc] = date_values[flag_low_idx]
            target["flag_low"][current_loc] = flag_low
            target["flag_retrace_ratio"][current_loc] = flag_retrace_ratio
            target["flag_width_pct"][current_loc] = flag_width_pct
            target["flag_upper_slope"][current_loc] = float(upper_slope)
            target["flag_lower_slope"][current_loc] = float(lower_slope)
            target["flag_upper_line_value"][current_loc] = projected_upper_line
            target["flag_lower_line_value"][current_loc] = projected_lower_line
            target["left_state_start_date"][current_loc] = date_values[left_state_start_idx]
            target["left_state_end_date"][current_loc] = date_values[left_state_end_idx]
            target["left_state_bars"][current_loc] = int(left_state_end_idx - left_state_start_idx + 1)
            target["flag_shape_ok"][current_loc] = flag_shape_ok
            target["flag_retrace_ok"][current_loc] = flag_retrace_ok
            target["flag_channel_ok"][current_loc] = flag_channel_ok
            target["bull_flag_candidate"][current_loc] = bull_flag_candidate
            target["breakout_candle"][current_loc] = breakout_candle
            target["signal_candle"][current_loc] = breakout_candle

        annotations = make_empty_annotations()
        for run_start_idx, run_end_idx in self._iter_true_runs(narrow_uptrend_state):
            if run_end_idx + 1 >= row_count:
                continue
            if run_end_idx - run_start_idx + 1 < cfg.narrow_trend_min_run_bars:
                continue

            peak_idx = int(run_end_idx)
            flagpole_start_idx = self._select_flagpole_start_index(
                pivot_low_indices,
                low_values,
                peak_idx=peak_idx,
                lookback_bars=cfg.flagpole_lookback_bars,
            )
            if flagpole_start_idx is None:
                continue

            flag_start_idx = int(run_end_idx + 1)
            breakout_start_idx = flag_start_idx + cfg.min_flag_bars
            breakout_end_idx = min(row_count - 1, flag_start_idx + cfg.max_flag_bars)
            if breakout_start_idx > breakout_end_idx:
                continue

            for current_loc in range(breakout_start_idx, breakout_end_idx + 1):
                record_setup(
                    annotations,
                    current_loc=current_loc,
                    flagpole_start_idx=flagpole_start_idx,
                    peak_idx=peak_idx,
                    flag_start_idx=flag_start_idx,
                    left_state_start_idx=int(run_start_idx),
                    left_state_end_idx=int(run_end_idx),
                )
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
        df["ema_20"] = ticker_group["close"].transform(lambda series: self._rolling_ema(series, 20))

        fast_window, mid_window, slow_window = cfg.ma_windows
        df["bullish_stack"] = (
            df[f"sma_{fast_window}"].gt(df[f"sma_{mid_window}"])
            & df[f"sma_{mid_window}"].gt(df[f"sma_{slow_window}"])
        )
        df["bullish_stack_run_length"] = ticker_group["bullish_stack"].transform(self._consecutive_true_run_length)
        df["stack_spread_pct"] = self._safe_ratio(
            df[f"sma_{fast_window}"] - df[f"sma_{slow_window}"], df["close"]
        ).replace([np.inf, -np.inf], np.nan)
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

        state_feature_columns = [
            "narrow_uptrend_state",
            "narrow_uptrend_run_length",
            "narrow_state_bear_ratio",
            "narrow_state_ema20_above_ratio",
            "narrow_state_max_consecutive_bear_bars",
            "narrow_state_peak_upper_shadow_pct",
        ]
        for column in state_feature_columns:
            if column == "narrow_uptrend_state":
                df[column] = False
            elif column in {"narrow_uptrend_run_length", "narrow_state_max_consecutive_bear_bars"}:
                df[column] = 0
            else:
                df[column] = np.nan

        row_count = len(df)
        date_columns = [
            "flagpole_start_date",
            "flag_peak_date",
            "flag_start_date",
            "flag_end_date",
            "flag_low_date",
            "left_state_start_date",
            "left_state_end_date",
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
        int_columns = ["flagpole_bars", "flag_bars", "left_state_bars"]
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
            state_annotations = self._compute_narrow_state_features(group_frame)
            for column, values in state_annotations.items():
                df.loc[positions, column] = values
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

    def inspect_signal(
        self,
        ticker: str,
        signal_date: str | pd.Timestamp,
        *,
        lookback: int = 60,
        lookahead: int = 10,
    ) -> dict[str, pd.DataFrame | dict[str, object]]:
        inspection = super().inspect_signal(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        target_date = pd.to_datetime(signal_date)
        ticker = str(ticker)
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        signal_rows = scored[
            scored["ticker"].astype(str).eq(ticker) & scored["date"].eq(target_date)
        ].reset_index(drop=True)
        if signal_rows.empty:
            return inspection

        signal_row = inspection["signal_row"].copy()
        source_row = signal_rows.iloc[[0]]
        extra_columns = [
            "ema_20",
            "narrow_uptrend_state",
            "narrow_uptrend_run_length",
            "narrow_state_bear_ratio",
            "narrow_state_ema20_above_ratio",
            "narrow_state_max_consecutive_bear_bars",
            "narrow_state_peak_upper_shadow_pct",
            "left_state_start_date",
            "left_state_end_date",
            "left_state_bars",
        ]
        for column in extra_columns:
            if column in source_row.columns:
                signal_row[column] = source_row[column].iat[0]
        inspection["signal_row"] = signal_row

        inspection["summary"].update(
            {
                "left_trend_mode": "narrow_state",
                "narrow_uptrend_run_length": None
                if pd.isna(source_row["narrow_uptrend_run_length"].iat[0])
                else int(source_row["narrow_uptrend_run_length"].iat[0]),
                "narrow_state_bear_ratio": float(source_row["narrow_state_bear_ratio"].iat[0])
                if pd.notna(source_row["narrow_state_bear_ratio"].iat[0])
                else np.nan,
                "narrow_state_ema20_above_ratio": float(source_row["narrow_state_ema20_above_ratio"].iat[0])
                if pd.notna(source_row["narrow_state_ema20_above_ratio"].iat[0])
                else np.nan,
                "left_state_start_date": source_row["left_state_start_date"].iat[0],
                "left_state_end_date": source_row["left_state_end_date"].iat[0],
            }
        )
        checklist = inspection["condition_checklist"].copy()
        left_gate = pd.DataFrame(
            [{"condition": "left_trend_gate", "value": bool(pd.notna(source_row["left_state_end_date"].iat[0]))}]
        )
        inspection["condition_checklist"] = pd.concat([checklist.iloc[:5], left_gate, checklist.iloc[5:]], ignore_index=True)
        return inspection

    def get_candidates(self, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if "signal_candle" not in self.stock_candle_df.columns:
            self.add_signals()

        base_columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "left_state_start_date",
            "left_state_end_date",
            "left_state_bars",
            "flagpole_start_date",
            "flag_peak_date",
            "flag_peak_high",
            "flagpole_return",
            "flag_bars",
            "flag_low",
            "flag_retrace_ratio",
            "flag_width_pct",
            "narrow_uptrend_run_length",
            "narrow_state_bear_ratio",
            "narrow_state_ema20_above_ratio",
            "narrow_state_max_consecutive_bear_bars",
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
            "left_state_start_date",
            "left_state_end_date",
            "left_state_bars",
            "flag_peak_high",
            "flag_low",
            "flagpole_return",
            "flag_retrace_ratio",
            "flag_bars",
            "narrow_uptrend_run_length",
            "narrow_state_bear_ratio",
            "narrow_state_ema20_above_ratio",
            "narrow_state_max_consecutive_bear_bars",
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

    def plot_signal_context(
        self,
        ticker: str,
        signal_date: str | pd.Timestamp,
        *,
        lookback: int = 60,
        lookahead: int = 10,
    ) -> go.Figure:
        figure = super().plot_signal_context(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        inspection = self.inspect_signal(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        signal_row = inspection["signal_row"]
        price_window = inspection["price_window"]
        if signal_row.empty or price_window.empty:
            return figure

        if "ema_20" in price_window.columns:
            figure.add_trace(
                go.Scatter(
                    x=price_window["date"],
                    y=price_window["ema_20"],
                    mode="lines",
                    name="EMA 20",
                    line=dict(color="teal", width=1.2, dash="dash"),
                ),
                row=1,
                col=1,
            )

        left_state_start_date = signal_row["left_state_start_date"].iat[0]
        left_state_end_date = signal_row["left_state_end_date"].iat[0]
        flag_peak_date = signal_row["flag_peak_date"].iat[0]

        def marker_y(date_value: pd.Timestamp | object, price_column: str, fallback_column: str) -> float:
            if pd.isna(date_value):
                return np.nan
            matching_rows = price_window[price_window["date"].eq(pd.Timestamp(date_value))]
            if matching_rows.empty:
                return np.nan
            if price_column in matching_rows.columns and pd.notna(matching_rows[price_column].iat[0]):
                return float(matching_rows[price_column].iat[0])
            if fallback_column in matching_rows.columns and pd.notna(matching_rows[fallback_column].iat[0]):
                return float(matching_rows[fallback_column].iat[0])
            return np.nan

        if pd.notna(left_state_start_date):
            left_state_start_y = marker_y(left_state_start_date, "low", "close")
            if pd.notna(left_state_start_y):
                figure.add_trace(
                    go.Scatter(
                        x=[left_state_start_date],
                        y=[left_state_start_y],
                        mode="markers+text",
                        marker=dict(size=12, symbol="circle", color="dodgerblue"),
                        text=["Left State Start"],
                        textposition="bottom left",
                        name="Left State Start",
                    ),
                    row=1,
                    col=1,
                )

        left_state_end_matches_peak = (
            pd.notna(left_state_end_date)
            and pd.notna(flag_peak_date)
            and pd.Timestamp(left_state_end_date) == pd.Timestamp(flag_peak_date)
        )
        if left_state_end_matches_peak and pd.notna(signal_row["flag_peak_high"].iat[0]):
            figure.add_trace(
                go.Scatter(
                    x=[left_state_end_date],
                    y=[signal_row["flag_peak_high"].iat[0]],
                    mode="markers+text",
                    marker=dict(size=13, symbol="diamond-wide", color="crimson"),
                    text=["Left State End / Flag Peak"],
                    textposition="top left",
                    name="Left State End / Flag Peak",
                ),
                row=1,
                col=1,
            )
        elif pd.notna(left_state_end_date):
            left_state_end_y = marker_y(left_state_end_date, "high", "close")
            if pd.notna(left_state_end_y):
                figure.add_trace(
                    go.Scatter(
                        x=[left_state_end_date],
                        y=[left_state_end_y],
                        mode="markers+text",
                        marker=dict(size=13, symbol="diamond-wide", color="crimson"),
                        text=["Left State End"],
                        textposition="top left",
                        name="Left State End",
                    ),
                    row=1,
                    col=1,
                )
        return figure
