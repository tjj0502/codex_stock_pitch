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
class GapBreakoutStrategyConfig:
    """
    Parameter bundle for a trend-following bullish gap strategy.

    Default behavior follows the more conservative "confirm" path:
    1. require an existing up-trend before the gap day
    2. detect a bullish gap with strong activity and controlled extension
    3. wait 1-3 sessions for confirmation without the gap being filled
    4. enter on the next open
    5. exit on a close below the gap stop, a close below MA10, or time stop
    """

    universe: str = "all_a"
    entry_mode: str = "confirm"
    confirm_level_mode: str = "gap_close"
    allowed_gap_types: tuple[str, ...] = ("breakaway", "continuation")

    min_listing_bars: int = 120
    min_avg_turnover_20: float = 100_000_000.0

    ma_windows: tuple[int, int, int] = (10, 20, 60)
    trend_return_40: float = 0.10
    trend_strength_window: int = 20
    min_closes_above_sma20: int = 12

    min_gap_pct: float = 0.015
    volume_multiple: float = 1.50
    turnover_multiple: float = 1.80
    max_gap_upper_shadow_pct: float = 0.35
    max_extension_from_sma20: float = 0.12
    max_consecutive_up_days_before_gap: int = 6
    platform_lookback: int = 20
    confirm_window: int = 3

    stop_reference: str = "gap_low"
    max_holding_days: int = 10
    enable_gap_stop: bool = True
    enable_ma10_exit: bool = True
    enable_time_stop: bool = True

    def __post_init__(self) -> None:
        if self.universe not in {"all_a", "csi500", "hs300"}:
            raise ValueError("universe must be one of 'all_a', 'csi500', or 'hs300'.")
        if self.entry_mode not in {"gap_day", "confirm"}:
            raise ValueError("entry_mode must be either 'gap_day' or 'confirm'.")
        if self.confirm_level_mode not in {"gap_close", "gap_midpoint"}:
            raise ValueError("confirm_level_mode must be either 'gap_close' or 'gap_midpoint'.")
        if not self.allowed_gap_types:
            raise ValueError("allowed_gap_types must contain at least one gap type.")
        invalid_gap_types = set(self.allowed_gap_types) - {"breakaway", "continuation"}
        if invalid_gap_types:
            raise ValueError(f"Unsupported gap types: {sorted(invalid_gap_types)}")
        if self.min_listing_bars < 20:
            raise ValueError("min_listing_bars must be at least 20.")
        if self.min_avg_turnover_20 <= 0:
            raise ValueError("min_avg_turnover_20 must be positive.")
        if len(self.ma_windows) != 3 or any(window < 2 for window in self.ma_windows):
            raise ValueError("ma_windows must contain exactly three integers >= 2.")
        if tuple(sorted(self.ma_windows)) != self.ma_windows:
            raise ValueError("ma_windows must be sorted from fast to slow.")
        if self.trend_strength_window < 5:
            raise ValueError("trend_strength_window must be at least 5.")
        if self.min_closes_above_sma20 < 1 or self.min_closes_above_sma20 > self.trend_strength_window:
            raise ValueError("min_closes_above_sma20 must be between 1 and trend_strength_window.")
        if self.min_gap_pct <= 0:
            raise ValueError("min_gap_pct must be positive.")
        if self.volume_multiple <= 0:
            raise ValueError("volume_multiple must be positive.")
        if self.turnover_multiple <= 0:
            raise ValueError("turnover_multiple must be positive.")
        if not 0 <= self.max_gap_upper_shadow_pct <= 1:
            raise ValueError("max_gap_upper_shadow_pct must be in [0, 1].")
        if self.max_extension_from_sma20 <= 0:
            raise ValueError("max_extension_from_sma20 must be positive.")
        if self.max_consecutive_up_days_before_gap < 0:
            raise ValueError("max_consecutive_up_days_before_gap must be non-negative.")
        if self.platform_lookback < 5:
            raise ValueError("platform_lookback must be at least 5.")
        if self.confirm_window < 1:
            raise ValueError("confirm_window must be at least 1.")
        if self.stop_reference not in {"gap_low", "gap_midpoint"}:
            raise ValueError("stop_reference must be either 'gap_low' or 'gap_midpoint'.")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days must be at least 1.")
        if not any([self.enable_gap_stop, self.enable_ma10_exit, self.enable_time_stop]):
            raise ValueError("At least one exit rule must be enabled.")


class GapBreakoutContinuationResearcher(BlueChipRangeReversionResearcher):
    """
    Research helper for a bullish gap-breakout continuation setup.

    The strategy deliberately ignores most random gaps and focuses on:
    - a pre-existing up-trend
    - a clean upward gap
    - strong activity on the gap day
    - either immediate entry or a short confirmation window
    """

    REQUIRED_COLUMNS = REQUIRED_COLUMNS
    NUMERIC_COLUMNS = NUMERIC_COLUMNS
    STRING_COLUMNS = STRING_COLUMNS
    FEATURE_COLUMNS = [
        "listing_bars",
        "is_st_name",
        "suspended_day",
        "avg_volume_20",
        "avg_turnover_20",
        "liquidity_ok",
        "sma_10",
        "sma_20",
        "sma_60",
        "ret_40d",
        "close_above_sma20_count_20",
        "trend_filter_ok",
        "extension_from_sma20_pct",
        "extension_ok",
        "consecutive_up_days",
        "prev_consecutive_up_days",
        "consecutive_up_days_ok",
        "prev_high",
        "platform_high_20",
        "up_gap",
        "gap_pct",
        "gap_volume_ok",
        "gap_turnover_ok",
        "gap_activity_ok",
        "gap_upper_shadow_pct",
        "gap_upper_shadow_ok",
        "breakaway_gap",
        "continuation_gap",
        "allowed_gap_type",
        "gap_day_candidate",
        "gap_day_date",
        "gap_type",
        "gap_prev_high",
        "gap_day_low",
        "gap_day_high",
        "gap_day_close",
        "gap_midpoint",
        "gap_fill_level",
        "confirm_level_price",
        "confirm_deadline_date",
        "gap_unfilled",
        "confirm_level_ok",
        "signal_gap_age",
        "signal_day_setup_ready",
    ]
    SIGNAL_COLUMNS = [
        "entry_reference_price",
        "signal_hard_stop_price",
        "signal_take_profit_price",
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
        "gap_pct",
        "signal_gap_age",
        "extension_from_sma20_pct",
        "ret_40d",
        "avg_turnover_20",
        "prev_consecutive_up_days",
        "close_above_sma20_count_20",
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
        "gap_day_date",
        "gap_type",
        "gap_prev_high",
        "gap_day_low",
        "gap_day_high",
        "gap_day_close",
        "gap_midpoint",
        "gap_pct",
        "signal_gap_age",
        "avg_turnover_20",
        "ret_40d",
        "close_above_sma20_count_20",
        "extension_from_sma20_pct",
        "prev_consecutive_up_days",
        "entry_date",
        "entry_open",
        "signal_hard_stop_price",
        "signal_take_profit_price",
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
        config: GapBreakoutStrategyConfig | None = None,
        *,
        copy: bool = True,
    ) -> None:
        super().__init__(stock_candle_df, config=config or GapBreakoutStrategyConfig(), copy=copy)
        self.stock_candle_df.attrs["strategy_name"] = "gap_breakout_continuation"
        self.stock_candle_df.attrs["strategy_universe"] = self.config.universe
        self.trade_df.attrs["strategy_name"] = "gap_breakout_continuation"
        self.trade_df.attrs["strategy_universe"] = self.config.universe

    @staticmethod
    def _rolling_ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False, min_periods=window).mean()

    @staticmethod
    def _rolling_return(series: pd.Series, window: int) -> pd.Series:
        shifted = series.shift(window)
        result = series.div(shifted) - 1.0
        return result.where(series.gt(0) & shifted.gt(0))

    @staticmethod
    def _consecutive_true_counts(values: pd.Series) -> pd.Series:
        counts = np.zeros(len(values), dtype=np.int32)
        running = 0
        boolean_values = values.fillna(False).to_numpy(dtype=bool, copy=False)
        for idx, flag in enumerate(boolean_values):
            if flag:
                running += 1
            else:
                running = 0
            counts[idx] = running
        return pd.Series(counts, index=values.index, dtype="int32")

    @staticmethod
    def _nat_array(length: int) -> np.ndarray:
        return np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]")

    @staticmethod
    def _int_array(length: int) -> np.ndarray:
        return np.full(length, -1, dtype=np.int32)

    @classmethod
    def _finalize_nullable_int(cls, values: np.ndarray) -> pd.Series:
        return pd.Series(values.astype("int64", copy=False)).where(lambda s: s.ge(0), pd.NA).astype("Int64")

    def _annotate_signal_setup(self, group: pd.DataFrame) -> dict[str, np.ndarray]:
        cfg = self.config
        length = len(group)
        dates = group["date"].to_numpy(copy=False)
        lows = group["low"].to_numpy(dtype=float, copy=False)
        highs = group["high"].to_numpy(dtype=float, copy=False)
        closes = group["close"].to_numpy(dtype=float, copy=False)
        trend_ok = group["trend_filter_ok"].fillna(False).to_numpy(dtype=bool, copy=False)
        liquid_ok = group["liquidity_ok"].fillna(False).to_numpy(dtype=bool, copy=False)
        gap_candidates = group["gap_day_candidate"].fillna(False).to_numpy(dtype=bool, copy=False)
        breakaway_flags = group["breakaway_gap"].fillna(False).to_numpy(dtype=bool, copy=False)

        gap_day_date = self._nat_array(length)
        gap_type = np.full(length, "", dtype=object)
        gap_prev_high = np.full(length, np.nan, dtype=float)
        gap_day_low = np.full(length, np.nan, dtype=float)
        gap_day_high = np.full(length, np.nan, dtype=float)
        gap_day_close = np.full(length, np.nan, dtype=float)
        gap_midpoint = np.full(length, np.nan, dtype=float)
        gap_fill_level = np.full(length, np.nan, dtype=float)
        confirm_level_price = np.full(length, np.nan, dtype=float)
        confirm_deadline_date = self._nat_array(length)
        gap_unfilled = np.zeros(length, dtype=bool)
        confirm_level_ok = np.zeros(length, dtype=bool)
        signal_gap_age = self._int_array(length)
        signal_day_setup_ready = np.zeros(length, dtype=bool)

        for gap_idx in np.flatnonzero(gap_candidates):
            prev_high = float(group.iloc[gap_idx]["prev_high"])
            if not np.isfinite(prev_high) or prev_high <= 0:
                continue

            gap_low_value = float(lows[gap_idx])
            gap_high_value = float(highs[gap_idx])
            gap_close_value = float(closes[gap_idx])
            midpoint_value = (gap_high_value + gap_low_value) / 2.0
            confirm_value = gap_close_value if cfg.confirm_level_mode == "gap_close" else midpoint_value
            deadline_idx = min(length - 1, gap_idx + cfg.confirm_window)
            current_gap_type = "breakaway" if breakaway_flags[gap_idx] else "continuation"

            if cfg.entry_mode == "gap_day":
                candidate_target_indices = [gap_idx]
            else:
                candidate_target_indices = list(range(gap_idx + 1, min(length, gap_idx + cfg.confirm_window + 1)))

            for target_idx in candidate_target_indices:
                if cfg.entry_mode == "confirm":
                    lows_since_gap = lows[gap_idx + 1 : target_idx + 1]
                    currently_unfilled = lows_since_gap.size == 0 or float(np.nanmin(lows_since_gap)) > prev_high
                    if not currently_unfilled:
                        break
                    current_confirm_ok = bool(closes[target_idx] > confirm_value)
                else:
                    currently_unfilled = True
                    current_confirm_ok = True

                if not liquid_ok[target_idx] or not trend_ok[target_idx] or not current_confirm_ok:
                    continue

                gap_day_date[target_idx] = dates[gap_idx]
                gap_type[target_idx] = current_gap_type
                gap_prev_high[target_idx] = prev_high
                gap_day_low[target_idx] = gap_low_value
                gap_day_high[target_idx] = gap_high_value
                gap_day_close[target_idx] = gap_close_value
                gap_midpoint[target_idx] = midpoint_value
                gap_fill_level[target_idx] = prev_high
                confirm_level_price[target_idx] = confirm_value
                confirm_deadline_date[target_idx] = dates[deadline_idx]
                gap_unfilled[target_idx] = currently_unfilled
                confirm_level_ok[target_idx] = current_confirm_ok
                signal_gap_age[target_idx] = target_idx - gap_idx
                signal_day_setup_ready[target_idx] = True
                break

        return {
            "gap_day_date": gap_day_date,
            "gap_type": gap_type,
            "gap_prev_high": gap_prev_high,
            "gap_day_low": gap_day_low,
            "gap_day_high": gap_day_high,
            "gap_day_close": gap_day_close,
            "gap_midpoint": gap_midpoint,
            "gap_fill_level": gap_fill_level,
            "confirm_level_price": confirm_level_price,
            "confirm_deadline_date": confirm_deadline_date,
            "gap_unfilled": gap_unfilled,
            "confirm_level_ok": confirm_level_ok,
            "signal_gap_age": signal_gap_age,
            "signal_day_setup_ready": signal_day_setup_ready,
        }

    def add_features(self) -> pd.DataFrame:
        cfg = self.config
        df = self._sort_for_calculation(self.stock_candle_df.copy())
        ticker_group = df.groupby("ticker", sort=False)

        df["listing_bars"] = ticker_group.cumcount() + 1
        df["is_st_name"] = df["name"].astype("string").str.contains("ST", case=False, na=False)
        df["suspended_day"] = (
            df["open"].le(0)
            | df["close"].le(0)
            | df["high"].le(0)
            | df["low"].le(0)
            | df["volume"].le(0)
            | df["turnover"].le(0)
        )

        df["avg_volume_20"] = ticker_group["volume"].transform(
            lambda series: series.rolling(20, min_periods=20).mean()
        )
        df["avg_turnover_20"] = ticker_group["turnover"].transform(
            lambda series: series.rolling(20, min_periods=20).mean()
        )
        df["liquidity_ok"] = (
            ~df["is_st_name"]
            & ~df["suspended_day"]
            & df["listing_bars"].ge(cfg.min_listing_bars)
            & df["avg_turnover_20"].ge(cfg.min_avg_turnover_20)
        )

        for window in cfg.ma_windows:
            df[f"sma_{window}"] = ticker_group["close"].transform(
                lambda series, current_window=window: series.rolling(current_window, min_periods=current_window).mean()
            )
        df["ret_40d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 40))
        df["close_above_sma20_count_20"] = (
            df["close"].gt(df["sma_20"]).groupby(df["ticker"], sort=False).transform(
                lambda series: series.astype(int).rolling(cfg.trend_strength_window, min_periods=cfg.trend_strength_window).sum()
            )
        )
        df["trend_filter_ok"] = (
            df["close"].gt(df["sma_20"])
            & df["sma_20"].gt(df["sma_60"])
            & df["ret_40d"].gt(cfg.trend_return_40)
            & df["close_above_sma20_count_20"].ge(cfg.min_closes_above_sma20)
        )

        df["extension_from_sma20_pct"] = (
            (df["close"] - df["sma_20"]).div(df["sma_20"].where(df["sma_20"].ne(0)))
        ).replace([np.inf, -np.inf], np.nan)
        df["extension_ok"] = df["extension_from_sma20_pct"].le(cfg.max_extension_from_sma20)

        close_up = ticker_group["close"].diff().gt(0)
        df["consecutive_up_days"] = close_up.groupby(df["ticker"], sort=False).transform(self._consecutive_true_counts)
        df["prev_consecutive_up_days"] = ticker_group["consecutive_up_days"].shift(1).fillna(0)
        df["consecutive_up_days_ok"] = df["prev_consecutive_up_days"].le(cfg.max_consecutive_up_days_before_gap)

        df["prev_high"] = ticker_group["high"].shift(1)
        df["platform_high_20"] = ticker_group["high"].shift(1).transform(
            lambda series: series.rolling(cfg.platform_lookback, min_periods=cfg.platform_lookback).max()
        )
        df["up_gap"] = df["low"].gt(df["prev_high"])
        df["gap_pct"] = (
            (df["low"] - df["prev_high"]).div(df["prev_high"].where(df["prev_high"].gt(0)))
        ).replace([np.inf, -np.inf], np.nan)
        df["gap_volume_ok"] = df["volume"].gt(df["avg_volume_20"] * cfg.volume_multiple)
        df["gap_turnover_ok"] = df["turnover"].gt(df["avg_turnover_20"] * cfg.turnover_multiple)
        df["gap_activity_ok"] = df["gap_volume_ok"] | df["gap_turnover_ok"]

        candle_range = (df["high"] - df["low"]).where(lambda series: series.gt(0))
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        df["gap_upper_shadow_pct"] = upper_shadow.div(candle_range).replace([np.inf, -np.inf], np.nan)
        df["gap_upper_shadow_ok"] = df["gap_upper_shadow_pct"].le(cfg.max_gap_upper_shadow_pct)
        df["breakaway_gap"] = df["up_gap"] & df["low"].gt(df["platform_high_20"])
        df["continuation_gap"] = df["up_gap"] & ~df["breakaway_gap"]
        df["allowed_gap_type"] = False
        if "breakaway" in cfg.allowed_gap_types:
            df["allowed_gap_type"] |= df["breakaway_gap"]
        if "continuation" in cfg.allowed_gap_types:
            df["allowed_gap_type"] |= df["continuation_gap"]

        df["gap_day_candidate"] = (
            df["liquidity_ok"]
            & df["trend_filter_ok"]
            & df["up_gap"]
            & df["gap_pct"].gt(cfg.min_gap_pct)
            & df["gap_activity_ok"]
            & df["gap_upper_shadow_ok"]
            & df["extension_ok"]
            & df["consecutive_up_days_ok"]
            & df["allowed_gap_type"]
        )

        defaults: dict[str, object] = {
            "gap_day_date": pd.NaT,
            "gap_type": "",
            "gap_prev_high": np.nan,
            "gap_day_low": np.nan,
            "gap_day_high": np.nan,
            "gap_day_close": np.nan,
            "gap_midpoint": np.nan,
            "gap_fill_level": np.nan,
            "confirm_level_price": np.nan,
            "confirm_deadline_date": pd.NaT,
            "gap_unfilled": False,
            "confirm_level_ok": False,
            "signal_gap_age": pd.NA,
            "signal_day_setup_ready": False,
        }
        for column, default_value in defaults.items():
            df[column] = default_value

        annotated_frames: list[pd.DataFrame] = []
        for _, group in df.groupby("ticker", sort=False):
            annotated = group.copy()
            setup = self._annotate_signal_setup(annotated)
            for column, values in setup.items():
                annotated[column] = values
            annotated_frames.append(annotated)

        enriched = pd.concat(annotated_frames, ignore_index=True) if annotated_frames else df
        if "signal_gap_age" in enriched.columns:
            raw_gap_age = enriched["signal_gap_age"].fillna(-1).astype("int32", copy=False).to_numpy()
            enriched["signal_gap_age"] = self._finalize_nullable_int(raw_gap_age)
        return self._store_output(enriched)

    def add_signals(self) -> pd.DataFrame:
        cfg = self.config
        df = self._sort_for_calculation(self.add_features().copy())
        df["entry_reference_price"] = df["close"].where(df["signal_day_setup_ready"])
        if cfg.stop_reference == "gap_low":
            df["signal_hard_stop_price"] = df["gap_day_low"].where(df["signal_day_setup_ready"])
        else:
            df["signal_hard_stop_price"] = df["gap_midpoint"].where(df["signal_day_setup_ready"])
        df["signal_take_profit_price"] = np.nan
        df["entry_signal"] = df["signal_day_setup_ready"] & df["signal_hard_stop_price"].notna()
        return self._store_output(df)

    def add_research_outcomes(self) -> pd.DataFrame:
        cfg = self.config
        df = self._sort_for_calculation(self.add_signals().copy())
        redundant_index_columns = [column for column in ("index", "level_0") if column in df.columns]
        if redundant_index_columns:
            df = df.drop(columns=redundant_index_columns)
        for column, default_value in [
            ("entry_signal_executed", False),
            ("entry_signal_suppressed", False),
            ("entry_date_next", pd.NaT),
            ("entry_open_next", np.nan),
            ("exit_signal_date", pd.NaT),
            ("exit_date_next", pd.NaT),
            ("exit_open_next", np.nan),
            ("exit_reason", pd.NA),
            ("holding_days", pd.NA),
            ("realized_open_to_open_return", np.nan),
            ("max_favorable_excursion", np.nan),
            ("max_adverse_excursion", np.nan),
        ]:
            df[column] = default_value

        df = df.reset_index(drop=False)
        for _, group in df.groupby("ticker", sort=False):
            group = group.reset_index(drop=True)
            next_search_loc = 0
            while True:
                candidate_locs = group.index[(group.index >= next_search_loc) & group["entry_signal"]]
                if candidate_locs.empty:
                    break

                signal_loc = int(candidate_locs[0])
                signal_idx = int(group.at[signal_loc, "index"])
                entry_loc = signal_loc + 1
                if entry_loc >= len(group):
                    break

                entry_row = group.iloc[entry_loc]
                entry_price = entry_row["open"]
                if pd.isna(entry_price) or entry_price <= 0:
                    next_search_loc = entry_loc + 1
                    continue

                stop_price = float(group.at[signal_loc, "signal_hard_stop_price"])
                exit_reason: str | None = None
                exit_signal_loc: int | None = None
                executed_exit_loc: int | None = None
                path_end_loc = len(group) - 1

                for eval_loc in range(entry_loc, len(group)):
                    eval_row = group.iloc[eval_loc]
                    close_price = eval_row["close"]
                    sma10_value = eval_row.get("sma_10", np.nan)

                    if cfg.enable_gap_stop and pd.notna(close_price) and close_price <= stop_price:
                        exit_reason = "gap_stop"
                        exit_signal_loc = eval_loc
                    elif cfg.enable_ma10_exit and pd.notna(close_price) and pd.notna(sma10_value) and close_price < sma10_value:
                        exit_reason = "ma10_exit"
                        exit_signal_loc = eval_loc
                    elif cfg.enable_time_stop and (eval_loc - entry_loc + 1) >= cfg.max_holding_days:
                        exit_reason = "time_stop"
                        exit_signal_loc = eval_loc

                    if exit_reason is not None:
                        executed_exit_loc = eval_loc + 1 if eval_loc + 1 < len(group) else None
                        path_end_loc = eval_loc
                        break

                path_slice = group.iloc[entry_loc : path_end_loc + 1]
                path_high = path_slice["high"].where(path_slice["high"].gt(0), path_slice["close"])
                path_low = path_slice["low"].where(path_slice["low"].gt(0), path_slice["close"])
                if entry_price > 0 and not path_slice.empty:
                    df.at[signal_idx, "max_favorable_excursion"] = float(path_high.max() / entry_price - 1.0)
                    df.at[signal_idx, "max_adverse_excursion"] = float(path_low.min() / entry_price - 1.0)

                df.at[signal_idx, "entry_signal_executed"] = True
                df.at[signal_idx, "entry_date_next"] = entry_row["date"]
                df.at[signal_idx, "entry_open_next"] = float(entry_price)
                df.at[signal_idx, "exit_reason"] = exit_reason if exit_reason is not None else "open_position"

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
        self.add_signals()
        df = self.stock_candle_df.copy()
        candidates = df[df["entry_signal"]].copy()
        if candidates.empty:
            return candidates
        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else candidates["date"].max()
        candidates = candidates[candidates["date"] == target_date].copy()
        columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "gap_day_date",
            "gap_type",
            "gap_pct",
            "signal_gap_age",
            "gap_day_low",
            "gap_prev_high",
            "gap_day_close",
            "confirm_level_price",
            "avg_turnover_20",
            "ret_40d",
            "extension_from_sma20_pct",
            "prev_consecutive_up_days",
            "entry_signal",
        ]
        selected = [column for column in columns if column in candidates.columns]
        return candidates.loc[:, selected].sort_values(
            ["gap_pct", "avg_turnover_20", "ticker"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )

    def get_next_session_candidates(
        self,
        as_of_date: str | pd.Timestamp | None = None,
        *,
        next_trade_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        self.add_signals()
        df = self._sort_for_calculation(self.stock_candle_df.copy())
        output_columns = [
            "signal_date",
            "planned_entry_date",
            "ticker",
            "ts_code",
            "name",
            "entry_reference_price",
            "planned_hard_stop_price",
            "gap_day_date",
            "gap_type",
            "gap_pct",
            "signal_gap_age",
            "gap_day_low",
            "gap_prev_high",
            "gap_day_close",
            "avg_turnover_20",
            "ret_40d",
            "entry_signal_live",
        ]
        if df.empty:
            return pd.DataFrame(columns=output_columns)

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else df["date"].max()
        planned_entry_date = pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)
        candidates = df[df["date"].eq(target_date) & df["entry_signal"]].copy()
        if candidates.empty:
            return pd.DataFrame(columns=output_columns)

        candidates["signal_date"] = candidates["date"]
        candidates["planned_entry_date"] = planned_entry_date
        candidates["planned_hard_stop_price"] = candidates["signal_hard_stop_price"]
        candidates["entry_signal_live"] = True
        return candidates.loc[:, output_columns].sort_values(
            ["gap_pct", "avg_turnover_20", "ticker"],
            ascending=[False, False, True],
            kind="mergesort",
            ignore_index=True,
        )

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

        self.add_research_outcomes()
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        trade_df = self.trade_df.copy()
        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else scored["date"].max()
        planned_exit_date = pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)

        positions = positions_df.copy()
        positions["ticker"] = positions["ticker"].astype("string")
        positions["entry_date"] = pd.to_datetime(positions["entry_date"], errors="coerce")
        positions["entry_price"] = pd.to_numeric(positions["entry_price"], errors="coerce")

        rows: list[dict[str, object]] = []
        for _, position in positions.iterrows():
            ticker = str(position["ticker"])
            entry_date = pd.to_datetime(position["entry_date"])
            ticker_frame = scored[scored["ticker"].astype(str).eq(ticker) & scored["date"].le(target_date)].copy()
            record = dict(position)
            record.update(
                {
                    "as_of_date": target_date,
                    "latest_bar_date": pd.NaT,
                    "signal_date_resolved": pd.NaT,
                    "gap_day_date": pd.NaT,
                    "gap_type": pd.NA,
                    "current_close": np.nan,
                    "current_sma10": np.nan,
                    "hard_stop_price": np.nan,
                    "trading_days_in_trade": pd.NA,
                    "days_until_time_stop": pd.NA,
                    "exit_signal": False,
                    "exit_reason": pd.NA,
                    "planned_exit_date": pd.NaT,
                    "action": "data_issue",
                    "issue": pd.NA,
                    "pnl_pct": np.nan,
                    "pnl_amount": np.nan,
                }
            )

            if ticker_frame.empty:
                record["issue"] = "ticker_missing_on_or_before_as_of_date"
                rows.append(record)
                continue

            latest_row = ticker_frame.iloc[-1]
            record["latest_bar_date"] = latest_row["date"]
            record["current_close"] = latest_row["close"]
            record["current_sma10"] = latest_row.get("sma_10", np.nan)

            matching_trades = trade_df[
                trade_df["ticker"].astype(str).eq(ticker) & trade_df["entry_date"].eq(entry_date)
            ].copy()
            if matching_trades.empty:
                record["issue"] = "unable_to_resolve_signal_from_trade_df"
                rows.append(record)
                continue

            trade_row = matching_trades.sort_values("signal_date", kind="mergesort").iloc[-1]
            record["signal_date_resolved"] = trade_row["signal_date"]
            record["gap_day_date"] = trade_row.get("gap_day_date", pd.NaT)
            record["gap_type"] = trade_row.get("gap_type", pd.NA)
            record["hard_stop_price"] = trade_row.get("signal_hard_stop_price", np.nan)

            entry_price = float(position["entry_price"]) if pd.notna(position["entry_price"]) else np.nan
            if pd.notna(entry_price) and entry_price > 0 and pd.notna(latest_row["close"]):
                record["pnl_pct"] = float(latest_row["close"] / entry_price - 1.0)
                record["pnl_amount"] = float(latest_row["close"] - entry_price)

            trade_days = int(ticker_frame["date"].ge(entry_date).sum())
            record["trading_days_in_trade"] = trade_days
            record["days_until_time_stop"] = max(int(self.config.max_holding_days - trade_days), 0)

            exit_reason: str | None = None
            if self.config.enable_gap_stop and pd.notna(record["hard_stop_price"]) and pd.notna(latest_row["close"]):
                if float(latest_row["close"]) <= float(record["hard_stop_price"]):
                    exit_reason = "gap_stop"
            if (
                exit_reason is None
                and self.config.enable_ma10_exit
                and pd.notna(latest_row["close"])
                and pd.notna(latest_row.get("sma_10", np.nan))
                and float(latest_row["close"]) < float(latest_row["sma_10"])
            ):
                exit_reason = "ma10_exit"
            if exit_reason is None and self.config.enable_time_stop and trade_days >= self.config.max_holding_days:
                exit_reason = "time_stop"

            if exit_reason is None:
                record["action"] = "hold"
            else:
                record["exit_signal"] = True
                record["exit_reason"] = exit_reason
                record["planned_exit_date"] = planned_exit_date
                record["action"] = "exit_next_open"

            rows.append(record)

        monitored = pd.DataFrame(rows)
        if monitored.empty:
            return monitored
        return monitored.sort_values(
            ["action", "ticker"],
            ascending=[True, True],
            kind="mergesort",
            ignore_index=True,
        )

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
        ticker = str(ticker)
        target_date = pd.to_datetime(signal_date)
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        ticker_frame = scored[scored["ticker"].astype(str).eq(ticker)].reset_index(drop=True)
        if ticker_frame.empty:
            raise ValueError(f"Ticker '{ticker}' is not present in stock_candle_df.")

        signal_rows = ticker_frame[ticker_frame["date"].eq(target_date)]
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
        start_loc = max(0, signal_loc - lookback)
        end_loc = min(len(ticker_frame), signal_loc + lookahead + 1)
        price_window = ticker_frame.iloc[start_loc:end_loc].copy().reset_index(drop=True)
        price_window["signal_marker"] = price_window["date"].eq(target_date)
        price_window["gap_day_marker"] = price_window["date"].eq(signal_row["gap_day_date"].iat[0])
        price_window["entry_marker"] = price_window["date"].eq(signal_row["entry_date_next"].iat[0])
        price_window["exit_marker"] = price_window["date"].eq(signal_row["exit_date_next"].iat[0])

        checklist = pd.DataFrame(
            {
                "condition": [
                    "liquidity_ok",
                    "trend_filter_ok",
                    "up_gap",
                    "gap_activity_ok",
                    "gap_upper_shadow_ok",
                    "extension_ok",
                    "consecutive_up_days_ok",
                    "allowed_gap_type",
                    "gap_unfilled",
                    "confirm_level_ok",
                    "entry_signal",
                    "entry_signal_executed",
                ],
                "value": [
                    bool(signal_row["liquidity_ok"].iat[0]),
                    bool(signal_row["trend_filter_ok"].iat[0]),
                    bool(signal_row["up_gap"].iat[0]),
                    bool(signal_row["gap_activity_ok"].iat[0]),
                    bool(signal_row["gap_upper_shadow_ok"].iat[0]),
                    bool(signal_row["extension_ok"].iat[0]),
                    bool(signal_row["consecutive_up_days_ok"].iat[0]),
                    bool(signal_row["allowed_gap_type"].iat[0]),
                    bool(signal_row["gap_unfilled"].iat[0]),
                    bool(signal_row["confirm_level_ok"].iat[0]),
                    bool(signal_row["entry_signal"].iat[0]),
                    bool(signal_row["entry_signal_executed"].iat[0]),
                ],
            }
        )

        summary = {
            "ticker": ticker,
            "signal_date": target_date,
            "gap_day_date": signal_row["gap_day_date"].iat[0],
            "gap_type": signal_row["gap_type"].iat[0],
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
            "sma_10",
            "sma_20",
            "sma_60",
            "ret_40d",
            "avg_turnover_20",
            "trend_filter_ok",
            "gap_day_date",
            "gap_type",
            "gap_prev_high",
            "gap_day_low",
            "gap_day_high",
            "gap_day_close",
            "gap_midpoint",
            "gap_pct",
            "signal_gap_age",
            "confirm_level_price",
            "signal_hard_stop_price",
            "entry_date_next",
            "entry_open_next",
            "entry_signal",
            "entry_signal_executed",
            "entry_signal_suppressed",
            "exit_signal_date",
            "exit_date_next",
            "exit_open_next",
            "exit_reason",
            "realized_open_to_open_return",
            "max_favorable_excursion",
            "max_adverse_excursion",
        ]
        selected_columns = [column for column in signal_columns if column in signal_row.columns]
        return {
            "summary": summary,
            "signal_row": signal_row.loc[:, selected_columns].reset_index(drop=True),
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
            row_heights=[0.58, 0.18, 0.24],
            subplot_titles=("Price Context", "Volume / Turnover", "Signal Conditions"),
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
            ("sma_10", "SMA 10", "darkorange", "solid"),
            ("sma_20", "SMA 20", "steelblue", "dash"),
            ("sma_60", "SMA 60", "slategray", "dot"),
        ]:
            if column in price_window.columns:
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.4, dash=dash),
                    ),
                    row=1,
                    col=1,
                )

        for column, name, color in [
            ("gap_prev_high", "Gap Fill Level", "firebrick"),
            ("gap_day_low", "Gap Low", "seagreen"),
            ("confirm_level_price", "Confirm Level", "mediumpurple"),
        ]:
            if column in price_window.columns and price_window[column].notna().any():
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=1.5),
                    ),
                    row=1,
                    col=1,
                )

        signal_date_value = signal_row["date"].iat[0]
        gap_day_date = signal_row["gap_day_date"].iat[0]
        figure.add_vline(x=signal_date_value, line_dash="dash", line_color="royalblue", row=1, col=1)
        if pd.notna(gap_day_date):
            figure.add_vline(x=gap_day_date, line_dash="dot", line_color="darkgreen", row=1, col=1)

        entry_date = signal_row["entry_date_next"].iat[0]
        if pd.notna(entry_date):
            entry_price = signal_row["entry_open_next"].iat[0]
            if pd.notna(entry_price):
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

        exit_date = signal_row["exit_date_next"].iat[0]
        exit_reason = None if pd.isna(signal_row["exit_reason"].iat[0]) else str(signal_row["exit_reason"].iat[0])
        if pd.notna(exit_date) and "exit_open_next" in signal_row.columns:
            exit_price = signal_row["exit_open_next"].iat[0]
            if pd.notna(exit_price):
                figure.add_trace(
                    go.Scatter(
                        x=[exit_date],
                        y=[exit_price],
                        mode="markers+text",
                        marker=dict(size=15, symbol="x", color="red"),
                        text=[f"Exit ({exit_reason})" if exit_reason else "Exit"],
                        textposition="top center",
                        name="Exit",
                    ),
                    row=1,
                    col=1,
                )

        if "volume" in price_window.columns:
            figure.add_trace(
                go.Bar(
                    x=price_window["date"],
                    y=price_window["volume"],
                    name="Volume",
                    marker_color="lightsteelblue",
                ),
                row=2,
                col=1,
            )
        if "avg_volume_20" in price_window.columns:
            figure.add_trace(
                go.Scatter(
                    x=price_window["date"],
                    y=price_window["avg_volume_20"],
                    mode="lines",
                    name="Avg Volume 20",
                    line=dict(color="navy", width=1.4),
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

        ret_40d = signal_row["ret_40d"].iat[0] if "ret_40d" in signal_row.columns else np.nan
        gap_pct = signal_row["gap_pct"].iat[0] if "gap_pct" in signal_row.columns else np.nan
        title = (
            f"{ticker} | signal {pd.Timestamp(signal_date_value).date()} | "
            f"gap {gap_pct:.2%} | ret40 {ret_40d:.2%} | exit {exit_reason or 'open'}"
            if pd.notna(gap_pct) and pd.notna(ret_40d)
            else f"{ticker} | signal {pd.Timestamp(signal_date_value).date()}"
        )
        figure.update_layout(
            height=980,
            width=1200,
            template="plotly_white",
            hovermode="x unified",
            title=title,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Volume", row=2, col=1)
        figure.update_yaxes(title_text="Met", row=3, col=1, range=[0, 1.2])
        return figure


__all__ = ["GapBreakoutContinuationResearcher", "GapBreakoutStrategyConfig"]
