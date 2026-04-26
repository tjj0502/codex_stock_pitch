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
class BullFlagDynamicExitConfig(BullFlagStrategyConfig):
    """
    Parameter bundle for bull-flag exit variants.

    These variants keep the baseline entry logic intact and only modify
    post-entry trade management.
    """

    tp1_fraction_of_target: float = 0.50
    breakeven_buffer_pct: float = 0.0
    trailing_stop_fraction_of_flagpole: float = 0.25

    ma_trail_window: int = 10
    ma_exit_buffer_pct: float = 0.0

    structure_trail_lookback: int = 5
    structure_trail_buffer_pct: float = 0.0

    vol_failure_threshold: float = 2.0
    close_retrace_pct: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.tp1_fraction_of_target < 1:
            raise ValueError("tp1_fraction_of_target must be in (0, 1).")
        if self.breakeven_buffer_pct < 0:
            raise ValueError("breakeven_buffer_pct must be non-negative.")
        if self.trailing_stop_fraction_of_flagpole <= 0:
            raise ValueError("trailing_stop_fraction_of_flagpole must be positive.")
        if self.ma_trail_window < 2:
            raise ValueError("ma_trail_window must be at least 2.")
        if self.ma_exit_buffer_pct < 0:
            raise ValueError("ma_exit_buffer_pct must be non-negative.")
        if self.structure_trail_lookback < 1:
            raise ValueError("structure_trail_lookback must be at least 1.")
        if self.structure_trail_buffer_pct < 0:
            raise ValueError("structure_trail_buffer_pct must be non-negative.")
        if self.vol_failure_threshold <= 0:
            raise ValueError("vol_failure_threshold must be positive.")
        if not 0 < self.close_retrace_pct < 1:
            raise ValueError("close_retrace_pct must be in (0, 1).")


class BullFlagDynamicExitResearcherBase(BullFlagContinuationResearcher):
    """
    Shared base for bull-flag dynamic exit variants.

    Design principles:
    - entry recognition is inherited unchanged from the static bull-flag model
    - TP1 must be hit before any dynamic management activates
    - all TP1-based logic becomes active from the next bar onward to avoid
      ambiguous intraday assumptions on daily data
    """

    OUTCOME_COLUMNS = BullFlagContinuationResearcher.OUTCOME_COLUMNS + [
        "dynamic_exit_variant",
        "tp1_price",
        "tp1_reached",
        "tp1_hit_date",
        "post_tp1_stop_price",
        "ma_trail_value",
        "structure_trail_value",
        "relative_volume_20_signal",
        "highest_close_since_tp1",
        "close_retrace_threshold",
    ]
    TRADE_COLUMNS = BullFlagContinuationResearcher.TRADE_COLUMNS + [
        "dynamic_exit_variant",
        "tp1_price",
        "tp1_reached",
        "tp1_hit_date",
        "post_tp1_stop_price",
        "ma_trail_value",
        "structure_trail_value",
        "relative_volume_20_signal",
        "highest_close_since_tp1",
        "close_retrace_threshold",
    ]
    FEATURE_ANALYSIS_COLUMNS = BullFlagContinuationResearcher.FEATURE_ANALYSIS_COLUMNS + [
        "tp1_reached",
    ]

    DYNAMIC_FEATURE_COLUMNS = [
        "ema_10",
        "ema_20",
        "volume_sma_20",
        "relative_volume_20",
        "prev_low",
    ]
    STRATEGY_NAME = "bull_flag_dynamic_exit_base"
    PROTECTIVE_STOP_REASON = "dynamic_stop"
    HAS_DYNAMIC_PROTECTIVE_STOP = False

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        config: BullFlagDynamicExitConfig | None = None,
        *,
        copy: bool = True,
    ) -> None:
        super().__init__(stock_candle_df, config=config or BullFlagDynamicExitConfig(), copy=copy)
        self.stock_candle_df.attrs["strategy_name"] = self.STRATEGY_NAME
        self.trade_df.attrs["strategy_name"] = self.STRATEGY_NAME

    def _ensure_dynamic_exit_features(self) -> pd.DataFrame:
        if self._has_columns(self.DYNAMIC_FEATURE_COLUMNS):
            return self.stock_candle_df

        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)
        if "ema_10" not in df.columns:
            df["ema_10"] = ticker_group["close"].transform(
                lambda series: series.ewm(span=10, adjust=False, min_periods=10).mean()
            )
        if "ema_20" not in df.columns:
            df["ema_20"] = ticker_group["close"].transform(
                lambda series: series.ewm(span=20, adjust=False, min_periods=20).mean()
            )
        if "volume_sma_20" not in df.columns:
            df["volume_sma_20"] = ticker_group["volume"].transform(
                lambda series: series.rolling(20, min_periods=20).mean()
            )
        if "relative_volume_20" not in df.columns:
            denominator = pd.to_numeric(df.get("volume_sma_20"), errors="coerce")
            numerator = pd.to_numeric(df.get("volume"), errors="coerce")
            df["relative_volume_20"] = np.where(
                denominator.gt(0),
                numerator.div(denominator),
                np.nan,
            )
        if "prev_low" not in df.columns:
            df["prev_low"] = ticker_group["low"].shift(1)
        return self._store_output(df)

    def _compute_tp1_price(self, signal_row: pd.Series, entry_price: float) -> float:
        final_target = signal_row.get("signal_take_profit_price", np.nan)
        if pd.isna(final_target) or pd.isna(entry_price) or entry_price <= 0 or final_target <= entry_price:
            return np.nan
        return float(entry_price + self.config.tp1_fraction_of_target * (final_target - entry_price))

    def _active_protective_stop(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
        base_stop_price: float,
    ) -> tuple[float, dict[str, float]]:
        return float(base_stop_price), {}

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return None, {}

    def _final_target_active(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> bool:
        return True

    def _time_stop_active(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> bool:
        return True

    def _evaluate_ma_trail(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
    ) -> tuple[str | None, dict[str, float]]:
        ema_column = f"ema_{self.config.ma_trail_window}"
        eval_row = group.iloc[eval_loc]
        ema_value = eval_row.get(ema_column, np.nan)
        trail_value = np.nan
        if pd.notna(ema_value):
            trail_value = float(ema_value * (1.0 - self.config.ma_exit_buffer_pct))
        if pd.notna(trail_value) and pd.notna(eval_row.get("close", np.nan)) and eval_row["close"] < trail_value:
            return "ma_trail_exit", {"ma_trail_value": trail_value}
        return None, {"ma_trail_value": trail_value}

    def _evaluate_volume_failure(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
    ) -> tuple[str | None, dict[str, float]]:
        eval_row = group.iloc[eval_loc]
        rv = eval_row.get("relative_volume_20", np.nan)
        metadata = {"relative_volume_20_signal": float(rv) if pd.notna(rv) else np.nan}
        if (
            pd.notna(rv)
            and rv >= self.config.vol_failure_threshold
            and pd.notna(eval_row.get("close", np.nan))
            and pd.notna(eval_row.get("open", np.nan))
            and pd.notna(eval_row.get("prev_low", np.nan))
            and eval_row["close"] < eval_row["open"]
            and eval_row["close"] < eval_row["prev_low"]
        ):
            return "volume_failure_exit", metadata
        return None, metadata

    def _evaluate_close_retrace(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        state: dict[str, object],
    ) -> tuple[str | None, dict[str, float]]:
        highest_close_prior = state.get("highest_close_since_tp1")
        highest_close = (
            float(highest_close_prior)
            if highest_close_prior is not None and pd.notna(highest_close_prior)
            else np.nan
        )
        threshold = np.nan
        if pd.notna(highest_close):
            threshold = float(highest_close * (1.0 - self.config.close_retrace_pct))
        eval_row = group.iloc[eval_loc]
        metadata = {
            "highest_close_since_tp1": highest_close,
            "close_retrace_threshold": threshold,
        }
        if pd.notna(threshold) and pd.notna(eval_row.get("close", np.nan)) and eval_row["close"] <= threshold:
            return "close_retrace_exit", metadata
        return None, metadata

    def _initialize_state(self) -> dict[str, object]:
        return {
            "tp1_hit_loc": None,
            "highest_high_since_tp1": np.nan,
            "highest_close_since_tp1": np.nan,
        }

    def _tp1_active(self, eval_loc: int, state: dict[str, object]) -> bool:
        tp1_hit_loc = state.get("tp1_hit_loc")
        return tp1_hit_loc is not None and eval_loc > int(tp1_hit_loc)

    def _build_path_record(
        self,
        *,
        eval_row: pd.Series,
        active_stop: float,
        tp1_active: bool,
        tp1_price: float,
        diagnostics: dict[str, float],
        exit_reason_candidate: str | None,
    ) -> dict[str, object]:
        return {
            "date": eval_row["date"],
            "active_protective_stop": active_stop,
            "tp1_active": tp1_active,
            "tp1_price": tp1_price,
            "ma_trail_value": diagnostics.get("ma_trail_value", np.nan),
            "structure_trail_value": diagnostics.get("structure_trail_value", np.nan),
            "relative_volume_20_signal": diagnostics.get("relative_volume_20_signal", np.nan),
            "highest_close_since_tp1": diagnostics.get("highest_close_since_tp1", np.nan),
            "close_retrace_threshold": diagnostics.get("close_retrace_threshold", np.nan),
            "exit_reason_candidate": exit_reason_candidate,
        }

    def _simulate_exit_state(
        self,
        group: pd.DataFrame,
        *,
        signal_loc: int,
        signal_row: pd.Series,
        collect_path: bool = False,
    ) -> dict[str, object]:
        cfg = self.config
        state = self._initialize_state()

        entry_loc = signal_loc + 2
        if entry_loc >= len(group):
            return {
                "entry_loc": None,
                "entry_price": np.nan,
                "tp1_price": np.nan,
                "tp1_reached": False,
                "tp1_hit_loc": None,
                "tp1_hit_date": pd.NaT,
                "post_tp1_stop_price": np.nan,
                "ma_trail_value": np.nan,
                "structure_trail_value": np.nan,
                "relative_volume_20_signal": np.nan,
                "highest_close_since_tp1": np.nan,
                "close_retrace_threshold": np.nan,
                "exit_signal_loc": None,
                "exit_reason": None,
                "executed_exit_loc": None,
                "exit_path": pd.DataFrame(),
            }

        entry_row = group.iloc[entry_loc]
        entry_price = entry_row.get("open", np.nan)
        if pd.isna(entry_price) or entry_price <= 0:
            return {
                "entry_loc": entry_loc,
                "entry_price": np.nan,
                "tp1_price": np.nan,
                "tp1_reached": False,
                "tp1_hit_loc": None,
                "tp1_hit_date": pd.NaT,
                "post_tp1_stop_price": np.nan,
                "ma_trail_value": np.nan,
                "structure_trail_value": np.nan,
                "relative_volume_20_signal": np.nan,
                "highest_close_since_tp1": np.nan,
                "close_retrace_threshold": np.nan,
                "exit_signal_loc": None,
                "exit_reason": None,
                "executed_exit_loc": None,
                "exit_path": pd.DataFrame(),
            }

        final_target = signal_row.get("signal_take_profit_price", np.nan)
        base_stop = signal_row.get("signal_hard_stop_price", np.nan)
        tp1_price = self._compute_tp1_price(signal_row, float(entry_price))

        exit_signal_loc: int | None = None
        exit_reason: str | None = None
        diagnostics_snapshot: dict[str, float] = {}
        path_records: list[dict[str, object]] = []

        for eval_loc in range(entry_loc, len(group)):
            eval_row = group.iloc[eval_loc]
            eval_low = eval_row.get("low", np.nan)
            eval_high = eval_row.get("high", np.nan)
            tp1_active = self._tp1_active(eval_loc, state)
            active_stop = float(base_stop)
            diagnostics: dict[str, float] = {}

            if tp1_active:
                active_stop, stop_meta = self._active_protective_stop(
                    group=group,
                    eval_loc=eval_loc,
                    signal_row=signal_row,
                    state=state,
                    entry_loc=entry_loc,
                    entry_price=float(entry_price),
                    base_stop_price=float(base_stop),
                )
                diagnostics.update(stop_meta)

            close_trigger_reason: str | None = None
            if tp1_active:
                close_trigger_reason, close_meta = self._post_tp1_exit_trigger(
                    group=group,
                    eval_loc=eval_loc,
                    signal_row=signal_row,
                    state=state,
                    entry_loc=entry_loc,
                    entry_price=float(entry_price),
                )
                diagnostics.update(close_meta)

            current_exit_reason: str | None = None
            dynamic_stop_engaged = (
                tp1_active
                and self.HAS_DYNAMIC_PROTECTIVE_STOP
                and pd.notna(active_stop)
                and pd.notna(base_stop)
                and float(active_stop) > float(base_stop) + 1e-12
            )
            if cfg.enable_hard_stop and pd.notna(eval_low) and pd.notna(active_stop) and eval_low <= active_stop:
                current_exit_reason = (
                    self.PROTECTIVE_STOP_REASON if dynamic_stop_engaged else "hard_stop"
                )
            elif (
                cfg.enable_take_profit
                and self._final_target_active(
                    group=group,
                    eval_loc=eval_loc,
                    signal_row=signal_row,
                    state=state,
                    entry_loc=entry_loc,
                    entry_price=float(entry_price),
                )
                and pd.notna(eval_high)
                and pd.notna(final_target)
                and eval_high >= final_target
            ):
                current_exit_reason = "take_profit"
            elif close_trigger_reason is not None:
                current_exit_reason = close_trigger_reason
            elif (
                cfg.enable_time_stop
                and cfg.max_holding_days is not None
                and self._time_stop_active(
                    group=group,
                    eval_loc=eval_loc,
                    signal_row=signal_row,
                    state=state,
                    entry_loc=entry_loc,
                    entry_price=float(entry_price),
                )
                and eval_loc - entry_loc + 1 >= cfg.max_holding_days
            ):
                current_exit_reason = "time_stop"

            path_record = self._build_path_record(
                eval_row=eval_row,
                active_stop=active_stop,
                tp1_active=tp1_active,
                tp1_price=tp1_price,
                diagnostics=diagnostics,
                exit_reason_candidate=current_exit_reason,
            )
            if collect_path:
                path_records.append(path_record)

            if current_exit_reason is not None:
                if state["tp1_hit_loc"] is None and pd.notna(tp1_price) and pd.notna(eval_high) and eval_high >= tp1_price:
                    state["tp1_hit_loc"] = eval_loc
                    state["highest_high_since_tp1"] = float(eval_high)
                    if pd.notna(eval_row.get("close", np.nan)):
                        state["highest_close_since_tp1"] = float(eval_row["close"])
                exit_signal_loc = eval_loc
                exit_reason = current_exit_reason
                diagnostics_snapshot = diagnostics
                break

            if state["tp1_hit_loc"] is None and pd.notna(tp1_price) and pd.notna(eval_high) and eval_high >= tp1_price:
                state["tp1_hit_loc"] = eval_loc
                state["highest_high_since_tp1"] = float(eval_high)
                if pd.notna(eval_row.get("close", np.nan)):
                    state["highest_close_since_tp1"] = float(eval_row["close"])
            elif state["tp1_hit_loc"] is not None:
                if pd.notna(eval_high):
                    prior_high = state.get("highest_high_since_tp1")
                    state["highest_high_since_tp1"] = (
                        max(float(prior_high), float(eval_high))
                        if prior_high is not None and pd.notna(prior_high)
                        else float(eval_high)
                    )
                if pd.notna(eval_row.get("close", np.nan)):
                    prior_close = state.get("highest_close_since_tp1")
                    state["highest_close_since_tp1"] = (
                        max(float(prior_close), float(eval_row["close"]))
                        if prior_close is not None and pd.notna(prior_close)
                        else float(eval_row["close"])
                    )

        executed_exit_loc: int | None = None
        if exit_signal_loc is not None and exit_signal_loc + 1 < len(group):
            executed_exit_loc = exit_signal_loc + 1
        elif exit_signal_loc is None:
            exit_reason = "open_position"

        tp1_hit_loc = state.get("tp1_hit_loc")
        return {
            "entry_loc": entry_loc,
            "entry_price": float(entry_price),
            "tp1_price": float(tp1_price) if pd.notna(tp1_price) else np.nan,
            "tp1_reached": tp1_hit_loc is not None,
            "tp1_hit_loc": tp1_hit_loc,
            "tp1_hit_date": group.iloc[int(tp1_hit_loc)]["date"] if tp1_hit_loc is not None else pd.NaT,
            "post_tp1_stop_price": float(diagnostics_snapshot.get("post_tp1_stop_price", np.nan)),
            "ma_trail_value": float(diagnostics_snapshot.get("ma_trail_value", np.nan)),
            "structure_trail_value": float(diagnostics_snapshot.get("structure_trail_value", np.nan)),
            "relative_volume_20_signal": float(diagnostics_snapshot.get("relative_volume_20_signal", np.nan)),
            "highest_close_since_tp1": float(diagnostics_snapshot.get("highest_close_since_tp1", np.nan)),
            "close_retrace_threshold": float(diagnostics_snapshot.get("close_retrace_threshold", np.nan)),
            "exit_signal_loc": exit_signal_loc,
            "exit_reason": exit_reason,
            "executed_exit_loc": executed_exit_loc,
            "exit_path": pd.DataFrame(path_records),
        }

    def add_research_outcomes(self) -> pd.DataFrame:
        self.add_signals()
        self._ensure_dynamic_exit_features()
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
        df["dynamic_exit_variant"] = self.STRATEGY_NAME
        df["tp1_price"] = np.nan
        df["tp1_reached"] = False
        df["tp1_hit_date"] = pd.NaT
        df["post_tp1_stop_price"] = np.nan
        df["ma_trail_value"] = np.nan
        df["structure_trail_value"] = np.nan
        df["relative_volume_20_signal"] = np.nan
        df["highest_close_since_tp1"] = np.nan
        df["close_retrace_threshold"] = np.nan

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
                simulation = self._simulate_exit_state(group, signal_loc=signal_loc, signal_row=signal_row, collect_path=False)
                entry_loc = simulation["entry_loc"]
                entry_price = simulation["entry_price"]
                if entry_loc is None or pd.isna(entry_price) or entry_price <= 0:
                    next_search_loc += 1
                    continue

                exit_signal_loc = simulation["exit_signal_loc"]
                executed_exit_loc = simulation["executed_exit_loc"]
                exit_reason = simulation["exit_reason"]

                path_end_loc = exit_signal_loc if exit_signal_loc is not None else len(group) - 1
                path_slice = group.iloc[int(entry_loc) : path_end_loc + 1]
                path_high = path_slice["high"].where(path_slice["high"].gt(0), path_slice["close"])
                path_low = path_slice["low"].where(path_slice["low"].gt(0), path_slice["close"])
                if not path_slice.empty:
                    df.at[signal_idx, "max_favorable_excursion"] = float(path_high.max() / entry_price - 1.0)
                    df.at[signal_idx, "max_adverse_excursion"] = float(path_low.min() / entry_price - 1.0)

                entry_row = group.iloc[int(entry_loc)]
                df.at[signal_idx, "entry_signal_executed"] = True
                df.at[signal_idx, "entry_date_next"] = entry_row["date"]
                df.at[signal_idx, "entry_open_next"] = float(entry_price)
                df.at[signal_idx, "exit_reason"] = exit_reason
                df.at[signal_idx, "dynamic_exit_variant"] = self.STRATEGY_NAME
                for column in [
                    "tp1_price",
                    "tp1_reached",
                    "tp1_hit_date",
                    "post_tp1_stop_price",
                    "ma_trail_value",
                    "structure_trail_value",
                    "relative_volume_20_signal",
                    "highest_close_since_tp1",
                    "close_retrace_threshold",
                ]:
                    df.at[signal_idx, column] = simulation[column]

                if exit_signal_loc is not None:
                    df.at[signal_idx, "exit_signal_date"] = group.iloc[int(exit_signal_loc)]["date"]

                if executed_exit_loc is not None:
                    exit_row = group.iloc[int(executed_exit_loc)]
                    exit_open = exit_row["open"]
                    if pd.notna(exit_open) and exit_open > 0:
                        df.at[signal_idx, "exit_date_next"] = exit_row["date"]
                        df.at[signal_idx, "exit_open_next"] = float(exit_open)
                        df.at[signal_idx, "realized_open_to_open_return"] = float(exit_open / entry_price - 1.0)
                        df.at[signal_idx, "holding_days"] = int(int(executed_exit_loc) - int(entry_loc))
                    suppressed_rows = group.iloc[signal_loc + 1 : int(executed_exit_loc)]
                    next_search_loc = int(executed_exit_loc)
                else:
                    df.at[signal_idx, "holding_days"] = int(len(group) - 1 - int(entry_loc))
                    suppressed_rows = group.iloc[signal_loc + 1 :]
                    next_search_loc = len(group)

                suppressed_indices = suppressed_rows.loc[suppressed_rows["entry_signal"], "index"]
                if not suppressed_indices.empty:
                    df.loc[suppressed_indices.astype(int), "entry_signal_suppressed"] = True

        return self._store_output(df)

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
            "dynamic_exit_variant",
            "tp1_price",
            "tp1_reached",
            "tp1_hit_date",
            "active_protective_stop",
            "post_tp1_stop_price",
            "ma_trail_value",
            "structure_trail_value",
            "relative_volume_20_signal",
            "highest_close_since_tp1",
            "close_retrace_threshold",
            "exit_signal",
            "exit_signal_date",
            "planned_exit_date",
            "exit_reason",
            "action",
            "issue",
        ]
        output_columns = list(dict.fromkeys(list(positions_df.columns) + computed_columns))

        self.add_signals()
        self._ensure_dynamic_exit_features()
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        if scored.empty:
            return pd.DataFrame(columns=output_columns)

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else scored["date"].max()
        scored = scored[scored["date"].le(target_date)].copy()
        if scored.empty:
            return pd.DataFrame(columns=output_columns)

        resolved_next_trade_date = (
            pd.to_datetime(next_trade_date) if next_trade_date is not None else target_date + pd.offsets.BDay(1)
        )

        positions = positions_df.copy()
        positions["ticker"] = positions["ticker"].astype("string")
        positions["entry_date"] = pd.to_datetime(positions["entry_date"], errors="coerce")
        positions["entry_price"] = pd.to_numeric(positions["entry_price"], errors="coerce")
        if "signal_date" in positions.columns:
            positions["signal_date"] = pd.to_datetime(positions["signal_date"], errors="coerce")
        if "shares" in positions.columns:
            positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce")

        records: list[dict[str, object]] = []
        for _, position in positions.iterrows():
            record = {column: position[column] for column in positions.columns}
            ticker = str(position["ticker"])
            entry_date = pd.Timestamp(position["entry_date"]) if pd.notna(position["entry_date"]) else pd.NaT
            entry_price = float(position["entry_price"]) if pd.notna(position["entry_price"]) else np.nan
            shares = float(position["shares"]) if "shares" in positions.columns and pd.notna(position["shares"]) else np.nan

            ticker_rows = scored[scored["ticker"].astype(str).eq(ticker)].copy().reset_index(drop=True)
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
            current_close = float(latest_row["close"]) if pd.notna(latest_row["close"]) else np.nan
            pnl_pct = float(current_close / entry_price - 1.0) if entry_price > 0 and pd.notna(current_close) else np.nan
            pnl_amount = (
                float((current_close - entry_price) * shares)
                if pd.notna(shares) and entry_price > 0 and pd.notna(current_close)
                else np.nan
            )

            trade_rows = ticker_rows[ticker_rows["date"].ge(entry_date)].copy() if pd.notna(entry_date) else pd.DataFrame()
            trading_days_in_trade = int(len(trade_rows))
            holding_days = (
                int((target_date.normalize() - entry_date.normalize()).days)
                if pd.notna(entry_date)
                else pd.NA
            )
            days_until_time_stop = (
                int(self.config.max_holding_days - trading_days_in_trade)
                if self.config.enable_time_stop and self.config.max_holding_days is not None
                else pd.NA
            )

            if "signal_date" in positions.columns and pd.notna(position.get("signal_date", pd.NaT)):
                matched_signal = ticker_rows[
                    ticker_rows["date"].eq(pd.Timestamp(position["signal_date"])) & ticker_rows["entry_signal"]
                ]
            else:
                matched_signal = ticker_rows[
                    ticker_rows["entry_date_next"].eq(entry_date) & ticker_rows["entry_signal"]
                ]

            if matched_signal.empty:
                record.update(
                    {
                        "name": latest_row.get("name", pd.NA),
                        "as_of_date": target_date,
                        "latest_bar_date": latest_row["date"],
                        "current_close": current_close,
                        "pnl_pct": pnl_pct,
                        "pnl_amount": pnl_amount,
                        "holding_days": holding_days,
                        "trading_days_in_trade": trading_days_in_trade,
                        "days_until_time_stop": days_until_time_stop,
                        "action": "review",
                        "issue": "matching_signal_not_found",
                    }
                )
                records.append(record)
                continue

            signal_loc = int(matched_signal.index[0])
            signal_row = ticker_rows.iloc[signal_loc]
            simulation = self._simulate_exit_state(
                ticker_rows,
                signal_loc=signal_loc,
                signal_row=signal_row,
                collect_path=True,
            )
            exit_path = simulation.get("exit_path", pd.DataFrame())
            latest_path = exit_path.iloc[-1] if not exit_path.empty else pd.Series(dtype="object")

            exit_signal_loc = simulation.get("exit_signal_loc")
            executed_exit_loc = simulation.get("executed_exit_loc")
            exit_signal_date = (
                ticker_rows.iloc[int(exit_signal_loc)]["date"] if exit_signal_loc is not None else pd.NaT
            )
            planned_exit_date = pd.NaT
            action = "hold"
            if exit_signal_loc is not None and executed_exit_loc is None:
                action = "exit_next_open"
                planned_exit_date = resolved_next_trade_date
            elif executed_exit_loc is not None:
                action = "exit_overdue"
                planned_exit_date = ticker_rows.iloc[int(executed_exit_loc)]["date"]

            record.update(
                {
                    "name": latest_row.get("name", pd.NA),
                    "as_of_date": target_date,
                    "latest_bar_date": latest_row["date"],
                    "signal_date_resolved": signal_row["date"],
                    "flag_peak_date": signal_row.get("flag_peak_date", pd.NaT),
                    "flag_peak_high": signal_row.get("flag_peak_high", np.nan),
                    "flag_low": signal_row.get("flag_low", np.nan),
                    "current_close": current_close,
                    "pnl_pct": pnl_pct,
                    "pnl_amount": pnl_amount,
                    "holding_days": holding_days,
                    "trading_days_in_trade": trading_days_in_trade,
                    "days_until_time_stop": days_until_time_stop,
                    "hard_stop_price": signal_row.get("signal_hard_stop_price", np.nan),
                    "take_profit_price": signal_row.get("signal_take_profit_price", np.nan),
                    "reward_to_risk": signal_row.get("reward_to_risk", np.nan),
                    "dynamic_exit_variant": self.STRATEGY_NAME,
                    "tp1_price": simulation.get("tp1_price", np.nan),
                    "tp1_reached": bool(simulation.get("tp1_reached", False)),
                    "tp1_hit_date": simulation.get("tp1_hit_date", pd.NaT),
                    "active_protective_stop": latest_path.get("active_protective_stop", np.nan),
                    "post_tp1_stop_price": simulation.get("post_tp1_stop_price", np.nan),
                    "ma_trail_value": latest_path.get("ma_trail_value", np.nan),
                    "structure_trail_value": latest_path.get("structure_trail_value", np.nan),
                    "relative_volume_20_signal": latest_path.get("relative_volume_20_signal", np.nan),
                    "highest_close_since_tp1": latest_path.get("highest_close_since_tp1", np.nan),
                    "close_retrace_threshold": latest_path.get("close_retrace_threshold", np.nan),
                    "exit_signal": exit_signal_loc is not None,
                    "exit_signal_date": exit_signal_date,
                    "planned_exit_date": planned_exit_date,
                    "exit_reason": simulation.get("exit_reason", pd.NA),
                    "action": action,
                    "issue": pd.NA,
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
        inspection = super().inspect_signal(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        target_date = pd.to_datetime(signal_date)
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        source_row = scored[
            scored["ticker"].astype(str).eq(str(ticker))
            & scored["date"].eq(target_date)
        ].iloc[[0]].copy()
        signal_row = inspection["signal_row"].copy()
        dynamic_columns = [
            "dynamic_exit_variant",
            "tp1_price",
            "tp1_reached",
            "tp1_hit_date",
            "post_tp1_stop_price",
            "ma_trail_value",
            "structure_trail_value",
            "relative_volume_20_signal",
            "highest_close_since_tp1",
            "close_retrace_threshold",
        ]
        for column in dynamic_columns:
            if column in source_row.columns and column not in signal_row.columns:
                signal_row[column] = source_row[column].to_numpy()
        inspection["signal_row"] = signal_row

        if str(inspection["summary"].get("review_mode", "executed")) != "executed":
            inspection["exit_path"] = pd.DataFrame()
            summary = dict(inspection["summary"])
            summary.update(
                {
                    "dynamic_exit_variant": self.STRATEGY_NAME,
                    "tp1_price": np.nan,
                    "tp1_reached": False,
                    "tp1_hit_date": pd.NaT,
                    "post_tp1_stop_price": np.nan,
                    "ma_trail_value": np.nan,
                    "structure_trail_value": np.nan,
                    "relative_volume_20_signal": np.nan,
                    "highest_close_since_tp1": np.nan,
                    "close_retrace_threshold": np.nan,
                }
            )
            inspection["summary"] = summary
            return inspection

        ticker_frame = scored[scored["ticker"].astype(str).eq(str(ticker))].reset_index(drop=True)
        signal_loc = int(ticker_frame.index[ticker_frame["date"] == target_date][0])
        simulation = self._simulate_exit_state(
            ticker_frame,
            signal_loc=signal_loc,
            signal_row=ticker_frame.iloc[signal_loc],
            collect_path=True,
        )
        inspection["exit_path"] = simulation["exit_path"]

        summary = dict(inspection["summary"])
        if not signal_row.empty:
            summary.update(
                {
                    "dynamic_exit_variant": str(signal_row["dynamic_exit_variant"].iat[0]),
                    "tp1_price": float(signal_row["tp1_price"].iat[0]) if pd.notna(signal_row["tp1_price"].iat[0]) else np.nan,
                    "tp1_reached": bool(signal_row["tp1_reached"].iat[0]),
                    "tp1_hit_date": signal_row["tp1_hit_date"].iat[0],
                    "post_tp1_stop_price": float(signal_row["post_tp1_stop_price"].iat[0]) if pd.notna(signal_row["post_tp1_stop_price"].iat[0]) else np.nan,
                    "ma_trail_value": float(signal_row["ma_trail_value"].iat[0]) if pd.notna(signal_row["ma_trail_value"].iat[0]) else np.nan,
                    "structure_trail_value": float(signal_row["structure_trail_value"].iat[0]) if pd.notna(signal_row["structure_trail_value"].iat[0]) else np.nan,
                    "relative_volume_20_signal": float(signal_row["relative_volume_20_signal"].iat[0]) if pd.notna(signal_row["relative_volume_20_signal"].iat[0]) else np.nan,
                    "highest_close_since_tp1": float(signal_row["highest_close_since_tp1"].iat[0]) if pd.notna(signal_row["highest_close_since_tp1"].iat[0]) else np.nan,
                    "close_retrace_threshold": float(signal_row["close_retrace_threshold"].iat[0]) if pd.notna(signal_row["close_retrace_threshold"].iat[0]) else np.nan,
                }
            )
        inspection["summary"] = summary
        return inspection

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
        if str(inspection["summary"].get("review_mode", "executed")) != "executed":
            return figure
        signal_row = inspection["signal_row"]
        price_window = inspection["price_window"].copy()
        exit_path = inspection.get("exit_path", pd.DataFrame()).copy()
        if signal_row.empty or price_window.empty:
            return figure

        if not exit_path.empty:
            price_window = price_window.merge(exit_path, on="date", how="left", suffixes=("", "_exit"))
            for column in [
                "tp1_price",
                "active_protective_stop",
                "ma_trail_value",
                "structure_trail_value",
                "relative_volume_20_signal",
                "highest_close_since_tp1",
                "close_retrace_threshold",
                "exit_reason_candidate",
            ]:
                exit_column = f"{column}_exit"
                if exit_column not in price_window.columns:
                    continue
                if column in price_window.columns:
                    price_window[column] = price_window[column].where(
                        price_window[column].notna(),
                        price_window[exit_column],
                    )
                    price_window = price_window.drop(columns=[exit_column])
                else:
                    price_window = price_window.rename(columns={exit_column: column})

        for column, name, color, dash in [
            ("tp1_price", "TP1", "darkorange", "dot"),
            ("active_protective_stop", "Active Stop", "firebrick", "dashdot"),
            ("ma_trail_value", "MA Trail", "forestgreen", "dash"),
            ("structure_trail_value", "Structure Trail", "mediumvioletred", "dash"),
            ("close_retrace_threshold", "Close Retrace Threshold", "peru", "dot"),
        ]:
            if column in price_window.columns and price_window[column].notna().any():
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line={"color": color, "dash": dash},
                    ),
                    row=1,
                    col=1,
                )

        for column, name, color in [
            ("ema_10", "EMA 10", "darkgreen"),
        ]:
            if column in price_window.columns and price_window[column].notna().any():
                figure.add_trace(
                    go.Scatter(
                        x=price_window["date"],
                        y=price_window[column],
                        mode="lines",
                        name=name,
                        line={"color": color, "dash": "dot"},
                    ),
                    row=1,
                    col=1,
                )

        if "exit_reason_candidate" in price_window.columns:
            trigger_window = price_window[price_window["exit_reason_candidate"].notna()]
            if not trigger_window.empty:
                figure.add_trace(
                    go.Scatter(
                        x=trigger_window["date"],
                        y=trigger_window["close"],
                        mode="markers",
                        name="Dynamic Exit Trigger",
                        marker={"color": "crimson", "size": 9, "symbol": "x"},
                    ),
                    row=1,
                    col=1,
                )
        return figure


class BullFlagBreakevenAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_breakeven_after_tp1"
    PROTECTIVE_STOP_REASON = "breakeven_stop"
    HAS_DYNAMIC_PROTECTIVE_STOP = True

    def _active_protective_stop(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
        base_stop_price: float,
    ) -> tuple[float, dict[str, float]]:
        breakeven_stop = float(entry_price * (1.0 + self.config.breakeven_buffer_pct))
        active_stop = float(max(base_stop_price, breakeven_stop))
        return active_stop, {"post_tp1_stop_price": active_stop}


class BullFlagTrailingAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_trailing_after_tp1"
    PROTECTIVE_STOP_REASON = "trailing_stop"
    HAS_DYNAMIC_PROTECTIVE_STOP = True

    def _active_protective_stop(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
        base_stop_price: float,
    ) -> tuple[float, dict[str, float]]:
        flagpole_length = signal_row.get("flagpole_length", np.nan)
        highest_high = state.get("highest_high_since_tp1")
        if (
            pd.isna(flagpole_length)
            or flagpole_length <= 0
            or highest_high is None
            or pd.isna(highest_high)
        ):
            active_stop = float(base_stop_price)
        else:
            trail_distance = float(self.config.trailing_stop_fraction_of_flagpole * flagpole_length)
            trailing_stop = float(float(highest_high) - trail_distance)
            active_stop = float(max(base_stop_price, trailing_stop))
        return active_stop, {"post_tp1_stop_price": active_stop}


class BullFlagTrailingStopOnlyAfterTp1Researcher(BullFlagTrailingAfterTp1Researcher):
    STRATEGY_NAME = "bull_flag_trailing_stop_only_after_tp1"

    def _final_target_active(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> bool:
        return not self._tp1_active(eval_loc, state)

    def _time_stop_active(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> bool:
        return not self._tp1_active(eval_loc, state)


class BullFlagMaTrailAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_ma_trail_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return self._evaluate_ma_trail(group=group, eval_loc=eval_loc)


class BullFlagStructureTrailAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_structure_trail_after_tp1"
    PROTECTIVE_STOP_REASON = "structure_trail_stop"
    HAS_DYNAMIC_PROTECTIVE_STOP = True

    def _active_protective_stop(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
        base_stop_price: float,
    ) -> tuple[float, dict[str, float]]:
        completed_start = max(entry_loc, eval_loc - self.config.structure_trail_lookback)
        completed = group.iloc[completed_start:eval_loc]
        structure_value = np.nan
        if not completed.empty:
            completed_lows = pd.to_numeric(completed["low"], errors="coerce")
            if completed_lows.notna().any():
                structure_value = float(completed_lows.min() * (1.0 - self.config.structure_trail_buffer_pct))
        active_stop = float(max(base_stop_price, structure_value)) if pd.notna(structure_value) else float(base_stop_price)
        return active_stop, {
            "post_tp1_stop_price": active_stop,
            "structure_trail_value": float(structure_value) if pd.notna(structure_value) else np.nan,
        }


class BullFlagVolumeFailureAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_volume_failure_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return self._evaluate_volume_failure(group=group, eval_loc=eval_loc)


class BullFlagCloseRetraceAfterTp1Researcher(BullFlagDynamicExitResearcherBase):
    STRATEGY_NAME = "bull_flag_close_retrace_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return self._evaluate_close_retrace(group=group, eval_loc=eval_loc, state=state)


class BullFlagTrailingVolumeFailureResearcher(BullFlagTrailingAfterTp1Researcher):
    STRATEGY_NAME = "bull_flag_trailing_volume_failure_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return self._evaluate_volume_failure(group=group, eval_loc=eval_loc)


class BullFlagTrailingCloseRetraceResearcher(BullFlagTrailingAfterTp1Researcher):
    STRATEGY_NAME = "bull_flag_trailing_close_retrace_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        return self._evaluate_close_retrace(group=group, eval_loc=eval_loc, state=state)


class BullFlagMaTrailVolumeFailureResearcher(BullFlagMaTrailAfterTp1Researcher):
    STRATEGY_NAME = "bull_flag_ma_trail_volume_failure_after_tp1"

    def _post_tp1_exit_trigger(
        self,
        *,
        group: pd.DataFrame,
        eval_loc: int,
        signal_row: pd.Series,
        state: dict[str, object],
        entry_loc: int,
        entry_price: float,
    ) -> tuple[str | None, dict[str, float]]:
        volume_reason, volume_meta = self._evaluate_volume_failure(group=group, eval_loc=eval_loc)
        if volume_reason is not None:
            return volume_reason, volume_meta
        ma_reason, ma_meta = self._evaluate_ma_trail(group=group, eval_loc=eval_loc)
        merged = dict(volume_meta)
        merged.update(ma_meta)
        return ma_reason, merged


__all__ = [
    "BullFlagDynamicExitConfig",
    "BullFlagBreakevenAfterTp1Researcher",
    "BullFlagTrailingAfterTp1Researcher",
    "BullFlagMaTrailAfterTp1Researcher",
    "BullFlagStructureTrailAfterTp1Researcher",
    "BullFlagVolumeFailureAfterTp1Researcher",
    "BullFlagCloseRetraceAfterTp1Researcher",
    "BullFlagTrailingVolumeFailureResearcher",
    "BullFlagTrailingCloseRetraceResearcher",
    "BullFlagMaTrailVolumeFailureResearcher",
]
