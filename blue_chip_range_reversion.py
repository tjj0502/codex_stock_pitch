from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "ts_code",
    "name",
    "weight",
    "constituent_trade_date",
    "open",
    "close",
    "high",
    "low",
    "pre_close",
    "volume",
    "turnover",
    "amplitude_pct",
    "change_pct",
    "change_amount",
]
NUMERIC_COLUMNS = [
    "weight",
    "open",
    "close",
    "high",
    "low",
    "pre_close",
    "volume",
    "turnover",
    "amplitude_pct",
    "change_pct",
    "change_amount",
]
STRING_COLUMNS = ["ticker", "ts_code", "name"]


@dataclass(frozen=True)
class RangeStrategyConfig:
    """
    Explicit parameter bundle for the blue-chip range reversion research flow.

    The class exists to keep all rule choices in one place instead of spreading
    thresholds across the implementation.
    """

    universe: str = "csi500"
    # Calculate the range
    range_window: int = 120
    upper_quantile: float = 0.9
    lower_quantile: float = 0.1
    min_amplitude: float = 0.20
    max_amplitude: float = 0.45
    max_abs_return_60: float = 0.15
    ma_dispersion_window: tuple[int, int, int] = (20, 60, 120)
    max_ma_dispersion: float = 0.08
    touch_zone_pct: float = 0.20
    min_lower_touches: int = 2
    min_upper_touches: int = 2
    entry_zone_threshold: float = 0.20

    # Make sure stop_loss_pct * take_profit_r_multiple > min_amplitude
    stop_loss_pct: float = 0.10
    breakdown_buffer: float = 0.03
    breakdown_confirm_days: int = 2
    take_profit_r_multiple: float = 2.0
    max_holding_days: int = 20

    def __post_init__(self) -> None:
        if self.universe not in {"csi500", "hs300"}:
            raise ValueError("universe must be either 'csi500' or 'hs300'.")
        if self.range_window < 2:
            raise ValueError("range_window must be at least 2.")
        if not 0 < self.lower_quantile < self.upper_quantile < 1:
            raise ValueError("lower_quantile and upper_quantile must satisfy 0 < lower < upper < 1.")
        if self.min_amplitude < 0 or self.max_amplitude <= self.min_amplitude:
            raise ValueError("min_amplitude must be non-negative and smaller than max_amplitude.")
        if len(self.ma_dispersion_window) != 3 or any(window < 2 for window in self.ma_dispersion_window):
            raise ValueError("ma_dispersion_window must contain exactly three integers >= 2.")
        if self.max_ma_dispersion <= 0:
            raise ValueError("max_ma_dispersion must be positive.")
        if not 0 < self.touch_zone_pct <= 0.5:
            raise ValueError("touch_zone_pct must be in (0, 0.5].")
        if self.min_lower_touches < 1 or self.min_upper_touches < 1:
            raise ValueError("touch counts must be at least 1.")
        if not 0 < self.entry_zone_threshold <= 0.5:
            raise ValueError("entry_zone_threshold must be in (0, 0.5].")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive.")
        if self.breakdown_buffer < 0:
            raise ValueError("breakdown_buffer must be non-negative.")
        if self.breakdown_confirm_days < 1:
            raise ValueError("breakdown_confirm_days must be at least 1.")
        if self.take_profit_r_multiple <= 0:
            raise ValueError("take_profit_r_multiple must be positive.")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days must be at least 1.")
        if self.stop_loss_pct * self.take_profit_r_multiple < self.min_amplitude:
            raise ValueError("too restrict constraint on profit")


class BlueChipRangeReversionResearcher:
    """
    Research helper for a long-only blue-chip range reversion strategy.

    The class follows the same style as the existing scorer classes:
    - keep a working dataframe on ``stock_candle_df``
    - enrich it in stages
    - expose inspection helpers for signal review

    The workflow is:
    1. ``add_features()`` builds range/trend/rebound context from daily candles
    2. ``add_signals()`` converts those features into raw entry signals on date ``t``
    3. ``add_research_outcomes()`` simulates one-position-per-stock event paths
       where entries happen at ``t+1`` open and exits also happen at a later open
    """

    REQUIRED_COLUMNS = REQUIRED_COLUMNS
    NUMERIC_COLUMNS = NUMERIC_COLUMNS
    STRING_COLUMNS = STRING_COLUMNS
    FEATURE_COLUMNS = [
        "ret_60d",
        "sma_5",
        "sma_20",
        "sma_60",
        "sma_120",
        "ma_dispersion",
        "range_upper",
        "range_lower",
        "range_mid",
        "range_width",
        "range_amplitude",
        "zone_position",
        "lower_touch_count",
        "upper_touch_count",
        "expected_upside_to_upper",
        "close_gt_open",
        "close_gt_prev_close",
        "close_gt_sma_5",
        "rebound_confirm_count",
        "range_candidate",
    ]
    SIGNAL_COLUMNS = [
        "entry_zone_ok",
        "expected_upside_ok",
        "rebound_confirmed",
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
        "range_upper",
        "range_lower",
        "range_mid",
        "range_amplitude",
        "zone_position",
        "expected_upside_to_upper",
        "ret_60d",
        "ma_dispersion",
        "lower_touch_count",
        "upper_touch_count",
        "rebound_confirm_count",
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
        config: RangeStrategyConfig | None = None,
        *,
        copy: bool = True,
    ) -> None:
        if not isinstance(stock_candle_df, pd.DataFrame):
            raise TypeError("stock_candle_df must be a pandas DataFrame.")
        self.config = config or RangeStrategyConfig()
        prepared = stock_candle_df.copy(deep=True) if copy else stock_candle_df
        self.stock_candle_df = self._prepare_input_frame(prepared)
        self.stock_candle_df.attrs.update(dict(stock_candle_df.attrs))
        self.stock_candle_df.attrs["strategy_name"] = "blue_chip_range_reversion"
        self.stock_candle_df.attrs["strategy_universe"] = self.config.universe
        self.stock_candle_df.attrs["constituent_history_mode"] = self.stock_candle_df.attrs.get(
            "constituent_history_mode", "latest_snapshot"
        )
        self.trade_df = pd.DataFrame(columns=self.TRADE_COLUMNS)
        # Precompute the full research state once during initialization so
        # downstream inspection/plotting calls do not repeatedly rebuild it.
        self.add_research_outcomes()

    @classmethod
    def _prepare_input_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the shared candle schema and coerce all core dtypes."""
        missing = [column for column in cls.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"stock_candle_df is missing required columns: {missing}")

        prepared = df.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])
        prepared["constituent_trade_date"] = pd.to_datetime(prepared["constituent_trade_date"])

        for column in cls.STRING_COLUMNS:
            prepared[column] = prepared[column].astype("string")
        for column in cls.NUMERIC_COLUMNS:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

        prepared = prepared.drop_duplicates(subset=["ticker", "date"], keep="last")
        return cls._sort_for_output(prepared)

    @staticmethod
    def _sort_for_calculation(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["ticker", "date"], kind="mergesort", ignore_index=True)

    @staticmethod
    def _sort_for_output(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)

    def _store_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Persist the latest enriched dataframe while preserving attrs metadata."""
        updated = self._sort_for_output(df)
        updated.attrs.update(dict(self.stock_candle_df.attrs))
        self.stock_candle_df = updated
        return self.stock_candle_df

    def _store_trade_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Persist the latest trade-level dataframe and attach sanity-check metadata."""
        if df.empty:
            updated = pd.DataFrame(columns=self.TRADE_COLUMNS)
        else:
            updated = df.sort_values(
                ["entry_date", "ticker", "signal_date"],
                kind="mergesort",
                ignore_index=True,
            )
            updated = updated.loc[:, [column for column in self.TRADE_COLUMNS if column in updated.columns]]

        updated.attrs.update(dict(self.stock_candle_df.attrs))
        total_trade_count = int(len(updated))
        open_trade_mask = updated["trade_status"].eq("open") if "trade_status" in updated.columns else pd.Series(dtype=bool)
        open_trade_count = int(open_trade_mask.sum()) if total_trade_count else 0
        updated.attrs["total_trade_count"] = total_trade_count
        updated.attrs["closed_trade_count"] = total_trade_count - open_trade_count
        updated.attrs["open_trade_count"] = open_trade_count
        updated.attrs["all_trades_closed"] = open_trade_count == 0
        updated.attrs["open_trade_tickers"] = (
            updated.loc[open_trade_mask, "ticker"].astype(str).tolist() if total_trade_count else []
        )
        updated.attrs["sanity_check_message"] = (
            "All executed buys have matching exits."
            if open_trade_count == 0
            else f"{open_trade_count} executed buys are still open at the end of the sample."
        )
        self.trade_df = updated
        return self.trade_df

    def _has_columns(self, columns: list[str]) -> bool:
        """Check whether the working dataframe already contains all requested columns."""
        return all(column in self.stock_candle_df.columns for column in columns)

    def _ensure_research_outcomes(self) -> None:
        """Materialize research outcome columns only when they are missing."""
        required_columns = self.SIGNAL_COLUMNS + self.OUTCOME_COLUMNS + ["range_candidate"]
        if not self._has_columns(required_columns):
            self.add_research_outcomes()

    @staticmethod
    def _rolling_return(series: pd.Series, window: int) -> pd.Series:
        shifted = series.shift(window)
        result = series.div(shifted) - 1.0
        return result.where(series.gt(0) & shifted.gt(0))

    @staticmethod
    def _rolling_sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=window).mean()

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator.div(denominator.where(denominator.ne(0)))

    def add_features(self) -> pd.DataFrame:
        """
        Derive all raw research features from daily candles.

        The feature set is intentionally transparent:
        - trend sanity checks: 60-day return and moving-average dispersion
        - range description: rolling upper/lower quantiles plus zone position
        - range quality: repeated lower/upper touches inside the lookback window
        - entry confirmation: simple price-action rebound checks
        """
        cfg = self.config
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        # Trend and smoothness features help reject names that are trending
        # too strongly instead of oscillating around a stable range.
        df["ret_60d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 60))
        df["sma_5"] = ticker_group["close"].transform(lambda series: self._rolling_sma(series, 5))
        df["sma_20"] = ticker_group["close"].transform(lambda series: self._rolling_sma(series, 20))
        df["sma_60"] = ticker_group["close"].transform(lambda series: self._rolling_sma(series, 60))
        df["sma_120"] = ticker_group["close"].transform(lambda series: self._rolling_sma(series, 120))

        dispersion_columns = [f"sma_{window}" for window in cfg.ma_dispersion_window]
        ma_max = df[dispersion_columns].max(axis=1)
        ma_min = df[dispersion_columns].min(axis=1)
        df["ma_dispersion"] = self._safe_ratio(ma_max - ma_min, df["close"]).replace([np.inf, -np.inf], np.nan)

        # The trading range is defined by rolling quantiles instead of raw
        # extrema so a single spike does not dominate the band definition.
        df["range_upper"] = ticker_group["high"].transform(
            lambda series: series.rolling(cfg.range_window, min_periods=cfg.range_window).quantile(cfg.upper_quantile)
        )
        df["range_lower"] = ticker_group["low"].transform(
            lambda series: series.rolling(cfg.range_window, min_periods=cfg.range_window).quantile(cfg.lower_quantile)
        )
        df["range_mid"] = (df["range_upper"] + df["range_lower"]) / 2.0
        df["range_width"] = df["range_upper"] - df["range_lower"]
        df["range_amplitude"] = self._safe_ratio(df["range_width"], df["close"]).replace([np.inf, -np.inf], np.nan)

        valid_width = df["range_width"].gt(0)
        # ``zone_position`` is the normalized location inside the current
        # range: 0 means near the lower band, 1 means near the upper band.
        df["zone_position"] = np.where(
            valid_width,
            (df["close"] - df["range_lower"]).div(df["range_width"]),
            0.5,
        )

        # Touch counts measure whether price has repeatedly visited both ends
        # of the range. Large amplitude alone is not enough; we want evidence
        # of actual back-and-forth movement.
        lower_touch = valid_width & df["zone_position"].le(cfg.touch_zone_pct)
        upper_touch = valid_width & df["zone_position"].ge(1.0 - cfg.touch_zone_pct)
        df["lower_touch_count"] = lower_touch.groupby(df["ticker"], sort=False).transform(
            lambda series: series.astype(int).rolling(cfg.range_window, min_periods=cfg.range_window).sum()
        )
        df["upper_touch_count"] = upper_touch.groupby(df["ticker"], sort=False).transform(
            lambda series: series.astype(int).rolling(cfg.range_window, min_periods=cfg.range_window).sum()
        )

        df["expected_upside_to_upper"] = self._safe_ratio(df["range_upper"] - df["close"], df["close"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df["close_gt_open"] = df["close"].gt(df["open"])
        df["close_gt_prev_close"] = df.groupby("ticker", sort=False)["close"].diff().gt(0)
        df["close_gt_sma_5"] = df["close"].gt(df["sma_5"])
        df["rebound_confirm_count"] = (
            df[["close_gt_open", "close_gt_prev_close", "close_gt_sma_5"]].fillna(False).sum(axis=1).astype(int)
        )

        # ``range_candidate`` is the coarse universe filter. A later signal
        # still needs to be near the lower band and show rebound confirmation.
        ma_complete = df[dispersion_columns].notna().all(axis=1)
        df["range_candidate"] = (
            df["ret_60d"].abs().le(cfg.max_abs_return_60)
            & df["range_amplitude"].between(cfg.min_amplitude, cfg.max_amplitude, inclusive="both")
            & df["ma_dispersion"].le(cfg.max_ma_dispersion)
            & df["lower_touch_count"].ge(cfg.min_lower_touches)
            & df["upper_touch_count"].ge(cfg.min_upper_touches)
            & valid_width
            & ma_complete
        )

        return self._store_output(df)

    def add_signals(self) -> pd.DataFrame:
        """
        Convert features into raw entry signals.

        Signals are formed after the close of day ``t`` and assume the trade
        would be entered on the next available open ``t+1``.
        """
        cfg = self.config
        self.add_features()
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        # These next-session fields make the timing explicit and are also
        # reused later by the event-study step.
        df["entry_date_next"] = ticker_group["date"].shift(-1)
        df["entry_open_next"] = ticker_group["open"].shift(-1)
        df["entry_zone_ok"] = df["zone_position"].le(cfg.entry_zone_threshold)
        df["expected_upside_ok"] = df["expected_upside_to_upper"].ge(cfg.stop_loss_pct * cfg.take_profit_r_multiple)
        df["rebound_confirmed"] = df["rebound_confirm_count"].ge(2)
        # Take-profit is capped by both the structural upper band and the 2R
        # target implied by the configured stop distance.
        df["signal_take_profit_price"] = np.minimum(
            df["range_upper"],
            df["entry_open_next"] * (1.0 + cfg.stop_loss_pct * cfg.take_profit_r_multiple),
        )
        df["signal_hard_stop_price"] = df["entry_open_next"] * (1.0 - cfg.stop_loss_pct)
        # ``entry_signal`` is still a raw signal; position overlap is handled
        # later by the single-stock event engine.
        df["entry_signal"] = (
            df["range_candidate"]
            & df["entry_zone_ok"]
            & df["expected_upside_ok"]
            & df["rebound_confirmed"]
            & df["entry_open_next"].gt(0)
            & df["entry_date_next"].notna()
        )

        return self._store_output(df)

    def add_research_outcomes(self) -> pd.DataFrame:
        """
        Replay each stock as an independent event-driven trade study.

        Rules implemented here:
        - a signal on ``t`` enters at ``t+1`` open
        - exit conditions are evaluated from each subsequent close
        - once an exit condition is hit, the position exits at the next open
        - while a position is open, later same-direction signals are recorded
          as suppressed instead of opening overlapping trades
        """
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
                # Step 1: find the next raw signal that is not already inside
                # an existing open position window.
                signal_row = group.iloc[next_search_loc]
                if not bool(signal_row["entry_signal"]):
                    next_search_loc += 1
                    continue

                signal_loc = next_search_loc
                signal_idx = int(signal_row["index"])
                entry_loc = signal_loc + 1
                if entry_loc >= len(group):
                    break

                entry_row = group.iloc[entry_loc]
                entry_price = entry_row["open"]
                if pd.isna(entry_price) or entry_price <= 0:
                    next_search_loc += 1
                    continue

                # Step 2: after entering on the next open, walk forward close
                # by close until one of the exit rules fires.
                take_profit_price = signal_row["signal_take_profit_price"]
                stop_price = signal_row["signal_hard_stop_price"]
                breakdown_streak = 0
                exit_signal_loc: int | None = None
                exit_reason: str | None = None

                for eval_loc in range(entry_loc, len(group)):
                    eval_row = group.iloc[eval_loc]
                    close_price = eval_row["close"]
                    lower_band = eval_row["range_lower"]

                    if pd.notna(close_price) and pd.notna(stop_price) and close_price <= stop_price:
                        exit_signal_loc = eval_loc
                        exit_reason = "hard_stop"
                        break

                    if (
                        pd.notna(close_price)
                        and pd.notna(lower_band)
                        and close_price <= lower_band * (1.0 - cfg.breakdown_buffer)
                    ):
                        breakdown_streak += 1
                    else:
                        breakdown_streak = 0

                    if breakdown_streak >= cfg.breakdown_confirm_days:
                        exit_signal_loc = eval_loc
                        exit_reason = "breakdown_stop"
                        break

                    if pd.notna(close_price) and pd.notna(take_profit_price) and close_price >= take_profit_price:
                        exit_signal_loc = eval_loc
                        exit_reason = "take_profit"
                        break

                    if eval_loc - entry_loc + 1 >= cfg.max_holding_days:
                        exit_signal_loc = eval_loc
                        exit_reason = "time_stop"
                        break

                executed_exit_loc: int | None = None
                if exit_signal_loc is not None and exit_signal_loc + 1 < len(group):
                    executed_exit_loc = exit_signal_loc + 1
                else:
                    # If the sample ends before an exit can be executed, keep
                    # the trade marked as still open for later inspection.
                    exit_reason = "open_position"

                # Step 3: summarize the whole path from entry to exit signal so
                # each signal row carries its own outcome statistics.
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

                # Step 4: any new raw signals that appeared while the position
                # was alive are kept for audit, but marked as suppressed.
                suppressed_indices = suppressed_rows.loc[suppressed_rows["entry_signal"], "index"]
                if not suppressed_indices.empty:
                    df.loc[suppressed_indices.astype(int), "entry_signal_suppressed"] = True

        return self._store_output(df)

    def add_trade_df(self) -> pd.DataFrame:
        """
        Build a trade-level dataframe from executed research signals.

        Each row represents one executed entry signal. Because this researcher
        does not size positions yet, ``pnl`` is reported on a one-share basis:
        ``exit_open - entry_open``. Later portfolio/backtest layers can scale
        that by actual shares.

        The returned dataframe is also stored on ``self.trade_df`` and carries
        sanity-check metadata in ``trade_df.attrs`` so we can quickly tell
        whether every executed buy was eventually closed.
        """
        # Trade extraction should reflect the latest working dataframe even if
        # the caller manually adjusted signal-related columns beforehand.
        self.add_research_outcomes()
        signal_frame = self._sort_for_calculation(self.stock_candle_df.copy())
        trades = signal_frame[signal_frame["entry_signal_executed"]].copy()

        if trades.empty:
            return self._store_trade_df(trades)

        trades["signal_date"] = trades["date"]
        trades["signal_open"] = trades["open"]
        trades["signal_high"] = trades["high"]
        trades["signal_low"] = trades["low"]
        trades["signal_close"] = trades["close"]
        trades["entry_date"] = trades["entry_date_next"]
        trades["entry_open"] = trades["entry_open_next"]
        trades["exit_date"] = trades["exit_date_next"]
        trades["exit_open"] = trades["exit_open_next"]

        closed_mask = (
            trades["exit_reason"].notna()
            & trades["exit_reason"].ne("open_position")
            & trades["exit_date"].notna()
            & trades["exit_open"].gt(0)
        )
        trades["trade_status"] = np.where(closed_mask, "closed", "open")
        trades["pnl_pct"] = trades["realized_open_to_open_return"]
        trades["pnl"] = np.where(
            closed_mask & trades["entry_open"].gt(0),
            trades["exit_open"] - trades["entry_open"],
            np.nan,
        )

        return self._store_trade_df(trades)

    def get_candidates(self, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Return all raw candidates on a signal date, sorted by attractiveness."""
        if "entry_signal" not in self.stock_candle_df.columns:
            self.add_signals()

        df = self.stock_candle_df.copy()
        candidates = df[df["entry_signal"]].copy()
        if candidates.empty:
            return candidates

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else candidates["date"].max()
        candidates = candidates[candidates["date"] == target_date].copy()
        candidates = candidates.sort_values(
            ["expected_upside_to_upper", "zone_position", "ticker"],
            ascending=[False, True, True],
            kind="mergesort",
            ignore_index=True,
        )
        columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "range_upper",
            "range_lower",
            "range_mid",
            "range_amplitude",
            "zone_position",
            "expected_upside_to_upper",
            "rebound_confirm_count",
            "entry_signal",
        ]
        optional_columns = [column for column in self.OUTCOME_COLUMNS if column in candidates.columns]
        selected_columns = [column for column in columns + optional_columns if column in candidates.columns]
        return candidates.loc[:, selected_columns].reset_index(drop=True)

    def inspect_signal(
        self,
        ticker: str,
        signal_date: str | pd.Timestamp,
        *,
        lookback: int = 60,
        lookahead: int = 10,
    ) -> dict[str, pd.DataFrame | dict[str, object]]:
        """
        Build a structured audit package for one executed signal.

        The returned payload is meant for debugging and strategy review:
        - summary: key timing and outcome fields
        - signal_row: the exact feature/signal row on the chosen date
        - condition_checklist: a human-readable checklist of rule matches
        - price_window: nearby candles plus entry/exit markers
        """
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

        start_loc = max(0, signal_loc - lookback)
        end_loc = min(len(ticker_frame), signal_loc + lookahead + 1)
        price_window = ticker_frame.iloc[start_loc:end_loc].copy().reset_index(drop=True)
        price_window["signal_marker"] = price_window["date"].eq(target_date)
        price_window["entry_marker"] = price_window["date"].eq(signal_row["entry_date_next"].iat[0])
        price_window["exit_marker"] = price_window["date"].eq(signal_row["exit_date_next"].iat[0])

        checklist = pd.DataFrame(
            {
                "condition": [
                    "range_candidate",
                    "entry_zone_ok",
                    "expected_upside_ok",
                    "rebound_confirmed",
                    "close_gt_open",
                    "close_gt_prev_close",
                    "close_gt_sma_5",
                    "entry_signal",
                    "entry_signal_executed",
                ],
                "value": [
                    bool(signal_row["range_candidate"].iat[0]),
                    bool(signal_row["entry_zone_ok"].iat[0]),
                    bool(signal_row["expected_upside_ok"].iat[0]),
                    bool(signal_row["rebound_confirmed"].iat[0]),
                    bool(signal_row["close_gt_open"].iat[0]),
                    bool(signal_row["close_gt_prev_close"].iat[0]),
                    bool(signal_row["close_gt_sma_5"].iat[0]),
                    bool(signal_row["entry_signal"].iat[0]),
                    bool(signal_row["entry_signal_executed"].iat[0]),
                ],
            }
        )

        summary = {
            "ticker": ticker,
            "signal_date": target_date,
            "raw_signal": bool(signal_row["entry_signal"].iat[0]),
            "executed_signal": bool(signal_row["entry_signal_executed"].iat[0]),
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
            "range_upper",
            "range_lower",
            "range_mid",
            "range_amplitude",
            "zone_position",
            "ret_60d",
            "ma_dispersion",
            "lower_touch_count",
            "upper_touch_count",
            "expected_upside_to_upper",
            "rebound_confirm_count",
            "entry_date_next",
            "entry_open_next",
            "signal_take_profit_price",
            "signal_hard_stop_price",
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
        """Plot the candle context, range bands, and signal checklist for one trade idea."""
        inspection = self.inspect_signal(ticker, signal_date, lookback=lookback, lookahead=lookahead)
        signal_row = inspection["signal_row"]
        price_window = inspection["price_window"]
        checklist = inspection["condition_checklist"]
        if signal_row.empty or price_window.empty:
            return go.Figure()

        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=("Price Context", "Signal Conditions"),
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
        for column, name, color in [
            ("range_upper", "Range Upper", "firebrick"),
            ("range_mid", "Range Mid", "darkorange"),
            ("range_lower", "Range Lower", "seagreen"),
        ]:
            if column in price_window.columns:
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

        entry_date = signal_row["entry_date_next"].iat[0]
        exit_date = signal_row["exit_date_next"].iat[0]
        exit_signal_date = signal_row["exit_signal_date"].iat[0] if "exit_signal_date" in signal_row.columns else pd.NaT
        exit_reason = (
            None if pd.isna(signal_row["exit_reason"].iat[0]) else str(signal_row["exit_reason"].iat[0])
        )
        figure.add_vline(x=signal_row["date"].iat[0], line_dash="dash", line_color="royalblue", row=1, col=1)
        if pd.notna(entry_date):
            entry_rows = price_window[price_window["date"] == entry_date]
            entry_price = signal_row["entry_open_next"].iat[0]
            if not entry_rows.empty and pd.notna(entry_price):
                figure.add_trace(
                    go.Scatter(
                        x=[entry_date],
                        y=[entry_price],
                        mode="markers+text",
                        marker=dict(size=14, symbol="triangle-up", color="green"),
                        text=["Entry"],
                        textposition="bottom center",
                        name="Entry",
                        hovertemplate="Entry<br>Date=%{x}<br>Price=%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
        if pd.notna(exit_signal_date):
            exit_signal_rows = price_window[price_window["date"] == exit_signal_date]
            if not exit_signal_rows.empty:
                figure.add_vline(
                    x=exit_signal_date,
                    line_dash="dot",
                    line_color="indianred",
                    row=1,
                    col=1,
                )
        if pd.notna(exit_date):
            exit_price = signal_row["exit_open_next"].iat[0] if "exit_open_next" in signal_row.columns else np.nan
            exit_rows = price_window[price_window["date"] == exit_date]
            if not exit_rows.empty and pd.notna(exit_price):
                figure.add_trace(
                    go.Scatter(
                        x=[exit_date],
                        y=[exit_price],
                        mode="markers+text",
                        marker=dict(size=16, symbol="x", color="red", line=dict(width=2, color="darkred")),
                        text=[f"Exit ({exit_reason})" if exit_reason else "Exit"],
                        textposition="top center",
                        name=f"Exit ({exit_reason})" if exit_reason else "Exit",
                        hovertemplate=(
                            "Exit<br>Date=%{x}<br>Price=%{y:.2f}<br>Reason="
                            + (exit_reason or "unknown")
                            + "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                figure.add_annotation(
                    x=exit_date,
                    y=exit_price,
                    text=f"Exit<br>{exit_reason}" if exit_reason else "Exit",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="red",
                    ax=30,
                    ay=-40,
                    row=1,
                    col=1,
                )
        elif exit_reason:
            latest_row = price_window.iloc[-1]
            figure.add_annotation(
                x=latest_row["date"],
                y=latest_row["close"],
                text=f"Status<br>{exit_reason}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="dimgray",
                ax=30,
                ay=-40,
                row=1,
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
            row=2,
            col=1,
        )
        summary = inspection["summary"]
        return_text = summary["realized_open_to_open_return"]
        title = (
            f"{ticker} | signal {pd.Timestamp(summary['signal_date']).date()} | "
            f"exit {summary['exit_reason']} | "
            f"return {return_text:.2%}"
            if pd.notna(return_text)
            else f"{ticker} | signal {pd.Timestamp(summary['signal_date']).date()}"
        )
        figure.update_layout(
            height=850,
            width=1200,
            template="plotly_white",
            hovermode="x unified",
            title=title,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Met", row=2, col=1, range=[0, 1.2])
        return figure
