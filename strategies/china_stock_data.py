from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tushare as ts


DateLike = Union[str, date, datetime]
PRICE_COLUMNS = [
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
INDEX_UNIVERSE_CONFIG = {
    "000905.SH": {"universe": "csi500", "label": "CSI500"},
    "000300.SH": {"universe": "hs300", "label": "HS300"},
}


def _normalize_date(value: DateLike) -> str:
    """Convert flexible date inputs into the ``YYYYMMDD`` format Tushare expects."""
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return pd.Timestamp(value).strftime("%Y%m%d")


def _get_tushare_client(token: str | None = None):
    """Build an authenticated Tushare pro client from an explicit token or env var."""
    resolved_token = token or os.getenv("TUSHARE_TOKEN")
    if not resolved_token:
        raise ValueError(
            "Tushare token is required. Pass token=... or set TUSHARE_TOKEN."
        )
    ts.set_token(resolved_token)
    return ts.pro_api()


def _index_metadata(index_code: str) -> dict[str, str]:
    """Map a Tushare index code to the repo's internal universe labels."""
    metadata = INDEX_UNIVERSE_CONFIG.get(index_code, {})
    return {
        "index_code": index_code,
        "universe": metadata.get("universe", index_code.lower()),
        "index_label": metadata.get("label", index_code),
    }


def _empty_price_frame() -> pd.DataFrame:
    """Return an empty frame that still matches the project-wide candle schema."""
    return pd.DataFrame(columns=PRICE_COLUMNS)


def get_index_constituents(
    index_code: str,
    token: str | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
) -> pd.DataFrame:
    """
    Fetch the latest available constituent snapshot for an index.

    The function intentionally keeps only the most recent ``trade_date`` returned
    by ``index_weight`` and applies that snapshot to the whole research window.
    This keeps the interface simple for research, but it also means historical
    studies inherit survivorship bias.
    """
    client = _get_tushare_client(token=token)
    resolved_end = _normalize_date(end_date or pd.Timestamp.today().date())
    resolved_start = _normalize_date(
        start_date or (pd.Timestamp(resolved_end) - pd.Timedelta(days=31))
    )
    metadata = _index_metadata(index_code)

    try:
        weights = client.index_weight(
            index_code=index_code,
            start_date=resolved_start,
            end_date=resolved_end,
        )
    except Exception as exc:
        raise PermissionError(
            f"Tushare index_weight access is required for {metadata['index_label']} constituents."
        ) from exc

    if weights.empty:
        empty = pd.DataFrame(columns=["ticker", "ts_code", "name", "trade_date", "weight"])
        empty.attrs.update(
            {
                "index_code": index_code,
                "universe": metadata["universe"],
                "index_label": metadata["index_label"],
                "constituent_history_mode": "latest_snapshot",
            }
        )
        return empty

    weights = weights.sort_values(["trade_date", "con_code"], ascending=[False, True])
    latest_trade_date = weights["trade_date"].iloc[0]
    weights = weights[weights["trade_date"] == latest_trade_date].copy()
    weights["ts_code"] = weights["con_code"]
    weights["ticker"] = weights["ts_code"].str.split(".").str[0]

    stock_basic = client.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name",
    )
    weights = weights.merge(stock_basic, on="ts_code", how="left")
    result = weights[["ticker", "ts_code", "name", "trade_date", "weight"]].drop_duplicates(
        subset=["ts_code"], keep="first"
    )
    result.attrs.update(
        {
            "index_code": index_code,
            "universe": metadata["universe"],
            "index_label": metadata["index_label"],
            "constituent_history_mode": "latest_snapshot",
            "constituent_trade_date": latest_trade_date,
        }
    )
    return result


def get_csi500_constituents(
    token: str | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for the CSI 500 latest constituent snapshot."""
    return get_index_constituents(
        "000905.SH",
        token=token,
        start_date=start_date,
        end_date=end_date,
    )


def get_hs300_constituents(
    token: str | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for the HS300 latest constituent snapshot."""
    return get_index_constituents(
        "000300.SH",
        token=token,
        start_date=start_date,
        end_date=end_date,
    )


def get_trade_calendar(
    token: str | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    *,
    exchange: str = "SSE",
    is_open: int | None = 1,
) -> pd.DataFrame:
    """
    Fetch the Chinese trading calendar from Tushare.

    Parameters
    ----------
    token:
        Optional Tushare token. Falls back to ``TUSHARE_TOKEN``.
    start_date, end_date:
        Date range to query. Defaults to the last 31 calendar days ending
        today if omitted.
    exchange:
        Exchange code passed to Tushare ``trade_cal``. ``"SSE"`` is a sensible
        default for mainland A-share workflows.
    is_open:
        When set to ``1`` or ``0``, keep only open or closed dates
        respectively. Pass ``None`` to keep all calendar rows.
    """
    client = _get_tushare_client(token=token)
    resolved_end = _normalize_date(end_date or pd.Timestamp.today().date())
    resolved_start = _normalize_date(
        start_date or (pd.Timestamp(resolved_end) - pd.Timedelta(days=31))
    )
    calendar = client.trade_cal(
        exchange=exchange,
        start_date=resolved_start,
        end_date=resolved_end,
    )
    if calendar.empty:
        return calendar

    calendar = calendar.copy()
    calendar["cal_date"] = pd.to_datetime(calendar["cal_date"], format="%Y%m%d")
    if "pretrade_date" in calendar.columns:
        calendar["pretrade_date"] = pd.to_datetime(calendar["pretrade_date"], format="%Y%m%d", errors="coerce")
    if is_open is not None:
        calendar = calendar[calendar["is_open"].eq(int(is_open))].copy()
    return calendar.sort_values("cal_date", kind="mergesort", ignore_index=True)


def get_next_trading_day(
    value: DateLike,
    token: str | None = None,
    *,
    exchange: str = "SSE",
    lookahead_days: int = 20,
) -> pd.Timestamp:
    """
    Return the next open trading day after ``value``.

    ``lookahead_days`` is a simple safety window for holiday stretches.
    """
    if lookahead_days < 1:
        raise ValueError("lookahead_days must be at least 1.")

    current_day = pd.Timestamp(value).normalize()
    search_start = current_day + pd.Timedelta(days=1)
    search_end = current_day + pd.Timedelta(days=lookahead_days)
    calendar = get_trade_calendar(
        token=token,
        start_date=search_start,
        end_date=search_end,
        exchange=exchange,
        is_open=1,
    )
    if calendar.empty:
        raise ValueError(
            f"No open trading day found between {search_start.date()} and {search_end.date()}."
        )
    return pd.Timestamp(calendar["cal_date"].iloc[0]).normalize()


def get_index_member_prices(
    index_code: str,
    sd: DateLike,
    ed: DateLike,
    token: str | None = None,
    pause_seconds: float = 1.3,
    max_calls_per_minute: int = 195,
) -> pd.DataFrame:
    """
    Fetch QFQ-adjusted daily price data for an index constituent snapshot.

    Step by step, this function:
    1. resolves the latest constituent list for the requested index
    2. fetches each member's QFQ-adjusted daily bars from Tushare
    3. normalizes the raw Tushare columns into the repo's candle schema
    4. attaches index-level metadata in ``DataFrame.attrs``

    The same latest constituent snapshot is applied to the full historical
    window, so historical results have survivorship bias by construction.
    """
    start_date = _normalize_date(sd)
    end_date = _normalize_date(ed)
    metadata = _index_metadata(index_code)
    constituents = get_index_constituents(
        index_code,
        token=token,
        start_date=pd.Timestamp(end_date) - pd.Timedelta(days=31),
        end_date=end_date,
    )
    client = _get_tushare_client(token=token)

    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    window_started_at = time.monotonic()
    calls_in_window = 0

    for _, row in constituents.iterrows():
        if max_calls_per_minute > 0 and calls_in_window >= max_calls_per_minute:
            elapsed = time.monotonic() - window_started_at
            if elapsed < 60:
                time.sleep(60 - elapsed)
            window_started_at = time.monotonic()
            calls_in_window = 0

        try:
            # Each constituent is fetched independently so partial failures do
            # not abort the full universe download.
            price_df = ts.pro_bar(
                api=client,
                ts_code=row["ts_code"],
                start_date=start_date,
                end_date=end_date,
                asset="E",
                adj="qfq",
            )
            calls_in_window += 1
        except Exception as exc:
            message = str(exc)
            if "50" in message:
                # Tushare occasionally signals per-minute throttling in the
                # error message. Retry once after a cool-down window.
                time.sleep(65)
                window_started_at = time.monotonic()
                calls_in_window = 0
                try:
                    price_df = ts.pro_bar(
                        api=client,
                        ts_code=row["ts_code"],
                        start_date=start_date,
                        end_date=end_date,
                        asset="E",
                        adj="qfq",
                    )
                    calls_in_window += 1
                except Exception:
                    failed_tickers.append(row["ticker"])
                    continue
            else:
                failed_tickers.append(row["ticker"])
                continue

        if price_df.empty:
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            continue

        # Align Tushare's field names with the candle schema used everywhere
        # else in the repo so scorers/backtesters can consume the result
        # without special-case logic.
        price_df = price_df.rename(
            columns={
                "trade_date": "date",
                "vol": "volume",
                "amount": "turnover",
                "pct_chg": "change_pct",
                "change": "change_amount",
            }
        )
        price_df["date"] = pd.to_datetime(price_df["date"], format="%Y%m%d")
        price_df["ticker"] = row["ticker"]
        price_df["name"] = row["name"]
        price_df["weight"] = row["weight"]
        price_df["constituent_trade_date"] = row["trade_date"]
        price_df["amplitude_pct"] = (
            (price_df["high"] - price_df["low"]) / price_df["pre_close"] * 100
        )
        frames.append(price_df)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not frames:
        empty_df = _empty_price_frame()
        empty_df.attrs.update(
            {
                "failed_tickers": failed_tickers,
                "price_adjustment": "qfq",
                "index_code": index_code,
                "universe": metadata["universe"],
                "index_label": metadata["index_label"],
                "constituent_history_mode": "latest_snapshot",
            }
        )
        return empty_df

    result = pd.concat(frames, ignore_index=True)
    result = result[PRICE_COLUMNS].sort_values(["date", "ticker"], ignore_index=True)
    result.attrs.update(
        {
            "failed_tickers": failed_tickers,
            "price_adjustment": "qfq",
            "index_code": index_code,
            "universe": metadata["universe"],
            "index_label": metadata["index_label"],
            "constituent_history_mode": "latest_snapshot",
        }
    )
    if "constituent_trade_date" in constituents.attrs:
        result.attrs["constituent_trade_date"] = constituents.attrs["constituent_trade_date"]
    return result


def get_csi500_member_prices(
    sd: DateLike,
    ed: DateLike,
    token: str | None = None,
    pause_seconds: float = 1.3,
    max_calls_per_minute: int = 195,
) -> pd.DataFrame:
    """
    Fetch QFQ-adjusted daily price data for the latest CSI 500 constituents
    between two dates.

    Parameters
    ----------
    sd:
        Start date. Accepts ``YYYY-MM-DD`` strings or Python date objects.
    ed:
        End date. Accepts ``YYYY-MM-DD`` strings or Python date objects.
    token:
        Tushare token. If omitted, the function reads ``TUSHARE_TOKEN``.
    pause_seconds:
        Delay after each per-ticker request to reduce throttling.
    max_calls_per_minute:
        Soft request cap used to stay under Tushare's rate limit.

    Returns
    -------
    pandas.DataFrame
        Daily OHLCV bars on a QFQ-adjusted basis using the existing
        scorer/backtester schema. The returned dataframe stores
        ``price_adjustment='qfq'`` in ``attrs``.
    """
    return get_index_member_prices(
        "000905.SH",
        sd=sd,
        ed=ed,
        token=token,
        pause_seconds=pause_seconds,
        max_calls_per_minute=max_calls_per_minute,
    )


def get_hs300_member_prices(
    sd: DateLike,
    ed: DateLike,
    token: str | None = None,
    pause_seconds: float = 1.3,
    max_calls_per_minute: int = 195,
) -> pd.DataFrame:
    """Convenience wrapper for HS300 QFQ-adjusted member prices."""
    return get_index_member_prices(
        "000300.SH",
        sd=sd,
        ed=ed,
        token=token,
        pause_seconds=pause_seconds,
        max_calls_per_minute=max_calls_per_minute,
    )


class DailyTechnicalScorer:
    """
    Score CSI 500 daily candles for next-session open-to-open long candidates.

    Future scoring methods to evaluate:
    - rolling_learned_weights
    - supervised_model
    """

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
    POSITIVE_INPUT_COLUMNS = [
        "open",
        "close",
        "high",
        "low",
        "pre_close",
        "volume",
        "turnover",
    ]
    FEATURE_COLUMNS = [
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ema_10_gap",
        "ema_20_gap",
        "breakout_20d",
        "drawdown_20d",
        "excess_ret_5d",
        "excess_ret_20d",
        "volume_ratio_5_20",
        "turnover_ratio_5_20",
        "volatility_10d",
        "atr_pct_14",
        "close_location",
        "body_to_range",
        "upper_shadow_pct",
    ]
    COMPONENT_FACTORS = {
        "trend_score": ["ret_10d", "ret_20d", "ema_10_gap", "ema_20_gap"],
        "relative_strength_score": [
            "breakout_20d",
            "excess_ret_5d",
            "excess_ret_20d",
        ],
        "liquidity_score": ["volume_ratio_5_20", "turnover_ratio_5_20"],
        "risk_score": ["volatility_10d", "atr_pct_14", "drawdown_20d"],
        "candle_quality_score": [
            "close_location",
            "body_to_range",
            "upper_shadow_pct",
        ],
    }
    COMPONENT_WEIGHTS = {
        "trend_score": 0.35,
        "relative_strength_score": 0.20,
        "liquidity_score": 0.15,
        "risk_score": 0.15,
        "candle_quality_score": 0.15,
    }
    PENALTY_FACTORS = {"volatility_10d", "atr_pct_14", "drawdown_20d", "upper_shadow_pct"}
    TODO_SCORING_METHODS = ("rolling_learned_weights", "supervised_model")

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        min_history: int = 60,
        copy: bool = True,
    ) -> None:
        if not isinstance(stock_candle_df, pd.DataFrame):
            raise TypeError("stock_candle_df must be a pandas DataFrame.")
        if min_history < 1:
            raise ValueError("min_history must be at least 1.")

        prepared = stock_candle_df.copy(deep=True) if copy else stock_candle_df
        self.min_history = int(min_history)
        self.stock_candle_df = self._prepare_input_frame(prepared)
        self.stock_candle_df.attrs.update(dict(stock_candle_df.attrs))

    @classmethod
    def _prepare_input_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in cls.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"stock_candle_df is missing required columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["constituent_trade_date"] = pd.to_datetime(df["constituent_trade_date"])

        for column in cls.STRING_COLUMNS:
            df[column] = df[column].astype("string")
        for column in cls.NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
        return cls._sort_for_output(df)

    @staticmethod
    def _sort_for_calculation(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["ticker", "date"], kind="mergesort", ignore_index=True)

    @staticmethod
    def _sort_for_output(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)

    def _store_output(self, df: pd.DataFrame) -> pd.DataFrame:
        updated = self._sort_for_output(df)
        updated.attrs.update(dict(self.stock_candle_df.attrs))
        self.stock_candle_df = updated
        return self.stock_candle_df

    @staticmethod
    def _rolling_return(series: pd.Series, window: int) -> pd.Series:
        shifted = series.shift(window)
        result = series.div(shifted) - 1.0
        return result.where(series.gt(0) & shifted.gt(0))

    @staticmethod
    def _rolling_ema_gap(series: pd.Series, span: int) -> pd.Series:
        ema = series.ewm(span=span, adjust=False).mean()
        return series.div(ema) - 1.0

    @staticmethod
    def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
        return (1.0 + series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0

    @staticmethod
    def _winsorized_percentile_rank(
        values: pd.Series,
        dates: pd.Series,
        *,
        valid_mask: pd.Series,
        ascending: bool,
    ) -> pd.Series:
        ranked_input = values.where(valid_mask)
        lower = ranked_input.groupby(dates).transform(lambda series: series.quantile(0.025))
        upper = ranked_input.groupby(dates).transform(lambda series: series.quantile(0.975))
        clipped = ranked_input.clip(lower=lower, upper=upper)
        return clipped.groupby(dates).rank(method="average", pct=True, ascending=ascending)

    def add_technical_features(self) -> pd.DataFrame:
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        daily_return = (df["close"].div(df["pre_close"]) - 1.0).where(
            df["close"].gt(0) & df["pre_close"].gt(0)
        )
        df["ret_5d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 5))
        df["ret_10d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 10))
        df["ret_20d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 20))
        df["ema_10_gap"] = ticker_group["close"].transform(
            lambda series: self._rolling_ema_gap(series, 10)
        )
        df["ema_20_gap"] = ticker_group["close"].transform(
            lambda series: self._rolling_ema_gap(series, 20)
        )

        prior_high_20d = ticker_group["high"].transform(
            lambda series: series.shift(1).rolling(20, min_periods=20).max()
        )
        breakout_ratio = df["close"].div(prior_high_20d) - 1.0
        df["breakout_20d"] = breakout_ratio.clip(lower=0.0)
        df["drawdown_20d"] = ((prior_high_20d - df["close"]) / prior_high_20d).clip(lower=0.0)

        volume_mean_5 = ticker_group["volume"].transform(
            lambda series: series.rolling(5, min_periods=5).mean()
        )
        volume_mean_20 = ticker_group["volume"].transform(
            lambda series: series.rolling(20, min_periods=20).mean()
        )
        turnover_mean_5 = ticker_group["turnover"].transform(
            lambda series: series.rolling(5, min_periods=5).mean()
        )
        turnover_mean_20 = ticker_group["turnover"].transform(
            lambda series: series.rolling(20, min_periods=20).mean()
        )
        df["volume_ratio_5_20"] = volume_mean_5.div(volume_mean_20)
        df["turnover_ratio_5_20"] = turnover_mean_5.div(turnover_mean_20)

        df["volatility_10d"] = daily_return.groupby(df["ticker"], sort=False).transform(
            lambda series: series.rolling(10, min_periods=10).std(ddof=0)
        )

        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["pre_close"]).abs(),
                (df["low"] - df["pre_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range.groupby(df["ticker"], sort=False).transform(
            lambda series: series.rolling(14, min_periods=14).mean()
        )
        df["atr_pct_14"] = atr_14.div(df["close"]) * 100.0

        trading_range = df["high"] - df["low"]
        zero_range = trading_range.eq(0)
        df["close_location"] = np.where(
            zero_range, 0.5, (df["close"] - df["low"]).div(trading_range)
        )
        df["body_to_range"] = np.where(
            zero_range, 0.0, (df["close"] - df["open"]).div(trading_range)
        )
        df["upper_shadow_pct"] = np.where(
            zero_range,
            0.0,
            (df["high"] - np.maximum(df["open"], df["close"])).div(trading_range),
        )

        universe_returns = (
            pd.DataFrame({"date": df["date"], "daily_return": daily_return})
            .groupby("date", sort=True)["daily_return"]
            .mean()
        )
        universe_frame = pd.DataFrame(index=universe_returns.index)
        universe_frame["universe_ret_5d"] = self._rolling_compound_return(universe_returns, 5)
        universe_frame["universe_ret_20d"] = self._rolling_compound_return(universe_returns, 20)
        df = df.merge(universe_frame, left_on="date", right_index=True, how="left")
        df["excess_ret_5d"] = (1.0 + df["ret_5d"]).div(1.0 + df["universe_ret_5d"]) - 1.0
        df["excess_ret_20d"] = (1.0 + df["ret_20d"]).div(1.0 + df["universe_ret_20d"]) - 1.0

        df["close_location"] = df["close_location"].clip(lower=0.0, upper=1.0)
        df["body_to_range"] = df["body_to_range"].clip(lower=-1.0, upper=1.0)
        df["upper_shadow_pct"] = df["upper_shadow_pct"].clip(lower=0.0, upper=1.0)
        df[self.FEATURE_COLUMNS] = df[self.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

        df = df.drop(columns=["universe_ret_5d", "universe_ret_20d"])
        return self._store_output(df)

    def add_research_targets(self) -> pd.DataFrame:
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        df["next_execution_date"] = ticker_group["date"].shift(-1)
        df["entry_open_next"] = ticker_group["open"].shift(-1)
        df["exit_open_next"] = ticker_group["open"].shift(-2)
        df["target_next_open_to_open"] = df["exit_open_next"].div(df["entry_open_next"]) - 1.0
        invalid_target = (df["entry_open_next"] <= 0) | (df["exit_open_next"] <= 0)
        df.loc[invalid_target, "target_next_open_to_open"] = np.nan

        return self._store_output(df)

    def add_technical_score(self, top_n: int | None = None) -> pd.DataFrame:
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be at least 1 when provided.")

        self.add_technical_features()
        df = self._sort_for_calculation(self.stock_candle_df)
        factor_columns = [factor for factors in self.COMPONENT_FACTORS.values() for factor in factors]
        history_count = df.groupby("ticker", sort=False).cumcount() + 1
        positive_inputs = df[self.POSITIVE_INPUT_COLUMNS].gt(0).all(axis=1)
        factor_completeness = df[factor_columns].notna().all(axis=1)
        eligible_mask = (history_count >= self.min_history) & positive_inputs & factor_completeness
        df["technical_score_eligible"] = eligible_mask

        factor_ranks: dict[str, pd.Series] = {}
        for factor in factor_columns:
            factor_ranks[factor] = self._winsorized_percentile_rank(
                df[factor],
                df["date"],
                valid_mask=eligible_mask & df[factor].notna(),
                ascending=factor not in self.PENALTY_FACTORS,
            )

        for component_name, factors in self.COMPONENT_FACTORS.items():
            component_frame = pd.concat([factor_ranks[factor] for factor in factors], axis=1)
            component_frame.columns = factors
            df[component_name] = component_frame.mean(axis=1, skipna=False)

        weighted_score = pd.Series(0.0, index=df.index, dtype="float64")
        for component_name, weight in self.COMPONENT_WEIGHTS.items():
            weighted_score = weighted_score + df[component_name] * weight
        df["technical_score"] = weighted_score.where(eligible_mask) * 100.0

        ranked = self._sort_for_output(df)
        technical_rank = ranked.groupby("date", sort=False)["technical_score"].rank(
            method="first",
            ascending=False,
        )
        ranked["technical_rank"] = technical_rank.where(ranked["technical_score"].notna()).astype("Int64")

        if top_n is not None:
            ranked["selected_top_n"] = ranked["technical_rank"].le(top_n).fillna(False)
        elif "selected_top_n" in ranked.columns:
            ranked = ranked.drop(columns=["selected_top_n"])

        return self._store_output(ranked)

    def get_top_candidates(
        self,
        top_n: int,
        as_of_date: str | pd.Timestamp | None = None,
        exclude_top_quantile: float = 0.0,
    ) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if exclude_top_quantile < 0 or exclude_top_quantile >= 1:
            raise ValueError("exclude_top_quantile must be between 0 and 1.")
        if "technical_score" not in self.stock_candle_df.columns:
            self.add_technical_score()

        df = self.stock_candle_df.copy()
        scored = df[df["technical_score"].notna()].copy()
        if scored.empty:
            return scored

        target_date = pd.to_datetime(as_of_date) if as_of_date is not None else scored["date"].max()
        candidates = scored[scored["date"] == target_date].copy()
        candidates = candidates.sort_values(
            ["technical_rank", "technical_score", "ticker"],
            ascending=[True, False, True],
            kind="mergesort",
            ignore_index=True,
        )
        if exclude_top_quantile > 0:
            excluded_count = int(np.ceil(len(candidates) * exclude_top_quantile))
            candidates = candidates.iloc[excluded_count:].reset_index(drop=True)
        columns = [
            "date",
            "ticker",
            "ts_code",
            "name",
            "technical_rank",
            "technical_score",
            "trend_score",
            "relative_strength_score",
            "liquidity_score",
            "risk_score",
            "candle_quality_score",
        ]
        optional_columns = [
            "next_execution_date",
            "entry_open_next",
            "exit_open_next",
            "target_next_open_to_open",
            "selected_top_n",
        ]
        selected_columns = [column for column in columns + optional_columns if column in candidates.columns]
        return candidates.loc[:, selected_columns].head(top_n).reset_index(drop=True)


@dataclass(frozen=True)
class RangeStrategyConfig:
    """
    Explicit parameter bundle for the blue-chip range reversion research flow.

    The class exists to keep all rule choices in one place instead of spreading
    thresholds across the implementation.
    """
    universe: str = "csi500"
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
    REQUIRED_COLUMNS = DailyTechnicalScorer.REQUIRED_COLUMNS
    NUMERIC_COLUMNS = DailyTechnicalScorer.NUMERIC_COLUMNS
    STRING_COLUMNS = DailyTechnicalScorer.STRING_COLUMNS
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

                    if pd.notna(close_price) and pd.notna(lower_band) and close_price <= lower_band * (1.0 - cfg.breakdown_buffer):
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
        Build a structured audit package for one signal.

        The returned payload is meant for debugging and strategy review:
        - summary: key timing and outcome fields
        - signal_row: the exact feature/signal row on the chosen date
        - condition_checklist: a human-readable checklist of rule matches
        - price_window: nearby candles plus entry/exit markers
        """
        if lookback < 0 or lookahead < 0:
            raise ValueError("lookback and lookahead must be non-negative.")

        self.add_research_outcomes()
        target_date = pd.to_datetime(signal_date)
        ticker = str(ticker)
        scored = self._sort_for_calculation(self.stock_candle_df.copy())
        ticker_frame = scored[scored["ticker"].astype(str).eq(ticker)].reset_index(drop=True)
        if ticker_frame.empty:
            raise ValueError(f"Ticker '{ticker}' is not present in stock_candle_df.")

        signal_rows = ticker_frame[ticker_frame["date"] == target_date]
        if signal_rows.empty:
            raise ValueError(f"Ticker '{ticker}' does not have data on {target_date.date()}.")

        signal_row = signal_rows.iloc[[0]].copy().reset_index(drop=True)
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
            "entry_open_next": float(signal_row["entry_open_next"].iat[0]) if pd.notna(signal_row["entry_open_next"].iat[0]) else np.nan,
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
        figure.add_vline(x=signal_row["date"].iat[0], line_dash="dash", line_color="royalblue", row=1, col=1)
        if pd.notna(entry_date):
            entry_rows = price_window[price_window["date"] == entry_date]
            entry_price = signal_row["entry_open_next"].iat[0]
            if not entry_rows.empty and pd.notna(entry_price):
                figure.add_trace(
                    go.Scatter(
                        x=[entry_date],
                        y=[entry_price],
                        mode="markers",
                        marker=dict(size=11, symbol="triangle-up", color="green"),
                        name="Entry",
                    ),
                    row=1,
                    col=1,
                )
        if pd.notna(exit_date):
            exit_price = signal_row["exit_open_next"].iat[0] if "exit_open_next" in signal_row.columns else np.nan
            if pd.notna(exit_price):
                figure.add_trace(
                    go.Scatter(
                        x=[exit_date],
                        y=[exit_price],
                        mode="markers",
                        marker=dict(size=11, symbol="x", color="red"),
                        name="Exit",
                    ),
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
        )
        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Met", row=2, col=1, range=[0, 1.2])
        return figure
