from __future__ import annotations

from datetime import date, datetime
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import tushare as ts


DateLike = Union[str, date, datetime]


def _normalize_date(value: DateLike) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return pd.Timestamp(value).strftime("%Y%m%d")


def _get_tushare_client(token: str | None = None):
    resolved_token = token or os.getenv("TUSHARE_TOKEN")
    if not resolved_token:
        raise ValueError(
            "Tushare token is required. Pass token=... or set TUSHARE_TOKEN."
        )
    ts.set_token(resolved_token)
    return ts.pro_api()


def get_csi500_constituents(
    token: str | None = None,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
) -> pd.DataFrame:
    """
    Fetch the latest CSI 500 constituents from Tushare ``index_weight``.

    This requires the active Tushare account to have access to the
    ``index_weight`` endpoint.
    """
    client = _get_tushare_client(token=token)
    resolved_end = _normalize_date(end_date or pd.Timestamp.today().date())
    resolved_start = _normalize_date(
        start_date or (pd.Timestamp(resolved_end) - pd.Timedelta(days=31))
    )

    try:
        weights = client.index_weight(
            index_code="000905.SH",
            start_date=resolved_start,
            end_date=resolved_end,
        )
    except Exception as exc:
        raise PermissionError(
            "Tushare index_weight access is required for CSI 500 constituents."
        ) from exc

    if weights.empty:
        return pd.DataFrame(
            columns=["ticker", "ts_code", "name", "trade_date", "weight"]
        )

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
    return weights[["ticker", "ts_code", "name", "trade_date", "weight"]].drop_duplicates(
        subset=["ts_code"], keep="first"
    )


def get_csi500_member_prices(
    sd: DateLike,
    ed: DateLike,
    token: str | None = None,
    pause_seconds: float = 1.3,
    max_calls_per_minute: int = 195,
) -> pd.DataFrame:
    """
    Fetch daily price data for the latest CSI 500 constituents between two dates.

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
    """
    start_date = _normalize_date(sd)
    end_date = _normalize_date(ed)
    constituents = get_csi500_constituents(
        token=token,
        start_date=pd.Timestamp(end_date) - pd.Timedelta(days=31),
        end_date=end_date,
    )
    client = _get_tushare_client(token=token)
    print("!")

    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    window_started_at = time.monotonic()
    calls_in_window = 0

    for _, row in constituents.iterrows():

        try:
            price_df = client.daily(
                ts_code=row["ts_code"],
                start_date=start_date,
                end_date=end_date,
            )
            calls_in_window += 1
        except Exception as exc:
            message = str(exc)
            if "50" in message and "分钟" in message:
                time.sleep(65)
                window_started_at = time.monotonic()
                calls_in_window = 0
                try:
                    price_df = client.daily(
                        ts_code=row["ts_code"],
                        start_date=start_date,
                        end_date=end_date,
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
        empty_df = pd.DataFrame(
            columns=[
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
        )
        empty_df.attrs["failed_tickers"] = failed_tickers
        return empty_df

    result = pd.concat(frames, ignore_index=True)
    result = result[
        [
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
    ].sort_values(["date", "ticker"], ignore_index=True)
    result.attrs["failed_tickers"] = failed_tickers
    return result


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
    ) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
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
