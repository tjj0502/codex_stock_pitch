from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from strategies.china_stock_data import PRICE_COLUMNS, _get_tushare_client, _normalize_date


ETF_PRICE_COLUMNS = PRICE_COLUMNS


@dataclass(frozen=True)
class ETFUniverseMember:
    ticker: str
    name: str
    theme: str
    ts_code: str | None = None


DEFAULT_ETF_UNIVERSE: tuple[ETFUniverseMember, ...] = (
    ETFUniverseMember("510300", "CSI300 ETF", "broad", "510300.SH"),
    ETFUniverseMember("510500", "CSI500 ETF", "broad", "510500.SH"),
    ETFUniverseMember("159915", "ChiNext ETF", "growth", "159915.SZ"),
    ETFUniverseMember("588000", "STAR50 ETF", "growth", "588000.SH"),
    ETFUniverseMember("512480", "Semiconductor ETF", "technology", "512480.SH"),
    ETFUniverseMember("512660", "Military ETF", "cyclical", "512660.SH"),
    ETFUniverseMember("512880", "Securities ETF", "financials", "512880.SH"),
    ETFUniverseMember("515790", "Solar ETF", "new_energy", "515790.SH"),
    ETFUniverseMember("516160", "New Energy ETF", "new_energy", "516160.SH"),
    ETFUniverseMember("512170", "Medical ETF", "healthcare", "512170.SH"),
    ETFUniverseMember("510880", "Dividend ETF", "defensive", "510880.SH"),
    ETFUniverseMember("512690", "Liquor ETF", "consumption", "512690.SH"),
)


@dataclass(frozen=True)
class ETFHeatRotationConfig:
    min_history: int = 20
    min_ret_5d: float = 0.0
    min_ret_10d: float = 0.0
    min_turnover_ratio_5_20: float = 0.8

    momentum_score_weight: float = 0.40
    relative_strength_score_weight: float = 0.25
    liquidity_score_weight: float = 0.15
    risk_score_weight: float = 0.10
    candle_quality_score_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.min_history < 1:
            raise ValueError("min_history must be at least 1.")
        if self.min_turnover_ratio_5_20 < 0:
            raise ValueError("min_turnover_ratio_5_20 must be non-negative.")
        weights = [
            self.momentum_score_weight,
            self.relative_strength_score_weight,
            self.liquidity_score_weight,
            self.risk_score_weight,
            self.candle_quality_score_weight,
        ]
        if any(weight < 0 for weight in weights):
            raise ValueError("All score weights must be non-negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one ETF heat score weight must be positive.")


def _call_with_retries(
    func: Callable[..., Any],
    *args,
    max_attempts: int = 4,
    retry_wait_seconds: float = 1.0,
    **kwargs,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - mainly exercised during live fetches
            last_error = exc
            if attempt >= max_attempts:
                break
            if retry_wait_seconds > 0:
                time.sleep(retry_wait_seconds)
    if last_error is None:  # pragma: no cover - defensive only
        raise RuntimeError("Retry helper failed without capturing an exception.")
    raise last_error


def _normalize_ts_code(value: str) -> str:
    text = str(value).strip().upper()
    if "." in text:
        return text
    if not text.isdigit():
        raise ValueError(f"ETF ticker must be a numeric code or ts_code, got {value!r}.")
    if text.startswith("5"):
        return f"{text}.SH"
    return f"{text}.SZ"


def _normalize_etf_universe(
    universe: Iterable[ETFUniverseMember | str] | None = None,
) -> list[ETFUniverseMember]:
    if universe is None:
        return list(DEFAULT_ETF_UNIVERSE)

    metadata_map = {member.ticker: member for member in DEFAULT_ETF_UNIVERSE}
    normalized: list[ETFUniverseMember] = []
    for item in universe:
        if isinstance(item, ETFUniverseMember):
            normalized.append(item)
            continue
        ticker = str(item)
        default_member = metadata_map.get(ticker)
        if default_member is not None:
            normalized.append(default_member)
            continue
        ts_code = _normalize_ts_code(ticker)
        normalized.append(ETFUniverseMember(ticker=ticker, name=ticker, theme="custom", ts_code=ts_code))

    deduped: dict[str, ETFUniverseMember] = {}
    for member in normalized:
        deduped[str(member.ts_code or member.ticker)] = member
    return list(deduped.values())


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ETF_PRICE_COLUMNS)


def _coerce_price_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_price_frame()

    frame = df.copy()
    missing = [column for column in ETF_PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"ETF price frame is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["constituent_trade_date"] = pd.to_datetime(frame["constituent_trade_date"], errors="coerce")
    for column in ("ticker", "ts_code", "name"):
        frame[column] = frame[column].astype("string")
    numeric_columns = [column for column in ETF_PRICE_COLUMNS if column not in {"date", "ticker", "ts_code", "name"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    frame = frame.loc[:, ETF_PRICE_COLUMNS]
    return frame.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)


def fetch_etf_basic_snapshot(
    *,
    token: str | None = None,
    market: str = "E",
    status: str = "L",
) -> pd.DataFrame:
    client = _get_tushare_client(token=token)
    raw = _call_with_retries(client.fund_basic, market=market, status=status)
    if raw is None or raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    frame["ts_code"] = frame["ts_code"].astype("string")
    frame["ticker"] = frame["ts_code"].str.split(".").str[0]
    frame["name"] = frame["name"].astype("string")
    frame["benchmark"] = frame.get("benchmark", pd.Series(index=frame.index, dtype="string")).astype("string")
    etf_mask = frame["name"].str.contains("ETF", case=False, na=False)
    frame = frame[etf_mask].copy()
    return frame.sort_values(["ticker"], kind="mergesort", ignore_index=True)


def fetch_single_etf_price_history(
    ticker: str,
    *,
    name: str | None = None,
    sd: str | pd.Timestamp = "20180101",
    ed: str | pd.Timestamp | None = None,
    adjust: str = "",
    token: str | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be one of '', 'qfq', or 'hfq'.")

    ts_code = _normalize_ts_code(ticker)
    symbol = ts_code.split(".")[0]
    start_date = _normalize_date(sd)
    end_date = _normalize_date(ed or pd.Timestamp.today())
    resolved_client = client or _get_tushare_client(token=token)

    raw = _call_with_retries(
        resolved_client.fund_daily,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
    )
    if raw is None or raw.empty:
        return _empty_price_frame()

    frame = raw.rename(
        columns={
            "trade_date": "date",
            "vol": "volume",
            "amount": "turnover",
            "pct_chg": "change_pct",
            "change": "change_amount",
        }
    ).copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame["ticker"] = symbol
    frame["ts_code"] = ts_code
    frame["name"] = str(name or symbol)
    frame["weight"] = 1.0
    frame["constituent_trade_date"] = frame["date"].max()

    for column in (
        "open",
        "close",
        "high",
        "low",
        "pre_close",
        "volume",
        "turnover",
        "change_pct",
        "change_amount",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["amplitude_pct"] = np.where(
        frame["pre_close"].gt(0),
        (frame["high"] - frame["low"]).div(frame["pre_close"]) * 100.0,
        np.nan,
    )

    frame = frame.dropna(subset=["date", "open", "close", "high", "low", "pre_close"])
    frame = frame.sort_values("date", kind="mergesort", ignore_index=True)

    if adjust:
        adj_df = _call_with_retries(
            resolved_client.fund_adj,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if adj_df is not None and not adj_df.empty:
            adj_df = adj_df.copy()
            adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"], format="%Y%m%d")
            adj_df = adj_df.rename(columns={"trade_date": "date"})
            adj_df["adj_factor"] = pd.to_numeric(adj_df["adj_factor"], errors="coerce")
            frame = frame.merge(adj_df[["date", "adj_factor"]], on="date", how="left")
            valid_adj = frame["adj_factor"].dropna()
            if not valid_adj.empty:
                first_factor = float(valid_adj.iloc[0])
                latest_factor = float(valid_adj.iloc[-1])
                if adjust == "qfq" and latest_factor != 0:
                    scale = frame["adj_factor"] / latest_factor
                elif adjust == "hfq" and first_factor != 0:
                    scale = frame["adj_factor"] / first_factor
                else:
                    scale = pd.Series(np.nan, index=frame.index, dtype="float64")
                for column in ("open", "high", "low", "close"):
                    frame[column] = frame[column] * scale
                frame["pre_close"] = frame["close"].shift(1)
                if len(frame) > 0:
                    frame.loc[0, "pre_close"] = frame.loc[0, "close"] - frame.loc[0, "change_amount"] * scale.iloc[0]
                frame["change_amount"] = frame["close"] - frame["pre_close"]
                frame["change_pct"] = np.where(
                    frame["pre_close"].gt(0),
                    frame["change_amount"].div(frame["pre_close"]) * 100.0,
                    np.nan,
                )
                frame["amplitude_pct"] = np.where(
                    frame["pre_close"].gt(0),
                    (frame["high"] - frame["low"]).div(frame["pre_close"]) * 100.0,
                    np.nan,
                )
            frame = frame.drop(columns=["adj_factor"], errors="ignore")

    return _coerce_price_frame(frame)


def fetch_etf_price_panel(
    universe: Iterable[ETFUniverseMember | str] | None = None,
    *,
    sd: str | pd.Timestamp = "2018-01-01",
    ed: str | pd.Timestamp | None = None,
    adjust: str = "",
    pause_seconds: float = 0.0,
    token: str | None = None,
) -> pd.DataFrame:
    members = _normalize_etf_universe(universe)
    frames: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    client = _get_tushare_client(token=token)
    for member in members:
        try:
            history = fetch_single_etf_price_history(
                member.ts_code or member.ticker,
                name=member.name,
                sd=sd,
                ed=ed,
                adjust=adjust,
                token=token,
                client=client,
            )
        except Exception:
            failed_tickers.append(member.ts_code or member.ticker)
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            continue
        if history.empty:
            failed_tickers.append(member.ts_code or member.ticker)
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            continue
        frames.append(history)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not frames:
        empty = _empty_price_frame()
        empty.attrs.update(
            {
                "source": "tushare.fund_daily",
                "adjust": adjust,
                "tickers": [member.ts_code or member.ticker for member in members],
                "failed_tickers": failed_tickers,
            }
        )
        return empty

    panel = pd.concat(frames, ignore_index=True)
    panel = _coerce_price_frame(panel)
    panel.attrs.update(
        {
            "source": "tushare.fund_daily",
            "adjust": adjust,
            "tickers": [member.ts_code or member.ticker for member in members],
            "failed_tickers": failed_tickers,
        }
    )
    return panel


class ETFHeatRotationScorer:
    REQUIRED_COLUMNS = ETF_PRICE_COLUMNS
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
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ema_5_gap",
        "ema_10_gap",
        "breakout_20d",
        "drawdown_20d",
        "excess_ret_5d",
        "excess_ret_10d",
        "volume_ratio_5_20",
        "turnover_ratio_5_20",
        "volatility_10d",
        "atr_pct_14",
        "close_location",
        "body_to_range",
        "upper_shadow_pct",
    ]
    COMPONENT_FACTORS = {
        "momentum_score": ["ret_3d", "ret_5d", "ret_10d", "ema_10_gap"],
        "relative_strength_score": ["breakout_20d", "excess_ret_5d", "excess_ret_10d"],
        "liquidity_score": ["volume_ratio_5_20", "turnover_ratio_5_20"],
        "risk_score": ["volatility_10d", "atr_pct_14", "drawdown_20d"],
        "candle_quality_score": ["close_location", "body_to_range", "upper_shadow_pct"],
    }
    PENALTY_FACTORS = {"volatility_10d", "atr_pct_14", "drawdown_20d", "upper_shadow_pct"}

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        config: ETFHeatRotationConfig | None = None,
        copy: bool = True,
    ) -> None:
        if not isinstance(stock_candle_df, pd.DataFrame):
            raise TypeError("stock_candle_df must be a pandas DataFrame.")

        self.config = config or ETFHeatRotationConfig()
        prepared = stock_candle_df.copy(deep=True) if copy else stock_candle_df
        self.stock_candle_df = self._prepare_input_frame(prepared)

    @classmethod
    def _prepare_input_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in cls.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"stock_candle_df is missing required columns: {missing}")

        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["constituent_trade_date"] = pd.to_datetime(frame["constituent_trade_date"], errors="coerce")
        for column in cls.STRING_COLUMNS:
            frame[column] = frame[column].astype("string")
        for column in cls.NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
        return frame.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)

    @staticmethod
    def _sort_for_calculation(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["ticker", "date"], kind="mergesort", ignore_index=True)

    @staticmethod
    def _sort_for_output(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)

    def _store_output(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stock_candle_df = self._sort_for_output(df)
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

    def add_heat_features(self) -> pd.DataFrame:
        df = self._sort_for_calculation(self.stock_candle_df)
        ticker_group = df.groupby("ticker", sort=False)

        daily_return = (df["close"].div(df["pre_close"]) - 1.0).where(
            df["close"].gt(0) & df["pre_close"].gt(0)
        )
        df["ret_3d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 3))
        df["ret_5d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 5))
        df["ret_10d"] = ticker_group["close"].transform(lambda series: self._rolling_return(series, 10))
        df["ema_5_gap"] = ticker_group["close"].transform(lambda series: self._rolling_ema_gap(series, 5))
        df["ema_10_gap"] = ticker_group["close"].transform(lambda series: self._rolling_ema_gap(series, 10))

        prior_high_20d = ticker_group["high"].transform(
            lambda series: series.shift(1).rolling(20, min_periods=20).max()
        )
        breakout_ratio = df["close"].div(prior_high_20d) - 1.0
        df["breakout_20d"] = breakout_ratio.clip(lower=0.0)
        df["drawdown_20d"] = ((prior_high_20d - df["close"]) / prior_high_20d).clip(lower=0.0)

        volume_mean_5 = ticker_group["volume"].transform(lambda series: series.rolling(5, min_periods=5).mean())
        volume_mean_20 = ticker_group["volume"].transform(lambda series: series.rolling(20, min_periods=20).mean())
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
        df["close_location"] = np.where(zero_range, 0.5, (df["close"] - df["low"]).div(trading_range))
        df["body_to_range"] = np.where(zero_range, 0.0, (df["close"] - df["open"]).div(trading_range))
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
        universe_frame["universe_ret_10d"] = self._rolling_compound_return(universe_returns, 10)
        df = df.merge(universe_frame, left_on="date", right_index=True, how="left")
        df["excess_ret_5d"] = (1.0 + df["ret_5d"]).div(1.0 + df["universe_ret_5d"]) - 1.0
        df["excess_ret_10d"] = (1.0 + df["ret_10d"]).div(1.0 + df["universe_ret_10d"]) - 1.0

        df["close_location"] = df["close_location"].clip(lower=0.0, upper=1.0)
        df["body_to_range"] = df["body_to_range"].clip(lower=-1.0, upper=1.0)
        df["upper_shadow_pct"] = df["upper_shadow_pct"].clip(lower=0.0, upper=1.0)
        df[self.FEATURE_COLUMNS] = df[self.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        df = df.drop(columns=["universe_ret_5d", "universe_ret_10d"])
        return self._store_output(df)

    def add_technical_score(self, top_n: int | None = None) -> pd.DataFrame:
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be at least 1 when provided.")

        self.add_heat_features()
        df = self._sort_for_calculation(self.stock_candle_df)
        factor_columns = [factor for factors in self.COMPONENT_FACTORS.values() for factor in factors]
        history_count = df.groupby("ticker", sort=False).cumcount() + 1
        positive_inputs = df[self.POSITIVE_INPUT_COLUMNS].gt(0).all(axis=1)
        factor_completeness = df[factor_columns].notna().all(axis=1)
        trend_filter = (
            df["ret_5d"].ge(self.config.min_ret_5d)
            & df["ret_10d"].ge(self.config.min_ret_10d)
            & df["ema_10_gap"].gt(0)
            & df["turnover_ratio_5_20"].ge(self.config.min_turnover_ratio_5_20)
        )
        eligible_mask = (
            (history_count >= self.config.min_history)
            & positive_inputs
            & factor_completeness
            & trend_filter.fillna(False)
        )
        df["technical_score_eligible"] = eligible_mask
        df["rotation_eligible"] = eligible_mask

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
        component_weights = {
            "momentum_score": self.config.momentum_score_weight,
            "relative_strength_score": self.config.relative_strength_score_weight,
            "liquidity_score": self.config.liquidity_score_weight,
            "risk_score": self.config.risk_score_weight,
            "candle_quality_score": self.config.candle_quality_score_weight,
        }
        for component_name, weight in component_weights.items():
            weighted_score = weighted_score + df[component_name] * weight

        df["technical_score"] = weighted_score.where(eligible_mask) * 100.0
        df["heat_score"] = df["technical_score"]
        ranked = self._sort_for_output(df)
        technical_rank = ranked.groupby("date", sort=False)["technical_score"].rank(
            method="first",
            ascending=False,
        )
        ranked["technical_rank"] = technical_rank.where(ranked["technical_score"].notna()).astype("Int64")
        ranked["heat_rank"] = ranked["technical_rank"]

        if top_n is not None:
            ranked["selected_top_n"] = ranked["technical_rank"].le(top_n).fillna(False)
        elif "selected_top_n" in ranked.columns:
            ranked = ranked.drop(columns=["selected_top_n"])

        return self._store_output(ranked)

    def get_top_candidates(
        self,
        top_n: int,
        *,
        as_of_date: str | pd.Timestamp | None = None,
        exclude_top_quantile: float = 0.0,
    ) -> pd.DataFrame:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if exclude_top_quantile < 0 or exclude_top_quantile >= 1:
            raise ValueError("exclude_top_quantile must be between 0 and 1.")
        if "technical_score" not in self.stock_candle_df.columns:
            self.add_technical_score(top_n=top_n)

        scored = self.stock_candle_df[self.stock_candle_df["technical_score"].notna()].copy()
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
            "heat_rank",
            "heat_score",
            "momentum_score",
            "relative_strength_score",
            "liquidity_score",
            "risk_score",
            "candle_quality_score",
            "ret_5d",
            "ret_10d",
            "breakout_20d",
            "turnover_ratio_5_20",
        ]
        selected_columns = [column for column in columns if column in candidates.columns]
        return candidates.loc[:, selected_columns].head(top_n).reset_index(drop=True)


def build_rotation_membership_frame(scored_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    required_columns = {"date", "ticker", "technical_rank", "technical_score"}
    missing = [column for column in required_columns if column not in scored_df.columns]
    if missing:
        raise ValueError(f"scored_df is missing required columns: {missing}")

    frame = scored_df.copy()
    frame = frame[frame["technical_score"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "ticker", "technical_rank", "technical_score"])

    frame = frame.sort_values(
        ["date", "technical_rank", "technical_score", "ticker"],
        ascending=[True, True, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    selected = frame[frame["technical_rank"].le(top_n)].copy()
    return selected.reset_index(drop=True)
