from __future__ import annotations

from datetime import date, datetime
import os
import time
from typing import Union

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
