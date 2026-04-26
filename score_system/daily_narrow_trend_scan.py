from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from strategies import (
    BullFlagNarrowTrendContinuationResearcher,
    BullFlagNarrowTrendStrategyConfig,
    get_csi1000_member_prices,
    get_csi500_member_prices,
    get_hs300_member_prices,
)
from strategies.china_stock_data import PRICE_COLUMNS, get_next_trading_day


REPO_ROOT = Path(__file__).resolve().parents[1]
DATAFRAME_DIR = REPO_ROOT / "Dataframes"
OUTPUT_ROOT = REPO_ROOT / "strategy_archive" / "bull_flag_narrow_trend_continuation" / "outputs"

DEFAULT_LOOKBACK_CALENDAR_DAYS = 420
DEFAULT_UNIVERSES = ("hs300", "csi500", "csi1000")
DEFAULT_PAUSE_SECONDS = 0.0


@dataclass(frozen=True)
class UniverseScanSpec:
    universe: str
    fetcher: Callable[..., pd.DataFrame]
    cache_path: Path
    legacy_seed_path: Path | None = None


UNIVERSE_SPECS: dict[str, UniverseScanSpec] = {
    "hs300": UniverseScanSpec(
        universe="hs300",
        fetcher=get_hs300_member_prices,
        cache_path=DATAFRAME_DIR / "hs300_stock_price.csv",
    ),
    "csi500": UniverseScanSpec(
        universe="csi500",
        fetcher=get_csi500_member_prices,
        cache_path=DATAFRAME_DIR / "csi500_stock_price.csv",
    ),
    "csi1000": UniverseScanSpec(
        universe="csi1000",
        fetcher=get_csi1000_member_prices,
        cache_path=DATAFRAME_DIR / "csi1000_stock_price.csv",
        legacy_seed_path=DATAFRAME_DIR / "csi_1000_stock_price2.csv",
    ),
}

JUST_ENDED_COLUMNS = [
    "date",
    "ticker",
    "ts_code",
    "name",
    "previous_state_date",
    "previous_run_length",
    "previous_bear_ratio",
    "previous_ema20_above_ratio",
    "previous_peak_upper_shadow_pct",
    "close",
]


def build_narrow_trend_config(universe: str) -> BullFlagNarrowTrendStrategyConfig:
    return BullFlagNarrowTrendStrategyConfig(
        universe=universe,
        narrow_trend_lookback_bars=17,
        narrow_trend_max_bear_ratio=0.25,
        narrow_trend_min_run_bars=1,
        max_flag_retrace_ratio=0.30,
        max_flag_width_pct=0.12,
        min_flag_channel_slope_pct_per_bar=-0.012,
        max_flag_channel_slope_pct_per_bar=0.016,
        min_breakout_body_pct=0.4,
        max_breakout_upper_shadow_pct=0.25,
        max_breakout_lower_shadow_pct=0.50,
    )


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _coerce_price_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_price_frame()

    frame = df.copy()
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"price frame is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["constituent_trade_date"] = pd.to_datetime(frame["constituent_trade_date"], errors="coerce")
    for column in ("ticker", "ts_code", "name"):
        frame[column] = frame[column].astype("string")
    frame = frame.loc[:, PRICE_COLUMNS]
    frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    return frame.sort_values(["date", "ticker"], ignore_index=True)


def _read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_price_frame()
    return _coerce_price_frame(pd.read_csv(path))


def _filter_window(df: pd.DataFrame, window_start: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return _empty_price_frame()
    filtered = df[df["date"].between(window_start, end_date)].copy()
    if filtered.empty:
        return _empty_price_frame()
    return _coerce_price_frame(filtered)


def _window_start(end_date: pd.Timestamp, lookback_calendar_days: int) -> pd.Timestamp:
    return pd.Timestamp(end_date).normalize() - pd.Timedelta(days=lookback_calendar_days)


def update_universe_cache(
    spec: UniverseScanSpec,
    *,
    end_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    token: str | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_calls_per_minute: int = 195,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_end_date = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    window_start = _window_start(resolved_end_date, lookback_calendar_days)

    seeded_from_legacy = False
    cached = _read_price_csv(spec.cache_path)
    if cached.empty and spec.legacy_seed_path is not None and spec.legacy_seed_path.exists():
        cached = _read_price_csv(spec.legacy_seed_path)
        cached = _filter_window(cached, window_start, resolved_end_date)
        seeded_from_legacy = not cached.empty

    cached = _filter_window(cached, window_start, resolved_end_date)
    fetch_start: pd.Timestamp | None = None
    if cached.empty:
        fetch_start = window_start
    else:
        cached_max_date = pd.Timestamp(cached["date"].max()).normalize()
        if cached_max_date < resolved_end_date:
            fetch_start = cached_max_date + pd.Timedelta(days=1)

    fetched = _empty_price_frame()
    if fetch_start is not None and fetch_start <= resolved_end_date:
        fetched = spec.fetcher(
            sd=fetch_start,
            ed=resolved_end_date,
            token=token,
            pause_seconds=pause_seconds,
            max_calls_per_minute=max_calls_per_minute,
        )
        fetched = _coerce_price_frame(fetched)

    frames = [frame for frame in (cached, fetched) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_price_frame()
    combined = _coerce_price_frame(combined)
    combined = _filter_window(combined, window_start, resolved_end_date)

    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(spec.cache_path, index=False)

    metadata = {
        "universe": spec.universe,
        "cache_path": str(spec.cache_path),
        "seeded_from_legacy": seeded_from_legacy,
        "fetched_rows": int(len(fetched)),
        "row_count": int(len(combined)),
        "ticker_count": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "window_start": window_start,
        "window_end": resolved_end_date,
        "as_of_date": pd.Timestamp(combined["date"].max()).normalize() if not combined.empty else pd.NaT,
    }
    return combined, metadata


def compute_narrow_trend_just_ended(scored_df: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame(columns=JUST_ENDED_COLUMNS)

    frame = scored_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"], ignore_index=True)
    grouped = frame.groupby("ticker", sort=False)
    frame["narrow_uptrend_state"] = frame["narrow_uptrend_state"].fillna(False).astype(bool)
    previous_state = grouped["narrow_uptrend_state"].shift(1)
    frame["prev_narrow_uptrend_state"] = previous_state.eq(True)
    frame["previous_state_date"] = grouped["date"].shift(1)
    frame["previous_run_length"] = grouped["narrow_uptrend_run_length"].shift(1)
    frame["previous_bear_ratio"] = grouped["narrow_state_bear_ratio"].shift(1)
    frame["previous_ema20_above_ratio"] = grouped["narrow_state_ema20_above_ratio"].shift(1)
    frame["previous_peak_upper_shadow_pct"] = grouped["narrow_state_peak_upper_shadow_pct"].shift(1)

    target_date = pd.Timestamp(as_of_date).normalize()
    ended = frame[
        frame["date"].eq(target_date)
        & frame["prev_narrow_uptrend_state"].astype(bool)
        & ~frame["narrow_uptrend_state"].astype(bool)
    ].copy()
    if ended.empty:
        return pd.DataFrame(columns=JUST_ENDED_COLUMNS)

    selected = ended.loc[:, JUST_ENDED_COLUMNS].sort_values(["ticker"], kind="mergesort", ignore_index=True)
    return selected


def scan_universe(
    spec: UniverseScanSpec,
    *,
    end_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    token: str | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_calls_per_minute: int = 195,
    researcher_cls: type[BullFlagNarrowTrendContinuationResearcher] = BullFlagNarrowTrendContinuationResearcher,
) -> dict[str, object]:
    price_df, cache_meta = update_universe_cache(
        spec,
        end_date=end_date,
        lookback_calendar_days=lookback_calendar_days,
        token=token,
        pause_seconds=pause_seconds,
        max_calls_per_minute=max_calls_per_minute,
    )
    as_of_date = cache_meta["as_of_date"]
    if pd.isna(as_of_date):
        return {
            **cache_meta,
            "next_trade_date": pd.NaT,
            "candidates": pd.DataFrame(),
            "just_ended": pd.DataFrame(columns=JUST_ENDED_COLUMNS),
            "candidate_count": 0,
            "just_ended_count": 0,
            "candidate_tickers": [],
            "just_ended_tickers": [],
            "status": "ok",
            "error": "",
        }

    next_trade_date = get_next_trading_day(as_of_date, token=token)
    researcher = researcher_cls(price_df, config=build_narrow_trend_config(spec.universe))
    scored_df = researcher.add_signals().copy()
    candidates = researcher.get_next_session_candidates(
        as_of_date=as_of_date,
        next_trade_date=next_trade_date,
        entry_price_basis="follow_through_close",
    )
    just_ended = compute_narrow_trend_just_ended(scored_df, as_of_date)
    return {
        **cache_meta,
        "next_trade_date": next_trade_date,
        "candidates": candidates,
        "just_ended": just_ended,
        "candidate_count": int(len(candidates)),
        "just_ended_count": int(len(just_ended)),
        "candidate_tickers": candidates["ticker"].astype(str).tolist() if not candidates.empty else [],
        "just_ended_tickers": just_ended["ticker"].astype(str).tolist() if not just_ended.empty else [],
        "status": "ok",
        "error": "",
    }


def run_daily_scan(
    *,
    universes: Iterable[str] = DEFAULT_UNIVERSES,
    end_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    token: str | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_calls_per_minute: int = 195,
    universe_specs: dict[str, UniverseScanSpec] | None = None,
    output_root: Path | None = None,
    researcher_cls: type[BullFlagNarrowTrendContinuationResearcher] = BullFlagNarrowTrendContinuationResearcher,
) -> list[dict[str, object]]:
    specs = universe_specs or UNIVERSE_SPECS
    output_base = output_root or OUTPUT_ROOT
    scan_date = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    daily_output_dir = output_base / scan_date.strftime("%Y-%m-%d")
    daily_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for universe in universes:
        spec = specs[universe]
        try:
            result = scan_universe(
                spec,
                end_date=scan_date,
                lookback_calendar_days=lookback_calendar_days,
                token=token,
                pause_seconds=pause_seconds,
                max_calls_per_minute=max_calls_per_minute,
                researcher_cls=researcher_cls,
            )
        except Exception as exc:
            result = {
                "universe": universe,
                "cache_path": str(spec.cache_path),
                "seeded_from_legacy": False,
                "fetched_rows": 0,
                "row_count": 0,
                "ticker_count": 0,
                "window_start": _window_start(scan_date, lookback_calendar_days),
                "window_end": scan_date,
                "as_of_date": pd.NaT,
                "next_trade_date": pd.NaT,
                "candidates": pd.DataFrame(),
                "just_ended": pd.DataFrame(columns=JUST_ENDED_COLUMNS),
                "candidate_count": 0,
                "just_ended_count": 0,
                "candidate_tickers": [],
                "just_ended_tickers": [],
                "status": "error",
                "error": str(exc),
            }

        candidates_path = daily_output_dir / f"{universe}_candidates_{scan_date:%Y%m%d}.csv"
        just_ended_path = daily_output_dir / f"{universe}_narrow_trend_just_ended_{scan_date:%Y%m%d}.csv"
        candidates_df = result["candidates"]
        just_ended_df = result["just_ended"]
        if isinstance(candidates_df, pd.DataFrame):
            candidates_df.to_csv(candidates_path, index=False)
        if isinstance(just_ended_df, pd.DataFrame):
            just_ended_df.to_csv(just_ended_path, index=False)

        summary_rows.append(
            {
                "scan_date": scan_date,
                "universe": universe,
                "status": result["status"],
                "error": result["error"],
                "cache_path": result["cache_path"],
                "seeded_from_legacy": result["seeded_from_legacy"],
                "fetched_rows": result["fetched_rows"],
                "row_count": result["row_count"],
                "ticker_count": result["ticker_count"],
                "as_of_date": result["as_of_date"],
                "next_trade_date": result["next_trade_date"],
                "candidate_count": result["candidate_count"],
                "just_ended_count": result["just_ended_count"],
                "candidate_tickers": "|".join(result["candidate_tickers"]),
                "just_ended_tickers": "|".join(result["just_ended_tickers"]),
                "candidates_path": str(candidates_path),
                "just_ended_path": str(just_ended_path),
            }
        )
        results.append(result)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = daily_output_dir / f"daily_scan_summary_{scan_date:%Y%m%d}.csv"
    summary_df.to_csv(summary_path, index=False)
    return results


def format_scan_report(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"[{result['universe']}]")
        if result["status"] != "ok":
            lines.append(f"- status: error")
            lines.append(f"- error: {result['error']}")
            lines.append("- candidates: 0")
            lines.append("- narrow_trend_just_ended: 0")
            lines.append("")
            continue

        as_of_date = result["as_of_date"]
        next_trade_date = result["next_trade_date"]
        lines.append(f"- as_of_date: {pd.Timestamp(as_of_date).date() if pd.notna(as_of_date) else 'NaT'}")
        lines.append(
            f"- next_trade_date: {pd.Timestamp(next_trade_date).date() if pd.notna(next_trade_date) else 'NaT'}"
        )
        lines.append(f"- candidate_count: {result['candidate_count']}")
        lines.append(f"- just_ended_count: {result['just_ended_count']}")
        candidate_text = ", ".join(result["candidate_tickers"]) if result["candidate_tickers"] else "0"
        just_ended_text = ", ".join(result["just_ended_tickers"]) if result["just_ended_tickers"] else "0"
        lines.append(f"- candidates: {candidate_text}")
        lines.append(f"- narrow_trend_just_ended: {just_ended_text}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily narrow-trend bull-flag setup scan.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Scan date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
        help="Rolling calendar window kept in local CSV caches.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Pause between per-ticker Tushare requests. Defaults to 0.0 for the fastest daily scan.",
    )
    parser.add_argument(
        "--max-calls-per-minute",
        type=int,
        default=195,
        help="Soft cap for per-minute Tushare calls.",
    )
    parser.add_argument(
        "--universe",
        nargs="*",
        choices=sorted(UNIVERSE_SPECS.keys()),
        default=list(DEFAULT_UNIVERSES),
        help="Subset of universes to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN")
    results = run_daily_scan(
        universes=args.universe,
        end_date=args.end_date,
        lookback_calendar_days=args.lookback_calendar_days,
        token=token,
        pause_seconds=args.pause_seconds,
        max_calls_per_minute=args.max_calls_per_minute,
    )
    print(format_scan_report(results))


if __name__ == "__main__":
    main()
