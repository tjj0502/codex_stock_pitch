from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from score_system.daily_narrow_trend_scan import (
    DEFAULT_LOOKBACK_CALENDAR_DAYS,
    DEFAULT_PAUSE_SECONDS,
    UNIVERSE_SPECS,
    UniverseScanSpec,
    _read_positions_csv,
    _select_positions_for_universe,
    update_universe_cache,
)
from strategies.blue_chip_range_reversion import (
    BlueChipRangeReversionResearcher,
    RangeStrategyConfig,
)
from strategies.china_stock_data import get_next_trading_day


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "strategy_archive" / "blue_chip_range_reversion" / "outputs"
POSITIONS_PATH = REPO_ROOT / "strategy_archive" / "blue_chip_range_reversion" / "open_positions.csv"

DEFAULT_UNIVERSES = ("hs300", "csi500")
EXIT_COLUMNS = [
    "ticker",
    "ts_code",
    "name",
    "entry_date",
    "entry_price",
    "shares",
    "signal_date",
    "note",
    "as_of_date",
    "latest_bar_date",
    "signal_date_resolved",
    "signal_range_upper",
    "latest_range_lower",
    "latest_range_upper",
    "latest_zone_position",
    "current_close",
    "pnl_pct",
    "pnl_amount",
    "holding_days",
    "trading_days_in_trade",
    "days_until_time_stop",
    "hard_stop_price",
    "take_profit_price",
    "breakdown_streak",
    "exit_signal",
    "exit_signal_date",
    "planned_exit_date",
    "exit_reason",
    "action",
    "issue",
]


def build_blue_chip_config(universe: str) -> RangeStrategyConfig:
    return RangeStrategyConfig(
        universe=universe,
        range_window=20,
        upper_quantile=0.9,
        lower_quantile=0.1,
        min_amplitude=0.20,
        max_amplitude=0.45,
        min_return_60=0.0,
        max_abs_return_60=0.10,
        ma_dispersion_window=(20, 60, 120),
        max_ma_dispersion=0.08,
        touch_zone_pct=0.20,
        min_lower_touches=2,
        min_upper_touches=2,
        entry_zone_threshold=0.20,
        stop_loss_pct=0.10,
        breakdown_buffer=0.03,
        breakdown_confirm_days=2,
        take_profit_r_multiple=2.0,
        max_holding_days=20,
        enable_hard_stop=True,
        enable_breakdown_stop=True,
        enable_take_profit=True,
        enable_time_stop=True,
    )


def scan_universe(
    spec: UniverseScanSpec,
    *,
    end_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    token: str | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
    max_calls_per_minute: int = 195,
    positions_df: pd.DataFrame | None = None,
    researcher_cls: type[BlueChipRangeReversionResearcher] = BlueChipRangeReversionResearcher,
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
            "exits": pd.DataFrame(columns=EXIT_COLUMNS),
            "candidate_count": 0,
            "exit_count": 0,
            "candidate_tickers": [],
            "exit_tickers": [],
            "status": "ok",
            "error": "",
        }

    next_trade_date = get_next_trading_day(as_of_date, token=token)
    researcher = researcher_cls(price_df, config=build_blue_chip_config(spec.universe))
    candidates = researcher.get_next_session_candidates(
        as_of_date=as_of_date,
        next_trade_date=next_trade_date,
        entry_price_basis="close",
    )
    positions_for_universe = _select_positions_for_universe(positions_df, universe=spec.universe, price_df=price_df)
    monitored_positions = (
        researcher.monitor_positions(
            positions_for_universe,
            as_of_date=as_of_date,
            next_trade_date=next_trade_date,
        )
        if not positions_for_universe.empty
        else pd.DataFrame(columns=EXIT_COLUMNS)
    )
    exits = (
        monitored_positions[
            monitored_positions["action"].astype(str).isin(["exit_next_open", "exit_overdue"])
        ].copy()
        if not monitored_positions.empty and "action" in monitored_positions.columns
        else pd.DataFrame(columns=EXIT_COLUMNS)
    )
    if not exits.empty:
        exits = exits.reindex(columns=list(dict.fromkeys([*EXIT_COLUMNS, *exits.columns])), fill_value=pd.NA)

    return {
        **cache_meta,
        "next_trade_date": next_trade_date,
        "candidates": candidates,
        "exits": exits,
        "candidate_count": int(len(candidates)),
        "exit_count": int(len(exits)),
        "candidate_tickers": candidates["ticker"].astype(str).tolist() if not candidates.empty else [],
        "exit_tickers": exits["ticker"].astype(str).tolist() if not exits.empty else [],
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
    positions_path: Path | None = None,
    universe_specs: dict[str, UniverseScanSpec] | None = None,
    output_root: Path | None = None,
    researcher_cls: type[BlueChipRangeReversionResearcher] = BlueChipRangeReversionResearcher,
) -> list[dict[str, object]]:
    specs = universe_specs or {name: UNIVERSE_SPECS[name] for name in DEFAULT_UNIVERSES}
    output_base = output_root or OUTPUT_ROOT
    resolved_positions_path = positions_path or POSITIONS_PATH
    scan_date = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    daily_output_dir = output_base / scan_date.strftime("%Y-%m-%d")
    daily_output_dir.mkdir(parents=True, exist_ok=True)
    positions_df = _read_positions_csv(resolved_positions_path)

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
                positions_df=positions_df,
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
                "window_start": pd.Timestamp(scan_date) - pd.Timedelta(days=lookback_calendar_days),
                "window_end": scan_date,
                "as_of_date": pd.NaT,
                "next_trade_date": pd.NaT,
                "candidates": pd.DataFrame(),
                "exits": pd.DataFrame(columns=EXIT_COLUMNS),
                "candidate_count": 0,
                "exit_count": 0,
                "candidate_tickers": [],
                "exit_tickers": [],
                "status": "error",
                "error": str(exc),
            }

        candidates_path = daily_output_dir / f"{universe}_candidates_{scan_date:%Y%m%d}.csv"
        exits_path = daily_output_dir / f"{universe}_exits_{scan_date:%Y%m%d}.csv"
        candidates_df = result["candidates"]
        exits_df = result["exits"]
        if isinstance(candidates_df, pd.DataFrame):
            candidates_df.to_csv(candidates_path, index=False)
        if isinstance(exits_df, pd.DataFrame):
            exits_df.to_csv(exits_path, index=False)

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
                "exit_count": result["exit_count"],
                "candidate_tickers": "|".join(result["candidate_tickers"]),
                "exit_tickers": "|".join(result["exit_tickers"]),
                "candidates_path": str(candidates_path),
                "exits_path": str(exits_path),
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
            lines.append("- status: error")
            lines.append(f"- error: {result['error']}")
            lines.append("- candidates: 0")
            lines.append("- exits: 0")
            lines.append("")
            continue

        as_of_date = result["as_of_date"]
        next_trade_date = result["next_trade_date"]
        lines.append(f"- as_of_date: {pd.Timestamp(as_of_date).date() if pd.notna(as_of_date) else 'NaT'}")
        lines.append(
            f"- next_trade_date: {pd.Timestamp(next_trade_date).date() if pd.notna(next_trade_date) else 'NaT'}"
        )
        lines.append(f"- candidate_count: {result['candidate_count']}")
        lines.append(f"- exit_count: {result['exit_count']}")
        candidate_text = ", ".join(result["candidate_tickers"]) if result["candidate_tickers"] else "0"
        exit_text = ", ".join(result["exit_tickers"]) if result["exit_tickers"] else "0"
        lines.append(f"- candidates: {candidate_text}")
        lines.append(f"- exits: {exit_text}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily blue-chip range reversion scan.")
    parser.add_argument("--end-date", default=None, help="Scan date in YYYY-MM-DD format. Defaults to today.")
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
        help="Pause between per-ticker Tushare requests. Defaults to 0.0.",
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
        choices=sorted(DEFAULT_UNIVERSES),
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
