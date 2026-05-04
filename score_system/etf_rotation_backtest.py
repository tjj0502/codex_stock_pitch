from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtester import ScoringBacktester
from strategies import (
    DEFAULT_ETF_UNIVERSE,
    ETFHeatRotationConfig,
    ETFHeatRotationScorer,
    ETFUniverseMember,
    build_rotation_membership_frame,
    fetch_etf_price_panel,
)
from strategies.china_stock_data import PRICE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]
DATAFRAME_DIR = REPO_ROOT / "Dataframes"
OUTPUT_ROOT = REPO_ROOT / "strategy_archive" / "etf_rotation" / "backtests"
DEFAULT_CACHE_PATH = DATAFRAME_DIR / "etf_rotation_price.csv"
DEFAULT_LOOKBACK_CALENDAR_DAYS = 1_500


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _coerce_price_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_price_frame()

    frame = df.copy()
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"ETF cache frame is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["constituent_trade_date"] = pd.to_datetime(frame["constituent_trade_date"], errors="coerce")
    for column in ("ticker", "ts_code", "name"):
        frame[column] = frame[column].astype("string")
    numeric_columns = [column for column in PRICE_COLUMNS if column not in {"date", "ticker", "ts_code", "name"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    return frame.loc[:, PRICE_COLUMNS].sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)


def _read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_price_frame()
    return _coerce_price_frame(pd.read_csv(path))


def _window_start(end_date: pd.Timestamp, lookback_calendar_days: int) -> pd.Timestamp:
    return pd.Timestamp(end_date).normalize() - pd.Timedelta(days=lookback_calendar_days)


def _normalize_universe(universe: Iterable[ETFUniverseMember | str] | None = None) -> list[ETFUniverseMember]:
    if universe is None:
        return list(DEFAULT_ETF_UNIVERSE)

    default_map = {member.ticker: member for member in DEFAULT_ETF_UNIVERSE}
    normalized: dict[str, ETFUniverseMember] = {}
    for item in universe:
        if isinstance(item, ETFUniverseMember):
            normalized[item.ticker] = item
        else:
            ticker = str(item)
            normalized[ticker] = default_map.get(ticker, ETFUniverseMember(ticker=ticker, name=ticker, theme="custom"))
    return list(normalized.values())


def update_etf_rotation_cache(
    *,
    universe: Iterable[ETFUniverseMember | str] | None = None,
    end_date: str | pd.Timestamp | None = None,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    cache_path: Path = DEFAULT_CACHE_PATH,
    adjust: str = "",
    pause_seconds: float = 0.0,
    token: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    members = _normalize_universe(universe)
    requested_tickers = {member.ticker for member in members}
    resolved_end_date = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    window_start = _window_start(resolved_end_date, lookback_calendar_days)

    cached = _read_price_csv(cache_path)
    cache_before = cached.copy()
    if not cached.empty:
        cached = cached[cached["ticker"].astype(str).isin(requested_tickers)].copy()
        cached = cached[cached["date"].between(window_start, resolved_end_date)].copy()

    cached_tickers = set(cached["ticker"].astype(str).unique()) if not cached.empty else set()
    missing_members = [member for member in members if member.ticker not in cached_tickers]
    incremental_members = [member for member in members if member.ticker in cached_tickers]

    fetched_frames: list[pd.DataFrame] = []
    fetched_rows = 0
    if missing_members:
        missing_frame = fetch_etf_price_panel(
            missing_members,
            sd=window_start,
            ed=resolved_end_date,
            adjust=adjust,
            pause_seconds=pause_seconds,
            token=token,
        )
        fetched_rows += int(len(missing_frame))
        if not missing_frame.empty:
            fetched_frames.append(missing_frame)

    if not cached.empty and incremental_members:
        cached_max_date = pd.Timestamp(cached["date"].max()).normalize()
        fetch_start = cached_max_date + pd.Timedelta(days=1)
        if fetch_start <= resolved_end_date:
            incremental_frame = fetch_etf_price_panel(
                incremental_members,
                sd=fetch_start,
                ed=resolved_end_date,
                adjust=adjust,
                pause_seconds=pause_seconds,
                token=token,
            )
            fetched_rows += int(len(incremental_frame))
            if not incremental_frame.empty:
                fetched_frames.append(incremental_frame)
    elif cached.empty and not missing_members:
        full_frame = fetch_etf_price_panel(
            members,
            sd=window_start,
            ed=resolved_end_date,
            adjust=adjust,
            pause_seconds=pause_seconds,
            token=token,
        )
        fetched_rows += int(len(full_frame))
        if not full_frame.empty:
            fetched_frames.append(full_frame)

    combined = pd.concat([frame for frame in [cached, *fetched_frames] if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in [cached, *fetched_frames]
    ) else _empty_price_frame()
    combined = _coerce_price_frame(combined)
    combined = combined[combined["date"].between(window_start, resolved_end_date)].copy()
    combined = combined[combined["ticker"].astype(str).isin(requested_tickers)].copy()

    preserved = cache_before.copy()
    if not preserved.empty:
        preserved = preserved[~preserved["ticker"].astype(str).isin(requested_tickers)].copy()
    cache_to_write = pd.concat([preserved, combined], ignore_index=True) if not preserved.empty else combined.copy()
    cache_to_write = _coerce_price_frame(cache_to_write) if not cache_to_write.empty else _empty_price_frame()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_to_write.to_csv(cache_path, index=False)

    metadata = {
        "cache_path": str(cache_path),
        "universe_size": len(members),
        "row_count": int(len(combined)),
        "ticker_count": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "fetched_rows": fetched_rows,
        "window_start": window_start,
        "window_end": resolved_end_date,
        "as_of_date": pd.Timestamp(combined["date"].max()).normalize() if not combined.empty else pd.NaT,
    }
    return combined.reset_index(drop=True), metadata


def run_etf_rotation_backtest(
    *,
    universe: Iterable[ETFUniverseMember | str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    top_n: int = 3,
    exclude_top_quantile: float = 0.0,
    lookback_calendar_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    cache_path: Path = DEFAULT_CACHE_PATH,
    output_root: Path | None = None,
    adjust: str = "",
    pause_seconds: float = 0.0,
    token: str | None = None,
    initial_capital: float = 1_000_000.0,
    scorer_config: ETFHeatRotationConfig | None = None,
) -> dict[str, object]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    members = _normalize_universe(universe)
    resolved_end_date = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    if start_date is None:
        default_start = resolved_end_date - pd.Timedelta(days=min(lookback_calendar_days, 900))
        resolved_start_date = default_start.normalize()
    else:
        resolved_start_date = pd.Timestamp(start_date).normalize()
    if resolved_start_date > resolved_end_date:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    price_df, cache_meta = update_etf_rotation_cache(
        universe=members,
        end_date=resolved_end_date,
        lookback_calendar_days=max(
            lookback_calendar_days,
            int((resolved_end_date - resolved_start_date).days) + 60,
        ),
        cache_path=cache_path,
        adjust=adjust,
        pause_seconds=pause_seconds,
        token=token,
    )
    if price_df.empty:
        raise ValueError("ETF price panel is empty; unable to run rotation backtest.")

    scorer = ETFHeatRotationScorer(price_df, config=scorer_config or ETFHeatRotationConfig())
    backtester = ScoringBacktester(
        price_df,
        scorer=scorer,
        top_n=top_n,
        exclude_top_quantile=exclude_top_quantile,
        initial_capital=initial_capital,
        board_lot_size=100,
        price_limit_pct=1.0,
    )
    results = backtester.compute_metrics(start_date=resolved_start_date, end_date=resolved_end_date)

    scorer_frame = scorer.stock_candle_df.copy()
    available_dates = scorer_frame.loc[
        scorer_frame["date"].le(resolved_end_date),
        "date",
    ]
    latest_signal_date = available_dates.max() if not available_dates.empty else pd.NaT
    latest_selection = (
        scorer.get_top_candidates(
            top_n,
            as_of_date=latest_signal_date,
            exclude_top_quantile=exclude_top_quantile,
        )
        if pd.notna(latest_signal_date)
        else pd.DataFrame()
    )
    membership_frame = build_rotation_membership_frame(scorer_frame, top_n=top_n)
    membership_frame = membership_frame[
        membership_frame["date"].between(resolved_start_date, resolved_end_date)
    ].reset_index(drop=True)

    output_base = output_root or OUTPUT_ROOT
    run_dir_name = f"etf_heat_rotation_{resolved_start_date:%Y%m%d}_{resolved_end_date:%Y%m%d}"
    run_output_dir = output_base / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        **results["summary"],
        "top_n": top_n,
        "exclude_top_quantile": exclude_top_quantile,
        "cache_path": cache_meta["cache_path"],
        "price_rows": cache_meta["row_count"],
        "price_ticker_count": cache_meta["ticker_count"],
        "as_of_date": cache_meta["as_of_date"],
        "latest_signal_date": latest_signal_date,
    }
    summary_df = pd.DataFrame([summary_row])

    summary_path = run_output_dir / "backtest_summary.csv"
    portfolio_path = run_output_dir / "portfolio.csv"
    trades_path = run_output_dir / "trades.csv"
    holdings_path = run_output_dir / "holdings.csv"
    membership_path = run_output_dir / "rotation_membership.csv"
    latest_selection_path = run_output_dir / "latest_selection.csv"
    universe_path = run_output_dir / "etf_universe.csv"
    config_path = run_output_dir / "scorer_config.csv"

    summary_df.to_csv(summary_path, index=False)
    results["portfolio"].to_csv(portfolio_path, index=False)
    results["trades"].to_csv(trades_path, index=False)
    results["holdings"].to_csv(holdings_path, index=False)
    membership_frame.to_csv(membership_path, index=False)
    latest_selection.to_csv(latest_selection_path, index=False)
    pd.DataFrame([asdict(member) for member in members]).to_csv(universe_path, index=False)
    pd.DataFrame([asdict(scorer.config)]).to_csv(config_path, index=False)

    return {
        **results,
        "cache_metadata": cache_meta,
        "latest_selection": latest_selection,
        "membership": membership_frame,
        "output_dir": run_output_dir,
        "summary_path": summary_path,
    }


def format_backtest_report(result: dict[str, object]) -> str:
    summary = result["summary"]
    latest_selection: pd.DataFrame = result["latest_selection"]
    output_dir: Path = result["output_dir"]
    lines = [
        f"start_date: {pd.Timestamp(summary['start_date']).date() if pd.notna(summary['start_date']) else 'NaT'}",
        f"end_date: {pd.Timestamp(summary['end_date']).date() if pd.notna(summary['end_date']) else 'NaT'}",
        f"total_return: {summary['total_return']:.2%}",
        f"benchmark_total_return: {summary['benchmark_total_return']:.2%}",
        f"excess_return: {summary['excess_return']:.2%}",
        f"cagr: {summary['cagr']:.2%}" if pd.notna(summary["cagr"]) else "cagr: NaN",
        f"max_drawdown: {summary['max_drawdown']:.2%}",
        f"sharpe: {summary['sharpe']:.2f}" if pd.notna(summary["sharpe"]) else "sharpe: NaN",
        f"average_holdings_count: {summary['average_holdings_count']:.2f}",
        f"total_trades: {summary['total_trades']}",
        f"output_dir: {output_dir}",
    ]
    if latest_selection.empty:
        lines.append("latest_selection: 0")
    else:
        selection_text = "; ".join(
            f"{row.name}({row.ticker}, score={row.technical_score:.1f}, ret_5d={row.ret_5d:.2%}, ret_10d={row.ret_10d:.2%})"
            for row in latest_selection.itertuples(index=False)
        )
        lines.append(f"latest_selection: {selection_text}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF heat rotation backtest.")
    parser.add_argument("--start-date", default=None, help="Backtest start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=None, help="Backtest end date in YYYY-MM-DD format.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of ETFs to hold.")
    parser.add_argument(
        "--exclude-top-quantile",
        type=float,
        default=0.0,
        help="Exclude the hottest fraction of names before selecting top_n.",
    )
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=DEFAULT_LOOKBACK_CALENDAR_DAYS,
        help="Calendar lookback kept in the local ETF cache.",
    )
    parser.add_argument(
        "--ticker",
        nargs="*",
        default=None,
        help="Optional custom ETF ticker universe. Defaults to the built-in pool.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Pause between ETF history requests.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital for the backtest.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Tushare token. Defaults to the TUSHARE_TOKEN environment variable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe: Iterable[ETFUniverseMember | str] | None = args.ticker if args.ticker else None
    result = run_etf_rotation_backtest(
        universe=universe,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        exclude_top_quantile=args.exclude_top_quantile,
        lookback_calendar_days=args.lookback_calendar_days,
        pause_seconds=args.pause_seconds,
        token=args.token,
        initial_capital=args.initial_capital,
    )
    print(format_backtest_report(result))


if __name__ == "__main__":
    main()
