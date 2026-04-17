from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import ScoringBacktester
from strategies.china_stock_data import DailyTechnicalScorer, get_csi500_member_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the scoring backtest, reconstruct realized position episodes, "
            "and generate a report for the top losses."
        )
    )
    parser.add_argument("--start-date", required=True, help="Execution window start date, e.g. 2025-06-01.")
    parser.add_argument("--end-date", required=True, help="Execution window end date, e.g. 2025-12-31.")
    parser.add_argument(
        "--data-start-date",
        help=(
            "Optional earlier fetch start date to provide scoring history. "
            "If omitted, the script uses a buffer before --start-date."
        ),
    )
    parser.add_argument("--prices-csv", help="Optional CSV file containing get_csi500_member_prices output.")
    parser.add_argument("--top-n", type=int, default=10, help="Top N names to hold each day.")
    parser.add_argument(
        "--exclude-top-quantile",
        type=float,
        default=0.0,
        help=(
            "Exclude the highest-ranked X%% of same-day candidates before selecting top N. "
            "For example, 0.05 skips the top 5%% tail and starts picking after that."
        ),
    )
    parser.add_argument(
        "--top-k-losses",
        type=int,
        default=10,
        help="Number of worst episodes to include in the report.",
    )
    parser.add_argument("--min-history", type=int, default=60, help="Minimum history passed to the scorer.")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital used in the backtest.",
    )
    parser.add_argument(
        "--board-lot-size",
        type=int,
        default=100,
        help="Board lot size used for buy order rounding.",
    )
    parser.add_argument(
        "--price-limit-pct",
        type=float,
        default=0.10,
        help="Daily price-limit threshold used to block trades.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["net_pnl", "net_return_pct"],
        default="net_pnl",
        help="Loss metric used to rank the worst episodes.",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include positions still open at the end of the backtest using mark-to-market PnL.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Number of trading days before the signal date to include in the context plots.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=5,
        help="Number of trading days after execution to include in the context plots.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to score_system/output/top_losses_<timestamp>.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Pause between symbol fetches when downloading data.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout for market data downloads.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel worker count for market data downloads.",
    )
    return parser.parse_args()


def default_data_start_date(start_date: str | pd.Timestamp, min_history: int) -> str:
    buffered = pd.to_datetime(start_date) - pd.Timedelta(days=max(180, min_history * 4))
    return buffered.strftime("%Y-%m-%d")


def load_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.prices_csv:
        return pd.read_csv(args.prices_csv, low_memory=False)

    data_start_date = args.data_start_date or default_data_start_date(args.start_date, args.min_history)
    return get_csi500_member_prices(
        sd=data_start_date,
        ed=args.end_date,
        pause_seconds=args.pause_seconds,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )


def build_position_episodes(
    backtester: ScoringBacktester,
    results: dict[str, Any],
    *,
    include_open: bool,
) -> pd.DataFrame:
    trades = (
        results["trades"]
        .sort_values(["ticker", "date", "side"], ascending=[True, True, True], kind="mergesort")
        .reset_index(drop=True)
    )
    holdings = results["holdings"].sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    scored = backtester.stock_candle_df.copy()
    scored["ticker"] = scored["ticker"].astype(str)
    scored_lookup = scored.set_index(["ticker", "date"])
    calendar = pd.Index(backtester._market_calendar)

    episodes: list[dict[str, Any]] = []

    for ticker, ticker_trades in trades.groupby("ticker", sort=True):
        active_buy: dict[str, Any] | None = None

        for trade in ticker_trades.to_dict("records"):
            side = str(trade["side"])
            if side == "buy":
                if active_buy is not None:
                    raise ValueError(f"Unexpected overlapping buy episodes for ticker {ticker}.")
                active_buy = trade
                continue

            if side != "sell":
                continue

            if active_buy is None:
                continue

            entry_cash_outflow = -float(active_buy["cash_delta"])
            exit_cash_inflow = float(trade["cash_delta"])
            net_pnl = exit_cash_inflow - entry_cash_outflow
            net_return_pct = net_pnl / entry_cash_outflow if entry_cash_outflow > 0 else float("nan")
            holding_days = int(calendar.get_loc(trade["date"]) - calendar.get_loc(active_buy["date"]))

            signal_key = (str(ticker), pd.Timestamp(active_buy["signal_date"]))
            signal_name = None
            signal_rank = None
            signal_score = None
            if signal_key in scored_lookup.index:
                signal_row = scored_lookup.loc[signal_key]
                if isinstance(signal_row, pd.DataFrame):
                    signal_row = signal_row.iloc[-1]
                signal_name = signal_row.get("name")
                signal_rank = signal_row.get(backtester.rank_column)
                signal_score = signal_row.get(backtester.score_column)

            episodes.append(
                {
                    "ticker": str(ticker),
                    "name": None if pd.isna(signal_name) else str(signal_name),
                    "status": "closed",
                    "signal_date": pd.Timestamp(active_buy["signal_date"]),
                    "execution_date": pd.Timestamp(active_buy["date"]),
                    "exit_signal_date": pd.Timestamp(trade["signal_date"])
                    if pd.notna(trade["signal_date"])
                    else pd.NaT,
                    "exit_date": pd.Timestamp(trade["date"]),
                    "holding_days": holding_days,
                    "shares": int(active_buy["shares"]),
                    "signal_rank": int(signal_rank) if pd.notna(signal_rank) else pd.NA,
                    "signal_score": float(signal_score) if pd.notna(signal_score) else float("nan"),
                    "entry_open_price": float(active_buy["open_price"]),
                    "entry_execution_price": float(active_buy["execution_price"]),
                    "entry_gross_notional": float(active_buy["gross_notional"]),
                    "entry_total_cost": float(active_buy["total_cost"]),
                    "exit_open_price": float(trade["open_price"]),
                    "exit_execution_price": float(trade["execution_price"]),
                    "exit_gross_notional": float(trade["gross_notional"]),
                    "exit_total_cost": float(trade["total_cost"]),
                    "net_pnl": float(net_pnl),
                    "net_return_pct": float(net_return_pct),
                }
            )
            active_buy = None

        if active_buy is None or not include_open:
            continue

        ticker_holdings = holdings.loc[holdings["ticker"].astype(str).eq(str(ticker))].copy()
        if ticker_holdings.empty:
            continue

        last_holding = ticker_holdings.iloc[-1]
        mark_price = float(last_holding["mark_price"])
        shares = int(last_holding["shares"])
        gross_mark_value = shares * mark_price
        estimated_exit_costs = backtester._calculate_trade_costs("sell", gross_mark_value)
        estimated_exit_cash = gross_mark_value - estimated_exit_costs["total_cost"]
        entry_cash_outflow = -float(active_buy["cash_delta"])
        net_pnl = estimated_exit_cash - entry_cash_outflow
        net_return_pct = net_pnl / entry_cash_outflow if entry_cash_outflow > 0 else float("nan")
        holding_days = int(calendar.get_loc(last_holding["date"]) - calendar.get_loc(active_buy["date"]))

        signal_key = (str(ticker), pd.Timestamp(active_buy["signal_date"]))
        signal_name = None
        signal_rank = None
        signal_score = None
        if signal_key in scored_lookup.index:
            signal_row = scored_lookup.loc[signal_key]
            if isinstance(signal_row, pd.DataFrame):
                signal_row = signal_row.iloc[-1]
            signal_name = signal_row.get("name")
            signal_rank = signal_row.get(backtester.rank_column)
            signal_score = signal_row.get(backtester.score_column)

        episodes.append(
            {
                "ticker": str(ticker),
                "name": None if pd.isna(signal_name) else str(signal_name),
                "status": "open_mark_to_market",
                "signal_date": pd.Timestamp(active_buy["signal_date"]),
                "execution_date": pd.Timestamp(active_buy["date"]),
                "exit_signal_date": pd.NaT,
                "exit_date": pd.Timestamp(last_holding["date"]),
                "holding_days": holding_days,
                "shares": shares,
                "signal_rank": int(signal_rank) if pd.notna(signal_rank) else pd.NA,
                "signal_score": float(signal_score) if pd.notna(signal_score) else float("nan"),
                "entry_open_price": float(active_buy["open_price"]),
                "entry_execution_price": float(active_buy["execution_price"]),
                "entry_gross_notional": float(active_buy["gross_notional"]),
                "entry_total_cost": float(active_buy["total_cost"]),
                "exit_open_price": float("nan"),
                "exit_execution_price": mark_price,
                "exit_gross_notional": gross_mark_value,
                "exit_total_cost": float(estimated_exit_costs["total_cost"]),
                "net_pnl": float(net_pnl),
                "net_return_pct": float(net_return_pct),
            }
        )

    if not episodes:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "status",
                "signal_date",
                "execution_date",
                "exit_signal_date",
                "exit_date",
                "holding_days",
                "shares",
                "signal_rank",
                "signal_score",
                "entry_open_price",
                "entry_execution_price",
                "entry_gross_notional",
                "entry_total_cost",
                "exit_open_price",
                "exit_execution_price",
                "exit_gross_notional",
                "exit_total_cost",
                "net_pnl",
                "net_return_pct",
            ]
        )

    episodes_df = pd.DataFrame(episodes).sort_values(
        ["net_pnl", "execution_date", "ticker"], kind="mergesort"
    )
    return episodes_df.reset_index(drop=True)


def select_top_losses(
    episodes: pd.DataFrame,
    *,
    top_k: int,
    sort_by: str,
) -> pd.DataFrame:
    if episodes.empty:
        return episodes.copy()

    losses = episodes.loc[episodes[sort_by] < 0].copy()
    if losses.empty:
        return losses

    losses = losses.sort_values([sort_by, "net_pnl", "execution_date", "ticker"], kind="mergesort")
    return losses.head(top_k).reset_index(drop=True)


def safe_slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def frame_to_block(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "(empty)"
    snippet = frame.head(max_rows) if max_rows is not None else frame
    return snippet.to_string(index=False)


def write_loss_report(
    backtester: ScoringBacktester,
    results: dict[str, Any],
    top_losses: pd.DataFrame,
    output_dir: Path,
    *,
    lookback: int,
    lookahead: int,
    start_date: str,
    end_date: str,
) -> Path:
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Top Losses Report",
        "",
        f"- Execution window: `{start_date}` to `{end_date}`",
        (
            f"- Strategy: `top_n={backtester.top_n}`, "
            f"`exclude_top_quantile={backtester.exclude_top_quantile:.2%}` "
            f"with `{backtester.scorer.__class__.__name__}`"
        ),
        f"- Episodes analyzed: `{len(results['trades'])}` trade records, `{len(top_losses)}` losses in report",
        "",
    ]

    if top_losses.empty:
        report_lines.extend(
            [
                "No losing episodes were found for the chosen window and ranking rule.",
                "",
            ]
        )
    else:
        summary_columns = [
            "ticker",
            "name",
            "status",
            "signal_date",
            "execution_date",
            "exit_date",
            "holding_days",
            "signal_rank",
            "signal_score",
            "net_pnl",
            "net_return_pct",
        ]
        report_lines.extend(
            [
                "## Summary Table",
                "",
                "```text",
                frame_to_block(top_losses.loc[:, summary_columns]),
                "```",
                "",
            ]
        )

    for index, row in top_losses.iterrows():
        inspection = backtester.inspect_selection(
            row["ticker"],
            row["signal_date"],
            date_kind="signal",
            lookback=lookback,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
        )
        figure = backtester.plot_selection_context(
            row["ticker"],
            row["signal_date"],
            date_kind="signal",
            lookback=lookback,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
        )

        base_name = f"{index + 1:02d}_{safe_slug(str(row['ticker']))}_{pd.Timestamp(row['signal_date']).strftime('%Y%m%d')}"
        plot_path = plots_dir / f"{base_name}.html"
        figure.write_html(plot_path, include_plotlyjs="cdn", full_html=True)

        signal_table_path = tables_dir / f"{base_name}_signal.csv"
        ranking_table_path = tables_dir / f"{base_name}_ranking.csv"
        trade_table_path = tables_dir / f"{base_name}_trades.csv"
        price_table_path = tables_dir / f"{base_name}_price_window.csv"

        inspection["signal_row"].to_csv(signal_table_path, index=False)
        inspection["ranking_context"].to_csv(ranking_table_path, index=False)
        inspection["trade_rows"].to_csv(trade_table_path, index=False)
        inspection["price_window"].to_csv(price_table_path, index=False)

        report_lines.extend(
            [
                f"## {index + 1}. {row['ticker']} {'' if pd.isna(row['name']) else row['name']}",
                "",
                f"- Signal date: `{pd.Timestamp(row['signal_date']).date()}`",
                f"- Execution date: `{pd.Timestamp(row['execution_date']).date()}`",
                f"- Exit date: `{pd.Timestamp(row['exit_date']).date()}`",
                f"- Status: `{row['status']}`",
                f"- Signal rank / score: `{row['signal_rank']}` / `{row['signal_score']:.4f}`",
                f"- Holding days: `{int(row['holding_days'])}`",
                f"- Net PnL: `{row['net_pnl']:.2f}`",
                f"- Net return: `{row['net_return_pct']:.4%}`",
                f"- Plot: [plot_selection_context](plots/{plot_path.name})",
                f"- Signal table: [csv](tables/{signal_table_path.name})",
                f"- Ranking table: [csv](tables/{ranking_table_path.name})",
                f"- Trades table: [csv](tables/{trade_table_path.name})",
                f"- Price window table: [csv](tables/{price_table_path.name})",
                "",
                "### Signal Snapshot",
                "",
                "```text",
                frame_to_block(inspection["signal_row"].T.reset_index().rename(columns={"index": "field", 0: "value"})),
                "```",
                "",
                "### Ranking Context",
                "",
                "```text",
                frame_to_block(inspection["ranking_context"], max_rows=max(10, backtester.top_n)),
                "```",
                "",
                "### Trades",
                "",
                "```text",
                frame_to_block(inspection["trade_rows"]),
                "```",
                "",
                "### Price Window",
                "",
                "```text",
                frame_to_block(
                    inspection["price_window"][
                        [
                            column
                            for column in [
                                "date",
                                "open",
                                "high",
                                "low",
                                "close",
                                "close_return_from_signal_close",
                                "close_return_from_execution_open",
                                "intraday_return",
                            ]
                            if column in inspection["price_window"].columns
                        ]
                    ]
                ),
                "```",
                "",
            ]
        )

    report_path = output_dir / "top_losses_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def ensure_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "score_system" / "output" / f"top_losses_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    prices = load_prices(args)

    backtester = ScoringBacktester(
        prices,
        scorer=DailyTechnicalScorer,
        scorer_kwargs={"min_history": args.min_history},
        top_n=args.top_n,
        exclude_top_quantile=args.exclude_top_quantile,
        initial_capital=args.initial_capital,
        board_lot_size=args.board_lot_size,
        price_limit_pct=args.price_limit_pct,
    )
    results = backtester.compute_metrics(start_date=args.start_date, end_date=args.end_date)
    episodes = build_position_episodes(backtester, results, include_open=args.include_open)
    top_losses = select_top_losses(episodes, top_k=args.top_k_losses, sort_by=args.sort_by)

    output_dir = ensure_output_dir(args.output_dir)
    episodes.to_csv(output_dir / "all_episodes.csv", index=False)
    top_losses.to_csv(output_dir / "top_losses.csv", index=False)
    report_path = write_loss_report(
        backtester,
        results,
        top_losses,
        output_dir,
        lookback=args.lookback,
        lookahead=args.lookahead,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Saved episode table: {output_dir / 'all_episodes.csv'}")
    print(f"Saved top-loss table: {output_dir / 'top_losses.csv'}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
