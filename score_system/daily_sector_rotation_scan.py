from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from strategies import (
    DEFAULT_BOARD_TYPES,
    SectorRotationConfig,
    fetch_sector_board_changes,
    fetch_sector_board_snapshot,
    fetch_sector_constituents,
    format_sector_rotation_report,
    scan_hot_sectors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "strategy_archive" / "etf_rotation" / "outputs"


def run_daily_sector_rotation_scan(
    *,
    board_types: Iterable[str] = DEFAULT_BOARD_TYPES,
    top_sectors: int = 10,
    leaders_per_sector: int = 3,
    scan_date: str | pd.Timestamp | None = None,
    output_root: Path | None = None,
    sector_snapshot_fetcher: Callable[[str], pd.DataFrame] = fetch_sector_board_snapshot,
    board_change_fetcher: Callable[[], pd.DataFrame] = fetch_sector_board_changes,
    constituent_fetcher: Callable[[str, str], pd.DataFrame] = fetch_sector_constituents,
) -> list[dict[str, object]]:
    cfg = SectorRotationConfig(
        top_sector_count=top_sectors,
        leaders_per_sector=leaders_per_sector,
    )
    scan_day = pd.Timestamp(scan_date or pd.Timestamp.today()).normalize()
    output_base = output_root or OUTPUT_ROOT
    daily_output_dir = output_base / scan_day.strftime("%Y-%m-%d")
    daily_output_dir.mkdir(parents=True, exist_ok=True)

    results = scan_hot_sectors(
        board_types=board_types,
        config=cfg,
        sector_snapshot_fetcher=sector_snapshot_fetcher,
        board_change_fetcher=board_change_fetcher,
        constituent_fetcher=constituent_fetcher,
    )

    summary_rows: list[dict[str, object]] = []
    for result in results:
        board_type = result["board_type"]
        sector_heat_path = daily_output_dir / f"{board_type}_sector_heat_{scan_day:%Y%m%d}.csv"
        top_sectors_path = daily_output_dir / f"{board_type}_top_sectors_{scan_day:%Y%m%d}.csv"
        leaders_path = daily_output_dir / f"{board_type}_sector_leaders_{scan_day:%Y%m%d}.csv"

        sector_heat_df = result["sector_heat"]
        top_sectors_df = result["top_sectors"]
        leaders_df = result["leaders"]
        if isinstance(sector_heat_df, pd.DataFrame):
            sector_heat_df.to_csv(sector_heat_path, index=False)
        if isinstance(top_sectors_df, pd.DataFrame):
            top_sectors_df.to_csv(top_sectors_path, index=False)
        if isinstance(leaders_df, pd.DataFrame):
            leaders_df.to_csv(leaders_path, index=False)

        summary_rows.append(
            {
                "scan_date": scan_day,
                "board_type": board_type,
                "board_type_label": result["board_type_label"],
                "status": result["status"],
                "error": result["error"],
                "sector_count": result["sector_count"],
                "top_sector_count": result["top_sector_count"],
                "leader_count": result["leader_count"],
                "leader_errors": "|".join(result.get("leader_errors", [])),
                "sector_heat_path": str(sector_heat_path),
                "top_sectors_path": str(top_sectors_path),
                "leaders_path": str(leaders_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = daily_output_dir / f"daily_sector_rotation_summary_{scan_day:%Y%m%d}.csv"
    summary_df.to_csv(summary_path, index=False)
    return results


def format_scan_report(results: list[dict[str, object]]) -> str:
    return format_sector_rotation_report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily sector-heat and leader scan for ETF rotation research."
    )
    parser.add_argument(
        "--board-type",
        nargs="*",
        choices=["industry", "concept", "行业", "概念"],
        default=list(DEFAULT_BOARD_TYPES),
        help="Board types to scan. Defaults to both industry and concept boards.",
    )
    parser.add_argument(
        "--top-sectors",
        type=int,
        default=10,
        help="How many hot sectors to keep per board type.",
    )
    parser.add_argument(
        "--leaders-per-sector",
        type=int,
        default=3,
        help="How many leaders to keep inside each hot sector.",
    )
    parser.add_argument(
        "--scan-date",
        default=None,
        help="Optional output-label date in YYYY-MM-DD format. Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_daily_sector_rotation_scan(
        board_types=args.board_type,
        top_sectors=args.top_sectors,
        leaders_per_sector=args.leaders_per_sector,
        scan_date=args.scan_date,
    )
    print(format_scan_report(results))


if __name__ == "__main__":
    main()
