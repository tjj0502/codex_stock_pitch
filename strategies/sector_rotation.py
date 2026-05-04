from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable

try:
    import akshare as ak
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    ak = None

import numpy as np
import pandas as pd


DEFAULT_BOARD_TYPES = ("industry", "concept")
BOARD_TYPE_LABELS = {
    "industry": "行业",
    "concept": "概念",
}
BOARD_TYPE_ALIASES = {
    "industry": "industry",
    "concept": "concept",
    "行业": "industry",
    "概念": "concept",
}
SECTOR_HEAT_COLUMNS = [
    "board_type",
    "board_type_label",
    "sector_name",
    "sector_code",
    "latest_price",
    "change_amount",
    "pct_change",
    "total_market_cap",
    "turnover_pct",
    "advancers",
    "decliners",
    "breadth_ratio",
    "board_leader_name",
    "board_leader_pct_change",
    "main_net_inflow",
    "change_event_count",
    "most_active_ticker",
    "most_active_name",
    "most_active_direction",
    "heat_score",
    "heat_rank",
]
SECTOR_LEADER_COLUMNS = [
    "board_type",
    "board_type_label",
    "sector_name",
    "sector_code",
    "sector_heat_rank",
    "sector_heat_score",
    "ticker",
    "name",
    "last_price",
    "pct_change",
    "change_amount",
    "volume",
    "turnover_amount",
    "amplitude_pct",
    "high",
    "low",
    "open",
    "prev_close",
    "turnover_pct",
    "close_location",
    "body_to_range",
    "upper_shadow_pct",
    "leader_score",
    "leader_rank",
]


@dataclass(frozen=True)
class SectorRotationConfig:
    top_sector_count: int = 10
    leaders_per_sector: int = 3

    sector_pct_change_weight: float = 0.30
    sector_breadth_weight: float = 0.20
    sector_turnover_weight: float = 0.10
    sector_leader_weight: float = 0.10
    sector_main_flow_weight: float = 0.20
    sector_change_count_weight: float = 0.10

    leader_pct_change_weight: float = 0.35
    leader_turnover_amount_weight: float = 0.25
    leader_turnover_weight: float = 0.15
    leader_close_location_weight: float = 0.10
    leader_body_weight: float = 0.10
    leader_upper_shadow_penalty_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.top_sector_count < 1:
            raise ValueError("top_sector_count must be at least 1.")
        if self.leaders_per_sector < 1:
            raise ValueError("leaders_per_sector must be at least 1.")

        sector_weights = [
            self.sector_pct_change_weight,
            self.sector_breadth_weight,
            self.sector_turnover_weight,
            self.sector_leader_weight,
            self.sector_main_flow_weight,
            self.sector_change_count_weight,
        ]
        leader_weights = [
            self.leader_pct_change_weight,
            self.leader_turnover_amount_weight,
            self.leader_turnover_weight,
            self.leader_close_location_weight,
            self.leader_body_weight,
            self.leader_upper_shadow_penalty_weight,
        ]
        if any(weight < 0 for weight in sector_weights + leader_weights):
            raise ValueError("All sector rotation weights must be non-negative.")
        if sum(sector_weights) <= 0:
            raise ValueError("At least one sector heat weight must be positive.")
        if sum(leader_weights) <= 0:
            raise ValueError("At least one leader score weight must be positive.")


def _require_akshare():
    if ak is None:
        raise ImportError("akshare is required for sector rotation scans. Please install akshare.")
    return ak


def _normalize_board_type(value: str) -> str:
    normalized = BOARD_TYPE_ALIASES.get(str(value).strip().lower(), BOARD_TYPE_ALIASES.get(str(value).strip()))
    if normalized is None:
        raise ValueError(f"Unsupported board_type: {value!r}")
    return normalized


def _empty_sector_heat_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SECTOR_HEAT_COLUMNS)


def _empty_sector_leader_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SECTOR_LEADER_COLUMNS)


def _winsorized_rank(values: pd.Series, *, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    lower = valid.quantile(0.025)
    upper = valid.quantile(0.975)
    clipped = numeric.clip(lower=lower, upper=upper)
    return clipped.rank(method="average", pct=True, ascending=ascending)


def _neutralize_missing(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.5)


def _call_with_retries(
    func: Callable[..., pd.DataFrame],
    *args,
    max_attempts: int = 3,
    retry_wait_seconds: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - mostly exercised via live calls
            last_error = exc
            if attempt >= max_attempts:
                break
            if retry_wait_seconds > 0:
                time.sleep(retry_wait_seconds)
    if last_error is None:  # pragma: no cover - defensive only
        raise RuntimeError("Retry helper failed without capturing an exception.")
    raise last_error


def fetch_sector_board_snapshot(board_type: str) -> pd.DataFrame:
    resolved_board_type = _normalize_board_type(board_type)
    akshare = _require_akshare()

    if resolved_board_type == "industry":
        raw = _call_with_retries(akshare.stock_board_industry_name_em)
    else:
        raw = _call_with_retries(akshare.stock_board_concept_name_em)

    rename_map = {
        "板块名称": "sector_name",
        "板块代码": "sector_code",
        "最新价": "latest_price",
        "涨跌额": "change_amount",
        "涨跌幅": "pct_change",
        "总市值": "total_market_cap",
        "换手率": "turnover_pct",
        "上涨家数": "advancers",
        "下跌家数": "decliners",
        "领涨股票": "board_leader_name",
        "领涨股票-涨跌幅": "board_leader_pct_change",
    }
    missing = [column for column in rename_map if column not in raw.columns]
    if missing:
        raise ValueError(f"Board snapshot is missing expected columns: {missing}")

    frame = raw.loc[:, list(rename_map.keys())].rename(columns=rename_map).copy()
    frame["board_type"] = resolved_board_type
    frame["board_type_label"] = BOARD_TYPE_LABELS[resolved_board_type]

    numeric_columns = [
        "latest_price",
        "change_amount",
        "pct_change",
        "total_market_cap",
        "turnover_pct",
        "advancers",
        "decliners",
        "board_leader_pct_change",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    breadth_denominator = frame["advancers"] + frame["decliners"]
    frame["breadth_ratio"] = np.where(
        breadth_denominator.gt(0),
        frame["advancers"].div(breadth_denominator),
        np.nan,
    )
    return frame


def fetch_sector_board_changes() -> pd.DataFrame:
    akshare = _require_akshare()
    raw = _call_with_retries(akshare.stock_board_change_em)
    rename_map = {
        "板块名称": "sector_name",
        "主力净流入": "main_net_inflow",
        "板块异动总次数": "change_event_count",
        "板块异动最频繁个股及所属类型-股票代码": "most_active_ticker",
        "板块异动最频繁个股及所属类型-股票名称": "most_active_name",
        "板块异动最频繁个股及所属类型-买卖方向": "most_active_direction",
    }
    missing = [column for column in rename_map if column not in raw.columns]
    if missing:
        raise ValueError(f"Board changes frame is missing expected columns: {missing}")

    frame = raw.loc[:, list(rename_map.keys())].rename(columns=rename_map).copy()
    for column in ("main_net_inflow", "change_event_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["most_active_ticker"] = frame["most_active_ticker"].astype("string")
    frame["most_active_name"] = frame["most_active_name"].astype("string")
    frame["most_active_direction"] = frame["most_active_direction"].astype("string")
    frame = frame.sort_values(
        ["change_event_count", "main_net_inflow"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    )
    frame = frame.drop_duplicates(subset=["sector_name"], keep="first")
    return frame


def score_sector_heat(
    board_snapshot: pd.DataFrame,
    board_changes: pd.DataFrame | None = None,
    config: SectorRotationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SectorRotationConfig()
    if board_snapshot.empty:
        return _empty_sector_heat_frame()

    required_snapshot_columns = [
        "board_type",
        "board_type_label",
        "sector_name",
        "sector_code",
        "latest_price",
        "change_amount",
        "pct_change",
        "total_market_cap",
        "turnover_pct",
        "advancers",
        "decliners",
        "breadth_ratio",
        "board_leader_name",
        "board_leader_pct_change",
    ]
    missing_snapshot = [column for column in required_snapshot_columns if column not in board_snapshot.columns]
    if missing_snapshot:
        raise ValueError(f"board_snapshot is missing required columns: {missing_snapshot}")

    frame = board_snapshot.copy()
    if board_changes is not None and not board_changes.empty:
        required_change_columns = [
            "sector_name",
            "main_net_inflow",
            "change_event_count",
            "most_active_ticker",
            "most_active_name",
            "most_active_direction",
        ]
        missing_change = [column for column in required_change_columns if column not in board_changes.columns]
        if missing_change:
            raise ValueError(f"board_changes is missing required columns: {missing_change}")
        frame = frame.merge(
            board_changes.loc[:, required_change_columns],
            on="sector_name",
            how="left",
        )
    else:
        frame["main_net_inflow"] = np.nan
        frame["change_event_count"] = np.nan
        frame["most_active_ticker"] = pd.Series(dtype="string")
        frame["most_active_name"] = pd.Series(dtype="string")
        frame["most_active_direction"] = pd.Series(dtype="string")

    factor_weights = {
        "pct_change": cfg.sector_pct_change_weight,
        "breadth_ratio": cfg.sector_breadth_weight,
        "turnover_pct": cfg.sector_turnover_weight,
        "board_leader_pct_change": cfg.sector_leader_weight,
        "main_net_inflow": cfg.sector_main_flow_weight,
        "change_event_count": cfg.sector_change_count_weight,
    }

    score = pd.Series(0.0, index=frame.index, dtype="float64")
    for factor, weight in factor_weights.items():
        ranked = _winsorized_rank(frame[factor], ascending=True)
        score = score + _neutralize_missing(ranked) * weight
    frame["heat_score"] = score * 100.0

    frame = frame.sort_values(
        ["heat_score", "pct_change", "breadth_ratio", "sector_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    frame["heat_rank"] = pd.RangeIndex(start=1, stop=len(frame) + 1, step=1, dtype="int64")
    return frame.loc[:, SECTOR_HEAT_COLUMNS]


def fetch_sector_constituents(board_type: str, sector_name: str) -> pd.DataFrame:
    resolved_board_type = _normalize_board_type(board_type)
    akshare = _require_akshare()

    if resolved_board_type == "industry":
        raw = _call_with_retries(akshare.stock_board_industry_cons_em, symbol=sector_name)
    else:
        raw = _call_with_retries(akshare.stock_board_concept_cons_em, symbol=sector_name)

    rename_map = {
        "代码": "ticker",
        "名称": "name",
        "最新价": "last_price",
        "涨跌幅": "pct_change",
        "涨跌额": "change_amount",
        "成交量": "volume",
        "成交额": "turnover_amount",
        "振幅": "amplitude_pct",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "prev_close",
        "换手率": "turnover_pct",
    }
    missing = [column for column in rename_map if column not in raw.columns]
    if missing:
        raise ValueError(f"Sector constituents frame is missing expected columns: {missing}")

    frame = raw.loc[:, list(rename_map.keys())].rename(columns=rename_map).copy()
    frame["ticker"] = frame["ticker"].astype("string")
    frame["name"] = frame["name"].astype("string")
    frame["board_type"] = resolved_board_type
    frame["board_type_label"] = BOARD_TYPE_LABELS[resolved_board_type]
    frame["sector_name"] = sector_name
    frame["sector_code"] = pd.NA

    numeric_columns = [
        "last_price",
        "pct_change",
        "change_amount",
        "volume",
        "turnover_amount",
        "amplitude_pct",
        "high",
        "low",
        "open",
        "prev_close",
        "turnover_pct",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price_range = frame["high"] - frame["low"]
    zero_range = price_range.eq(0)
    frame["close_location"] = np.where(
        zero_range,
        0.5,
        (frame["last_price"] - frame["low"]).div(price_range),
    )
    frame["body_to_range"] = np.where(
        zero_range,
        0.0,
        (frame["last_price"] - frame["open"]).div(price_range),
    )
    frame["upper_shadow_pct"] = np.where(
        zero_range,
        0.0,
        (frame["high"] - np.maximum(frame["open"], frame["last_price"])).div(price_range),
    )
    frame["close_location"] = frame["close_location"].clip(lower=0.0, upper=1.0)
    frame["body_to_range"] = frame["body_to_range"].clip(lower=-1.0, upper=1.0)
    frame["upper_shadow_pct"] = frame["upper_shadow_pct"].clip(lower=0.0, upper=1.0)
    return frame


def score_sector_leaders(
    constituents: pd.DataFrame,
    config: SectorRotationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SectorRotationConfig()
    if constituents.empty:
        return _empty_sector_leader_frame()

    required_columns = [
        "board_type",
        "board_type_label",
        "sector_name",
        "ticker",
        "name",
        "last_price",
        "pct_change",
        "change_amount",
        "volume",
        "turnover_amount",
        "amplitude_pct",
        "high",
        "low",
        "open",
        "prev_close",
        "turnover_pct",
        "close_location",
        "body_to_range",
        "upper_shadow_pct",
    ]
    missing = [column for column in required_columns if column not in constituents.columns]
    if missing:
        raise ValueError(f"constituents is missing required columns: {missing}")

    frame = constituents.copy()
    factor_weights = {
        "pct_change": cfg.leader_pct_change_weight,
        "turnover_amount": cfg.leader_turnover_amount_weight,
        "turnover_pct": cfg.leader_turnover_weight,
        "close_location": cfg.leader_close_location_weight,
        "body_to_range": cfg.leader_body_weight,
    }

    score = pd.Series(0.0, index=frame.index, dtype="float64")
    for factor, weight in factor_weights.items():
        ranked = _winsorized_rank(frame[factor], ascending=True)
        score = score + _neutralize_missing(ranked) * weight

    upper_shadow_rank = _winsorized_rank(frame["upper_shadow_pct"], ascending=False)
    score = score + _neutralize_missing(upper_shadow_rank) * cfg.leader_upper_shadow_penalty_weight
    frame["leader_score"] = score * 100.0

    frame = frame.sort_values(
        ["leader_score", "pct_change", "turnover_amount", "ticker"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    frame["leader_rank"] = pd.RangeIndex(start=1, stop=len(frame) + 1, step=1, dtype="int64")
    if "sector_code" not in frame.columns:
        frame["sector_code"] = pd.NA
    return frame.loc[:, [column for column in SECTOR_LEADER_COLUMNS if column in frame.columns]]


def scan_hot_sectors(
    *,
    board_types: Iterable[str] = DEFAULT_BOARD_TYPES,
    config: SectorRotationConfig | None = None,
    sector_snapshot_fetcher: Callable[[str], pd.DataFrame] = fetch_sector_board_snapshot,
    board_change_fetcher: Callable[[], pd.DataFrame] = fetch_sector_board_changes,
    constituent_fetcher: Callable[[str, str], pd.DataFrame] = fetch_sector_constituents,
) -> list[dict[str, object]]:
    cfg = config or SectorRotationConfig()
    resolved_board_types = [_normalize_board_type(board_type) for board_type in board_types]
    try:
        board_changes = board_change_fetcher()
    except Exception:
        board_changes = pd.DataFrame(
            columns=[
                "sector_name",
                "main_net_inflow",
                "change_event_count",
                "most_active_ticker",
                "most_active_name",
                "most_active_direction",
            ]
        )

    results: list[dict[str, object]] = []
    for board_type in resolved_board_types:
        try:
            board_snapshot = sector_snapshot_fetcher(board_type)
            sector_heat = score_sector_heat(board_snapshot, board_changes=board_changes, config=cfg)
            top_sectors = sector_heat.head(cfg.top_sector_count).reset_index(drop=True)
        except Exception as exc:
            results.append(
                {
                    "board_type": board_type,
                    "board_type_label": BOARD_TYPE_LABELS[board_type],
                    "sector_heat": _empty_sector_heat_frame(),
                    "top_sectors": _empty_sector_heat_frame(),
                    "leaders": _empty_sector_leader_frame(),
                    "sector_count": 0,
                    "top_sector_count": 0,
                    "leader_count": 0,
                    "status": "error",
                    "error": str(exc),
                    "leader_errors": [],
                }
            )
            continue

        leader_frames: list[pd.DataFrame] = []
        leader_errors: list[str] = []
        for sector in top_sectors.itertuples(index=False):
            try:
                constituents = constituent_fetcher(board_type, sector.sector_name)
                leaders = score_sector_leaders(constituents, config=cfg).head(cfg.leaders_per_sector).copy()
                if leaders.empty:
                    continue
                leaders["sector_code"] = sector.sector_code
                leaders["sector_heat_rank"] = int(sector.heat_rank)
                leaders["sector_heat_score"] = float(sector.heat_score)
                leader_frames.append(leaders.loc[:, SECTOR_LEADER_COLUMNS])
            except Exception as exc:
                leader_errors.append(f"{sector.sector_name}: {exc}")

        leaders_df = (
            pd.concat(leader_frames, ignore_index=True)
            if leader_frames
            else _empty_sector_leader_frame()
        )
        results.append(
            {
                "board_type": board_type,
                "board_type_label": BOARD_TYPE_LABELS[board_type],
                "sector_heat": sector_heat,
                "top_sectors": top_sectors,
                "leaders": leaders_df,
                "sector_count": int(len(sector_heat)),
                "top_sector_count": int(len(top_sectors)),
                "leader_count": int(len(leaders_df)),
                "status": "ok",
                "error": "",
                "leader_errors": leader_errors,
            }
        )
    return results


def format_sector_rotation_report(
    results: list[dict[str, object]],
    *,
    top_sector_preview: int = 5,
    leaders_preview: int = 3,
) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"[{result['board_type_label']}]")
        if result["status"] != "ok":
            lines.append("- status: error")
            lines.append(f"- error: {result['error']}")
            lines.append("")
            continue

        sector_heat: pd.DataFrame = result["top_sectors"]
        leaders: pd.DataFrame = result["leaders"]
        lines.append(f"- sector_count: {result['sector_count']}")
        lines.append(f"- top_sector_count: {result['top_sector_count']}")
        lines.append(f"- leader_count: {result['leader_count']}")

        if sector_heat.empty:
            lines.append("- top_sectors: 0")
            lines.append("")
            continue

        preview_rows = sector_heat.head(top_sector_preview)
        sector_parts = [
            f"{row.sector_name}(heat={row.heat_score:.1f}, 涨幅={row.pct_change:.2f}%)"
            for row in preview_rows.itertuples(index=False)
        ]
        lines.append(f"- top_sectors: {'; '.join(sector_parts)}")

        if leaders.empty:
            lines.append("- leaders: 0")
        else:
            lines.append("- leaders:")
            for sector_name in preview_rows["sector_name"].tolist():
                sector_leaders = leaders[leaders["sector_name"] == sector_name].head(leaders_preview)
                if sector_leaders.empty:
                    continue
                leader_text = ", ".join(
                    f"{row.name}({row.ticker}, {row.pct_change:.2f}%, score={row.leader_score:.1f})"
                    for row in sector_leaders.itertuples(index=False)
                )
                lines.append(f"  {sector_name} -> {leader_text}")

        leader_errors: list[str] = result.get("leader_errors", [])
        if leader_errors:
            lines.append(f"- leader_errors: {'; '.join(leader_errors)}")
        lines.append("")
    return "\n".join(lines).rstrip()
