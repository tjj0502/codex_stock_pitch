import unittest

import numpy as np
import pandas as pd

from strategies.sector_rotation import (
    BOARD_TYPE_LABELS,
    SectorRotationConfig,
    format_sector_rotation_report,
    scan_hot_sectors,
    score_sector_heat,
    score_sector_leaders,
)


def make_sector_snapshot(board_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "board_type": [board_type, board_type, board_type],
            "board_type_label": [BOARD_TYPE_LABELS[board_type]] * 3,
            "sector_name": ["Semis", "Brokers", "Coal"],
            "sector_code": ["BK001", "BK002", "BK003"],
            "latest_price": [100.0, 100.0, 100.0],
            "change_amount": [4.0, 2.2, 0.8],
            "pct_change": [4.5, 2.8, 0.5],
            "total_market_cap": [8.0e11, 5.5e11, 4.0e11],
            "turnover_pct": [5.2, 3.5, 1.2],
            "advancers": [18, 14, 5],
            "decliners": [2, 6, 15],
            "breadth_ratio": [0.90, 0.70, 0.25],
            "board_leader_name": ["ChipAlpha", "BrokerAlpha", "CoalAlpha"],
            "board_leader_pct_change": [11.0, 7.2, 2.0],
        }
    )


def make_board_changes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector_name": ["Semis", "Brokers", "Coal"],
            "main_net_inflow": [5.0e9, 2.0e9, -3.0e8],
            "change_event_count": [24, 12, 3],
            "most_active_ticker": ["688001", "600030", "601225"],
            "most_active_name": ["ChipAlpha", "BrokerAlpha", "CoalAlpha"],
            "most_active_direction": ["大笔买入", "大笔买入", "大笔卖出"],
        }
    )


def make_sector_constituents(sector_name: str, board_type: str) -> pd.DataFrame:
    if sector_name == "Semis":
        records = [
            ("688001", "ChipAlpha", 25.0, 9.6, 2.2, 2.0e6, 8.0e8, 11.0, 25.5, 22.8, 23.1, 22.8, 7.0),
            ("688002", "ChipBeta", 18.3, 7.5, 1.3, 1.5e6, 5.5e8, 8.5, 18.4, 17.0, 17.2, 17.0, 5.6),
            ("688003", "ChipGamma", 12.8, 4.1, 0.5, 9.0e5, 2.4e8, 9.2, 13.2, 12.0, 12.6, 12.3, 4.8),
        ]
    elif sector_name == "Brokers":
        records = [
            ("600030", "BrokerAlpha", 20.2, 6.2, 1.1, 2.6e6, 9.2e8, 7.2, 20.4, 19.1, 19.3, 19.0, 4.6),
            ("601688", "BrokerBeta", 14.4, 4.8, 0.7, 1.9e6, 4.4e8, 6.0, 14.7, 13.8, 14.0, 13.7, 3.7),
            ("601211", "BrokerGamma", 11.8, 2.0, 0.2, 1.2e6, 1.5e8, 5.4, 12.2, 11.5, 11.9, 11.6, 2.1),
        ]
    else:
        records = [
            ("601225", "CoalAlpha", 18.0, 1.8, 0.3, 1.0e6, 1.7e8, 4.0, 18.4, 17.6, 17.9, 17.7, 2.5),
            ("601699", "CoalBeta", 13.0, 0.6, 0.1, 8.0e5, 9.5e7, 3.2, 13.1, 12.7, 12.9, 12.9, 1.8),
            ("600188", "CoalGamma", 9.1, -0.5, -0.1, 7.5e5, 6.2e7, 2.8, 9.2, 8.9, 9.0, 9.1, 1.3),
        ]

    frame = pd.DataFrame(
        records,
        columns=[
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
        ],
    )
    frame["board_type"] = board_type
    frame["board_type_label"] = BOARD_TYPE_LABELS[board_type]
    frame["sector_name"] = sector_name
    frame["sector_code"] = {
        "Semis": "BK001",
        "Brokers": "BK002",
        "Coal": "BK003",
    }[sector_name]
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
    return frame


class SectorRotationTests(unittest.TestCase):
    def test_score_sector_heat_prefers_strength_breadth_and_flow(self) -> None:
        scored = score_sector_heat(
            make_sector_snapshot("industry"),
            board_changes=make_board_changes(),
            config=SectorRotationConfig(top_sector_count=3, leaders_per_sector=2),
        )

        self.assertEqual(scored["sector_name"].tolist(), ["Semis", "Brokers", "Coal"])
        self.assertEqual(scored["heat_rank"].tolist(), [1, 2, 3])
        self.assertGreater(scored.loc[0, "heat_score"], scored.loc[1, "heat_score"])
        self.assertGreater(scored.loc[1, "heat_score"], scored.loc[2, "heat_score"])

    def test_score_sector_leaders_prefers_breakout_liquidity_and_close_quality(self) -> None:
        scored = score_sector_leaders(
            make_sector_constituents("Semis", "industry"),
            config=SectorRotationConfig(top_sector_count=2, leaders_per_sector=3),
        )

        self.assertEqual(scored["ticker"].tolist(), ["688001", "688002", "688003"])
        self.assertEqual(scored["leader_rank"].tolist(), [1, 2, 3])
        self.assertGreater(scored.loc[0, "leader_score"], scored.loc[1, "leader_score"])
        self.assertGreater(scored.loc[1, "leader_score"], scored.loc[2, "leader_score"])

    def test_scan_hot_sectors_returns_top_sectors_and_nested_leaders(self) -> None:
        def fake_snapshot_fetcher(board_type: str) -> pd.DataFrame:
            return make_sector_snapshot(board_type)

        def fake_constituent_fetcher(board_type: str, sector_name: str) -> pd.DataFrame:
            return make_sector_constituents(sector_name, board_type)

        results = scan_hot_sectors(
            board_types=("industry",),
            config=SectorRotationConfig(top_sector_count=2, leaders_per_sector=2),
            sector_snapshot_fetcher=fake_snapshot_fetcher,
            board_change_fetcher=make_board_changes,
            constituent_fetcher=fake_constituent_fetcher,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["top_sectors"]["sector_name"].tolist(), ["Semis", "Brokers"])
        self.assertEqual(result["leader_count"], 4)
        self.assertEqual(
            result["leaders"]["sector_name"].drop_duplicates().tolist(),
            ["Semis", "Brokers"],
        )

        report = format_sector_rotation_report(results)
        self.assertIn("[行业]", report)
        self.assertIn("Semis", report)
        self.assertIn("ChipAlpha", report)


if __name__ == "__main__":
    unittest.main()
