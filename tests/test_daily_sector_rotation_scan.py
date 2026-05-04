import tempfile
import unittest
from pathlib import Path

import pandas as pd

from score_system.daily_sector_rotation_scan import (
    format_scan_report,
    run_daily_sector_rotation_scan,
)
from tests.test_sector_rotation import (
    make_board_changes,
    make_sector_constituents,
    make_sector_snapshot,
)


class DailySectorRotationScanTests(unittest.TestCase):
    def test_run_daily_sector_rotation_scan_writes_outputs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"

            def fake_snapshot_fetcher(board_type: str) -> pd.DataFrame:
                return make_sector_snapshot(board_type)

            def fake_constituent_fetcher(board_type: str, sector_name: str) -> pd.DataFrame:
                return make_sector_constituents(sector_name, board_type)

            results = run_daily_sector_rotation_scan(
                board_types=("industry", "concept"),
                top_sectors=2,
                leaders_per_sector=2,
                scan_date="2026-04-24",
                output_root=output_root,
                sector_snapshot_fetcher=fake_snapshot_fetcher,
                board_change_fetcher=make_board_changes,
                constituent_fetcher=fake_constituent_fetcher,
            )

            daily_output_dir = output_root / "2026-04-24"
            self.assertEqual(len(results), 2)
            self.assertTrue((daily_output_dir / "industry_sector_heat_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "industry_top_sectors_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "industry_sector_leaders_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "concept_sector_heat_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "concept_top_sectors_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "concept_sector_leaders_20260424.csv").exists())
            self.assertTrue((daily_output_dir / "daily_sector_rotation_summary_20260424.csv").exists())

            summary_df = pd.read_csv(daily_output_dir / "daily_sector_rotation_summary_20260424.csv")
            self.assertEqual(summary_df["board_type"].tolist(), ["industry", "concept"])
            self.assertTrue((summary_df["status"] == "ok").all())
            self.assertTrue((summary_df["top_sector_count"] == 2).all())
            self.assertTrue((summary_df["leader_count"] == 4).all())

            report = format_scan_report(results)
            self.assertIn("[行业]", report)
            self.assertIn("[概念]", report)
            self.assertIn("Brokers", report)


if __name__ == "__main__":
    unittest.main()
