import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

import china_stock_data


def make_constituents_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "ts_code": "000002.SZ",
                "name": "BBB Corp",
                "trade_date": "20250331",
                "weight": 2.0,
            },
            {
                "ticker": "AAA",
                "ts_code": "000001.SZ",
                "name": "AAA Corp",
                "trade_date": "20250331",
                "weight": 1.0,
            },
        ]
    )


def make_price_frame(ts_code: str, rows: list[tuple[str, float, float, float, float, float, float, float]]) -> pd.DataFrame:
    records = []
    for trade_date, open_, high, low, close, pre_close, vol, amount in rows:
        change = close - pre_close
        pct_chg = change / pre_close * 100.0 if pre_close > 0 else 0.0
        records.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": pre_close,
                "change": change,
                "pct_chg": pct_chg,
                "vol": vol,
                "amount": amount,
            }
        )
    return pd.DataFrame(records)


class GetCsi500MemberPricesTest(unittest.TestCase):
    EXPECTED_COLUMNS = [
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

    @patch("china_stock_data.ts.pro_bar")
    @patch("china_stock_data._get_tushare_client")
    @patch("china_stock_data.get_csi500_constituents")
    def test_get_csi500_member_prices_fetches_qfq_adjusted_schema(
        self,
        mock_get_constituents: MagicMock,
        mock_get_client: MagicMock,
        mock_pro_bar: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = make_constituents_frame()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_pro_bar.side_effect = [
            make_price_frame(
                "000002.SZ",
                [
                    ("20250103", 20.0, 21.0, 19.0, 20.5, 19.5, 1000.0, 20500.0),
                    ("20250102", 19.0, 20.0, 18.5, 19.5, 19.0, 900.0, 17550.0),
                ],
            ),
            make_price_frame(
                "000001.SZ",
                [
                    ("20250103", 10.0, 10.6, 9.8, 10.4, 10.1, 1200.0, 12480.0),
                    ("20250102", 9.8, 10.2, 9.6, 10.1, 9.9, 1100.0, 11110.0),
                ],
            ),
        ]

        result = china_stock_data.get_csi500_member_prices(
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertEqual(result.columns.tolist(), self.EXPECTED_COLUMNS)
        self.assertEqual(result.attrs["price_adjustment"], "qfq")
        self.assertEqual(result.attrs["failed_tickers"], [])
        self.assertEqual(
            list(zip(result["date"].dt.strftime("%Y-%m-%d"), result["ticker"])),
            [
                ("2025-01-02", "AAA"),
                ("2025-01-02", "BBB"),
                ("2025-01-03", "AAA"),
                ("2025-01-03", "BBB"),
            ],
        )
        self.assertAlmostEqual(
            result.loc[(result["ticker"] == "AAA") & (result["date"] == pd.Timestamp("2025-01-03")), "amplitude_pct"].iat[0],
            (10.6 - 9.8) / 10.1 * 100.0,
        )
        self.assertEqual(
            mock_pro_bar.call_args_list,
            [
                call(
                    api=mock_client,
                    ts_code="000002.SZ",
                    start_date="20250102",
                    end_date="20250103",
                    asset="E",
                    adj="qfq",
                ),
                call(
                    api=mock_client,
                    ts_code="000001.SZ",
                    start_date="20250102",
                    end_date="20250103",
                    asset="E",
                    adj="qfq",
                ),
            ],
        )

    @patch("china_stock_data.ts.pro_bar")
    @patch("china_stock_data._get_tushare_client")
    @patch("china_stock_data.get_csi500_constituents")
    def test_get_csi500_member_prices_returns_empty_qfq_frame_when_no_prices(
        self,
        mock_get_constituents: MagicMock,
        mock_get_client: MagicMock,
        mock_pro_bar: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = make_constituents_frame().iloc[[0]].copy()
        mock_get_client.return_value = MagicMock()
        mock_pro_bar.return_value = pd.DataFrame()

        result = china_stock_data.get_csi500_member_prices(
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), self.EXPECTED_COLUMNS)
        self.assertEqual(result.attrs["failed_tickers"], [])
        self.assertEqual(result.attrs["price_adjustment"], "qfq")

    @patch("china_stock_data.ts.pro_bar")
    @patch("china_stock_data._get_tushare_client")
    @patch("china_stock_data.get_csi500_constituents")
    def test_get_csi500_member_prices_tracks_failed_tickers_with_qfq_fetches(
        self,
        mock_get_constituents: MagicMock,
        mock_get_client: MagicMock,
        mock_pro_bar: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = make_constituents_frame()
        mock_get_client.return_value = MagicMock()
        mock_pro_bar.side_effect = [
            Exception("request failed"),
            make_price_frame(
                "000001.SZ",
                [("20250103", 10.0, 10.6, 9.8, 10.4, 10.1, 1200.0, 12480.0)],
            ),
        ]

        result = china_stock_data.get_csi500_member_prices(
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertEqual(result["ticker"].tolist(), ["AAA"])
        self.assertEqual(result.attrs["failed_tickers"], ["BBB"])
        self.assertEqual(result.attrs["price_adjustment"], "qfq")


if __name__ == "__main__":
    unittest.main()
