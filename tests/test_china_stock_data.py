import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

import strategies.china_stock_data as china_stock_data


def make_constituents_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
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
    frame.attrs["constituent_trade_date"] = "20250331"
    return frame


def make_price_frame(
    ts_code: str,
    rows: list[tuple[str, float, float, float, float, float, float, float]],
) -> pd.DataFrame:
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


class IndexDataFetchTest(unittest.TestCase):
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

    @patch("strategies.china_stock_data._get_tushare_client")
    def test_get_index_constituents_returns_latest_snapshot_with_metadata(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.index_weight.return_value = pd.DataFrame(
            [
                {"trade_date": "20250331", "con_code": "000002.SZ", "weight": 2.0},
                {"trade_date": "20250331", "con_code": "000001.SZ", "weight": 1.0},
                {"trade_date": "20250324", "con_code": "000003.SZ", "weight": 3.0},
            ]
        )
        mock_client.stock_basic.return_value = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "AAA Corp"},
                {"ts_code": "000002.SZ", "name": "BBB Corp"},
                {"ts_code": "000003.SZ", "name": "CCC Corp"},
            ]
        )
        mock_get_client.return_value = mock_client

        result = china_stock_data.get_index_constituents(
            "000300.SH",
            start_date="2025-03-01",
            end_date="2025-03-31",
        )

        self.assertEqual(set(result["ticker"]), {"000001", "000002"})
        self.assertEqual(set(result["name"]), {"AAA Corp", "BBB Corp"})
        self.assertEqual(result.attrs["index_code"], "000300.SH")
        self.assertEqual(result.attrs["universe"], "hs300")
        self.assertEqual(result.attrs["index_label"], "HS300")
        self.assertEqual(result.attrs["constituent_history_mode"], "latest_snapshot")
        self.assertEqual(result.attrs["constituent_trade_date"], "20250331")
        mock_client.index_weight.assert_called_once_with(
            index_code="000300.SH",
            start_date="20250301",
            end_date="20250331",
        )

    @patch("strategies.china_stock_data._get_tushare_client")
    def test_get_trade_calendar_normalizes_dates_and_filters_open_days(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.trade_cal.return_value = pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20250403", "is_open": 1, "pretrade_date": "20250402"},
                {"exchange": "SSE", "cal_date": "20250404", "is_open": 0, "pretrade_date": "20250403"},
                {"exchange": "SSE", "cal_date": "20250407", "is_open": 1, "pretrade_date": "20250403"},
            ]
        )
        mock_get_client.return_value = mock_client

        result = china_stock_data.get_trade_calendar(
            start_date="2025-04-03",
            end_date="2025-04-07",
        )

        self.assertEqual(result["cal_date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-04-03", "2025-04-07"])
        self.assertTrue(result["is_open"].eq(1).all())
        mock_client.trade_cal.assert_called_once_with(
            exchange="SSE",
            start_date="20250403",
            end_date="20250407",
        )

    @patch("strategies.china_stock_data.get_trade_calendar")
    def test_get_next_trading_day_returns_first_open_date(
        self,
        mock_get_trade_calendar: MagicMock,
    ) -> None:
        mock_get_trade_calendar.return_value = pd.DataFrame(
            {
                "cal_date": pd.to_datetime(["2025-04-07", "2025-04-08"]),
                "is_open": [1, 1],
            }
        )

        result = china_stock_data.get_next_trading_day("2025-04-03")

        self.assertEqual(result, pd.Timestamp("2025-04-07"))
        mock_get_trade_calendar.assert_called_once()

    @patch("strategies.china_stock_data._get_tushare_client")
    def test_get_all_a_share_constituents_returns_latest_listed_snapshot(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.stock_basic.return_value = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "AAA Corp", "list_date": "20010101"},
                {"ts_code": "600000.SH", "symbol": "600000", "name": "BBB Corp", "list_date": "19991110"},
                {"ts_code": "430001.NQ", "symbol": "430001", "name": "Ignore Corp", "list_date": "20100101"},
            ]
        )
        mock_get_client.return_value = mock_client

        result = china_stock_data.get_all_a_share_constituents(end_date="2025-03-31")

        self.assertEqual(result["ticker"].tolist(), ["000001", "600000"])
        self.assertTrue((result["trade_date"] == "20250331").all())
        self.assertTrue((result["weight"] == 1.0).all())
        self.assertEqual(result.attrs["index_code"], "ALL_A")
        self.assertEqual(result.attrs["universe"], "all_a")
        self.assertEqual(result.attrs["index_label"], "All A Shares")
        self.assertEqual(result.attrs["constituent_history_mode"], "latest_snapshot")
        self.assertEqual(result.attrs["constituent_trade_date"], "20250331")
        mock_client.stock_basic.assert_called_once_with(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,list_date",
        )

    @patch("strategies.china_stock_data.ts.pro_bar")
    @patch("strategies.china_stock_data._get_tushare_client")
    @patch("strategies.china_stock_data.get_index_constituents")
    def test_get_index_member_prices_fetches_qfq_adjusted_schema_and_metadata(
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

        result = china_stock_data.get_index_member_prices(
            "000905.SH",
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertEqual(result.columns.tolist(), self.EXPECTED_COLUMNS)
        self.assertEqual(result.attrs["price_adjustment"], "qfq")
        self.assertEqual(result.attrs["failed_tickers"], [])
        self.assertEqual(result.attrs["index_code"], "000905.SH")
        self.assertEqual(result.attrs["universe"], "csi500")
        self.assertEqual(result.attrs["index_label"], "CSI500")
        self.assertEqual(result.attrs["constituent_history_mode"], "latest_snapshot")
        self.assertEqual(result.attrs["constituent_trade_date"], "20250331")
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
            result.loc[
                (result["ticker"] == "AAA") & (result["date"] == pd.Timestamp("2025-01-03")),
                "amplitude_pct",
            ].iat[0],
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

    @patch("strategies.china_stock_data.ts.pro_bar")
    @patch("strategies.china_stock_data._get_tushare_client")
    @patch("strategies.china_stock_data.get_index_constituents")
    def test_get_index_member_prices_returns_empty_frame_when_no_prices(
        self,
        mock_get_constituents: MagicMock,
        mock_get_client: MagicMock,
        mock_pro_bar: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = make_constituents_frame().iloc[[0]].copy()
        mock_get_client.return_value = MagicMock()
        mock_pro_bar.return_value = pd.DataFrame()

        result = china_stock_data.get_index_member_prices(
            "000300.SH",
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), self.EXPECTED_COLUMNS)
        self.assertEqual(result.attrs["failed_tickers"], [])
        self.assertEqual(result.attrs["price_adjustment"], "qfq")
        self.assertEqual(result.attrs["index_code"], "000300.SH")
        self.assertEqual(result.attrs["universe"], "hs300")
        self.assertEqual(result.attrs["constituent_history_mode"], "latest_snapshot")

    @patch("strategies.china_stock_data.ts.pro_bar")
    @patch("strategies.china_stock_data._get_tushare_client")
    @patch("strategies.china_stock_data.get_index_constituents")
    def test_get_index_member_prices_tracks_failed_tickers(
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

        result = china_stock_data.get_index_member_prices(
            "000905.SH",
            sd="2025-01-02",
            ed="2025-01-03",
            pause_seconds=0.0,
        )

        self.assertEqual(result["ticker"].tolist(), ["AAA"])
        self.assertEqual(result.attrs["failed_tickers"], ["BBB"])
        self.assertEqual(result.attrs["price_adjustment"], "qfq")

    @patch("strategies.china_stock_data._fetch_member_prices_for_constituents")
    @patch("strategies.china_stock_data.get_all_a_share_constituents")
    def test_get_all_a_share_member_prices_delegates_to_full_universe_helpers(
        self,
        mock_get_constituents: MagicMock,
        mock_fetch_member_prices: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = pd.DataFrame(
            [{"ticker": "000001", "ts_code": "000001.SZ", "name": "AAA Corp", "trade_date": "20250331", "weight": 1.0}]
        )
        mock_fetch_member_prices.return_value = pd.DataFrame()

        china_stock_data.get_all_a_share_member_prices("2025-01-01", "2025-01-10")

        mock_get_constituents.assert_called_once_with(token=None, end_date="2025-01-10")
        mock_fetch_member_prices.assert_called_once_with(
            mock_get_constituents.return_value,
            metadata=china_stock_data.ALL_A_UNIVERSE_METADATA,
            sd="2025-01-01",
            ed="2025-01-10",
            token=None,
            pause_seconds=1.3,
            max_calls_per_minute=195,
        )

    @patch("strategies.china_stock_data.get_index_member_prices")
    @patch("strategies.china_stock_data.get_index_constituents")
    def test_index_wrappers_delegate_to_generic_helpers(
        self,
        mock_get_constituents: MagicMock,
        mock_get_member_prices: MagicMock,
    ) -> None:
        mock_get_constituents.return_value = pd.DataFrame()
        mock_get_member_prices.return_value = pd.DataFrame()

        china_stock_data.get_csi500_constituents(start_date="2025-03-01", end_date="2025-03-31")
        china_stock_data.get_hs300_constituents(start_date="2025-03-01", end_date="2025-03-31")
        china_stock_data.get_csi1000_constituents(start_date="2025-03-01", end_date="2025-03-31")
        china_stock_data.get_csi500_member_prices("2025-01-01", "2025-01-10")
        china_stock_data.get_hs300_member_prices("2025-01-01", "2025-01-10")
        china_stock_data.get_csi1000_member_prices("2025-01-01", "2025-01-10")

        self.assertEqual(
            mock_get_constituents.call_args_list,
            [
                call("000905.SH", token=None, start_date="2025-03-01", end_date="2025-03-31"),
                call("000300.SH", token=None, start_date="2025-03-01", end_date="2025-03-31"),
                call("000852.SH", token=None, start_date="2025-03-01", end_date="2025-03-31"),
            ],
        )
        self.assertEqual(
            mock_get_member_prices.call_args_list,
            [
                call(
                    "000905.SH",
                    sd="2025-01-01",
                    ed="2025-01-10",
                    token=None,
                    pause_seconds=1.3,
                    max_calls_per_minute=195,
                ),
                call(
                    "000300.SH",
                    sd="2025-01-01",
                    ed="2025-01-10",
                    token=None,
                    pause_seconds=1.3,
                    max_calls_per_minute=195,
                ),
                call(
                    "000852.SH",
                    sd="2025-01-01",
                    ed="2025-01-10",
                    token=None,
                    pause_seconds=1.3,
                    max_calls_per_minute=195,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
