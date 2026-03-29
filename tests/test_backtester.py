import unittest

import pandas as pd
import plotly.graph_objects as go

from backtester import Backtester, ExecutionCostModel, ScoringBacktester


def make_quote_frame(
    ticker: str,
    dates: pd.DatetimeIndex,
    open_values: list[float],
    *,
    close_values: list[float] | None = None,
    pre_close_values: list[float] | None = None,
) -> pd.DataFrame:
    if close_values is None:
        close_values = open_values
    if pre_close_values is None:
        pre_close_values = [close_values[0], *close_values[:-1]]

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": open_values,
            "close": close_values,
            "pre_close": pre_close_values,
        }
    )


class StaticTargetBacktester(Backtester):
    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        target_membership_map: dict[pd.Timestamp | str, set[str] | list[str]],
        **kwargs,
    ) -> None:
        self.target_membership_map = {
            pd.Timestamp(date): {str(ticker) for ticker in tickers}
            for date, tickers in target_membership_map.items()
        }
        super().__init__(stock_candle_df, **kwargs)

    def _build_target_membership_map(self) -> dict[pd.Timestamp, set[str]]:
        return self.target_membership_map


class StaticScorer:
    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        rankings_by_date: dict[pd.Timestamp | str, list[str]],
        copy: bool = True,
    ) -> None:
        self.rankings_by_date = {
            pd.Timestamp(date): [str(ticker) for ticker in tickers]
            for date, tickers in rankings_by_date.items()
        }
        self.stock_candle_df = stock_candle_df.copy(deep=True) if copy else stock_candle_df

    def add_technical_score(self, top_n: int | None = None) -> pd.DataFrame:
        df = self.stock_candle_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["technical_score"] = pd.NA
        df["technical_rank"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

        for date, tickers in self.rankings_by_date.items():
            date_mask = df["date"].eq(date)
            total = len(tickers)
            for rank, ticker in enumerate(tickers, start=1):
                mask = date_mask & df["ticker"].astype(str).eq(ticker)
                df.loc[mask, "technical_score"] = float(total - rank + 1)
                df.loc[mask, "technical_rank"] = rank

        if top_n is not None:
            df["selected_top_n"] = df["technical_rank"].le(top_n).fillna(False)

        self.stock_candle_df = df.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)
        return self.stock_candle_df


class BacktesterTest(unittest.TestCase):
    ZERO_COSTS = ExecutionCostModel(
        commission_rate=0.0,
        min_commission=0.0,
        stamp_duty_rate=0.0,
        transfer_fee_rate=0.0,
        half_spread_bps=0.0,
    )

    def test_signal_timing_and_membership_only_rebalance(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        panel = pd.concat(
            [
                make_quote_frame("AAA", dates, [100.0, 100.0, 100.0, 100.0]),
                make_quote_frame("BBB", dates, [100.0, 100.0, 100.0, 100.0]),
                make_quote_frame("CCC", dates, [100.0, 100.0, 100.0, 100.0]),
            ],
            ignore_index=True,
        )
        targets = {
            dates[0]: {"AAA", "BBB"},
            dates[1]: {"BBB", "CCC"},
            dates[2]: {"CCC"},
        }

        backtester = StaticTargetBacktester(
            panel,
            target_membership_map=targets,
            initial_capital=200.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()
        trades = results["trades"]
        holdings = results["holdings"]

        self.assertEqual(
            list(zip(trades["date"], trades["side"], trades["ticker"], trades["shares"])),
            [
                (dates[1], "buy", "AAA", 1),
                (dates[1], "buy", "BBB", 1),
                (dates[2], "sell", "AAA", 1),
                (dates[2], "buy", "CCC", 1),
                (dates[3], "sell", "BBB", 1),
            ],
        )
        bbb_day_one = holdings[(holdings["date"] == dates[1]) & (holdings["ticker"] == "BBB")]
        bbb_day_two = holdings[(holdings["date"] == dates[2]) & (holdings["ticker"] == "BBB")]
        self.assertEqual(int(bbb_day_one["shares"].iat[0]), 1)
        self.assertEqual(int(bbb_day_two["shares"].iat[0]), 1)
        self.assertTrue((trades["date"] > dates[0]).all())
        self.assertEqual(
            int(results["portfolio"].loc[results["portfolio"]["date"] == dates[2], "holdings_count"].iat[0]),
            2,
        )

    def test_limit_opens_block_buys_and_sells(self) -> None:
        buy_dates = pd.date_range("2025-01-01", periods=2, freq="B")
        buy_panel = make_quote_frame(
            "AAA",
            buy_dates,
            [10.0, 11.0],
            close_values=[10.0, 11.0],
            pre_close_values=[10.0, 10.0],
        )
        buy_backtester = StaticTargetBacktester(
            buy_panel,
            target_membership_map={buy_dates[0]: {"AAA"}},
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        buy_results = buy_backtester.compute_metrics()
        self.assertTrue(buy_results["trades"].empty)
        self.assertTrue(buy_results["holdings"].empty)

        sell_dates = pd.date_range("2025-01-01", periods=3, freq="B")
        sell_panel = make_quote_frame(
            "AAA",
            sell_dates,
            [10.0, 10.0, 9.0],
            close_values=[10.0, 10.0, 9.0],
            pre_close_values=[10.0, 10.0, 10.0],
        )
        sell_backtester = StaticTargetBacktester(
            sell_panel,
            target_membership_map={sell_dates[0]: {"AAA"}, sell_dates[1]: set()},
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        sell_results = sell_backtester.compute_metrics()

        self.assertEqual(len(sell_results["trades"]), 1)
        self.assertEqual(sell_results["trades"]["side"].tolist(), ["buy"])
        self.assertEqual(
            int(
                sell_results["portfolio"].loc[
                    sell_results["portfolio"]["date"] == sell_dates[2], "holdings_count"
                ].iat[0]
            ),
            1,
        )

    def test_missing_execution_quote_freezes_position_until_quote_returns(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        aaa = make_quote_frame(
            "AAA",
            pd.DatetimeIndex([dates[0], dates[1], dates[3]]),
            [10.0, 10.0, 12.0],
            close_values=[10.0, 12.0, 12.0],
            pre_close_values=[10.0, 10.0, 12.0],
        )
        bbb = make_quote_frame(
            "BBB",
            dates,
            [20.0, 20.0, 20.0, 20.0],
            close_values=[20.0, 20.0, 20.0, 20.0],
            pre_close_values=[20.0, 20.0, 20.0, 20.0],
        )
        panel = pd.concat([aaa, bbb], ignore_index=True)

        backtester = StaticTargetBacktester(
            panel,
            target_membership_map={dates[0]: {"AAA"}, dates[1]: set(), dates[2]: set()},
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()
        holdings = results["holdings"]
        trades = results["trades"]
        portfolio = results["portfolio"]

        self.assertEqual(trades["side"].tolist(), ["buy", "sell"])
        self.assertEqual(trades["date"].tolist(), [dates[1], dates[3]])
        frozen_row = holdings[(holdings["date"] == dates[2]) & (holdings["ticker"] == "AAA")]
        self.assertAlmostEqual(float(frozen_row["mark_price"].iat[0]), 12.0)
        self.assertAlmostEqual(
            float(portfolio.loc[portfolio["date"] == dates[2], "nav"].iat[0]),
            120.0,
        )

    def test_cost_helpers_and_board_lot_rounding(self) -> None:
        dates = pd.date_range("2025-01-01", periods=1, freq="B")
        panel = make_quote_frame("AAA", dates, [10.0], close_values=[10.0], pre_close_values=[10.0])
        backtester = StaticTargetBacktester(panel, target_membership_map={}, board_lot_size=100)

        buy_costs = backtester._calculate_trade_costs("buy", 1_000.0)
        sell_costs = backtester._calculate_trade_costs("sell", 1_000.0)
        self.assertAlmostEqual(backtester._execution_price("buy", 10.0), 10.005)
        self.assertAlmostEqual(backtester._execution_price("sell", 10.0), 9.995)
        self.assertAlmostEqual(buy_costs["commission"], 5.0)
        self.assertAlmostEqual(buy_costs["transfer_fee"], 0.01)
        self.assertAlmostEqual(buy_costs["stamp_duty"], 0.0)
        self.assertAlmostEqual(buy_costs["total_cost"], 5.01)
        self.assertAlmostEqual(sell_costs["commission"], 5.0)
        self.assertAlmostEqual(sell_costs["transfer_fee"], 0.01)
        self.assertAlmostEqual(sell_costs["stamp_duty"], 0.5)
        self.assertAlmostEqual(sell_costs["total_cost"], 5.51)

        zero_cost_backtester = StaticTargetBacktester(
            panel,
            target_membership_map={},
            board_lot_size=100,
            costs=self.ZERO_COSTS,
        )
        shares = zero_cost_backtester._max_affordable_buy_shares(pd.Series({"open": 10.0}), 1_050.0)
        self.assertEqual(shares, 100)

    def test_compute_metrics_and_show_metrics_outputs(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        panel = make_quote_frame(
            "AAA",
            dates,
            [10.0, 10.0, 11.0],
            close_values=[10.0, 11.0, 11.0],
            pre_close_values=[10.0, 10.0, 11.0],
        )
        backtester = StaticTargetBacktester(
            panel,
            target_membership_map={dates[0]: {"AAA"}, dates[1]: set()},
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()
        summary = results["summary"]
        portfolio = results["portfolio"]

        self.assertEqual(
            set(results),
            {"summary", "portfolio", "returns", "benchmark_returns", "trades", "holdings"},
        )
        self.assertTrue(
            {
                "date",
                "nav",
                "cash",
                "gross_exposure",
                "holdings_count",
                "strategy_return",
                "benchmark_return",
                "drawdown",
                "daily_turnover",
                "transaction_cost",
                "cumulative_cost",
            }.issubset(portfolio.columns)
        )
        self.assertAlmostEqual(summary["total_return"], 0.1)
        self.assertAlmostEqual(summary["max_drawdown"], 0.0)
        self.assertEqual(summary["total_trades"], 2)
        self.assertAlmostEqual(summary["total_turnover"], 2.0)
        self.assertTrue((portfolio["drawdown"] == 0.0).all())
        self.assertTrue(results["benchmark_returns"].index.equals(results["returns"].index))

        figure = backtester.show_metrics()
        self.assertIsInstance(figure, go.Figure)
        self.assertGreaterEqual(len(figure.data), 7)

    def test_scoring_backtester_uses_scorer_rankings(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        panel = pd.concat(
            [
                make_quote_frame("AAA", dates, [100.0, 100.0, 100.0, 100.0]),
                make_quote_frame("BBB", dates, [100.0, 100.0, 100.0, 100.0]),
                make_quote_frame("CCC", dates, [100.0, 100.0, 100.0, 100.0]),
            ],
            ignore_index=True,
        )
        rankings = {
            dates[0]: ["AAA", "BBB", "CCC"],
            dates[1]: ["BBB", "CCC", "AAA"],
            dates[2]: ["CCC", "AAA", "BBB"],
        }

        backtester = ScoringBacktester(
            panel,
            scorer=StaticScorer,
            scorer_kwargs={"rankings_by_date": rankings},
            top_n=2,
            initial_capital=200.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()
        trades = results["trades"]

        self.assertEqual(
            list(zip(trades["date"], trades["side"], trades["ticker"])),
            [
                (dates[1], "buy", "AAA"),
                (dates[1], "buy", "BBB"),
                (dates[2], "sell", "AAA"),
                (dates[2], "buy", "CCC"),
                (dates[3], "sell", "BBB"),
                (dates[3], "buy", "AAA"),
            ],
        )
        held_on_final_day = results["holdings"][results["holdings"]["date"] == dates[3]]
        self.assertEqual(held_on_final_day["ticker"].tolist(), ["AAA", "CCC"])
        self.assertIn("technical_score", backtester.scorer.stock_candle_df.columns)
        self.assertIn("technical_rank", backtester.scorer.stock_candle_df.columns)

    def test_scoring_backtester_can_skip_the_top_quantile_before_top_n_selection(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        panel = pd.concat(
            [
                make_quote_frame("AAA", dates, [100.0, 100.0, 100.0]),
                make_quote_frame("BBB", dates, [100.0, 100.0, 100.0]),
                make_quote_frame("CCC", dates, [100.0, 100.0, 100.0]),
            ],
            ignore_index=True,
        )
        rankings = {
            dates[0]: ["AAA", "BBB", "CCC"],
            dates[1]: ["AAA", "BBB", "CCC"],
        }

        backtester = ScoringBacktester(
            panel,
            scorer=StaticScorer,
            scorer_kwargs={"rankings_by_date": rankings},
            top_n=1,
            exclude_top_quantile=0.2,
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        results = backtester.compute_metrics()

        self.assertEqual(results["trades"]["ticker"].tolist(), ["BBB"])
        self.assertEqual(results["trades"]["side"].tolist(), ["buy"])

    def test_inspect_selection_returns_signal_trade_and_price_context(self) -> None:
        dates = pd.date_range("2025-01-01", periods=5, freq="B")
        aaa = make_quote_frame(
            "AAA",
            dates,
            [10.0, 10.5, 11.0, 11.5, 12.0],
            close_values=[10.0, 10.8, 11.2, 11.7, 12.1],
            pre_close_values=[10.0, 10.0, 10.8, 11.2, 11.7],
        )
        aaa["high"] = aaa["close"] + 0.3
        aaa["low"] = aaa["open"] - 0.3
        bbb = make_quote_frame(
            "BBB",
            dates,
            [20.0, 20.0, 20.0, 20.0, 20.0],
            close_values=[20.0, 20.0, 20.0, 20.0, 20.0],
            pre_close_values=[20.0, 20.0, 20.0, 20.0, 20.0],
        )
        bbb["high"] = bbb["close"] + 0.2
        bbb["low"] = bbb["open"] - 0.2
        panel = pd.concat([aaa, bbb], ignore_index=True)
        rankings = {
            dates[0]: ["AAA", "BBB"],
            dates[1]: ["AAA", "BBB"],
            dates[2]: ["BBB", "AAA"],
            dates[3]: ["BBB", "AAA"],
        }

        backtester = ScoringBacktester(
            panel,
            scorer=StaticScorer,
            scorer_kwargs={"rankings_by_date": rankings},
            top_n=1,
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        inspection = backtester.inspect_selection("AAA", dates[0], lookback=1, lookahead=2)

        self.assertEqual(inspection["summary"]["signal_date"], dates[0])
        self.assertEqual(inspection["summary"]["execution_date"], dates[1])
        self.assertTrue(inspection["summary"]["selected"])
        self.assertEqual(inspection["summary"]["event_type"], "buy")
        self.assertEqual(inspection["summary"]["rank"], 1)
        self.assertEqual(inspection["trade_rows"]["side"].tolist(), ["buy"])
        self.assertIn("close_return_from_signal_close", inspection["price_window"].columns)
        self.assertEqual(inspection["signal_row"]["ticker"].iat[0], "AAA")
        self.assertEqual(inspection["ranking_context"]["ticker"].tolist(), ["AAA", "BBB"])

        execution_view = backtester.inspect_selection("AAA", dates[1], date_kind="execution")
        self.assertEqual(execution_view["summary"]["signal_date"], dates[0])
        self.assertEqual(execution_view["summary"]["execution_date"], dates[1])

    def test_plot_selection_context_returns_plotly_figure(self) -> None:
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        aaa = make_quote_frame(
            "AAA",
            dates,
            [10.0, 10.2, 10.4, 10.6],
            close_values=[10.1, 10.3, 10.5, 10.7],
            pre_close_values=[10.0, 10.1, 10.3, 10.5],
        )
        aaa["high"] = aaa["close"] + 0.2
        aaa["low"] = aaa["open"] - 0.2
        bbb = make_quote_frame(
            "BBB",
            dates,
            [20.0, 20.1, 20.2, 20.3],
            close_values=[20.0, 20.1, 20.2, 20.3],
            pre_close_values=[20.0, 20.0, 20.1, 20.2],
        )
        bbb["high"] = bbb["close"] + 0.2
        bbb["low"] = bbb["open"] - 0.2
        panel = pd.concat([aaa, bbb], ignore_index=True)
        rankings = {
            dates[0]: ["AAA", "BBB"],
            dates[1]: ["AAA", "BBB"],
            dates[2]: ["BBB", "AAA"],
        }

        backtester = ScoringBacktester(
            panel,
            scorer=StaticScorer,
            scorer_kwargs={"rankings_by_date": rankings},
            top_n=1,
            initial_capital=100.0,
            board_lot_size=1,
            costs=self.ZERO_COSTS,
        )
        figure = backtester.plot_selection_context("AAA", dates[0], lookback=1, lookahead=2)

        self.assertIsInstance(figure, go.Figure)
        self.assertGreaterEqual(len(figure.data), 2)


if __name__ == "__main__":
    unittest.main()
