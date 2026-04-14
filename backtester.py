from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class ExecutionCostModel:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    half_spread_bps: float = 5.0


class Backtester(ABC):
    REQUIRED_COLUMNS = ["date", "ticker", "open", "close", "pre_close"]
    NUMERIC_COLUMNS = [
        "open",
        "close",
        "pre_close",
        "high",
        "low",
        "volume",
        "turnover",
        "weight",
        "amplitude_pct",
        "change_pct",
        "change_amount",
    ]
    STRING_COLUMNS = ["ticker", "ts_code", "name"]

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        initial_capital: float = 1_000_000.0,
        board_lot_size: int = 100,
        price_limit_pct: float = 0.10,
        benchmark: str = "equal_weight_universe",
        costs: ExecutionCostModel | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if board_lot_size < 1:
            raise ValueError("board_lot_size must be at least 1.")
        if price_limit_pct <= 0:
            raise ValueError("price_limit_pct must be positive.")

        self.stock_candle_df = self._prepare_input_frame(stock_candle_df)
        self.initial_capital = float(initial_capital)
        self.board_lot_size = int(board_lot_size)
        self.price_limit_pct = float(price_limit_pct)
        self.benchmark = benchmark
        self.costs = costs or ExecutionCostModel()

        self._quotes_by_date = self._build_quotes_by_date()
        self._market_calendar = sorted(self.stock_candle_df["date"].drop_duplicates().tolist())
        self._prev_date_map = {
            self._market_calendar[index]: self._market_calendar[index - 1]
            for index in range(1, len(self._market_calendar))
        }
        self._benchmark_returns = self._build_benchmark_returns()
        self._target_membership_cache: dict[pd.Timestamp, set[str]] | None = None
        self._simulation_cache: dict[tuple[pd.Timestamp | None, pd.Timestamp | None], dict[str, Any]] = {}

    @classmethod
    def _prepare_input_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("stock_candle_df must be a pandas DataFrame.")

        missing = [column for column in cls.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"stock_candle_df is missing required columns: {missing}")

        prepared = df.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])

        if "constituent_trade_date" in prepared.columns:
            prepared["constituent_trade_date"] = pd.to_datetime(
                prepared["constituent_trade_date"], errors="coerce"
            )

        for column in cls.STRING_COLUMNS:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("string")
        for column in cls.NUMERIC_COLUMNS:
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

        prepared = prepared.drop_duplicates(subset=["date", "ticker"], keep="last")
        return prepared.sort_values(["date", "ticker"], kind="mergesort", ignore_index=True)

    def _build_quotes_by_date(self) -> dict[pd.Timestamp, pd.DataFrame]:
        quotes: dict[pd.Timestamp, pd.DataFrame] = {}
        for date, group in self.stock_candle_df.groupby("date", sort=True):
            quotes[pd.Timestamp(date)] = group.set_index("ticker", drop=False).sort_index()
        return quotes

    def _build_benchmark_returns(self) -> pd.Series:
        if self.benchmark != "equal_weight_universe":
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")

        daily_returns = (self.stock_candle_df["close"] / self.stock_candle_df["pre_close"] - 1.0).where(
            self.stock_candle_df["close"].gt(0) & self.stock_candle_df["pre_close"].gt(0)
        )
        benchmark = (
            pd.DataFrame({"date": self.stock_candle_df["date"], "benchmark_return": daily_returns})
            .groupby("date", sort=True)["benchmark_return"]
            .mean()
            .fillna(0.0)
        )
        return benchmark.sort_index()

    def _normalize_window_key(
        self,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        start = pd.to_datetime(start_date) if start_date is not None else None
        end = pd.to_datetime(end_date) if end_date is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("start_date must be earlier than or equal to end_date.")
        return start, end

    def _get_window_dates(
        self,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> list[pd.Timestamp]:
        dates = [
            date
            for date in self._market_calendar
            if (start_date is None or date >= start_date) and (end_date is None or date <= end_date)
        ]
        if not dates:
            raise ValueError("No market dates fall inside the requested window.")
        return dates

    def _get_target_membership_map(self) -> dict[pd.Timestamp, set[str]]:
        if self._target_membership_cache is None:
            raw_targets = self._build_target_membership_map()
            self._target_membership_cache = {
                pd.Timestamp(date): {str(ticker) for ticker in tickers}
                for date, tickers in raw_targets.items()
            }
        return self._target_membership_cache

    @abstractmethod
    def _build_target_membership_map(self) -> Mapping[pd.Timestamp, set[str]]:
        raise NotImplementedError

    @staticmethod
    def _round_price(value: float) -> float:
        return round(float(value), 2)

    def _get_quote(self, date: pd.Timestamp, ticker: str) -> pd.Series | None:
        quotes = self._quotes_by_date.get(date)
        if quotes is None or ticker not in quotes.index:
            return None
        row = quotes.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    def _is_tradeable(self, quote: pd.Series | None) -> bool:
        if quote is None:
            return False

        open_price = quote.get("open")
        pre_close = quote.get("pre_close")
        if pd.isna(open_price) or pd.isna(pre_close) or open_price <= 0 or pre_close <= 0:
            return False

        limit_up = self._round_price(pre_close * (1.0 + self.price_limit_pct))
        limit_down = self._round_price(pre_close * (1.0 - self.price_limit_pct))
        rounded_open = self._round_price(open_price)
        return rounded_open not in {limit_up, limit_down}

    def _get_mark_price(
        self,
        date: pd.Timestamp,
        ticker: str,
        last_mark_prices: dict[str, float],
    ) -> float | None:
        quote = self._get_quote(date, ticker)
        if quote is not None:
            close_price = quote.get("close")
            if pd.notna(close_price) and close_price > 0:
                return float(close_price)
            open_price = quote.get("open")
            if pd.notna(open_price) and open_price > 0:
                return float(open_price)
        return last_mark_prices.get(ticker)

    def _calculate_trade_costs(self, side: str, gross_notional: float) -> dict[str, float]:
        if gross_notional <= 0:
            return {
                "commission": 0.0,
                "transfer_fee": 0.0,
                "stamp_duty": 0.0,
                "total_cost": 0.0,
            }

        commission = max(gross_notional * self.costs.commission_rate, self.costs.min_commission)
        transfer_fee = gross_notional * self.costs.transfer_fee_rate
        stamp_duty = gross_notional * self.costs.stamp_duty_rate if side == "sell" else 0.0
        total_cost = commission + transfer_fee + stamp_duty
        return {
            "commission": float(commission),
            "transfer_fee": float(transfer_fee),
            "stamp_duty": float(stamp_duty),
            "total_cost": float(total_cost),
        }

    def _execution_price(self, side: str, open_price: float) -> float:
        slippage = self.costs.half_spread_bps / 10_000.0
        multiplier = 1.0 + slippage if side == "buy" else 1.0 - slippage
        return float(open_price * multiplier)

    def _max_affordable_buy_shares(self, quote: pd.Series, budget: float) -> int:
        if budget <= 0:
            return 0

        execution_price = self._execution_price("buy", float(quote["open"]))
        max_shares = int(budget // execution_price)
        max_shares = (max_shares // self.board_lot_size) * self.board_lot_size
        while max_shares > 0:
            gross_notional = max_shares * execution_price
            total_outflow = gross_notional + self._calculate_trade_costs("buy", gross_notional)["total_cost"]
            if total_outflow <= budget + 1e-9:
                return int(max_shares)
            max_shares -= self.board_lot_size
        return 0

    def _build_trade_record(
        self,
        *,
        date: pd.Timestamp,
        signal_date: pd.Timestamp | None,
        ticker: str,
        side: str,
        shares: int,
        open_price: float,
        execution_price: float,
        gross_notional: float,
        costs: dict[str, float],
        cash_delta: float,
    ) -> dict[str, Any]:
        return {
            "date": date,
            "signal_date": signal_date,
            "ticker": ticker,
            "side": side,
            "shares": int(shares),
            "open_price": float(open_price),
            "execution_price": float(execution_price),
            "gross_notional": float(gross_notional),
            "commission": float(costs["commission"]),
            "transfer_fee": float(costs["transfer_fee"]),
            "stamp_duty": float(costs["stamp_duty"]),
            "total_cost": float(costs["total_cost"]),
            "cash_delta": float(cash_delta),
        }

    def _empty_result(
        self, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None
    ) -> dict[str, Any]:
        portfolio = pd.DataFrame(
            columns=[
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
                "strategy_nav_norm",
                "benchmark_nav_norm",
            ]
        )
        trades = pd.DataFrame(
            columns=[
                "date",
                "signal_date",
                "ticker",
                "side",
                "shares",
                "open_price",
                "execution_price",
                "gross_notional",
                "commission",
                "transfer_fee",
                "stamp_duty",
                "total_cost",
                "cash_delta",
            ]
        )
        holdings = pd.DataFrame(columns=["date", "ticker", "shares", "mark_price", "market_value", "weight"])
        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "excess_return": 0.0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": 0.0,
            "calmar": np.nan,
            "daily_win_rate": np.nan,
            "average_holdings_count": 0.0,
            "total_turnover": 0.0,
            "total_trades": 0,
            "total_transaction_costs": 0.0,
            "information_ratio": np.nan,
        }
        empty_returns = pd.Series(dtype="float64", name="strategy_return")
        empty_benchmark = pd.Series(dtype="float64", name="benchmark_return")
        return {
            "summary": summary,
            "portfolio": portfolio,
            "returns": empty_returns,
            "benchmark_returns": empty_benchmark,
            "trades": trades,
            "holdings": holdings,
        }

    def _compute_summary(
        self,
        portfolio: pd.DataFrame,
        trades: pd.DataFrame,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> dict[str, Any]:
        if portfolio.empty:
            return self._empty_result(start_date, end_date)["summary"]

        strategy_returns = portfolio["strategy_return"]
        benchmark_returns = portfolio["benchmark_return"]
        active_returns = strategy_returns - benchmark_returns
        total_return = portfolio["nav"].iat[-1] / self.initial_capital - 1.0
        benchmark_total_return = portfolio["benchmark_nav_norm"].iat[-1] - 1.0
        excess_return = total_return - benchmark_total_return

        periods = len(portfolio)
        years = periods / 252.0
        cagr = np.nan
        if years > 0 and 1.0 + total_return > 0:
            cagr = (1.0 + total_return) ** (1.0 / years) - 1.0

        volatility = strategy_returns.std(ddof=0) * np.sqrt(252.0)
        sharpe = np.nan
        if strategy_returns.std(ddof=0) > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252.0)

        downside_returns = strategy_returns[strategy_returns < 0]
        sortino = np.nan
        if len(downside_returns) > 0 and downside_returns.std(ddof=0) > 0:
            sortino = strategy_returns.mean() / downside_returns.std(ddof=0) * np.sqrt(252.0)

        max_drawdown = float(-portfolio["drawdown"].min()) if not portfolio["drawdown"].empty else 0.0
        calmar = np.nan
        if max_drawdown > 0 and pd.notna(cagr):
            calmar = cagr / max_drawdown

        information_ratio = np.nan
        if active_returns.std(ddof=0) > 0:
            information_ratio = active_returns.mean() / active_returns.std(ddof=0) * np.sqrt(252.0)

        return {
            "start_date": portfolio["date"].iat[0],
            "end_date": portfolio["date"].iat[-1],
            "total_return": float(total_return),
            "benchmark_total_return": float(benchmark_total_return),
            "excess_return": float(excess_return),
            "cagr": float(cagr) if pd.notna(cagr) else np.nan,
            "annualized_volatility": float(volatility),
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "sortino": float(sortino) if pd.notna(sortino) else np.nan,
            "max_drawdown": float(max_drawdown),
            "calmar": float(calmar) if pd.notna(calmar) else np.nan,
            "daily_win_rate": float((strategy_returns > 0).mean()) if len(strategy_returns) else np.nan,
            "average_holdings_count": float(portfolio["holdings_count"].mean()),
            "total_turnover": float(portfolio["daily_turnover"].sum()),
            "total_trades": int(len(trades)),
            "total_transaction_costs": float(portfolio["transaction_cost"].sum()),
            "information_ratio": float(information_ratio) if pd.notna(information_ratio) else np.nan,
        }

    def _run_simulation(
        self, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None
    ) -> dict[str, Any]:
        window_dates = self._get_window_dates(start_date, end_date)
        target_membership = self._get_target_membership_map()

        cash = float(self.initial_capital)
        positions: dict[str, int] = {}
        last_mark_prices: dict[str, float] = {}
        trade_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        portfolio_records: list[dict[str, Any]] = []
        prev_nav = float(self.initial_capital)
        cumulative_cost = 0.0

        for index, current_date in enumerate(window_dates):
            signal_date = self._prev_date_map.get(current_date)
            target_tickers = set(target_membership.get(signal_date, set())) if signal_date is not None else set()
            day_trade_notional = 0.0
            day_transaction_cost = 0.0

            held_tickers = {ticker for ticker, shares in positions.items() if shares > 0}
            exit_tickers = sorted(held_tickers - target_tickers)

            for ticker in exit_tickers:
                shares = positions.get(ticker, 0)
                quote = self._get_quote(current_date, ticker)
                if shares <= 0 or not self._is_tradeable(quote):
                    continue

                open_price = float(quote["open"])
                execution_price = self._execution_price("sell", open_price)
                gross_notional = shares * execution_price
                costs = self._calculate_trade_costs("sell", gross_notional)
                cash_delta = gross_notional - costs["total_cost"]
                cash += cash_delta
                day_trade_notional += gross_notional
                day_transaction_cost += costs["total_cost"]
                trade_records.append(
                    self._build_trade_record(
                        date=current_date,
                        signal_date=signal_date,
                        ticker=ticker,
                        side="sell",
                        shares=shares,
                        open_price=open_price,
                        execution_price=execution_price,
                        gross_notional=gross_notional,
                        costs=costs,
                        cash_delta=cash_delta,
                    )
                )
                positions.pop(ticker, None)

            held_after_sells = {ticker for ticker, shares in positions.items() if shares > 0}
            entrant_candidates = sorted(target_tickers - held_after_sells)
            executable_entries = [
                ticker
                for ticker in entrant_candidates
                if self._is_tradeable(self._get_quote(current_date, ticker))
            ]

            remaining_entries = executable_entries.copy()
            for ticker in executable_entries:
                quote = self._get_quote(current_date, ticker)
                if quote is None:
                    remaining_entries.pop(0)
                    continue

                names_left = len(remaining_entries)
                budget = cash / names_left if names_left > 0 else 0.0
                shares = self._max_affordable_buy_shares(quote, budget)
                if shares > 0:
                    open_price = float(quote["open"])
                    execution_price = self._execution_price("buy", open_price)
                    gross_notional = shares * execution_price
                    costs = self._calculate_trade_costs("buy", gross_notional)
                    cash_delta = -(gross_notional + costs["total_cost"])
                    if cash + cash_delta >= -1e-9:
                        cash += cash_delta
                        day_trade_notional += gross_notional
                        day_transaction_cost += costs["total_cost"]
                        positions[ticker] = positions.get(ticker, 0) + shares
                        trade_records.append(
                            self._build_trade_record(
                                date=current_date,
                                signal_date=signal_date,
                                ticker=ticker,
                                side="buy",
                                shares=shares,
                                open_price=open_price,
                                execution_price=execution_price,
                                gross_notional=gross_notional,
                                costs=costs,
                                cash_delta=cash_delta,
                            )
                        )
                remaining_entries.pop(0)

            day_holdings: list[dict[str, Any]] = []
            gross_exposure = 0.0
            for ticker, shares in sorted(positions.items()):
                if shares <= 0:
                    continue
                mark_price = self._get_mark_price(current_date, ticker, last_mark_prices)
                if mark_price is None:
                    continue
                last_mark_prices[ticker] = float(mark_price)
                market_value = shares * float(mark_price)
                gross_exposure += market_value
                day_holdings.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "shares": int(shares),
                        "mark_price": float(mark_price),
                        "market_value": float(market_value),
                    }
                )

            nav = cash + gross_exposure
            strategy_return = nav / prev_nav - 1.0 if prev_nav > 0 else 0.0
            benchmark_return = float(self._benchmark_returns.get(current_date, 0.0))
            if index == 0:
                benchmark_return = 0.0

            cumulative_cost += day_transaction_cost
            daily_turnover = day_trade_notional / prev_nav if prev_nav > 0 else 0.0

            portfolio_records.append(
                {
                    "date": current_date,
                    "nav": float(nav),
                    "cash": float(cash),
                    "gross_exposure": float(gross_exposure),
                    "holdings_count": int(len(day_holdings)),
                    "strategy_return": float(strategy_return),
                    "benchmark_return": float(benchmark_return),
                    "daily_turnover": float(daily_turnover),
                    "transaction_cost": float(day_transaction_cost),
                    "cumulative_cost": float(cumulative_cost),
                }
            )

            for holding in day_holdings:
                holding["weight"] = holding["market_value"] / nav if nav > 0 else 0.0
            holding_records.extend(day_holdings)
            prev_nav = nav

        portfolio = pd.DataFrame(portfolio_records)
        if portfolio.empty:
            return self._empty_result(start_date, end_date)

        portfolio["strategy_nav_norm"] = portfolio["nav"] / self.initial_capital
        portfolio["benchmark_nav_norm"] = (1.0 + portfolio["benchmark_return"]).cumprod()
        portfolio["drawdown"] = portfolio["nav"] / portfolio["nav"].cummax() - 1.0
        trades = pd.DataFrame(trade_records)
        holdings = pd.DataFrame(holding_records)
        returns = portfolio.set_index("date")["strategy_return"].rename("strategy_return")
        benchmark_returns = portfolio.set_index("date")["benchmark_return"].rename("benchmark_return")
        summary = self._compute_summary(portfolio, trades, start_date, end_date)
        return {
            "summary": summary,
            "portfolio": portfolio,
            "returns": returns,
            "benchmark_returns": benchmark_returns,
            "trades": trades,
            "holdings": holdings,
        }

    def compute_metrics(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        window_key = self._normalize_window_key(start_date, end_date)
        if window_key not in self._simulation_cache:
            self._simulation_cache[window_key] = self._run_simulation(*window_key)

        cached = self._simulation_cache[window_key]
        return {
            "summary": dict(cached["summary"]),
            "portfolio": cached["portfolio"].copy(),
            "returns": cached["returns"].copy(),
            "benchmark_returns": cached["benchmark_returns"].copy(),
            "trades": cached["trades"].copy(),
            "holdings": cached["holdings"].copy(),
        }

    def show_metrics(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> go.Figure:
        results = self.compute_metrics(start_date=start_date, end_date=end_date)
        portfolio = results["portfolio"]
        if portfolio.empty:
            return go.Figure()

        rolling_sharpe = (
            portfolio["strategy_return"].rolling(63, min_periods=20).mean()
            / portfolio["strategy_return"].rolling(63, min_periods=20).std(ddof=0)
        ) * np.sqrt(252.0)

        figure = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Strategy vs Benchmark",
                "Drawdown",
                "Rolling 63-Day Sharpe",
                "Holdings / Turnover / Costs",
            ),
            specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        )

        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["strategy_nav_norm"],
                mode="lines",
                name="Strategy NAV",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["benchmark_nav_norm"],
                mode="lines",
                name="Benchmark NAV",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=portfolio["date"], y=portfolio["drawdown"], mode="lines", name="Drawdown"),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=rolling_sharpe,
                mode="lines",
                name="Rolling Sharpe (63D)",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["holdings_count"],
                mode="lines",
                name="Holdings Count",
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Bar(
                x=portfolio["date"],
                y=portfolio["daily_turnover"],
                name="Daily Turnover",
                opacity=0.45,
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["cumulative_cost"],
                mode="lines",
                name="Cumulative Cost",
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

        figure.update_yaxes(title_text="Normalized NAV", row=1, col=1)
        figure.update_yaxes(title_text="Drawdown", row=2, col=1)
        figure.update_yaxes(title_text="Sharpe", row=3, col=1)
        figure.update_yaxes(title_text="Holdings / Turnover", row=4, col=1, secondary_y=False)
        figure.update_yaxes(title_text="Cumulative Cost", row=4, col=1, secondary_y=True)
        figure.update_layout(
            height=1100,
            width=1200,
            template="plotly_white",
            hovermode="x unified",
            title="Backtest Metrics",
        )
        return figure


class ScoringBacktester(Backtester):
    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        scorer: Any,
        top_n: int,
        exclude_top_quantile: float = 0.0,
        score_column: str = "technical_score",
        rank_column: str = "technical_rank",
        scorer_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if exclude_top_quantile < 0 or exclude_top_quantile >= 1:
            raise ValueError("exclude_top_quantile must be between 0 and 1.")
        if not score_column:
            raise ValueError("score_column must be a non-empty string.")
        if not rank_column:
            raise ValueError("rank_column must be a non-empty string.")

        self.top_n = int(top_n)
        self.exclude_top_quantile = float(exclude_top_quantile)
        self.score_column = str(score_column)
        self.rank_column = str(rank_column)
        self.scorer_kwargs = dict(scorer_kwargs or {})
        self.scorer = self._initialize_scorer(stock_candle_df, scorer)
        self._ensure_score_columns()

        super().__init__(self.scorer.stock_candle_df, **kwargs)

    @classmethod
    def _normalize_date_ticker_pairs(cls, df: pd.DataFrame) -> pd.DataFrame:
        prepared = cls._prepare_input_frame(df)
        return prepared.loc[:, ["date", "ticker"]].reset_index(drop=True)

    def _initialize_scorer(self, stock_candle_df: pd.DataFrame, scorer: Any) -> Any:
        if isinstance(scorer, type):
            scorer_instance = scorer(stock_candle_df.copy(deep=True), **self.scorer_kwargs)
        else:
            scorer_instance = scorer

        scorer_frame = getattr(scorer_instance, "stock_candle_df", None)
        if not isinstance(scorer_frame, pd.DataFrame):
            raise TypeError("scorer must expose a pandas DataFrame on stock_candle_df.")

        expected_pairs = self._normalize_date_ticker_pairs(stock_candle_df)
        actual_pairs = self._normalize_date_ticker_pairs(scorer_frame)
        if not expected_pairs.equals(actual_pairs):
            raise ValueError(
                "scorer.stock_candle_df must align with the provided stock_candle_df on date/ticker rows."
            )

        return scorer_instance

    def _ensure_score_columns(self) -> None:
        scorer_frame = getattr(self.scorer, "stock_candle_df", None)
        if not isinstance(scorer_frame, pd.DataFrame):
            raise TypeError("scorer must expose a pandas DataFrame on stock_candle_df.")

        missing = [
            column
            for column in (self.score_column, self.rank_column)
            if column not in scorer_frame.columns
        ]
        if missing:
            scoring_method = getattr(self.scorer, "add_technical_score", None)
            if not callable(scoring_method):
                raise ValueError(
                    "scorer must provide score/rank columns or implement add_technical_score(top_n=...)."
                )
            scoring_method(top_n=self.top_n)
            scorer_frame = getattr(self.scorer, "stock_candle_df", None)

        missing_after_score = [
            column
            for column in (self.score_column, self.rank_column)
            if column not in scorer_frame.columns
        ]
        if missing_after_score:
            raise ValueError(
                f"scorer did not populate required score columns: {missing_after_score}"
            )

    def _build_target_membership_map(self) -> Mapping[pd.Timestamp, set[str]]:
        scored = self.stock_candle_df.copy()
        valid_mask = scored[self.score_column].notna() & scored[self.rank_column].notna()
        eligible = scored.loc[valid_mask].copy()
        if eligible.empty:
            return {}

        targets: dict[pd.Timestamp, set[str]] = {}
        for date, group in eligible.groupby("date", sort=True):
            ranked_group = group.sort_values(
                [self.score_column, self.rank_column, "ticker"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            if self.exclude_top_quantile > 0:
                excluded_count = int(np.ceil(len(ranked_group) * self.exclude_top_quantile))
                ranked_group = ranked_group.iloc[excluded_count:].reset_index(drop=True)
            selected = ranked_group.head(self.top_n)
            targets[pd.Timestamp(date)] = set(selected["ticker"].astype(str))

        return targets

    def _component_columns(self) -> list[str]:
        scorer_components = getattr(self.scorer, "COMPONENT_FACTORS", None)
        if isinstance(scorer_components, Mapping):
            return [str(column) for column in scorer_components.keys() if column in self.stock_candle_df.columns]
        return [
            column
            for column in self.stock_candle_df.columns
            if column.endswith("_score") and column != self.score_column
        ]

    def _feature_columns(self) -> list[str]:
        scorer_features = getattr(self.scorer, "FEATURE_COLUMNS", None)
        if scorer_features is None:
            return []
        return [str(column) for column in scorer_features if column in self.stock_candle_df.columns]

    def _resolve_signal_date(self, date: str | pd.Timestamp, *, date_kind: str) -> pd.Timestamp:
        normalized_date = pd.to_datetime(date)
        if date_kind == "signal":
            signal_date = normalized_date
        elif date_kind == "execution":
            if normalized_date not in self._prev_date_map:
                raise ValueError("execution date must have a prior market date in the dataset.")
            signal_date = self._prev_date_map[normalized_date]
        else:
            raise ValueError("date_kind must be either 'signal' or 'execution'.")

        if signal_date not in self._quotes_by_date:
            raise ValueError("The requested date is not present in the market calendar.")
        return pd.Timestamp(signal_date)

    def _get_execution_date(self, signal_date: pd.Timestamp) -> pd.Timestamp | None:
        calendar = pd.Index(self._market_calendar)
        if signal_date not in calendar:
            return None

        signal_position = int(calendar.get_loc(signal_date))
        if signal_position + 1 >= len(calendar):
            return None
        return pd.Timestamp(calendar[signal_position + 1])

    @staticmethod
    def _event_label(
        *,
        selected: bool,
        trade_rows: pd.DataFrame,
        shares_before: int,
        shares_after: int,
        execution_tradeable: bool,
    ) -> str:
        if not trade_rows.empty:
            sides = trade_rows["side"].astype(str).tolist()
            if len(sides) == 1:
                return sides[0]
            return "+".join(sides)
        if selected and shares_before > 0 and shares_after > 0:
            return "keep"
        if selected and shares_before == 0 and shares_after == 0:
            return "blocked_entry" if not execution_tradeable else "selected_no_fill"
        if (not selected) and shares_before > 0 and shares_after > 0:
            return "blocked_exit" if not execution_tradeable else "held_without_signal"
        return "not_selected"

    def inspect_selection(
        self,
        ticker: str,
        date: str | pd.Timestamp,
        *,
        date_kind: str = "signal",
        lookback: int = 20,
        lookahead: int = 5,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        if lookback < 0 or lookahead < 0:
            raise ValueError("lookback and lookahead must be non-negative.")

        ticker = str(ticker)
        signal_date = self._resolve_signal_date(date, date_kind=date_kind)
        execution_date = self._get_execution_date(signal_date)

        scored = self.stock_candle_df.copy()
        ticker_frame = (
            scored.loc[scored["ticker"].astype(str).eq(ticker)]
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True)
        )
        if ticker_frame.empty:
            raise ValueError(f"Ticker '{ticker}' is not present in the backtest universe.")

        signal_row = ticker_frame.loc[ticker_frame["date"].eq(signal_date)].copy()
        if signal_row.empty:
            raise ValueError(f"Ticker '{ticker}' does not have data on {signal_date.date()}.")

        results = self.compute_metrics(start_date=start_date, end_date=end_date)
        trades = results["trades"]
        holdings = results["holdings"]
        portfolio = results["portfolio"]
        target_membership = self._get_target_membership_map()

        ranking_columns = [
            column
            for column in [
                "date",
                "ticker",
                "name",
                self.rank_column,
                self.score_column,
                *self._component_columns(),
                "selected_top_n",
            ]
            if column in scored.columns
        ]
        ranking_context = (
            scored.loc[scored["date"].eq(signal_date) & scored[self.score_column].notna(), ranking_columns]
            .sort_values([self.rank_column, self.score_column, "ticker"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        if not ranking_context.empty:
            ranking_context["selected_by_strategy"] = ranking_context["ticker"].astype(str).isin(
                target_membership.get(signal_date, set())
            )

        trade_rows = pd.DataFrame(columns=trades.columns)
        if not trades.empty and execution_date is not None:
            trade_rows = trades.loc[
                trades["ticker"].astype(str).eq(ticker)
                & trades["signal_date"].eq(signal_date)
                & trades["date"].eq(execution_date)
            ].copy()

        shares_before = 0
        shares_after = 0
        if not holdings.empty:
            before_rows = holdings.loc[
                holdings["ticker"].astype(str).eq(ticker) & holdings["date"].eq(signal_date)
            ]
            after_rows = holdings.loc[
                holdings["ticker"].astype(str).eq(ticker)
                & holdings["date"].eq(execution_date if execution_date is not None else signal_date)
            ]
            if not before_rows.empty:
                shares_before = int(before_rows["shares"].sum())
            if not after_rows.empty:
                shares_after = int(after_rows["shares"].sum())

        execution_quote = self._get_quote(execution_date, ticker) if execution_date is not None else None
        execution_tradeable = bool(self._is_tradeable(execution_quote))
        signal_selected = ticker in target_membership.get(signal_date, set())
        event_type = self._event_label(
            selected=signal_selected,
            trade_rows=trade_rows,
            shares_before=shares_before,
            shares_after=shares_after,
            execution_tradeable=execution_tradeable,
        )

        signal_position = int(ticker_frame.index[ticker_frame["date"].eq(signal_date)][0])
        execution_positions = ticker_frame.index[ticker_frame["date"].eq(execution_date)] if execution_date is not None else []
        execution_position = int(execution_positions[0]) if len(execution_positions) else signal_position
        window_start = max(0, signal_position - lookback)
        window_end = min(len(ticker_frame), execution_position + lookahead + 1)
        price_window = ticker_frame.iloc[window_start:window_end].copy().reset_index(drop=True)
        signal_close = float(signal_row["close"].iat[0]) if "close" in signal_row.columns else np.nan
        execution_open = (
            float(execution_quote["open"])
            if execution_quote is not None and pd.notna(execution_quote.get("open"))
            else np.nan
        )
        if pd.notna(signal_close) and signal_close > 0 and "close" in price_window.columns:
            price_window["close_return_from_signal_close"] = price_window["close"] / signal_close - 1.0
        if pd.notna(execution_open) and execution_open > 0 and "close" in price_window.columns:
            price_window["close_return_from_execution_open"] = price_window["close"] / execution_open - 1.0
        if "open" in price_window.columns and "close" in price_window.columns:
            price_window["intraday_return"] = (
                price_window["close"] / price_window["open"] - 1.0
            ).where(price_window["open"].gt(0))

        future_outcomes: dict[str, float] = {}
        if pd.notna(execution_open) and execution_open > 0 and "close" in ticker_frame.columns:
            for horizon in (1, 3, 5):
                future_index = execution_position + horizon
                if future_index < len(ticker_frame):
                    future_close = ticker_frame["close"].iat[future_index]
                    if pd.notna(future_close) and future_close > 0:
                        future_outcomes[f"close_return_{horizon}d_from_execution_open"] = float(
                            future_close / execution_open - 1.0
                        )

        signal_columns = [
            column
            for column in [
                "date",
                "ticker",
                "ts_code",
                "name",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
                self.rank_column,
                self.score_column,
                *self._component_columns(),
                *self._feature_columns(),
                "selected_top_n",
            ]
            if column in signal_row.columns
        ]
        signal_details = signal_row.loc[:, signal_columns].reset_index(drop=True)
        portfolio_window = portfolio.loc[
            portfolio["date"].between(signal_date, execution_date if execution_date is not None else signal_date)
        ].copy()

        summary = {
            "ticker": ticker,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selected": bool(signal_selected),
            "event_type": event_type,
            "exclude_top_quantile": float(self.exclude_top_quantile),
            "score": float(signal_row[self.score_column].iat[0])
            if self.score_column in signal_row.columns and pd.notna(signal_row[self.score_column].iat[0])
            else np.nan,
            "rank": int(signal_row[self.rank_column].iat[0])
            if self.rank_column in signal_row.columns and pd.notna(signal_row[self.rank_column].iat[0])
            else None,
            "shares_before_execution": int(shares_before),
            "shares_after_execution": int(shares_after),
            "execution_tradeable": execution_tradeable,
            "execution_quote_available": execution_quote is not None,
            "trade_count": int(len(trade_rows)),
        }
        summary.update(future_outcomes)

        return {
            "summary": summary,
            "signal_row": signal_details,
            "ranking_context": ranking_context,
            "trade_rows": trade_rows.reset_index(drop=True),
            "price_window": price_window,
            "portfolio_window": portfolio_window.reset_index(drop=True),
        }

    def plot_selection_context(
        self,
        ticker: str,
        date: str | pd.Timestamp,
        *,
        date_kind: str = "signal",
        lookback: int = 20,
        lookahead: int = 5,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> go.Figure:
        inspection = self.inspect_selection(
            ticker,
            date,
            date_kind=date_kind,
            lookback=lookback,
            lookahead=lookahead,
            start_date=start_date,
            end_date=end_date,
        )
        price_window = inspection["price_window"]
        signal_row = inspection["signal_row"]
        if price_window.empty or signal_row.empty:
            return go.Figure()

        component_columns = [
            column for column in self._component_columns() if column in signal_row.columns
        ]
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Price Context", "Signal Components"),
            row_heights=[0.7, 0.3],
        )

        if {"open", "high", "low", "close"}.issubset(price_window.columns):
            figure.add_trace(
                go.Candlestick(
                    x=price_window["date"],
                    open=price_window["open"],
                    high=price_window["high"],
                    low=price_window["low"],
                    close=price_window["close"],
                    name=str(ticker),
                ),
                row=1,
                col=1,
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=price_window["date"],
                    y=price_window["close"],
                    mode="lines+markers",
                    name=str(ticker),
                ),
                row=1,
                col=1,
            )

        summary = inspection["summary"]
        signal_date = summary["signal_date"]
        execution_date = summary["execution_date"]
        figure.add_vline(x=signal_date, line_dash="dash", line_color="royalblue", row=1, col=1)
        if execution_date is not None:
            figure.add_vline(x=execution_date, line_dash="dot", line_color="firebrick", row=1, col=1)

        trade_rows = inspection["trade_rows"]
        if not trade_rows.empty:
            figure.add_trace(
                go.Scatter(
                    x=trade_rows["date"],
                    y=trade_rows["execution_price"],
                    mode="markers+text",
                    text=trade_rows["side"].str.upper(),
                    textposition="top center",
                    marker=dict(size=11, symbol="diamond", color="black"),
                    name="Trades",
                ),
                row=1,
                col=1,
            )

        if component_columns:
            component_values = signal_row.iloc[0][component_columns].astype(float)
            figure.add_trace(
                go.Bar(
                    x=component_columns,
                    y=component_values,
                    text=[f"{value:.2f}" for value in component_values],
                    textposition="outside",
                    name="Component Scores",
                ),
                row=2,
                col=1,
            )

        title = (
            f"{ticker} | signal {signal_date.date()} | execution "
            f"{execution_date.date() if execution_date is not None else 'N/A'} | "
            f"event {summary['event_type']} | rank {summary['rank']} | score {summary['score']:.2f}"
            if pd.notna(summary["score"])
            else f"{ticker} | signal {signal_date.date()} | event {summary['event_type']}"
        )
        figure.update_layout(
            height=850,
            width=1200,
            template="plotly_white",
            hovermode="x unified",
            title=title,
            xaxis_rangeslider_visible=False,
        )
        figure.update_yaxes(title_text="Price", row=1, col=1)
        figure.update_yaxes(title_text="Score", row=2, col=1)
        return figure


class TradePlanBacktester(Backtester):
    """
    Backtest a precomputed trade plan against daily candle data.

    This adapter is designed for research flows such as
    ``BlueChipRangeReversionResearcher.add_trade_df()`` where:
    - the researcher already decided which trades should exist
    - entries are intended for one specific execution date
    - exits may need to be retried if the next open is not tradeable

    The key difference from ``ScoringBacktester`` is that we do not rebuild a
    daily target portfolio from rankings. Instead, we take the trade plan as
    the source of truth and simulate whether those planned entries/exits could
    actually be executed under cash, lot-size, spread, fee, and limit-open
    constraints.
    """

    TRADE_REQUIRED_COLUMNS = ["signal_date", "ticker", "entry_date", "exit_date", "exit_reason"]
    TRADE_DATE_COLUMNS = ["signal_date", "entry_date", "exit_signal_date", "exit_date"]
    TRADE_STRING_COLUMNS = ["ticker", "ts_code", "name", "exit_reason", "trade_status"]
    ACTUAL_TRADE_COLUMNS = [
        "planned_trade_id",
        "entry_order_status",
        "actual_entry_date",
        "actual_entry_open_price",
        "actual_entry_execution_price",
        "target_entry_budget",
        "actual_entry_shares",
        "actual_entry_gross_notional",
        "actual_entry_total_cost",
        "exit_order_status",
        "actual_exit_date",
        "actual_exit_open_price",
        "actual_exit_execution_price",
        "actual_exit_gross_notional",
        "actual_exit_total_cost",
        "actual_exit_reason",
        "actual_trade_status",
        "actual_holding_days",
        "actual_mark_price",
        "actual_pnl",
        "actual_pnl_pct",
    ]

    def __init__(
        self,
        stock_candle_df: pd.DataFrame,
        *,
        trade_df: pd.DataFrame | None = None,
        researcher: Any | None = None,
        researcher_kwargs: Mapping[str, Any] | None = None,
        fixed_entry_notional: float | None = None,
        **kwargs: Any,
    ) -> None:
        if trade_df is None and researcher is None:
            raise ValueError("Provide either trade_df or researcher.")
        if fixed_entry_notional is not None and fixed_entry_notional <= 0:
            raise ValueError("fixed_entry_notional must be positive when provided.")

        self.researcher_kwargs = dict(researcher_kwargs or {})
        self.fixed_entry_notional = (
            float(fixed_entry_notional) if fixed_entry_notional is not None else None
        )
        self.researcher = self._initialize_researcher(stock_candle_df, researcher) if researcher is not None else None

        source_trade_df = trade_df
        if source_trade_df is None:
            build_trade_df = getattr(self.researcher, "add_trade_df", None)
            if not callable(build_trade_df):
                raise ValueError("researcher must implement add_trade_df().")
            source_trade_df = build_trade_df()

        self.planned_trade_df = self._prepare_trade_df(source_trade_df)
        backtest_frame = (
            self.researcher.stock_candle_df
            if self.researcher is not None and isinstance(getattr(self.researcher, "stock_candle_df", None), pd.DataFrame)
            else stock_candle_df
        )
        super().__init__(backtest_frame, **kwargs)

    @classmethod
    def _normalize_date_ticker_pairs(cls, df: pd.DataFrame) -> pd.DataFrame:
        prepared = cls._prepare_input_frame(df)
        return prepared.loc[:, ["date", "ticker"]].reset_index(drop=True)

    def _initialize_researcher(self, stock_candle_df: pd.DataFrame, researcher: Any) -> Any:
        if isinstance(researcher, type):
            researcher_instance = researcher(stock_candle_df.copy(deep=True), **self.researcher_kwargs)
        else:
            researcher_instance = researcher

        researcher_frame = getattr(researcher_instance, "stock_candle_df", None)
        if not isinstance(researcher_frame, pd.DataFrame):
            raise TypeError("researcher must expose a pandas DataFrame on stock_candle_df.")

        expected_pairs = self._normalize_date_ticker_pairs(stock_candle_df)
        actual_pairs = self._normalize_date_ticker_pairs(researcher_frame)
        if not expected_pairs.equals(actual_pairs):
            raise ValueError(
                "researcher.stock_candle_df must align with the provided stock_candle_df on date/ticker rows."
            )

        return researcher_instance

    @classmethod
    def _prepare_trade_df(cls, trade_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(trade_df, pd.DataFrame):
            raise TypeError("trade_df must be a pandas DataFrame.")

        missing = [column for column in cls.TRADE_REQUIRED_COLUMNS if column not in trade_df.columns]
        if missing:
            raise ValueError(f"trade_df is missing required columns: {missing}")

        prepared = trade_df.copy()
        for column in cls.TRADE_DATE_COLUMNS:
            if column in prepared.columns:
                prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
        for column in cls.TRADE_STRING_COLUMNS:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("string")

        prepared = prepared.drop_duplicates(subset=["ticker", "signal_date", "entry_date"], keep="last")
        if prepared["entry_date"].isna().any():
            raise ValueError("trade_df must have a non-null entry_date for every planned trade.")

        prepared = prepared.sort_values(
            ["ticker", "entry_date", "signal_date"],
            kind="mergesort",
            ignore_index=True,
        )
        cls._validate_non_overlapping_trade_plan(prepared)

        if "planned_trade_id" in prepared.columns:
            prepared = prepared.drop(columns=["planned_trade_id"])
        prepared.insert(0, "planned_trade_id", np.arange(len(prepared), dtype=int))
        return prepared.sort_values(
            ["entry_date", "ticker", "signal_date"],
            kind="mergesort",
            ignore_index=True,
        )

    @staticmethod
    def _validate_non_overlapping_trade_plan(trade_df: pd.DataFrame) -> None:
        for ticker, group in trade_df.groupby("ticker", sort=False):
            ordered = group.sort_values(["entry_date", "signal_date"], kind="mergesort").reset_index(drop=True)
            for index in range(1, len(ordered)):
                previous_exit = ordered.loc[index - 1, "exit_date"]
                current_entry = ordered.loc[index, "entry_date"]
                if pd.isna(previous_exit):
                    raise ValueError(
                        f"trade_df contains overlapping or unbounded trades for ticker '{ticker}'."
                    )
                if current_entry < previous_exit:
                    raise ValueError(
                        f"trade_df contains overlapping planned trades for ticker '{ticker}'."
                    )

    def _build_target_membership_map(self) -> Mapping[pd.Timestamp, set[str]]:
        return {}

    @staticmethod
    def _initialize_strategy_trades(trade_plan: pd.DataFrame) -> pd.DataFrame:
        strategy_trades = trade_plan.copy()
        strategy_trades["entry_order_status"] = "not_attempted"
        strategy_trades["actual_entry_date"] = pd.NaT
        strategy_trades["actual_entry_open_price"] = np.nan
        strategy_trades["actual_entry_execution_price"] = np.nan
        strategy_trades["target_entry_budget"] = np.nan
        strategy_trades["actual_entry_shares"] = 0
        strategy_trades["actual_entry_gross_notional"] = np.nan
        strategy_trades["actual_entry_total_cost"] = 0.0
        strategy_trades["exit_order_status"] = "not_triggered"
        strategy_trades["actual_exit_date"] = pd.NaT
        strategy_trades["actual_exit_open_price"] = np.nan
        strategy_trades["actual_exit_execution_price"] = np.nan
        strategy_trades["actual_exit_gross_notional"] = np.nan
        strategy_trades["actual_exit_total_cost"] = 0.0
        strategy_trades["actual_exit_reason"] = pd.Series(pd.NA, index=strategy_trades.index, dtype="string")
        strategy_trades["actual_trade_status"] = "not_entered"
        strategy_trades["actual_holding_days"] = pd.Series(pd.NA, index=strategy_trades.index, dtype="Int64")
        strategy_trades["actual_mark_price"] = np.nan
        strategy_trades["actual_pnl"] = np.nan
        strategy_trades["actual_pnl_pct"] = np.nan
        return strategy_trades.set_index("planned_trade_id", drop=False)

    def _entry_budget(self, cash: float, remaining_entry_count: int) -> tuple[float, float]:
        if self.fixed_entry_notional is not None:
            raw_budget = float(self.fixed_entry_notional)
            return raw_budget, min(float(cash), raw_budget)

        if remaining_entry_count <= 0:
            return 0.0, 0.0

        raw_budget = float(cash) / float(remaining_entry_count)
        return raw_budget, raw_budget

    def _get_window_trade_plan(
        self,
        start_date: pd.Timestamp | None,
        end_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        trade_plan = self.planned_trade_df.copy()
        if start_date is not None:
            trade_plan = trade_plan.loc[trade_plan["entry_date"].ge(start_date)]
        if end_date is not None:
            trade_plan = trade_plan.loc[trade_plan["entry_date"].le(end_date)]
        return trade_plan.sort_values(
            ["entry_date", "ticker", "signal_date"],
            kind="mergesort",
            ignore_index=True,
        )

    @staticmethod
    def _status_for_untradeable_entry(quote: pd.Series | None) -> str:
        if quote is None:
            return "blocked_no_quote"
        return "blocked_limit"

    @staticmethod
    def _status_for_untradeable_exit(quote: pd.Series | None) -> str:
        if quote is None:
            return "pending_no_quote"
        return "pending_limit"

    def _date_distance(self, start: pd.Timestamp, end: pd.Timestamp) -> int | None:
        if start not in self._market_calendar or end not in self._market_calendar:
            return None
        return int(self._market_calendar.index(end) - self._market_calendar.index(start))

    def _update_closed_trade_metrics(self, strategy_trades: pd.DataFrame, trade_id: int) -> None:
        shares = int(strategy_trades.at[trade_id, "actual_entry_shares"])
        entry_execution_price = strategy_trades.at[trade_id, "actual_entry_execution_price"]
        exit_execution_price = strategy_trades.at[trade_id, "actual_exit_execution_price"]
        entry_cost = strategy_trades.at[trade_id, "actual_entry_total_cost"]
        exit_cost = strategy_trades.at[trade_id, "actual_exit_total_cost"]
        if shares <= 0 or pd.isna(entry_execution_price) or pd.isna(exit_execution_price):
            return

        entry_notional = float(entry_execution_price) * shares
        exit_notional = float(exit_execution_price) * shares
        net_pnl = exit_notional - exit_cost - entry_notional - entry_cost
        entry_outflow = entry_notional + entry_cost
        strategy_trades.at[trade_id, "actual_pnl"] = float(net_pnl)
        strategy_trades.at[trade_id, "actual_pnl_pct"] = (
            float(net_pnl / entry_outflow) if entry_outflow > 0 else np.nan
        )

    def _finalize_open_trade_metrics(
        self,
        strategy_trades: pd.DataFrame,
        positions: dict[str, dict[str, Any]],
        last_mark_prices: dict[str, float],
        window_dates: list[pd.Timestamp],
    ) -> None:
        if not window_dates:
            return

        final_date = window_dates[-1]
        for ticker, position in positions.items():
            trade_id = int(position["planned_trade_id"])
            shares = int(position["shares"])
            if shares <= 0:
                continue

            mark_price = last_mark_prices.get(ticker)
            if mark_price is None:
                mark_price = self._get_mark_price(final_date, ticker, last_mark_prices)
            if mark_price is not None:
                strategy_trades.at[trade_id, "actual_mark_price"] = float(mark_price)

            entry_date = strategy_trades.at[trade_id, "actual_entry_date"]
            if pd.notna(entry_date):
                holding_days = self._date_distance(pd.Timestamp(entry_date), final_date)
                if holding_days is not None:
                    strategy_trades.at[trade_id, "actual_holding_days"] = holding_days

            if pd.notna(mark_price):
                entry_execution_price = strategy_trades.at[trade_id, "actual_entry_execution_price"]
                entry_cost = strategy_trades.at[trade_id, "actual_entry_total_cost"]
                if pd.notna(entry_execution_price):
                    entry_notional = float(entry_execution_price) * shares
                    mtm_value = float(mark_price) * shares
                    net_pnl = mtm_value - entry_notional - entry_cost
                    entry_outflow = entry_notional + entry_cost
                    strategy_trades.at[trade_id, "actual_pnl"] = float(net_pnl)
                    strategy_trades.at[trade_id, "actual_pnl_pct"] = (
                        float(net_pnl / entry_outflow) if entry_outflow > 0 else np.nan
                    )

            if strategy_trades.at[trade_id, "exit_order_status"] == "not_triggered":
                strategy_trades.at[trade_id, "exit_order_status"] = "open_position"
            if pd.isna(strategy_trades.at[trade_id, "actual_exit_reason"]):
                planned_reason = strategy_trades.at[trade_id, "exit_reason"]
                if pd.isna(planned_reason):
                    planned_reason = "open_position"
                strategy_trades.at[trade_id, "actual_exit_reason"] = str(planned_reason)
            strategy_trades.at[trade_id, "actual_trade_status"] = "open"

    @staticmethod
    def _compute_trade_summary(strategy_trades: pd.DataFrame) -> dict[str, Any]:
        planned_trade_count = int(len(strategy_trades))
        entered_mask = strategy_trades["entry_order_status"].eq("executed") if planned_trade_count else pd.Series(dtype=bool)
        closed_mask = strategy_trades["actual_trade_status"].eq("closed") if planned_trade_count else pd.Series(dtype=bool)
        open_mask = strategy_trades["actual_trade_status"].eq("open") if planned_trade_count else pd.Series(dtype=bool)

        closed_returns = (
            strategy_trades.loc[closed_mask, "actual_pnl_pct"].astype(float).dropna() if planned_trade_count else pd.Series(dtype=float)
        )
        closed_pnl = (
            strategy_trades.loc[closed_mask, "actual_pnl"].astype(float).dropna() if planned_trade_count else pd.Series(dtype=float)
        )
        gross_profit = float(closed_pnl[closed_pnl.gt(0)].sum()) if not closed_pnl.empty else 0.0
        gross_loss = float(-closed_pnl[closed_pnl.lt(0)].sum()) if not closed_pnl.empty else 0.0
        profit_factor = np.nan
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = np.inf

        return {
            "planned_trade_count": planned_trade_count,
            "entered_trade_count": int(entered_mask.sum()) if planned_trade_count else 0,
            "closed_trade_count": int(closed_mask.sum()) if planned_trade_count else 0,
            "open_trade_count": int(open_mask.sum()) if planned_trade_count else 0,
            "entry_fill_rate": float(entered_mask.mean()) if planned_trade_count else np.nan,
            "trade_win_rate": float((closed_returns > 0).mean()) if len(closed_returns) else np.nan,
            "average_trade_return": float(closed_returns.mean()) if len(closed_returns) else np.nan,
            "median_trade_return": float(closed_returns.median()) if len(closed_returns) else np.nan,
            "profit_factor": float(profit_factor) if pd.notna(profit_factor) else profit_factor,
            "fixed_entry_notional": np.nan,
        }

    def _run_simulation(
        self, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None
    ) -> dict[str, Any]:
        window_dates = self._get_window_dates(start_date, end_date)
        trade_plan = self._get_window_trade_plan(start_date, end_date)
        strategy_trades = self._initialize_strategy_trades(trade_plan)

        cash = float(self.initial_capital)
        positions: dict[str, dict[str, Any]] = {}
        pending_exit_ids: set[int] = set()
        last_mark_prices: dict[str, float] = {}
        trade_records: list[dict[str, Any]] = []
        holding_records: list[dict[str, Any]] = []
        portfolio_records: list[dict[str, Any]] = []
        prev_nav = float(self.initial_capital)
        cumulative_cost = 0.0

        entries_by_date = {
            pd.Timestamp(date): group["planned_trade_id"].astype(int).tolist()
            for date, group in trade_plan.groupby("entry_date", sort=True)
        }
        exit_schedule = (
            trade_plan.loc[trade_plan["exit_date"].notna(), ["planned_trade_id", "exit_date"]]
            if not trade_plan.empty
            else pd.DataFrame(columns=["planned_trade_id", "exit_date"])
        )
        exits_by_date = {
            pd.Timestamp(date): group["planned_trade_id"].astype(int).tolist()
            for date, group in exit_schedule.groupby("exit_date", sort=True)
        }

        for index, current_date in enumerate(window_dates):
            day_trade_notional = 0.0
            day_transaction_cost = 0.0

            for trade_id in exits_by_date.get(current_date, []):
                ticker = str(strategy_trades.at[trade_id, "ticker"])
                position = positions.get(ticker)
                if position is not None and int(position["planned_trade_id"]) == int(trade_id):
                    pending_exit_ids.add(int(trade_id))

            for trade_id in sorted(pending_exit_ids):
                ticker = str(strategy_trades.at[trade_id, "ticker"])
                position = positions.get(ticker)
                if position is None or int(position["planned_trade_id"]) != int(trade_id):
                    continue

                shares = int(position["shares"])
                quote = self._get_quote(current_date, ticker)
                if shares <= 0 or not self._is_tradeable(quote):
                    strategy_trades.at[trade_id, "exit_order_status"] = self._status_for_untradeable_exit(quote)
                    continue

                open_price = float(quote["open"])
                execution_price = self._execution_price("sell", open_price)
                gross_notional = shares * execution_price
                costs = self._calculate_trade_costs("sell", gross_notional)
                cash_delta = gross_notional - costs["total_cost"]
                cash += cash_delta
                day_trade_notional += gross_notional
                day_transaction_cost += costs["total_cost"]

                trade_record = self._build_trade_record(
                    date=current_date,
                    signal_date=pd.Timestamp(strategy_trades.at[trade_id, "signal_date"]),
                    ticker=ticker,
                    side="sell",
                    shares=shares,
                    open_price=open_price,
                    execution_price=execution_price,
                    gross_notional=gross_notional,
                    costs=costs,
                    cash_delta=cash_delta,
                )
                trade_record["planned_trade_id"] = int(trade_id)
                trade_records.append(trade_record)

                strategy_trades.at[trade_id, "actual_exit_date"] = current_date
                strategy_trades.at[trade_id, "actual_exit_open_price"] = open_price
                strategy_trades.at[trade_id, "actual_exit_execution_price"] = execution_price
                strategy_trades.at[trade_id, "actual_exit_gross_notional"] = gross_notional
                strategy_trades.at[trade_id, "actual_exit_total_cost"] = costs["total_cost"]
                strategy_trades.at[trade_id, "actual_exit_reason"] = str(strategy_trades.at[trade_id, "exit_reason"])
                strategy_trades.at[trade_id, "exit_order_status"] = "executed"
                strategy_trades.at[trade_id, "actual_trade_status"] = "closed"

                entry_date = strategy_trades.at[trade_id, "actual_entry_date"]
                if pd.notna(entry_date):
                    holding_days = self._date_distance(pd.Timestamp(entry_date), current_date)
                    if holding_days is not None:
                        strategy_trades.at[trade_id, "actual_holding_days"] = holding_days
                self._update_closed_trade_metrics(strategy_trades, trade_id)

                positions.pop(ticker, None)

            pending_exit_ids = {
                trade_id
                for trade_id in pending_exit_ids
                if str(strategy_trades.at[trade_id, "ticker"]) in positions
            }

            todays_entries = entries_by_date.get(current_date, [])
            executable_entry_ids: list[int] = []
            for trade_id in todays_entries:
                ticker = str(strategy_trades.at[trade_id, "ticker"])
                if ticker in positions:
                    strategy_trades.at[trade_id, "entry_order_status"] = "blocked_existing_position"
                    continue

                quote = self._get_quote(current_date, ticker)
                if not self._is_tradeable(quote):
                    strategy_trades.at[trade_id, "entry_order_status"] = self._status_for_untradeable_entry(quote)
                    continue
                executable_entry_ids.append(int(trade_id))

            remaining_entry_ids = executable_entry_ids.copy()
            for trade_id in todays_entries:
                if int(trade_id) not in executable_entry_ids:
                    continue

                ticker = str(strategy_trades.at[trade_id, "ticker"])
                quote = self._get_quote(current_date, ticker)
                if quote is None:
                    strategy_trades.at[trade_id, "entry_order_status"] = "blocked_no_quote"
                    remaining_entry_ids.remove(int(trade_id))
                    continue

                raw_budget, effective_budget = self._entry_budget(cash, len(remaining_entry_ids))
                strategy_trades.at[trade_id, "target_entry_budget"] = float(raw_budget)
                shares = self._max_affordable_buy_shares(quote, effective_budget)
                if shares <= 0:
                    if (
                        self.fixed_entry_notional is not None
                        and cash >= raw_budget - 1e-9
                    ):
                        strategy_trades.at[trade_id, "entry_order_status"] = "skipped_position_budget"
                    else:
                        strategy_trades.at[trade_id, "entry_order_status"] = "skipped_cash_constraint"
                    remaining_entry_ids.remove(int(trade_id))
                    continue

                open_price = float(quote["open"])
                execution_price = self._execution_price("buy", open_price)
                gross_notional = shares * execution_price
                costs = self._calculate_trade_costs("buy", gross_notional)
                cash_delta = -(gross_notional + costs["total_cost"])
                if cash + cash_delta < -1e-9:
                    strategy_trades.at[trade_id, "entry_order_status"] = "skipped_cash_constraint"
                    remaining_entry_ids.remove(int(trade_id))
                    continue

                cash += cash_delta
                day_trade_notional += gross_notional
                day_transaction_cost += costs["total_cost"]
                positions[ticker] = {"shares": int(shares), "planned_trade_id": int(trade_id)}
                trade_record = self._build_trade_record(
                    date=current_date,
                    signal_date=pd.Timestamp(strategy_trades.at[trade_id, "signal_date"]),
                    ticker=ticker,
                    side="buy",
                    shares=shares,
                    open_price=open_price,
                    execution_price=execution_price,
                    gross_notional=gross_notional,
                    costs=costs,
                    cash_delta=cash_delta,
                )
                trade_record["planned_trade_id"] = int(trade_id)
                trade_records.append(trade_record)

                strategy_trades.at[trade_id, "entry_order_status"] = "executed"
                strategy_trades.at[trade_id, "actual_entry_date"] = current_date
                strategy_trades.at[trade_id, "actual_entry_open_price"] = open_price
                strategy_trades.at[trade_id, "actual_entry_execution_price"] = execution_price
                strategy_trades.at[trade_id, "actual_entry_shares"] = int(shares)
                strategy_trades.at[trade_id, "actual_entry_gross_notional"] = gross_notional
                strategy_trades.at[trade_id, "actual_entry_total_cost"] = costs["total_cost"]
                strategy_trades.at[trade_id, "actual_trade_status"] = "open"
                remaining_entry_ids.remove(int(trade_id))

            day_holdings: list[dict[str, Any]] = []
            gross_exposure = 0.0
            for ticker, position in sorted(positions.items()):
                shares = int(position["shares"])
                if shares <= 0:
                    continue

                mark_price = self._get_mark_price(current_date, ticker, last_mark_prices)
                if mark_price is None:
                    continue

                last_mark_prices[ticker] = float(mark_price)
                market_value = shares * float(mark_price)
                gross_exposure += market_value
                day_holdings.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "shares": shares,
                        "mark_price": float(mark_price),
                        "market_value": float(market_value),
                    }
                )

            nav = cash + gross_exposure
            strategy_return = nav / prev_nav - 1.0 if prev_nav > 0 else 0.0
            benchmark_return = float(self._benchmark_returns.get(current_date, 0.0))
            if index == 0:
                benchmark_return = 0.0

            cumulative_cost += day_transaction_cost
            daily_turnover = day_trade_notional / prev_nav if prev_nav > 0 else 0.0
            portfolio_records.append(
                {
                    "date": current_date,
                    "nav": float(nav),
                    "cash": float(cash),
                    "gross_exposure": float(gross_exposure),
                    "holdings_count": int(len(day_holdings)),
                    "strategy_return": float(strategy_return),
                    "benchmark_return": float(benchmark_return),
                    "daily_turnover": float(daily_turnover),
                    "transaction_cost": float(day_transaction_cost),
                    "cumulative_cost": float(cumulative_cost),
                }
            )

            for holding in day_holdings:
                holding["weight"] = holding["market_value"] / nav if nav > 0 else 0.0
            holding_records.extend(day_holdings)
            prev_nav = nav

        portfolio = pd.DataFrame(portfolio_records)
        if portfolio.empty:
            result = self._empty_result(start_date, end_date)
            empty_strategy_trades = strategy_trades.reset_index(drop=True)
            result["strategy_trades"] = empty_strategy_trades
            result["trade_summary"] = self._compute_trade_summary(empty_strategy_trades)
            result["trade_summary"]["fixed_entry_notional"] = (
                float(self.fixed_entry_notional) if self.fixed_entry_notional is not None else np.nan
            )
            result["summary"].update(result["trade_summary"])
            return result

        portfolio["strategy_nav_norm"] = portfolio["nav"] / self.initial_capital
        portfolio["benchmark_nav_norm"] = (1.0 + portfolio["benchmark_return"]).cumprod()
        portfolio["drawdown"] = portfolio["nav"] / portfolio["nav"].cummax() - 1.0
        trades = pd.DataFrame(trade_records)
        holdings = pd.DataFrame(holding_records)
        returns = portfolio.set_index("date")["strategy_return"].rename("strategy_return")
        benchmark_returns = portfolio.set_index("date")["benchmark_return"].rename("benchmark_return")

        self._finalize_open_trade_metrics(strategy_trades, positions, last_mark_prices, window_dates)
        strategy_trade_output = strategy_trades.reset_index(drop=True).sort_values(
            ["entry_date", "ticker", "signal_date"],
            kind="mergesort",
            ignore_index=True,
        )
        strategy_trade_output.attrs.update(dict(self.planned_trade_df.attrs))
        trade_summary = self._compute_trade_summary(strategy_trade_output)
        trade_summary["fixed_entry_notional"] = (
            float(self.fixed_entry_notional) if self.fixed_entry_notional is not None else np.nan
        )
        summary = self._compute_summary(portfolio, trades, start_date, end_date)
        summary.update(trade_summary)

        return {
            "summary": summary,
            "portfolio": portfolio,
            "returns": returns,
            "benchmark_returns": benchmark_returns,
            "trades": trades,
            "holdings": holdings,
            "strategy_trades": strategy_trade_output,
            "trade_summary": trade_summary,
        }

    def compute_metrics(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        window_key = self._normalize_window_key(start_date, end_date)
        base_result = super().compute_metrics(start_date=start_date, end_date=end_date)
        cached = self._simulation_cache[window_key]
        base_result["strategy_trades"] = cached["strategy_trades"].copy()
        base_result["trade_summary"] = dict(cached["trade_summary"])
        return base_result

    def show_metrics(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> go.Figure:
        results = self.compute_metrics(start_date=start_date, end_date=end_date)
        portfolio = results["portfolio"]
        strategy_trades = results["strategy_trades"]
        trade_summary = results["trade_summary"]
        if portfolio.empty:
            return go.Figure()

        rolling_sharpe = (
            portfolio["strategy_return"].rolling(63, min_periods=20).mean()
            / portfolio["strategy_return"].rolling(63, min_periods=20).std(ddof=0)
        ) * np.sqrt(252.0)

        planned_entries = pd.Series(dtype="float64")
        actual_entries = pd.Series(dtype="float64")
        actual_exits = pd.Series(dtype="float64")
        realized_pnl = pd.Series(dtype="float64")
        if not strategy_trades.empty:
            planned_entries = (
                strategy_trades.groupby("entry_date", sort=True)["planned_trade_id"]
                .count()
                .astype(float)
            )
            actual_entries = (
                strategy_trades.loc[strategy_trades["actual_entry_date"].notna()]
                .groupby("actual_entry_date", sort=True)["planned_trade_id"]
                .count()
                .astype(float)
            )
            actual_exits = (
                strategy_trades.loc[strategy_trades["actual_exit_date"].notna()]
                .groupby("actual_exit_date", sort=True)["planned_trade_id"]
                .count()
                .astype(float)
            )
            realized_pnl = (
                strategy_trades.loc[
                    strategy_trades["actual_trade_status"].eq("closed")
                    & strategy_trades["actual_exit_date"].notna()
                    & strategy_trades["actual_pnl"].notna()
                ]
                .groupby("actual_exit_date", sort=True)["actual_pnl"]
                .sum()
                .astype(float)
            )

        figure = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                "Strategy vs Benchmark",
                "Drawdown",
                "Rolling 63-Day Sharpe",
                "Holdings / Turnover / Costs",
                "Trade Plan Execution",
            ),
            specs=[[{}], [{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        )

        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["strategy_nav_norm"],
                mode="lines",
                name="Strategy NAV",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["benchmark_nav_norm"],
                mode="lines",
                name="Benchmark NAV",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["drawdown"],
                mode="lines",
                name="Drawdown",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=rolling_sharpe,
                mode="lines",
                name="Rolling Sharpe (63D)",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["holdings_count"],
                mode="lines",
                name="Holdings Count",
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Bar(
                x=portfolio["date"],
                y=portfolio["daily_turnover"],
                name="Daily Turnover",
                opacity=0.45,
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=portfolio["date"],
                y=portfolio["cumulative_cost"],
                mode="lines",
                name="Cumulative Cost",
            ),
            row=4,
            col=1,
            secondary_y=True,
        )
        figure.add_trace(
            go.Bar(
                x=planned_entries.index,
                y=planned_entries.values,
                name="Planned Entries",
                opacity=0.35,
            ),
            row=5,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Bar(
                x=actual_entries.index,
                y=actual_entries.values,
                name="Actual Entries",
                opacity=0.5,
            ),
            row=5,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Bar(
                x=actual_exits.index,
                y=actual_exits.values,
                name="Actual Exits",
                opacity=0.5,
            ),
            row=5,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=realized_pnl.index,
                y=realized_pnl.values,
                mode="lines+markers",
                name="Realized PnL",
            ),
            row=5,
            col=1,
            secondary_y=True,
        )

        fixed_notional = trade_summary.get("fixed_entry_notional")
        fixed_notional_text = (
            f"{fixed_notional:,.0f}" if pd.notna(fixed_notional) else "dynamic"
        )
        win_rate = trade_summary.get("trade_win_rate")
        win_rate_text = f"{win_rate:.1%}" if pd.notna(win_rate) else "N/A"
        profit_factor = trade_summary.get("profit_factor")
        if pd.isna(profit_factor):
            profit_factor_text = "N/A"
        elif np.isinf(profit_factor):
            profit_factor_text = "inf"
        else:
            profit_factor_text = f"{profit_factor:.2f}"

        title = (
            "Trade Plan Backtest Metrics"
            f" | planned {trade_summary.get('planned_trade_count', 0)}"
            f" | entered {trade_summary.get('entered_trade_count', 0)}"
            f" | closed {trade_summary.get('closed_trade_count', 0)}"
            f" | fill {trade_summary.get('entry_fill_rate', np.nan):.1%}"
            if pd.notna(trade_summary.get("entry_fill_rate", np.nan))
            else "Trade Plan Backtest Metrics"
        )
        figure.update_layout(
            height=1350,
            width=1250,
            template="plotly_white",
            hovermode="x unified",
            barmode="group",
            title=title,
        )
        figure.add_annotation(
            x=1.0,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            text=(
                f"Fixed entry notional: {fixed_notional_text}<br>"
                f"Trade win rate: {win_rate_text}<br>"
                f"Profit factor: {profit_factor_text}<br>"
                f"Avg trade return: "
                f"{trade_summary.get('average_trade_return', np.nan):.2%}"
                if pd.notna(trade_summary.get("average_trade_return", np.nan))
                else (
                    f"Fixed entry notional: {fixed_notional_text}<br>"
                    f"Trade win rate: {win_rate_text}<br>"
                    f"Profit factor: {profit_factor_text}<br>"
                    "Avg trade return: N/A"
                )
            ),
        )
        figure.update_yaxes(title_text="Normalized NAV", row=1, col=1)
        figure.update_yaxes(title_text="Drawdown", row=2, col=1)
        figure.update_yaxes(title_text="Sharpe", row=3, col=1)
        figure.update_yaxes(title_text="Holdings / Turnover", row=4, col=1, secondary_y=False)
        figure.update_yaxes(title_text="Cumulative Cost", row=4, col=1, secondary_y=True)
        figure.update_yaxes(title_text="Trade Count", row=5, col=1, secondary_y=False)
        figure.update_yaxes(title_text="Realized PnL", row=5, col=1, secondary_y=True)
        return figure
