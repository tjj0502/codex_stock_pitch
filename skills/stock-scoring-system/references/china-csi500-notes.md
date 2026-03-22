# China CSI 500 Notes

## Table of Contents

- Current repo context
- Suggested first score for this repo
- China-specific event mapping
- Implementation rules

## Current repo context

- The current data access file is `china_stock_data.py`.
- `get_csi500_constituents()` pulls the latest available CSI 500 membership from Tushare `index_weight`.
- `get_csi500_member_prices()` then fetches daily prices for that latest constituent set over the requested history.

Inference for this repo: the current price helper is fine for collecting recent member prices, but it is not a historical-universe engine. If it is used directly in a backtest, it introduces survivorship bias because past dates are filtered through the latest constituent list.

## Suggested first score for this repo

Start with a monthly cross-sectional score on the CSI 500 universe:

- `technical = 40%`
- `fundamental = 45%`
- `event_or_regime = 15%`

Suggested technical block:

- 20-day reversal
- 60-day momentum
- 120-day momentum
- 250-day momentum with a recent-gap or skip-month treatment
- 60-day realized volatility penalty
- simple liquidity or turnover stability metric

Suggested fundamental block:

- value ratio such as book-to-market or earnings yield
- profitability ratio
- asset-growth or investment penalty
- leverage penalty
- accrual or cash-quality metric
- `F-score` style composite if enough statement fields are available

Suggested event or regime block:

- global or China-specific geopolitical risk
- Fed meeting surprise or meeting-window flag
- trade policy uncertainty
- broad financial-stress regime variable

## China-specific event mapping

- Fed policy still matters for China-linked equities through global liquidity, growth, and FX channels. Use it as a market regime or interact it with export, duration, or leverage exposure.
- Trade Policy Uncertainty is especially relevant for export-heavy, hardware, semiconductor, shipping, and industrial names.
- Country-specific GPR data for China, Russia, and Ukraine are useful when the strategy needs to reflect geopolitical spillovers rather than only domestic fundamentals.
- If local policy events become central to the strategy, add a separate China policy calendar rather than overloading the Fed proxy.

## Implementation rules

- Rebuild CSI 500 membership on each rebalance date if the task is historical backtesting.
- Use trade-date aligned point-in-time joins for prices, fundamentals, and events.
- Lag fundamentals by filing availability, not by fiscal period alone.
- Normalize scores within the active universe on each rebalance date.
- Compare raw scores with sector-neutral scores before accepting the model.
- Report turnover and implementation cost alongside every decile or long-short result.
- Prefer a simple monthly model first. Add short-term technical factors only after a cost-aware test.

