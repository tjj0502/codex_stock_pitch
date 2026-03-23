# Technical Scoring System

## Purpose

`DailyTechnicalScorer` is a daily, long-only ranking system built on the dataframe returned by `get_csi500_member_prices`.

The intended live workflow is:

1. Use all data available through the latest completed trading session.
2. Run the scorer before the next market open.
3. Buy the top `N` ranked names at the next open.
4. Re-rank again before the following open.

For research only, the target attached to row `t` is:

`target_next_open_to_open = open[t+2] / open[t+1] - 1`

That alignment matches the real trading workflow:

- signal known after close of `t`
- entry at open of `t+1`
- exit or rebalance at open of `t+2`

## Review Summary

I reviewed each feature group, the cross-sectional ranking logic, the component aggregation, and the final score construction.

What I checked:

- no lookahead in feature engineering
- correct target alignment for next-open trading
- correct direction for reward features versus penalty features
- deterministic ranking behavior when scores tie
- handling of zero-range candles and missing history
- robustness against bad denominators that can otherwise create `inf`

What I changed during the review:

- invalid historical denominators now become `NaN` instead of leaking `inf` into the score
- a regression test was added so rows with broken return denominators stay unscored

Bottom line:

- the current implementation is internally consistent for a pre-open, daily-rebalance technical ranking system
- the main remaining caveats are data assumptions, not coding mistakes

## Input Contract

The scorer expects the exact columns produced by `get_csi500_member_prices`:

- `date`
- `ticker`
- `ts_code`
- `name`
- `weight`
- `constituent_trade_date`
- `open`
- `close`
- `high`
- `low`
- `pre_close`
- `volume`
- `turnover`
- `amplitude_pct`
- `change_pct`
- `change_amount`

On initialization the class:

- validates required columns
- coerces dates and numeric fields
- drops duplicate `ticker/date` rows, keeping the last one
- stores the working dataframe on `stock_candle_df`

## Raw Features

### 1. Trend Features

These measure whether the stock is already in a sustained upward move.

- `ret_5d = close[t] / close[t-5] - 1`
- `ret_10d = close[t] / close[t-10] - 1`
- `ret_20d = close[t] / close[t-20] - 1`
- `ema_10_gap = close[t] / ema_10[t] - 1`
- `ema_20_gap = close[t] / ema_20[t] - 1`

Interpretation:

- higher is better
- strong positive values mean the stock is trading above its recent history and above its short and medium trend anchors

Note:

- `ret_5d` is computed and kept for inspection, but it is not part of the final v1 score

### 2. Breakout and Relative Strength Features

These measure whether a stock is leading the universe and breaking higher versus recent resistance.

- `breakout_20d = max(close[t] / prior_20d_high[t] - 1, 0)`
- `drawdown_20d = max((prior_20d_high[t] - close[t]) / prior_20d_high[t], 0)`
- `excess_ret_5d = ((1 + ret_5d) / (1 + universe_ret_5d)) - 1`
- `excess_ret_20d = ((1 + ret_20d) / (1 + universe_ret_20d)) - 1`

Interpretation:

- `breakout_20d` rewards names that are above the prior 20-day high
- `drawdown_20d` penalizes names that are well below the prior 20-day high
- `excess_ret_5d` and `excess_ret_20d` reward stocks outperforming the equal-weight universe

Design choice:

- the universe benchmark is equal-weight daily return from the provided dataframe, not the `weight` column
- this is intentional because `weight` is snapshot metadata, not a historical daily weighting series

### 3. Liquidity Features

These measure whether participation is improving.

- `volume_ratio_5_20 = mean(volume, 5) / mean(volume, 20)`
- `turnover_ratio_5_20 = mean(turnover, 5) / mean(turnover, 20)`

Interpretation:

- higher is better
- these features reward names where recent activity is stronger than their own 20-day baseline

Why both:

- `volume` captures share participation
- `turnover` captures money flow and naturally scales with price

### 4. Risk Features

These penalize unstable or stretched names.

- `volatility_10d = rolling_std(close / pre_close - 1, 10)`
- `atr_pct_14 = rolling_mean(true_range, 14) / close * 100`
- `drawdown_20d` is reused here as a risk penalty

Where:

- `true_range = max(high - low, abs(high - pre_close), abs(low - pre_close))`

Interpretation:

- lower is better
- this block favors smoother names with smaller daily turbulence and less distance below their recent highs

### 5. Candle Quality Features

These measure the quality of the most recent daily bar.

- `close_location = (close - low) / (high - low)`
- `body_to_range = (close - open) / (high - low)`
- `upper_shadow_pct = (high - max(open, close)) / (high - low)`

Interpretation:

- higher `close_location` is better because it means the stock closed near the high of the day
- higher `body_to_range` is better because it means the body was bullish and meaningful relative to the full range
- lower `upper_shadow_pct` is better because it means less rejection from the highs

Special handling:

- if `high == low`, then:
  - `close_location = 0.5`
  - `body_to_range = 0`
  - `upper_shadow_pct = 0`

## From Raw Features to Component Scores

Raw features are not used directly.

For each trading date:

1. Each factor is winsorized cross-sectionally at the 2.5% and 97.5% quantiles.
2. Each winsorized factor is converted into a percentile rank inside that day.
3. Penalty factors are ranked in the reverse direction so that lower raw risk becomes a higher score.

Penalty factors are:

- `volatility_10d`
- `atr_pct_14`
- `drawdown_20d`
- `upper_shadow_pct`

This means every factor is normalized onto the same direction:

- higher rank always means better from a long-only perspective

## Component Scores

Each component score is the simple average of its normalized factor ranks.

### Trend Score

`trend_score = mean(ret_10d, ret_20d, ema_10_gap, ema_20_gap)`

What it rewards:

- persistent upward movement
- price holding above short and medium EMAs

### Relative Strength Score

`relative_strength_score = mean(breakout_20d, excess_ret_5d, excess_ret_20d)`

What it rewards:

- stocks outperforming the equal-weight universe
- stocks making fresh progress above resistance

### Liquidity Score

`liquidity_score = mean(volume_ratio_5_20, turnover_ratio_5_20)`

What it rewards:

- improving participation and money flow

### Risk Score

`risk_score = mean(inverse volatility_10d, inverse atr_pct_14, inverse drawdown_20d)`

What it rewards:

- smoother names
- less noisy tape
- less distance below recent highs

### Candle Quality Score

`candle_quality_score = mean(close_location, body_to_range, inverse upper_shadow_pct)`

What it rewards:

- closes near the top of the day
- bullish body structure
- limited rejection from highs

## Final Technical Score

The final score is:

`technical_score = 100 * (0.35 * trend_score + 0.20 * relative_strength_score + 0.15 * liquidity_score + 0.15 * risk_score + 0.15 * candle_quality_score)`

Interpretation of the weights:

- trend is the dominant block because the system is trying to stay with short-term strength
- relative strength is second because leadership versus the universe matters
- liquidity, risk, and candle quality are confirmation and quality filters

After the score is computed:

- stocks are ranked descending within each `date`
- `technical_rank = 1` is the best stock on that date
- `selected_top_n` is set only if `add_technical_score(top_n=...)` was called

Important note:

- `selected_top_n` reflects the `top_n` used in the last scoring call
- `get_top_candidates(top_n=...)` only returns the top rows; it does not recompute `selected_top_n`

## Eligibility and Safety Guards

A row is eligible for scoring only when all of the following are true:

- the ticker has at least `min_history` observations
- current-row `open`, `close`, `high`, `low`, `pre_close`, `volume`, and `turnover` are all positive
- all required scoring features are finite and non-missing

Additional safeguards:

- rows with invalid return denominators now produce `NaN` features instead of `inf`
- rows with missing or invalid features remain unscored
- this keeps broken rows out of the percentile ranking and out of the final score

## Important Caveats

These are not coding errors, but they matter for interpretation.

### 1. Raw Prices

The scorer uses the raw daily prices returned by `get_csi500_member_prices`.

Implication:

- corporate actions can distort long historical momentum features
- if you later want production-grade historical research, adjusted prices are worth adding

### 2. Survivorship Bias

`get_csi500_member_prices` currently pulls the latest CSI 500 members and then fetches their historical prices.

Implication:

- historical score evaluation will be survivorship-biased
- live scoring is still fine, but backtests will look cleaner than reality until historical constituent membership is added

### 3. Universe Proxy

The relative-strength benchmark is the equal-weight return of the available dataframe.

Implication:

- it is a practical internal benchmark
- it is not the same thing as a true historical CSI 500 total-return series

### 4. Signal Timing

This system is intentionally end-of-day signal generation plus next-open execution.

Implication:

- do not interpret the score as an intraday prediction
- do not use same-day open information that would not have been known before the open

## Current TODOs

These are intentionally not implemented in v1:

- `rolling_learned_weights`
- `supervised_model`

The current version is a transparent factor-mix model so every component can be inspected directly in the dataframe.
