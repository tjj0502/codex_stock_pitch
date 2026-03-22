---
name: stock-scoring-system
description: Design cross-sectional stock scoring and ranking systems that combine technical, fundamental, and event or regime signals for an actively traded equity market. Use when Codex needs to research, specify, implement, backtest, or refine a stock-selection model, especially for universes such as CSI 500 or other broad stock baskets, and when the work involves factor definitions, score normalization, feature weighting, regime overlays, point-in-time data rules, or evaluation methodology.
---

# Stock Scoring System

## Overview

Use this skill to turn multiple stock signals into one testable ranking score. Keep the base model cross-sectional, point-in-time, and simple enough to audit before adding more signals.

Read [references/online-research.md](references/online-research.md) when choosing signals or citing evidence. Read [references/china-csi500-notes.md](references/china-csi500-notes.md) when the task uses this repo's current China market setup.

## Workflow

### 1. Fix the objective before selecting features

- Define the market, universe, benchmark, rebalance cadence, holding period, and turnover budget.
- Decide whether the score should be pure stock selection or whether sector, style, and macro bets are allowed.
- Prefer monthly or weekly rebalancing first. Only use higher-frequency signals if the user explicitly wants that tradeoff and the cost model supports it.

### 2. Build signal pillars

Start with three pillars and keep each pillar interpretable.

- Technical:
  Use medium-horizon trend, short-horizon reversal, volatility, and liquidity or tradability features.
- Fundamental:
  Use value, profitability, quality, investment intensity, leverage, and accounting-strength features.
- Event or regime:
  Use measurable time series such as policy surprises, geopolitical risk, policy uncertainty, trade uncertainty, and financial-stress proxies.

Treat event variables as overlays or interactions, not as free-form notes. A market-level event series should usually modify stock scores through exposure maps, sector tilts, or dynamic weights.

### 3. Convert raw features into comparable scores

- Use only information available on the scoring date.
- Lag financial statements by a realistic filing delay.
- Winsorize or clip outliers before standardizing.
- Standardize by date within the active universe, typically with z-scores or percentile ranks.
- Flip the sign so higher is always better.
- Neutralize sector, size, or beta exposures when the target is idiosyncratic stock selection rather than style timing.

### 4. Combine signals conservatively

Prefer a two-stage combination:

1. Average normalized signals within each pillar.
2. Combine pillar scores with explicit weights.

Good default starting weights:

- `technical = 0.45`
- `fundamental = 0.40`
- `event = 0.15`

Adjust weights only after seeing subperiod results, turnover, and correlation between pillars.

### 5. Validate like a ranking model, not a narrative

Check at minimum:

- rank IC and Pearson IC by rebalance date
- top-minus-bottom decile or quintile spread
- hit rate and stability by year
- turnover and estimated costs
- sector-neutral and cap-neutral performance
- drawdowns during stress windows
- sensitivity to lag assumptions, winsorization, and weighting

Reject models that work only before costs, only in one regime, or only because they load on unintended sector or size bets.

### 6. Produce concrete deliverables

When asked to design or extend a model, return:

- the score formula
- the exact field mapping
- the lag and missing-data rules
- the rebalance and holding schedule
- the validation plan
- the main failure modes

## Practical Rules

- Use slower signals unless there is evidence the faster signal survives costs.
- Keep the first version parsimonious. Add features only when they improve out-of-sample behavior or reduce known failure modes.
- Use event features to scale or reroute exposures during stress, not to replace the core cross-sectional model.
- Prefer breadth over precision in early versions: several decent, weakly correlated signals are usually better than one overfit signal.
- State clearly when a recommendation is an inference from the research notes instead of a direct source claim.
