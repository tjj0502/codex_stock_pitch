# Trend Pullback Experiment Log

Last updated: `2026-04-17`

## Scope

This log captures the strategy experiments run so far for the trend-pullback continuation setup.

Common test bed:

- Data: [stock_price.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/Dataframes/stock_price.csv)
- Universe: `csi500`
- Window: `2020-01-01` to `2026-03-16`
- Backtest sizing: `initial_capital=1,000,000`, `fixed_entry_notional=20,000`, `board_lot_size=100`

Reference baseline config before later sweeps:

- `ma_windows=(20, 60, 120)`
- `min_trend_bars=10`
- `pivot_window=1`
- `max_pullback_bars=40`
- `max_signal_delay_after_third_low=5`
- `min_signal_body_pct=0.50`
- `max_signal_upper_shadow_pct=0.25`
- `max_signal_lower_shadow_pct=0.35`
- `stop_buffer_pct=0.01`
- `min_reward_r=1.50`
- `take_profit_fraction_of_trend_move=0.50`

## Baseline

Baseline metrics under the reference config:

| Metric | Value |
| --- | ---: |
| planned_trade_count | 50 |
| closed_trade_count | 49 |
| trade_win_rate | 46.94% |
| average_trade_return | 3.21% |
| total_return | 3.20% |
| sharpe | 0.85 |
| profit_factor | 2.21 |
| max_drawdown | 0.64% |
| hard_stop_rate | 53.06% |
| take_profit_rate | 46.94% |

## Funnel Analysis

Signal funnel under the baseline config:

| Stage | Count |
| --- | ---: |
| `post_trend_phase` | 466,995 |
| `three_push_pullback` | 6,531 |
| `signal_candle` | 326 |
| `follow_through_confirmed` | 223 |
| `signal_date >= trend_end_date` | 114 |
| `reward_to_risk >= 1.5` | 56 |
| raw entry-ready signals | 56 |
| final planned trades in `trade_df` | 50 |

Key takeaways:

- The strategy is not primarily bottlenecked by follow-through.
- The biggest drop-offs are:
  - `three_push_pullback -> signal_candle`
  - `follow_through_confirmed -> after_trend_end`
  - `after_trend_end -> reward_to_risk >= 1.5`
- The current frequency issue is mainly driven by strict signal-candle quality, timing after `trend_end`, and reward-to-risk filtering.

Additional baseline timing diagnostics:

- Follow-through signals after `trend_end`: `114`
- Follow-through signals before `trend_end`: `109`
- Estimated raw entries per year:
  - current (`delay=5`, `rr=1.5`): about `8.0`
  - `rr=1.2`: about `10.3`
  - `rr=1.0`: about `12.4`
  - removing timing filter while keeping `rr=1.5`: about `13.7`

## Experiment 1: `max_signal_delay_after_third_low x min_reward_r`

Archive: [grid_delay_rr.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/trend_pullback_continuation/experiment_logs/grid_delay_rr.csv)

Grid:

- `max_signal_delay_after_third_low = [5, 8, 10]`
- `min_reward_r = [1.0, 1.2, 1.5]`

Summary:

| delay | min_reward_r | planned trades | win rate | total return | sharpe | profit factor | max DD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 5 | 1.2 | 61 | 45.76% | 3.29% | 0.76 | 1.96 | 0.95% |
| 5 | 1.0 | 75 | 46.58% | 3.52% | 0.77 | 1.81 | 0.97% |
| 8 | 1.5 | 55 | 42.59% | 2.32% | 0.58 | 1.66 | 0.79% |
| 8 | 1.2 | 70 | 41.18% | 2.33% | 0.51 | 1.51 | 0.98% |
| 8 | 1.0 | 86 | 44.05% | 3.22% | 0.65 | 1.59 | 1.08% |
| 10 | 1.5 | 58 | 42.11% | 2.33% | 0.57 | 1.63 | 0.79% |
| 10 | 1.2 | 73 | 42.25% | 2.67% | 0.57 | 1.57 | 0.98% |
| 10 | 1.0 | 91 | 43.82% | 3.29% | 0.65 | 1.57 | 1.11% |

Conclusion:

- The best overall balance in this grid remains `delay=5`, `min_reward_r=1.5`.
- Lowering `min_reward_r` is the cleanest way to increase frequency.
- Extending `max_signal_delay_after_third_low` beyond `5` increases frequency a bit, but consistently hurts quality.
- Recommendation from this sweep:
  - Keep `delay=5`.
  - If more frequency is needed, lower `min_reward_r` before widening the signal-delay window.

## Experiment 2: stricter `delay x min_reward_r` quality sweep

Archive: [grid_strict_quality.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/trend_pullback_continuation/experiment_logs/grid_strict_quality.csv)

Grid:

- `max_signal_delay_after_third_low = [3, 4, 5]`
- `min_reward_r = [1.5, 1.8, 2.0]`

Selected rows:

| delay | min_reward_r | planned trades | win rate | total return | sharpe | profit factor | max DD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 4 | 1.5 | 46 | 44.44% | 2.79% | 0.81 | 2.13 | 0.64% |
| 3 | 1.5 | 39 | 44.74% | 1.94% | 0.64 | 1.90 | 0.71% |
| 5 | 1.8 | 41 | 42.50% | 1.86% | 0.58 | 1.78 | 0.99% |
| 5 | 2.0 | 35 | 44.12% | 2.25% | 0.72 | 2.26 | 0.79% |

Conclusion:

- Raising `min_reward_r` above `1.5` reduces sample size quickly.
- `min_reward_r=2.0` does lift `profit_factor` slightly over the baseline, but the gain is modest relative to the drop in frequency and return.
- Cutting `delay` below `5` does not improve the strategy enough to justify the lower opportunity set.
- Recommendation from this sweep:
  - `min_reward_r=1.5` still looks like the practical default.
  - `min_reward_r=2.0` only makes sense if the explicit goal is ultra-selective, lower-frequency quality.

## Experiment 3: `take_profit_fraction_of_trend_move`

Archive: [grid_take_profit.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/trend_pullback_continuation/experiment_logs/grid_take_profit.csv)

Grid:

- fixed `delay=5`
- fixed `min_reward_r=1.5`
- `take_profit_fraction_of_trend_move = [0.4, 0.5, 0.6]`

Summary:

| tp fraction | planned trades | win rate | total return | sharpe | profit factor | max DD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 39 | 47.37% | 1.50% | 0.51 | 1.73 | 0.66% |
| 0.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 0.6 | 60 | 40.35% | 2.75% | 0.54 | 1.71 | 1.62% |

Conclusion:

- `0.5` is the best of the tested take-profit fractions.
- `0.4` is too conservative and cuts upside too aggressively.
- `0.6` pushes the target too far and increases both hard stops and drawdown.
- Recommendation from this sweep:
  - Keep `take_profit_fraction_of_trend_move=0.5`.

## Experiment 4: `min_signal_body_pct`

Archive: [grid_signal_body.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/trend_pullback_continuation/experiment_logs/grid_signal_body.csv)

Grid:

- fixed `delay=5`
- fixed `min_reward_r=1.5`
- fixed `take_profit_fraction_of_trend_move=0.5`
- `min_signal_body_pct = [0.5, 0.6, 0.7]`

Summary:

| min body | planned trades | win rate | total return | sharpe | profit factor | max DD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 0.6 | 39 | 51.28% | 3.11% | 0.96 | 2.58 | 0.46% |
| 0.7 | 32 | 53.12% | 2.68% | 0.93 | 2.71 | 0.46% |

Conclusion:

- Requiring a stronger signal-candle body is an effective way to improve quality.
- `0.7` gives the highest `profit_factor`, but frequency gets very low.
- `0.6` is the best balance found so far:
  - strong improvement in `profit_factor`
  - best `sharpe`
  - only a small drop in total return
  - meaningfully lower drawdown

## Current Recommendation

If the goal is to improve `profit_factor` without making the strategy too thin, the best current candidate is:

- `max_signal_delay_after_third_low = 5`
- `min_reward_r = 1.5`
- `take_profit_fraction_of_trend_move = 0.5`
- `min_signal_body_pct = 0.6`

Expected profile from the tested sweep:

| Metric | Candidate (`body=0.6`) |
| --- | ---: |
| planned_trade_count | 39 |
| trade_win_rate | 51.28% |
| total_return | 3.11% |
| sharpe | 0.96 |
| profit_factor | 2.58 |
| max_drawdown | 0.46% |

High-quality, lower-frequency alternative:

- `min_signal_body_pct = 0.7`

Higher-frequency compromise:

- `min_reward_r = 1.2`
- keep `delay=5`
- keep `take_profit_fraction_of_trend_move=0.5`
