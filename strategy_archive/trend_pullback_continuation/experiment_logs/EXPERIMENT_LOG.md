# 顺趋势回调策略实验日志

最后更新：`2026-04-17`

## 范围

这份日志记录 `trend pullback continuation` 策略到目前为止的实验结果。

统一测试环境：

- 数据：`Dataframes/stock_price.csv`
- 股票池：`csi500`
- 区间：`2020-01-01` 到 `2026-03-16`
- 回测口径：`initial_capital=1,000,000`，`fixed_entry_notional=20,000`，`board_lot_size=100`

基线参数：

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

## 基线

基线结果：

| 指标 | 数值 |
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

## 漏斗分析

基线参数下的信号漏斗：

| 阶段 | 数量 |
| --- | ---: |
| `post_trend_phase` | 466,995 |
| `three_push_pullback` | 6,531 |
| `signal_candle` | 326 |
| `follow_through_confirmed` | 223 |
| `signal_date >= trend_end_date` | 114 |
| `reward_to_risk >= 1.5` | 56 |
| 原始可入场信号 | 56 |
| `trade_df` 最终计划交易 | 50 |

主要结论：

- 这个策略的主瓶颈不是 follow-through。
- 最大的样本损耗来自：
  - `three_push_pullback -> signal_candle`
  - `follow_through_confirmed -> after_trend_end`
  - `after_trend_end -> reward_to_risk >= 1.5`
- 也就是说，频率太低主要是因为：
  - signal K 质量要求较严
  - 必须在 `trend_end` 之后
  - `reward_to_risk` 过滤较严

额外的基线诊断：

- `trend_end` 之后的 follow-through 信号：`114`
- `trend_end` 之前的 follow-through 信号：`109`
- 粗略估算的年化原始入场数：
  - 当前基线（`delay=5`, `rr=1.5`）：约 `8.0`
  - `rr=1.2`：约 `10.3`
  - `rr=1.0`：约 `12.4`
  - 去掉时机过滤但保留 `rr=1.5`：约 `13.7`

## 实验 1：`max_signal_delay_after_third_low x min_reward_r`

归档文件：`grid_delay_rr.csv`

参数网格：

- `max_signal_delay_after_third_low = [5, 8, 10]`
- `min_reward_r = [1.0, 1.2, 1.5]`

结果摘要：

| delay | min_reward_r | 计划交易数 | 胜率 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
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

结论：

- 这一轮里整体最平衡的仍然是 `delay=5`、`min_reward_r=1.5`
- 如果只是想提频，最直接的办法是先降 `min_reward_r`
- 把 `max_signal_delay_after_third_low` 放宽到 `5` 以上，会带来一点样本，但质量会持续变差

建议：

- 保持 `delay=5`
- 如果一定要提频，优先降低 `min_reward_r`，不要先放宽 signal delay

## 实验 2：更严格的 `delay x min_reward_r` 质量扫描

归档文件：`grid_strict_quality.csv`

参数网格：

- `max_signal_delay_after_third_low = [3, 4, 5]`
- `min_reward_r = [1.5, 1.8, 2.0]`

选取的代表性结果：

| delay | min_reward_r | 计划交易数 | 胜率 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 4 | 1.5 | 46 | 44.44% | 2.79% | 0.81 | 2.13 | 0.64% |
| 3 | 1.5 | 39 | 44.74% | 1.94% | 0.64 | 1.90 | 0.71% |
| 5 | 1.8 | 41 | 42.50% | 1.86% | 0.58 | 1.78 | 0.99% |
| 5 | 2.0 | 35 | 44.12% | 2.25% | 0.72 | 2.26 | 0.79% |

结论：

- `min_reward_r` 提到 `1.5` 以上后，样本会掉得很快
- `min_reward_r=2.0` 虽然让 `profit_factor` 略高于基线，但提升幅度不大，不足以覆盖频率和收益的损失
- `delay` 缩到 `5` 以下，并没有带来足够大的质量改善

建议：

- `min_reward_r=1.5` 仍然是最实用的默认值
- `min_reward_r=2.0` 只适合明确追求超低频、超挑剔质量的场景

## 实验 3：`take_profit_fraction_of_trend_move`

归档文件：`grid_take_profit.csv`

参数网格：

- 固定 `delay=5`
- 固定 `min_reward_r=1.5`
- 扫描 `take_profit_fraction_of_trend_move = [0.4, 0.5, 0.6]`

结果：

| 止盈比例 | 计划交易数 | 胜率 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 39 | 47.37% | 1.50% | 0.51 | 1.73 | 0.66% |
| 0.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 0.6 | 60 | 40.35% | 2.75% | 0.54 | 1.71 | 1.62% |

结论：

- `0.5` 是这一轮里最好的止盈比例
- `0.4` 太保守，明显砍掉了上行空间
- `0.6` 太远，止盈更难打到，硬止损和回撤都会变大

建议：

- 保持 `take_profit_fraction_of_trend_move=0.5`

## 实验 4：`min_signal_body_pct`

归档文件：`grid_signal_body.csv`

参数网格：

- 固定 `delay=5`
- 固定 `min_reward_r=1.5`
- 固定 `take_profit_fraction_of_trend_move=0.5`
- 扫描 `min_signal_body_pct = [0.5, 0.6, 0.7]`

结果：

| 最小实体占比 | 计划交易数 | 胜率 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 50 | 46.94% | 3.20% | 0.85 | 2.21 | 0.64% |
| 0.6 | 39 | 51.28% | 3.11% | 0.96 | 2.58 | 0.46% |
| 0.7 | 32 | 53.12% | 2.68% | 0.93 | 2.71 | 0.46% |

结论：

- 提高 signal K 实体要求，确实能改善质量
- `0.7` 的 `profit_factor` 最高，但样本已经变得很薄
- `0.6` 是更合理的平衡点：
  - `profit_factor` 提升明显
  - `sharpe` 最好
  - 总收益只小幅下降
  - 回撤明显更低

## 当前建议

如果目标是在不把样本压得太薄的前提下提升 `profit_factor`，当时最好的候选组合是：

- `max_signal_delay_after_third_low = 5`
- `min_reward_r = 1.5`
- `take_profit_fraction_of_trend_move = 0.5`
- `min_signal_body_pct = 0.6`

对应大概画像：

| 指标 | 候选值（`body=0.6`） |
| --- | ---: |
| planned_trade_count | 39 |
| trade_win_rate | 51.28% |
| total_return | 3.11% |
| sharpe | 0.96 |
| profit_factor | 2.58 |
| max_drawdown | 0.46% |

更高质量、但更低频的替代：

- `min_signal_body_pct = 0.7`

偏高频一点的折中：

- `min_reward_r = 1.2`
- 同时保持 `delay=5`
- 同时保持 `take_profit_fraction_of_trend_move=0.5`
