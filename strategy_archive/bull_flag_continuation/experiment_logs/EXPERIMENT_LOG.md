# 牛旗策略实验日志

最后更新：`2026-04-19`

## 范围

这份日志记录独立 `bull flag continuation` 策略的迭代研究过程。

统一测试环境：

- 数据：`Dataframes/stock_price.csv`
- 股票池：`csi500`
- 区间：`2020-01-01` 到 `2026-03-16`
- 回测资金与下单口径：`initial_capital=1,000,000`，`fixed_entry_notional=20,000`，`board_lot_size=100`

初始基线参数：

- `ma_windows=(20, 60, 120)`
- `pivot_window=1`
- `flagpole_lookback_bars=20`
- `min_flagpole_bars=5`
- `max_flagpole_bars=20`
- `min_flagpole_return=0.12`
- `min_flag_bars=4`
- `max_flag_bars=15`
- `max_flag_retrace_ratio=0.40`
- `max_flag_channel_slope_pct_per_bar=0.008`
- `max_flag_width_pct=0.12`
- `min_breakout_body_pct=0.60`
- `max_breakout_upper_shadow_pct=0.25`
- `max_breakout_lower_shadow_pct=0.35`
- `measured_move_fraction=0.75`
- `min_reward_r=1.50`
- `stop_buffer_pct=0.01`

## 基线

默认 bull flag 参数下的结果：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 335 |
| entered_trade_count | 327 |
| closed_trade_count | 320 |
| open_trade_count | 7 |
| entry_fill_rate | 97.61% |
| trade_win_rate | 35.00% |
| average_trade_return | 0.04% |
| profit_factor | 1.01 |
| sharpe | 0.04 |
| max_drawdown | 5.18% |
| total_return | 0.41% |
| benchmark_total_return | 257.64% |
| excess_return | -257.23% |
| hard_stop_rate | 67.68% |
| take_profit_rate | 32.32% |

初始结论：

- 牛旗版本的优点是样本量足够。
- 但默认参数太松，质量基本接近打平。
- 第一优先级不是再提频，而是先把形态定义收紧，尤其是旗面回撤深度。

当时的下一步计划：

- 先扫 `max_flag_retrace_ratio`。

## 第 1 轮：`max_flag_retrace_ratio`

本轮做了什么：

- 其他参数全部固定在基线值。
- 扫描 `max_flag_retrace_ratio = [0.25, 0.30, 0.35, 0.40]`。

结果：

| 回撤上限 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 | 止损占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 82 | 42.31% | 1.31% | 1.84% | 0.27 | 1.28 | 2.04% | 59.76% |
| 0.30 | 162 | 39.35% | 1.23% | 4.19% | 0.47 | 1.31 | 2.49% | 63.13% |
| 0.35 | 239 | 34.78% | 0.00% | 0.48% | 0.05 | 1.01 | 3.71% | 67.66% |
| 0.40 | 335 | 35.00% | 0.04% | 0.41% | 0.04 | 1.01 | 5.18% | 67.68% |

结论：

- `0.35` 和 `0.40` 明显太松，质量会直接塌掉。
- `0.30` 是频率和质量最平衡的一档。
- `0.25` 更干净，但机会掉得有点多。

下一步计划：

- 把 `max_flag_retrace_ratio` 固定到 `0.30`。
- 继续研究 signal K，本轮先从实体大小开始。

## 第 2 轮：粗扫 `min_breakout_body_pct`

本轮做了什么：

- 固定 `max_flag_retrace_ratio=0.30`
- 扫描 `min_breakout_body_pct = [0.60, 0.70, 0.80]`

结果：

| 最小实体占比 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 162 | 39.35% | 1.23% | 4.19% | 0.47 | 1.31 | 2.49% |
| 0.70 | 115 | 36.11% | 0.65% | 1.90% | 0.27 | 1.17 | 2.26% |
| 0.80 | 60 | 37.50% | 1.32% | 1.79% | 0.41 | 1.33 | 1.19% |

结论：

- signal K 确实需要足够实体，但不是实体越大越好。
- `0.70` 这一档明显变差。
- `0.80` 质量略有改善，但样本太薄。
- 当前最实用的起点仍然是 `0.60`。

下一步计划：

- 保持 `min_breakout_body_pct=0.60`
- 继续研究 wick，先看上影线。

## 第 3 轮：`max_breakout_upper_shadow_pct`

本轮做了什么：

- 固定 `max_flag_retrace_ratio=0.30`
- 固定 `min_breakout_body_pct=0.60`
- 扫描 `max_breakout_upper_shadow_pct = [0.15, 0.25, 0.35]`

结果：

| 上影线占比上限 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 122 | 33.62% | 0.53% | 1.58% | 0.22 | 1.13 | 2.32% |
| 0.25 | 162 | 39.35% | 1.23% | 4.19% | 0.47 | 1.31 | 2.49% |
| 0.35 | 174 | 39.76% | 1.54% | 5.08% | 0.55 | 1.38 | 2.64% |

结论：

- 上影线控制有用，但不能太苛刻。
- 带一点上影的 breakout bar 依然可以是好信号。
- `0.35` 比 `0.25` 更好，说明之前对“干净突破”的要求有点过头。

下一步计划：

- 把 `max_breakout_upper_shadow_pct` 固定为 `0.35`
- 继续看下影线容忍度。

## 第 4 轮：`max_breakout_lower_shadow_pct`

本轮做了什么：

- 固定 `max_flag_retrace_ratio=0.30`
- 固定 `min_breakout_body_pct=0.60`
- 固定 `max_breakout_upper_shadow_pct=0.35`
- 扫描 `max_breakout_lower_shadow_pct = [0.20, 0.35, 0.50]`

结果：

| 下影线占比上限 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 140 | 41.35% | 1.80% | 4.71% | 0.59 | 1.44 | 2.10% |
| 0.35 | 174 | 39.76% | 1.54% | 5.08% | 0.55 | 1.38 | 2.64% |
| 0.50 | 176 | 40.48% | 1.70% | 5.65% | 0.60 | 1.43 | 2.50% |

结论：

- 下影线和上影线的行为不一样。
- 下影特别短并不是必须条件。
- 放宽到 `0.50` 并没有伤害策略，反而在总收益和 Sharpe 上更强。
- 这说明一些好 breakout 会在日内先回踩，再强势收回来。

下一步计划：

- 保持 `max_breakout_lower_shadow_pct=0.50`
- 在新的 wick 设定下，再精细回看实体阈值。

## 第 5 轮：细扫 `min_breakout_body_pct`

本轮做了什么：

- 固定 `max_flag_retrace_ratio=0.30`
- 固定 `max_breakout_upper_shadow_pct=0.35`
- 固定 `max_breakout_lower_shadow_pct=0.50`
- 扫描 `min_breakout_body_pct = [0.50, 0.55, 0.60]`

结果：

| 最小实体占比 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 201 | 37.31% | 1.29% | 5.03% | 0.49 | 1.33 | 2.51% |
| 0.55 | 185 | 38.98% | 1.53% | 5.40% | 0.56 | 1.39 | 2.52% |
| 0.60 | 176 | 40.48% | 1.70% | 5.65% | 0.60 | 1.43 | 2.50% |

结论：

- 在更合理的 wick 组合下，实体阈值在 `0.50 -> 0.60` 之间是单调改善的。
- `0.60` 目前仍是最平衡的一档。
- 说明第 2 轮里“实体不单调”的现象，有一部分是被更差的 wick 条件干扰了。

下一步计划：

- 把 `retrace=0.30`、`body=0.60`、`upper=0.35`、`lower=0.50` 作为当前最优候选。
- 再做一轮解释型分析，直接看赢单和亏单的 signal K 到底长什么样。

## 第 6 轮：signal K 解释型分析

使用配置：

- `max_flag_retrace_ratio=0.30`
- `min_breakout_body_pct=0.60`
- `max_breakout_upper_shadow_pct=0.35`
- `max_breakout_lower_shadow_pct=0.50`

本轮做了什么：

- 对当前最优候选下的已平仓交易，比较赢单和亏单的 signal K 特征。
- 重点看：
  - `signal_body_pct`
  - `signal_upper_shadow_pct`
  - `signal_lower_shadow_pct`
  - `flag_retrace_ratio`
  - `reward_to_risk`

赢亏对比：

| 特征 | 赢单均值 | 亏单均值 | 赢单中位数 | 亏单中位数 |
| --- | ---: | ---: | ---: | ---: |
| `signal_body_pct` | 0.750 | 0.765 | 0.732 | 0.750 |
| `signal_upper_shadow_pct` | 0.123 | 0.108 | 0.120 | 0.097 |
| `signal_lower_shadow_pct` | 0.127 | 0.127 | 0.100 | 0.119 |
| `flag_retrace_ratio` | 0.236 | 0.243 | 0.238 | 0.257 |
| `reward_to_risk` | 3.434 | 2.889 | 2.145 | 2.458 |

一些分桶观察：

- `signal_body_pct`
  - 胜率最好的桶反而是最低的有效实体分位，也就是大约 `0.60` 到 `0.67`
  - 特别夸张的大实体不是最优桶
- `signal_upper_shadow_pct`
  - 在允许范围内，上影线最高的那个分位反而胜率最好
  - 说明不能把“几乎没有上影”当成必须条件
- `signal_lower_shadow_pct`
  - 最好的桶是中等下影，大约 `0.04` 到 `0.11`
  - 下影特别短和特别长都不如中间值

解释：

- 好的 signal K 不是完美光头阳线。
- 更像是：
  - 明显偏多
  - 实体够大
  - 没有明显 rejection
  - 但允许日内上下探一探
- 结构背景仍然重要：
  - 更浅的旗面更好
  - 更高的 `reward_to_risk` 确实有帮助

当时的下一步计划：

- signal K 这块已经比较清楚了。
- 如果继续迭代，下一步应该去看：
  - breakout bar 的 close location
  - 或者 follow-through 质量

## 当前最佳候选

目前通过 signal K 方向研究得到的最好参数组合：

- `max_flag_retrace_ratio = 0.30`
- `min_breakout_body_pct = 0.60`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`

对应大概画像：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 176 |
| win rate | 40.48% |
| average trade return | 1.70% |
| total return | 5.65% |
| sharpe | 0.60 |
| profit factor | 1.43 |
| max_drawdown | 2.50% |
| hard_stop_rate | 62.43% |
| take_profit_rate | 37.57% |

## 阶段性认识

经过前 6 轮，当前认识是：

1. 最重要的结构修正是把旗面深度收紧到 `0.30`
2. signal K 很重要，但“完美 breakout bar”这个想法是错的
3. 当前策略真正想要的是：
   - 足够实体
   - 允许一定 wick
   - 上影不能太苛刻
   - 下影中等反而比完全没有更合理
4. 下一步更有价值的优化，应该来自 close location 或 follow-through 质量，而不是继续死抠 body / wick

## 代码检查

本轮做了什么：

- 通读 [bull_flag_continuation.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_continuation.py)
- 重跑 bull flag 定向测试：`10 passed`
- 重点看了：
  - `reward_to_risk` 算法是否有错
  - `signal -> follow-through -> t+2 entry` 时序是否对齐
  - `inspect / plot / trade_df` 是否会对不上
  - `add_signals()` / `add_trade_df()` 是否有缓存旧列的问题

结论：

- 没发现会直接让回测结论失真的 blocking bug
- 当前代码可以继续做 follow-through 研究
- 有一个非阻塞观察点：
  - 如果同一根 breakout bar 能被多个 pivot high 解释，后面的有效 peak 会覆盖前面的注释
  - 目前把它看成“最近的有效旗面优先”，不当成 correctness bug

下一步计划：

- 从 signal K 继续推进到 follow-through 质量

## 第 7 轮：follow-through 诊断分析

使用配置：

- `max_flag_retrace_ratio=0.30`
- `min_breakout_body_pct=0.60`
- `max_breakout_upper_shadow_pct=0.35`
- `max_breakout_lower_shadow_pct=0.50`

本轮做了什么：

- 把每笔已执行交易和它的 `t+1` follow-through bar 拼在一起
- 衍生了这些特征：
  - `ft_close_gt_open`
  - `ft_close_gt_signal_high`
  - `ft_close_gt_signal_close`
  - `ft_body_pct`
  - `ft_upper_shadow_pct`
  - `ft_lower_shadow_pct`
  - `ft_close_position`
  - `ft_return_vs_signal_close`

关键观察：

- `ft_close_gt_signal_close` 比基线更有信息量
- follow-through 的下影线 surprisingly 有用：
  - **中等**下影明显好于接近 0 或很长下影
- follow-through 也不是越“完美”越好

代表性结果：

- `ft_close_gt_signal_close`
  - 为真：胜率 `44.16%`，均值收益 `2.87%`
  - 为假：胜率 `37.50%`，均值收益 `1.20%`
- `ft_lower_shadow_pct`
  - 最优四分位大约在 `0.12` 到 `0.22`
  - 该桶胜率 `63.64%`，均值收益 `6.53%`

结论：

- follow-through bar 里确实有额外信息
- 最稳健、最值得先测试的简单过滤条件是：
  - `follow-through close > signal close`

下一步计划：

- 直接把这个条件放进回测口径里试效果

## 第 8 轮：简单 follow-through 收盘过滤

使用配置：

- 与第 7 轮相同

本轮做了什么：

- 比较以下几种版本：
  - baseline
  - 要求 `ft_close > ft_open`
  - 要求 `ft_close > signal_close`
  - 要求 `ft_close > signal_high`

结果：

| 版本 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 176 | 40.48% | 1.70% | 5.65% | 0.60 | 1.43 | 2.50% |
| `ft_close > ft_open` | 76 | 43.06% | 2.09% | 2.53% | 0.46 | 1.41 | 1.49% |
| `ft_close > signal_close` | 80 | 44.00% | 2.63% | 3.43% | 0.58 | 1.53 | 1.48% |
| `ft_close > signal_high` | 57 | 44.44% | 2.56% | 2.48% | 0.49 | 1.48 | 1.35% |

结论：

- 最干净的简单升级是 `ft_close > signal_close`
- 它可以改善：
  - 胜率
  - 平均单笔收益
  - Profit Factor
  - 回撤
- 代价是样本量明显下降
- `ft_close > signal_high` 更激进，但提升幅度不足以覆盖额外掉样本的代价

下一步计划：

- 看看如果再加更“完美”的 follow-through 条件，会不会只是把样本压得太薄

## 第 9 轮：更严格的 follow-through 组合

使用配置：

- 与第 8 轮相同

本轮做了什么：

- 比较：
  - baseline
  - 要求 `ft_close > signal_close`
  - 要求 `ft_close > signal_close` 且 `ft_body_pct >= 0.40`

结果：

| 版本 | 计划交易数 | 胜率 | 平均单笔收益 | 总收益 | Sharpe | Profit Factor | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 176 | 40.48% | 1.70% | 5.65% | 0.60 | 1.43 | 2.50% |
| `ft_close > signal_close` | 80 | 44.00% | 2.63% | 3.43% | 0.58 | 1.53 | 1.48% |
| `ft_close > signal_close` 且 `ft_body_pct >= 0.40` | 35 | 46.88% | 2.83% | 1.59% | 0.45 | 1.47 | 0.82% |

结论：

- `ft_close > signal_close` 这个条件是真有用的
- 再叠一个“follow-through 必须有明显实体”的条件，样本会变得过薄
- 这和 signal K 的结论一致：
  - 要求强，但不能要求得太完美

## 更新后的最佳候选

目前有两个实用版本：

### 平衡版

- `max_flag_retrace_ratio = 0.30`
- `min_breakout_body_pct = 0.60`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`

画像：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 176 |
| win rate | 40.48% |
| average trade return | 1.70% |
| total return | 5.65% |
| sharpe | 0.60 |
| profit factor | 1.43 |
| max_drawdown | 2.50% |

### 更高质量的 follow-through 版本

- 平衡版基础上额外要求：
  - `follow-through close > signal close`

画像：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 80 |
| win rate | 44.00% |
| average trade return | 2.63% |
| total return | 3.43% |
| sharpe | 0.58 |
| profit factor | 1.53 |
| max_drawdown | 1.48% |

## 当前认识

经过 follow-through 这一轮后，当前认识更新为：

1. 最重要的结构优化仍然是 `max_flag_retrace_ratio=0.30`
2. 好的 signal K 是“足够强”，不是“完美”
3. 好的 follow-through 也是一样：
   - `close > signal close` 这个确认条件很有价值
   - 但继续叠更苛刻的“完美 K”条件，收益不大，掉样本更严重
4. 当前最值得落地的实现改动是：
   - 增加一个可选参数，让策略支持 `t+1 close > signal close`
5. 如果后面继续研究，优先方向会是：
   - close location / continuation-through-close 特征
   - 而不是继续把 body / wick 阈值越拧越紧

## 第 10 轮：把 follow-through 条件做成可选参数

本轮做了什么：

- 给 bull flag 策略新增一个可选开关：
  - `require_follow_through_close_gt_signal_close: bool = False`
- 当这个开关打开时，`entry_signal` 还需要额外满足：
  - `follow_through_close > signal_close`
- 这个条件已经同步接入：
  - signal generation
  - `inspect_signal`
  - `get_next_session_candidates`
  - trade-level 输出

为什么这样做：

- 第 7 到第 9 轮已经证明：`follow-through close > signal close` 是当前最干净的 follow-through 升级
- 它可以改善 `profit_factor`，同时降低回撤，而且不依赖过度理想化的 K 线形态

验证：

- bull flag 定向测试：`11 passed`
- 全量测试：`71 passed`

实现结论：

- 现在代码层面已经支持两种版本：
  - 平衡版 bull flag
  - 更高质量的 follow-through 版 bull flag
- 这个条件默认关闭，所以不会改变当前默认策略行为

## 第 11 轮：修复背景失效后旧 setup 被复用的问题，并重新回测当前最优参数

本轮做了什么：

- 检查 bull flag 识别逻辑在背景过滤上的边界行为
- 发现旧逻辑虽然要求：
  - `flag peak` 当天必须处于 `bullish_stack`
  - breakout 当天也必须处于 `bullish_stack`
- 但如果旗面中间某一天 `bullish_stack` 先断掉、后面又恢复，旧 setup 仍可能被继续复用
- 已修复为：
  - 从 `flag_start` 到 breakout 候选这整段里，`bullish_stack` 只要中途断过一次，这套旧 bull flag 直接失效
- 同时补了一个定向测试，验证“背景中途断掉后 setup 不会复活”
- 顺手修复了 `trend_pullback_continuation.py` 里一个无关但会影响全量测试的 typo：
  - `signal_quality_ok` 原先误写成了 `df["z"]`
  - 现已改回 `df["signal_body_pct"]`

验证：

- bull flag 定向测试：`12 passed`
- 全量测试：`72 passed`

重新回测的参数：

- `max_flag_retrace_ratio = 0.30`
- `min_breakout_body_pct = 0.60`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`

回测窗口与口径：

- 数据：`Dataframes/stock_price.csv`
- 时间：`2020-01-01` 到 `2026-03-16`
- 回测参数：
  - `initial_capital = 1,000,000`
  - `fixed_entry_notional = 20,000`
  - `board_lot_size = 100`

修复后结果：

| 指标 | 修复后 |
| --- | ---: |
| planned_trade_count | 196 |
| entered_trade_count | 191 |
| closed_trade_count | 188 |
| open_trade_count | 3 |
| entry_fill_rate | 97.45% |
| win rate | 40.96% |
| average trade return | 1.78% |
| total return | 6.64% |
| sharpe | 0.66 |
| profit factor | 1.44 |
| max_drawdown | 2.48% |
| hard_stop_rate | 61.70% |
| take_profit_rate | 38.30% |

与修复前日志基线对比：

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| planned_trade_count | 176 | 196 | +20 |
| win rate | 40.48% | 40.96% | +0.48pct |
| average trade return | 1.70% | 1.78% | +0.08pct |
| total return | 5.65% | 6.64% | +0.99pct |
| sharpe | 0.60 | 0.66 | +0.06 |
| profit factor | 1.43 | 1.44 | +0.01 |
| max_drawdown | 2.50% | 2.48% | -0.02pct |
| hard_stop_rate | 62.43% | 61.70% | -0.73pct |
| take_profit_rate | 37.57% | 38.30% | +0.73pct |

本轮结论：

- 这次修复没有让策略变差，反而结果略有改善
- 从最终表现看，当前最优参数在修复后仍然成立，甚至更稳一点：
  - 胜率略升
  - `profit_factor` 略升
  - 总收益和 `sharpe` 略升
  - 回撤略降
- 也就是说，之前这个“背景失效后旧 setup 可能被复用”的边界问题，不是这套策略收益的主要来源；修掉之后策略画像没有被破坏

下一步计划：

- 继续在当前修复后的版本上研究 follow-through 与 breakout close 质量
- 重点看：
  - breakout 收盘在当日振幅中的位置
  - follow-through bar 的 close location
  - 是否值得把 `follow_through_close > signal_close` 纳入默认版本

## 第 12 轮：开始研究“当前仍在运行中的多头趋势环境”的敏感度

本轮目标：

- 不再只看 signal K 和 follow-through
- 转而回答一个更根本的问题：
  - bull flag 对“正在运行中的多头背景”到底敏感在哪里？
  - 是不是越强的趋势背景越好？
  - 还是说，过热的趋势环境反而更差？

本轮代码改动：

- 新增研究工具：[bull_flag_environment_sensitivity.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/score_system/bull_flag_environment_sensitivity.py)
- 这个工具做两件事：
  1. 用当前 bull flag 策略先跑出一份完整 `trade_df`
  2. 在不重复重建 researcher 的前提下，对“更严格的背景过滤”做快速 post-hoc grid search
- 同时会给每笔 trade 附加背景诊断特征，包括：
  - `signal_bullish_stack_run_length`
  - `peak_bullish_stack_run_length`
  - `signal_stack_spread_pct`
  - `peak_stack_spread_pct`
  - `signal_sma20_return_5`
  - `peak_sma20_return_5`
  - `signal_sma60_return_10`
  - `peak_sma60_return_10`
  - `signal_close_to_sma120_pct`
  - `peak_high_to_sma120_pct`

额外说明：

- 这一步只是研究工具，不改变默认 bull flag 策略行为
- 目的只是为了先把“背景强弱”的信息量看清楚

输出文件：

- [bull_flag_environment_trade_frame.pkl](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/bull_flag_environment_trade_frame.pkl)
- [bull_flag_environment_trade_frame_with_actuals.pkl](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/bull_flag_environment_trade_frame_with_actuals.pkl)
- [bull_flag_environment_feature_buckets.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/bull_flag_environment_feature_buckets.csv)

## 第 13 轮：先做基线分桶，判断背景强弱到底怎么影响结果

本轮做了什么：

- 基于当前修复后的“最佳基础配置”跑出完整 trade frame
- 按四分位数对背景特征分桶，观察：
  - 胜率
  - 平均单笔收益
  - `hard_stop_rate`
  - `take_profit_rate`

关键发现：

1. `signal_stack_spread_pct` 是最有信息量的背景特征之一
   - 越小越好
   - 说明 signal 当天如果 `SMA20` 和 `SMA120` 已经拉得很开，反而更像“趋势过热”
   - 这和“越强的背景越好”相反

2. `peak_high_to_sma120_pct` 也有明显模式
   - `flag peak` 距离慢均线越远，表现越差
   - 说明旗杆冲得太离谱，后面旗形更容易失败

3. `peak_sma60_return_10` 的信息量非常高
   - 太低不算好
   - 但太高更差
   - 最好的不是“最陡的中期趋势”，而是一个中等偏强、但不过热的上行斜率

4. `bullish_stack_run_length` 有信息，但没有前面几个强
   - 不是“多头排列持续越久越好”
   - 更像是中等时长最好，过久反而进入成熟/拥挤阶段

阶段性认识：

- bull flag 对背景的敏感点，不是“强趋势”本身
- 而是“处在健康的持续上行里，但还没热到离谱”
- 这套策略想要的是：
  - 有趋势
  - 但不过热
  - 不是极度发散、成熟过头的末端趋势

## 第 14 轮：单因子背景过滤网格

本轮做了什么：

- 分别单独扫描这些背景过滤：
  - `signal_stack_spread_pct__max`
  - `peak_high_to_sma120_pct__max`
  - `peak_sma60_return_10__max`
  - `signal_bullish_stack_run_length__min`
  - `signal_bullish_stack_run_length__max`
  - `peak_bullish_stack_run_length__max`

对应输出：

- [signal_stack_spread_max.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/signal_stack_spread_max.csv)
- [peak_high_to_sma120_max.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/peak_high_to_sma120_max.csv)
- [peak_sma60_return_10_max.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/peak_sma60_return_10_max.csv)
- [signal_stack_run_min.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/signal_stack_run_min.csv)
- [signal_stack_run_max.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/signal_stack_run_max.csv)
- [peak_stack_run_max.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/peak_stack_run_max.csv)

单因子最重要的结论：

1. 最强的单因子是 `signal_stack_spread_pct__max`
   - `<= 0.10` 时，质量提升非常明显
   - 画像：
     - `planned_trade_count = 62`
     - `win_rate = 51.61%`
     - `average_trade_return = 4.08%`
     - `profit_factor = 2.53`
     - `total_return = 4.66%`
     - `sharpe = 1.12`
     - `max_drawdown = 0.83%`
   - 这版非常像“高质量、低频率”的 bull flag

2. 最平衡的单因子是 `peak_sma60_return_10__max`
   - 这一项很特别：
     - 不像 `signal_stack_spread_pct` 那样极端压缩样本
     - 却能同时改善 `profit_factor`、`sharpe` 和 `total_return`
   - 粗扫里最好的点落在 `0.05 ~ 0.06`

3. `run_length` 相关过滤价值有限
   - 能带来一点形状变化
   - 但整体不如“热度/离均线距离”这类过滤有效

阶段性认识：

- 这轮几乎已经可以确定：
  - 背景敏感度的核心不是趋势持续多久
  - 而是趋势是否已经过热

## 第 15 轮：双因子组合验证

本轮做了什么：

- 检查单因子里最有信息量的过滤之间是“互补”还是“重复”
- 主要测了：
  - `signal_stack_spread_pct__max x peak_sma60_return_10__max`
  - `signal_stack_spread_pct__max x peak_high_to_sma120_pct__max`
  - `signal_stack_spread_pct__max x signal_sma20_return_5__min`

对应输出：

- [spread_x_peak_sma60.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/spread_x_peak_sma60.csv)
- [spread_x_peak_distance.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/spread_x_peak_distance.csv)
- [spread_x_signal_sma20_min.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/spread_x_signal_sma20_min.csv)

关键发现：

1. `signal_stack_spread_pct__max` 与 `peak_sma60_return_10__max` 高度重合
   - 一旦把 `signal_stack_spread_pct` 卡到比较低，`peak_sma60_return_10` 再加上去几乎没提供额外增益
   - 说明这两个特征描述的是同一个核心现象：
     - 趋势是否已经过热 / 发散

2. `signal_stack_spread_pct__max` 与 `peak_high_to_sma120_pct__max` 也有明显重合
   - 组合以后确实还能再提高一点质量
   - 但代价是样本继续变薄
   - 更像是把“高质量低频版”压得更极端

3. `signal_sma20_return_5__min` 可以作为“别太弱”的补充条件
   - 它能进一步清掉一部分弱 breakout
   - 但本质上是在拿频率换质量
   - 并没有像 `peak_sma60_return_10__max` 那样形成特别平衡的提升

阶段性认识：

- 这轮以后我基本确定：
  - 组合过滤还有边际价值
  - 但信息增量已经不大
  - 如果继续堆更多环境过滤，本质上只是在做不同版本的“更薄、更纯”

## 第 16 轮：细扫 `peak_sma60_return_10`，找平衡点

本轮做了什么：

- 因为 `peak_sma60_return_10__max` 是最平衡的单因子
- 所以对它做精细网格：
  - `0.045 / 0.050 / 0.055 / 0.060 / 0.070`

输出：

- [peak_sma60_return_10_fine_grid.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/peak_sma60_return_10_fine_grid.csv)

细扫结果：

| 阈值 | planned_trade_count | win_rate | avg trade return | profit_factor | total_return | sharpe | max_drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.045` | 129 | 44.35% | 2.64% | 1.78 | 6.14% | 0.86 | 1.71% |
| `0.050` | 139 | 45.52% | 2.89% | 1.87 | 7.35% | 0.97 | 1.56% |
| `0.055` | 150 | 44.83% | 2.86% | 1.84 | 7.88% | 0.97 | 1.43% |
| `0.060` | 156 | 44.00% | 2.62% | 1.76 | 7.65% | 0.91 | 1.83% |
| `0.070` | 165 | 44.03% | 2.46% | 1.67 | 7.52% | 0.84 | 1.61% |

本轮结论：

- 如果要一个“最平衡”的背景增强版本，最佳点是：
  - `max_peak_sma60_return_10 = 0.055`
- 它的画像是：
  - `planned_trade_count = 150`
  - `win_rate = 44.83%`
  - `average_trade_return = 2.86%`
  - `profit_factor = 1.84`
  - `total_return = 7.88%`
  - `sharpe = 0.97`
  - `max_drawdown = 1.43%`

这比当前修复后的基础版：

- `planned_trade_count = 196`
- `win_rate = 40.96%`
- `average_trade_return = 1.78%`
- `profit_factor = 1.44`
- `total_return = 6.64%`
- `sharpe = 0.66`
- `max_drawdown = 2.48%`

明显更平衡。

## 第 17 轮：把有效的背景过滤正式接进策略

本轮代码改动：

### 策略参数

在 [bull_flag_continuation.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_continuation.py) 中新增了 3 个可选背景过滤参数：

- `max_signal_stack_spread_pct`
- `min_signal_sma20_return_5`
- `max_peak_sma60_return_10`

默认值都为 `None`，所以：

- 不会影响当前默认 bull flag 行为
- 只有显式打开时才会生效

### 诊断特征

策略现在会直接产出这些背景诊断列：

- `bullish_stack_run_length`
- `stack_spread_pct`
- `sma20_return_5`
- `sma60_return_10`
- `signal_bullish_stack_run_length`
- `signal_stack_spread_pct`
- `signal_sma20_return_5`
- `peak_bullish_stack_run_length`
- `peak_sma60_return_10`

### 信号过滤开关

同时新增这些布尔列：

- `signal_stack_spread_ok`
- `signal_sma20_return_ok`
- `peak_sma60_return_ok`
- `trend_environment_ok`

`entry_signal` 现在在原有条件基础上，还会额外要求：

- `trend_environment_ok`

### 检查/展示层

这些新列也同步接入了：

- `get_candidates()`
- `get_next_session_candidates()`
- `inspect_signal()`
- 条件 checklist
- `trade_df`

### 测试

新增了 bull flag 定向测试，覆盖：

- `max_signal_stack_spread_pct` 会阻止过度发散的 setup
- `max_peak_sma60_return_10` 会阻止过热峰值环境

验证结果：

- bull flag 测试：`14 passed`
- 全量测试：`74 passed`

## 第 18 轮：用正式接入策略的参数重新实跑确认

本轮目的：

- 确认“post-hoc 过滤得到的好结果”在真正接进策略以后依然成立

实跑了两版：

### 版本 A：平衡增强版

- 参数：
  - `max_peak_sma60_return_10 = 0.055`

结果：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 150 |
| entered_trade_count | 147 |
| closed_trade_count | 145 |
| win_rate | 44.83% |
| average_trade_return | 2.86% |
| profit_factor | 1.84 |
| total_return | 7.88% |
| sharpe | 0.97 |
| max_drawdown | 1.43% |
| hard_stop_rate | 57.93% |
| take_profit_rate | 42.07% |

### 版本 B：高质量低频版

- 参数：
  - `max_signal_stack_spread_pct = 0.10`

结果：

| 指标 | 数值 |
| --- | ---: |
| planned_trade_count | 62 |
| entered_trade_count | 62 |
| closed_trade_count | 62 |
| win_rate | 51.61% |
| average_trade_return | 4.08% |
| profit_factor | 2.53 |
| total_return | 4.66% |
| sharpe | 1.12 |
| max_drawdown | 0.83% |
| hard_stop_rate | 50.00% |
| take_profit_rate | 50.00% |

## 当前最终认识

经过这轮关于“当前仍在运行中的多头趋势环境”的敏感度研究，结论已经比较清楚：

1. bull flag 不是越强趋势越好
   - 真正有效的是：
     - 有趋势
     - 但不要过热
     - 不要离慢均线太远
     - 不要让中期均线斜率冲得太离谱

2. “过热过滤”比“趋势持续时长过滤”更重要
   - `run_length` 有信息，但不是主因
   - `spread / 离慢均线距离 / 中期均线斜率` 才是核心

3. 最平衡的背景过滤是：
   - `max_peak_sma60_return_10 = 0.055`
   - 它是当前最值得默认尝试的“增强版 bull flag”

4. 如果目标是极致质量，而不是频率：
   - `max_signal_stack_spread_pct = 0.10`
   - 这版更像高质量、低频率的精选版

5. 继续往下叠更多背景过滤，已经开始明显进入“拿样本换更漂亮指标”的阶段
   - 还能继续把 `profit_factor` 往上抬
   - 但代价是交易数迅速变薄
   - 所以这轮可以认为已经到了“继续复杂化，边际收益不高”的位置

## 当前建议

如果你回来后想继续往实战版本推进，我建议先按两条路线分开：

### 路线 A：平衡增强版

- 在当前最佳基础配置上额外加：
  - `max_peak_sma60_return_10 = 0.055`

适合：

- 希望保留相对像样的频率
- 同时改善：
  - 总收益
  - `profit_factor`
  - `sharpe`
  - 回撤

### 路线 B：高质量精选版

- 在当前最佳基础配置上额外加：
  - `max_signal_stack_spread_pct = 0.10`

适合：

- 更在意质量和稳定度
- 可以接受明显更低的交易频率

## 这轮为何停在这里

我继续试过把：

- `signal_stack_spread_pct`
- `peak_high_to_sma120_pct`
- `signal_sma20_return_5`
- `peak_sma60_return_10`

做双因子、三因子组合。

结论是：

- 的确还能做出更漂亮的 `profit_factor`
- 但开始明显牺牲 trade count
- 不再像 `max_peak_sma60_return_10=0.055` 这样同时改善多个核心指标

所以这轮我判断已经到了：

- “认识足够清晰”
- “继续试不会产生同级别的新结论”

的阶段，因此在这里收束。

## 第 19 轮：动态止盈退出变体代码架构

本轮做了什么：

- 没有覆盖现有的 `bull_flag_continuation.py` 基线版本。
- 新增了独立模块：
  - `strategies/bull_flag_exit_variants.py`
- 在这个模块里实现了两类**与当前 TradePlanBacktester 兼容**的动态退出版本：
  - `BullFlagBreakevenAfterTp1Researcher`
  - `BullFlagTrailingAfterTp1Researcher`
- 新增参数配置：
  - `BullFlagDynamicExitConfig`
  - `tp1_fraction_of_target`
  - `breakeven_buffer_pct`
  - `trailing_stop_fraction_of_flagpole`
- 新增测试：
  - `tests/test_bull_flag_exit_variants.py`
- 同时更新导出：
  - `strategies/__init__.py`

这轮的关键技术判断：

1. 现有 `TradePlanBacktester` 是“单笔计划单、单次完整退出”的模型
   - 它要求同一 ticker 的 trade plan 不能重叠
   - 因此**不能**用两条并行 planned trade 去伪装“50% 先止盈、50% 后续再跑”

2. 所以动态退出先拆成两层
   - 第一层：完全兼容当前回测器的“单次完整退出”版本
     - TP1 后保本 stop
     - TP1 后 trailing stop
   - 第二层：如果以后要认真做分批止盈
     - 必须单独做 multi-leg / partial-exit 研究层
     - 不能直接硬塞进当前 backtester

3. 这次动态退出的日线语义约定
   - 先定义 `TP1`
   - 如果某根日线第一次打到 `TP1`
   - 新的保本 stop / trailing stop **从下一根 bar 才开始生效**
   - 这样可以避免在同一根日线里做不现实的 intraday 顺序猜测

单测验证结果：

- `python -m pytest tests/test_bull_flag_exit_variants.py -q`
  - `4 passed`
- `python -m pytest tests/test_bull_flag_continuation.py -q`
  - `14 passed`

结论：

- 退出变体的代码结构已经搭好，且没有污染静态基线版本。
- 现在可以在不改 entry 逻辑的前提下，单独比较不同 exit policy。

下一步计划：

- 用当前 bull flag 最优背景/信号配置，先做一轮真实样本对比：
  - 静态止盈基线
  - `TP1 -> 保本 stop`
  - `TP1 -> trailing stop`

## 第 20 轮：动态退出初测（权威口径）

本轮做了什么：

- 使用当前 bull flag 较优配置作为固定 entry 基线：
  - `max_flag_retrace_ratio=0.30`
  - `min_breakout_body_pct=0.60`
  - `max_breakout_upper_shadow_pct=0.35`
  - `max_breakout_lower_shadow_pct=0.50`
  - `max_peak_sma60_return_10=0.055`
- 对比三种退出版本：
  - `baseline_static`
  - `tp1_breakeven`
  - `tp1_trailing`
- 这轮结果以**全历史输入 + 回测窗口裁切 + backtester 直接吃 `trade_df`**为准
  - 这样不会丢失 2020 年前的均线/结构预热
  - 也避免 `researcher` 传给 backtester 时重复重算 trade plan
- 输出文件：
  - `strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_variant_comparison_tp1_full_history.csv`

结果：

| 版本 | 计划交易数 | 已入场 | 已平仓 | 胜率 | 平均单笔收益 | Profit Factor | 总收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_static` | 150 | 147 | 145 | 44.83% | 2.86% | 1.84 | 7.88% | 0.97 | 1.43% |
| `tp1_breakeven` | 150 | 147 | 146 | 48.63% | 2.63% | 2.02 | 6.84% | 1.01 | 1.45% |
| `tp1_trailing` | 150 | 147 | 146 | 59.59% | 2.77% | 2.24 | 7.36% | 1.27 | 0.89% |

结论：

1. 动态退出是有价值的
   - 不改 entry，只改 exit，指标已经明显变化

2. `TP1 -> 保本 stop`
   - 比静态基线：
     - 胜率更高
     - `profit_factor` 更高
     - `sharpe` 小幅更高
   - 但代价是：
     - 总收益下降
     - 更像“防守型”改进

3. `TP1 -> trailing stop`
   - 是这轮最有意思的版本
   - 相比静态基线：
     - 胜率显著更高
     - `profit_factor` 更高
     - `sharpe` 明显更高
     - 最大回撤显著更低
   - 代价是：
     - 总收益略低于静态基线
   - 说明：
     - 对 bull flag 来说，**让盈利单有机会继续跑，但不把利润吐回去太多**
     - 比单纯“摸到静态目标就全部走人”更有吸引力

4. 当前阶段的判断
   - 如果更看重绝对收益，静态基线仍然有竞争力
   - 如果更看重风险调整后表现，`tp1_trailing` 已经非常值得继续往下挖

下一步计划：

- 优先继续细化 `tp1_trailing`
  - 先扫：
    - `tp1_fraction_of_target`
    - `trailing_stop_fraction_of_flagpole`
- 暂时不直接做 partial exit
  - 因为那需要单独的 multi-leg 研究层
  - 不适合混在当前单次完整退出回测器里一起写

## 第 21 轮：动态退出扩展与次日早报

本轮做了什么：

- 沿用“**不覆盖现有 bull flag 基线，只在独立 exit 路径扩展**”的原则，继续完善动态退出模块。
- 更新了 [bull_flag_exit_variants.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_exit_variants.py)：
  - 加入更通用的 TP1 后退出钩子
  - 支持 `MA trail`、`structure trail`、`volume failure`、`close retrace`
  - 支持 overlay 版本：
    - `trailing + volume failure`
    - `trailing + close retrace`
    - `MA trail + volume failure`
- 新增/完善了 [bull_flag_exit_variant_grid_search.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/score_system/bull_flag_exit_variant_grid_search.py)，让 exit 研究可以复用同一份 entry 信号框架。
- 修复了一个真实可视化问题：
  - `plot_signal_context()` 里，`exit_path` merge 后动态列会带 suffix，导致 `TP1` 线画不出来。
  - 现在会在 merge 后自动回填 `tp1_price / active_protective_stop / ma_trail_value / structure_trail_value / close_retrace_threshold`。
- 补强了 [test_bull_flag_exit_variants.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/tests/test_bull_flag_exit_variants.py)：
  - 增加 `MA trail / structure trail / volume failure / close retrace / overlay` 场景
  - 修复测试 helper，避免动态列缓存住旧值导致 volume/EMA 类测试失真
- 验证结果：
  - `python -m pytest tests/test_bull_flag_exit_variants.py -q` -> `10 passed`
  - `python -m pytest -q` -> `84 passed`

本轮有一个过程调整：

- 原计划想一次性跑完整 `trailing / MA / structure / volume / close retrace / overlay` 全部网格。
- 但全量网格在全历史样本上耗时过长，单次大实验超过 1 小时被终止。
- 因此改成更稳妥的“**先做一次 entry 预计算，再逐个 family 跑代表性版本**”。
- 这次的晨报数据，都是在同一 entry 基线上逐个实跑得到的，不是口头推测。

固定的 bull flag entry 基线：

- `max_flag_retrace_ratio=0.30`
- `min_breakout_body_pct=0.60`
- `max_breakout_upper_shadow_pct=0.35`
- `max_breakout_lower_shadow_pct=0.50`
- `max_peak_sma60_return_10=0.055`

统一回测口径：

- 数据：`Dataframes/stock_price.csv`
- 股票池：`csi500`
- 区间：`2020-01-01` 到 `2026-03-16`
- 资金口径：`initial_capital=1,000,000`，`fixed_entry_notional=20,000`，`board_lot_size=100`

代表性退出版本结果：

| 版本 | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | 总收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 静态基线 | 150 | 44.83% | 2.86% | 1.84 | 7.88% | 0.97 | 1.43% |
| `TP1 -> trailing` 默认版（`tp1=0.5`, `trail=0.25`） | 150 | 59.59% | 2.77% | 2.24 | 7.36% | 1.27 | 0.89% |
| `TP1 -> trailing` 更早更紧（`tp1=0.4`, `trail=0.2`） | 150 | 61.64% | 2.23% | 2.09 | 5.91% | 1.15 | 0.99% |
| `TP1 -> trailing` 同步更紧（`tp1=0.5`, `trail=0.2`） | 150 | 59.59% | 2.54% | 2.15 | 6.77% | 1.21 | 0.81% |
| `TP1 -> MA trail`（`EMA10`, `buffer=0`） | 150 | 50.00% | 2.38% | 1.93 | 6.17% | 0.98 | 1.18% |
| `TP1 -> structure trail`（`lookback=5`, `buffer=0`） | 150 | 53.42% | 2.57% | 2.01 | 6.70% | 1.06 | 1.04% |
| `TP1 -> volume failure`（`threshold=2.0`） | 150 | 45.52% | 2.68% | 1.86 | 7.32% | 0.99 | 1.43% |
| `TP1 -> close retrace`（`5%`） | 150 | 48.63% | 2.39% | 1.92 | 6.26% | 0.98 | 1.58% |
| `trailing + volume failure` | 150 | 59.59% | 2.77% | 2.24 | 7.36% | 1.27 | 0.89% |
| `trailing + close retrace` | 150 | 59.59% | 2.76% | 2.24 | 7.36% | 1.27 | 0.89% |

归档输出：

- 基线和默认 trailing 对照：[exit_research_baseline_report.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_research_baseline_report.csv)
- MA 代表版本：[exit_probe_ma_trail.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_ma_trail.csv)
- structure 代表版本：[exit_probe_structure_trail.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_structure_trail.csv)
- volume 代表版本：[exit_probe_volume_failure.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_volume_failure.csv)
- close retrace 代表版本：[exit_probe_close_retrace.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_close_retrace.csv)
- trailing 定向试验：
  - [exit_probe_trailing_0p4_0p2.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_0p4_0p2.csv)
  - [exit_probe_trailing_0p5_0p2.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_0p5_0p2.csv)
- overlay：
  - [exit_probe_trailing_plus_volume.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_plus_volume.csv)
  - [exit_probe_trailing_plus_close.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_plus_close.csv)
- 汇总晨报表：[exit_variant_morning_report.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_variant_morning_report.csv)

结论：

1. 这轮最清楚的结论仍然是：**动态退出里最值得继续用的是 trailing 家族**。
   - 它不是把总收益做到最高
   - 但它把 `胜率 / Profit Factor / Sharpe / 最大回撤` 这一整组指标一起抬起来了

2. `TP1 -> trailing` 默认版（`tp1=0.5`, `trail=0.25`）目前仍然是最平衡的版本。
   - 更早启动、更紧的 trailing 会继续提高胜率
   - 但会明显砍掉总收益和平均单笔收益
   - 说明这条线上已经开始出现“过度保护利润”的副作用

3. `MA trail` 和 `close retrace` 都是“能改善一些风险指标，但不如 trailing 有效率”的版本。
   - 它们都比静态基线更稳一些
   - 但综合表现仍落后于默认 trailing

4. `structure trail` 是一个合格的备选保守版。
   - 比 `MA trail` 和 `close retrace` 更强
   - 但整体仍不如默认 trailing
   - 如果以后你想要一个“更贴价格结构、而不是贴高点回撤”的风格，它值得保留

5. `volume failure` 单独拿来做主要退出，帮助很有限。
   - 它几乎没有明显抬高 `Profit Factor`
   - overlay 到 trailing 上，结果几乎不变
   - 这说明在当前 bull flag 结构里，真正管用的主导力量还是价格本身，不是量能失败这条规则

6. `trailing + close retrace` 是目前唯一一个比默认 trailing **略好一点点** 的 overlay。
   - 提升非常小：
     - `profit_factor` 从 `2.2429` 到 `2.2435`
     - `sharpe` 从 `1.2719` 到 `1.2742`
   - 但它至少说明：
     - `close` 级别的利润保护和 trailing 本身并不冲突
     - 只是当前这档阈值下，增益非常有限

当前阶段结论：

- **主推版本**：
  - bull flag entry
  - `TP1 -> trailing stop`
  - 参数维持：
    - `tp1_fraction_of_target = 0.50`
    - `trailing_stop_fraction_of_flagpole = 0.25`
- **备选保守版**：
  - `TP1 -> structure trail`
- **可以保留观察但暂不主推**：
  - `MA trail`
  - `close retrace`
- **暂时不值得继续复杂化**：
  - `volume failure` 作为主退出
  - `trailing + volume failure`

下一步计划：

- 先不继续把 exit family 扩得更花。
- 如果继续深挖，优先只做一件事：
  - 细扫 `trailing + close retrace`
  - 看这点微弱增益是不是稳健存在，还是只是样本噪音
- 如果那条线也没有再明显改善，就把当前最优 exit 定稿为：
  - `TP1 -> trailing stop (0.5, 0.25)`

## 第 22 轮：`TP1` 后纯止损 stop-only 版本

本轮做了什么：

- 围绕一个新 thesis 做平行实验：
  - 如果 bull flag 的 thesis 是对的
  - 到达 `TP1` 后，趋势很可能还没结束
  - 那么不一定要继续保留静态最终止盈
  - 可以尝试：`TP1` 之后只靠原始/动态止损离场
- 更新了 [bull_flag_exit_variants.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_exit_variants.py)：
  - 在动态退出基类里新增两个小钩子：
    - `final_target_active(...)`
    - `time_stop_active(...)`
  - 默认实现都返回 `True`
  - 这样 exit family 可以按阶段决定“某类退出是否继续有效”
  - 新增平行 researcher：
    - `BullFlagTrailingStopOnlyAfterTp1Researcher`
  - 这个版本继承默认 trailing 主线，但只改一点：
    - `TP1` 之后，关闭 `take_profit`
    - `TP1` 之后，关闭 `time_stop`
    - 只保留：
      - 原始 `hard_stop`
      - `trailing_stop`
- 顺手补了一个更细的日线语义修正：
  - 如果某根 bar 第一次打到 `TP1`
  - 同时也打到了最终止盈
  - 现在仍然按 `take_profit` 解释
  - 但也会把 `tp1_hit_date` 正确记下来
  - 这样更符合“`TP1` 当根只做记录，stop-only 从下一根才生效”的定义
- 更新了 [strategies/__init__.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/__init__.py)，把 stop-only researcher 导出
- 补强了 [test_bull_flag_exit_variants.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/tests/test_bull_flag_exit_variants.py)：
  - `TP1` 前 stop-only 和默认 trailing 行为一致
  - `TP1` 后不再 `take_profit`
  - `TP1` 后不再 `time_stop`
  - `TP1` 后仍可被 trailing stop 打出
  - `TP1` 命中当天同时 hit target 时，仍按 `take_profit`
  - entry timing 与默认 trailing 完全一致
- 验证结果：
  - `python -m pytest tests/test_bull_flag_exit_variants.py -q` -> `16 passed`
  - `python -m pytest -q` -> `90 passed`

固定的 bull flag entry 基线：

- `max_flag_retrace_ratio=0.30`
- `min_breakout_body_pct=0.60`
- `max_breakout_upper_shadow_pct=0.35`
- `max_breakout_lower_shadow_pct=0.50`
- `max_peak_sma60_return_10=0.055`

固定的 trailing 参数：

- `tp1_fraction_of_target=0.50`
- `trailing_stop_fraction_of_flagpole=0.25`

统一回测口径：

- 数据：`Dataframes/stock_price.csv`
- 股票池：`csi500`
- 区间：`2020-01-02` 到 `2026-03-16`
- 资金口径：`initial_capital=1,000,000`，`fixed_entry_notional=20,000`，`board_lot_size=100`
- 为了节省重算时间，这次实验复用了已有缓存：
  - `exit_research_stock_frame.pkl`
  - `exit_research_base_signal_frame.pkl`

结果对比：

| 版本 | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | 总收益 | Sharpe | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TP1 -> trailing` 默认版 | 150 | 59.59% | 2.77% | 2.2429 | 7.3607% | 1.2719 | 0.8948% |
| `TP1` 后纯止损 stop-only | 150 | 59.59% | 2.8448% | 2.2767 | 7.5664% | 1.2468 | 0.8831% |

归档输出：

- 新版本单独结果：[exit_probe_trailing_stop_only_after_tp1.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_stop_only_after_tp1.csv)
- 新版本 trade df：[exit_probe_trailing_stop_only_after_tp1_trade_df.pkl](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_stop_only_after_tp1_trade_df.pkl)
- 和默认 trailing 的两行对照：[exit_probe_trailing_stop_only_comparison.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/exit_probe_trailing_stop_only_comparison.csv)

结论：

1. 这个 thesis 是有价值的。
   - `TP1` 之后取消静态止盈，并没有把策略搞坏
   - 相反，它让平均单笔收益、`profit_factor`、`total_return`、`max_drawdown` 都略有改善

2. stop-only 版本说明：
   - 当前默认 trailing 的静态最终止盈，确实会截掉一部分后续还能继续走的单子
   - 把最终止盈拿掉以后，系统能靠 trailing stop 多吃一点趋势延续的钱

3. 但 stop-only 并没有全面压过默认 trailing。
   - `sharpe` 从 `1.2719` 小幅回落到 `1.2468`
   - 说明虽然单笔赚钱厚了一点，资金曲线的节奏并没有更平滑

4. 当前阶段的判断：
   - stop-only 是一个**值得保留的强候选版本**
   - 但它还没有强到可以无争议替换默认 trailing 主线
   - 更准确地说：
     - 如果更偏向“让盈利单尽量多跑”，它很有吸引力
     - 如果更看重目前最稳的风险调整后表现，默认 trailing 仍然更均衡

当前阶段结论：

- **默认主推版本暂时不变**：
  - `TP1 -> trailing stop (0.5, 0.25)`
- **新增强候选版本**：
  - `BullFlagTrailingStopOnlyAfterTp1Researcher`
  - 适合继续往“让盈利单更充分延展”这条线上挖

下一步计划：

- 如果继续深挖 exit，最值得做的是：
  - 只围绕 stop-only 版本继续小范围研究
  - 例如：
    - trailing 的松紧
    - 是否要在 stop-only 里保留某种弱形式的保护阈值
- 如果后续 stop-only 能把 `sharpe` 也拉回来，再考虑把它升级成新的主推荐版本
