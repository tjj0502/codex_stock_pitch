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

## 第 13 轮：只做负斜率 flag，以及 slope 区间 grid search（`Dataframes/stock_price2.csv`）

本轮做了什么：

- 这轮只研究 bull flag 旗面的 slope 方向，不改 entry 其余部分，也不改当前主线的 trailing exit。
- 数据明确改成了：`Dataframes/stock_price2.csv`
- 研究问题分两步：
  1. 如果只做负斜率 flag，也就是把 slope 限定在 `-0.008` 到 `0`，结果会不会更好
  2. 如果把 slope 上下界继续放宽或收紧，是否能找到更好的区间

改了哪些代码：

- 在 [bull_flag_continuation.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_continuation.py) 里新增了 `min_flag_channel_slope_pct_per_bar`
- 旗面 slope 过滤从“对称绝对值限制”改成了“上下界区间限制”
  - 旧逻辑等价于：`[-max_slope, +max_slope]`
  - 新逻辑可以直接表达：
    - 只做负斜率：`[-0.008, 0]`
    - 轻微负到小幅正：`[-0.008, 0.004]`
- 这样做以后，默认配置仍兼容旧行为，但我们终于可以明确测试“负斜率限定”到底有没有价值

补了哪些测试：

- 默认对称 slope 限制下，轻微正斜率仍然允许通过
- 当 slope 区间限制为 `[-0.008, 0]` 时，同样的轻微正斜率 setup 会被拒绝

统一回测口径：

- 数据：`Dataframes/stock_price2.csv`
- 股票池：`csi500`
- 区间：`2020-01-02` 到 `2026-03-16`
- 其余固定参数：
  - `max_flag_retrace_ratio = 0.30`
  - `min_breakout_body_pct = 0.60`
  - `max_breakout_upper_shadow_pct = 0.35`
  - `max_breakout_lower_shadow_pct = 0.50`
  - `max_peak_sma60_return_10 = 0.055`
  - `tp1_fraction_of_target = 0.50`
  - `trailing_stop_fraction_of_flagpole = 0.25`

本轮 grid：

- 负斜率限定：
  - `[-0.004, 0]`
  - `[-0.006, 0]`
  - `[-0.008, 0]`
  - `[-0.010, 0]`
  - `[-0.012, 0]`
- 放宽正斜率容忍：
  - `[-0.008, 0.002]`
  - `[-0.008, 0.004]`
  - `[-0.008, 0.008]`（当前旧默认等价口径）

关键结果：

| slope 区间 | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | Sharpe | 最大回撤 | 总收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `[-0.008, 0.004]` | 129 | 60.63% | 3.03% | 2.3884 | 1.3521 | 0.86% | 6.96% |
| `[-0.008, 0.008]` | 165 | 59.26% | 2.67% | 2.1445 | 1.2386 | 1.02% | 7.84% |
| `[-0.008, 0.002]` | 98 | 57.73% | 2.34% | 1.9910 | 0.9350 | 1.27% | 3.88% |
| `[-0.004, 0]` | 47 | 54.35% | 1.56% | 1.6302 | 0.4131 | 1.18% | 1.07% |
| `[-0.006, 0]` | 55 | 50.00% | 1.25% | 1.4750 | 0.3661 | 1.47% | 1.00% |
| `[-0.008, 0]` | 68 | 52.24% | 1.43% | 1.5426 | 0.4525 | 1.78% | 1.49% |
| `[-0.010, 0]` | 75 | 47.30% | 0.57% | 1.1644 | 0.1413 | 2.29% | 0.47% |
| `[-0.012, 0]` | 78 | 48.05% | 0.69% | 1.1957 | 0.1811 | 2.17% | 0.64% |

归档输出：

- 全部结果表：[slope_grid_stock_price2.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/slope_grid_stock_price2.csv)

结论：

1. **只做负斜率 flag，这条路在 `stock_price2.csv` 上并不好。**
   - 你提的核心版本 `[-0.008, 0]` 计划交易数只剩 `68`
   - `profit_factor = 1.5426`
   - `sharpe = 0.4525`
   - `total_return = 1.49%`
   - 和当前对称默认 `[-0.008, 0.008]` 比，质量和收益都明显更差

2. **把正斜率完全禁掉，会误杀太多本来有用的 setup。**
   - 这说明我们之前担心的事情是真实存在的：
     - 日线里不少肉眼看起来接近横盘的旗面，拟合后会带一点轻微正斜率
     - 如果把正斜率一刀切掉，会把这些其实仍然合理的 continuation setup 一起杀掉

3. **最好的 tested 区间不是“纯负”，而是“轻微负到小幅正”。**
   - 本轮最优是：`[-0.008, 0.004]`
   - 它相比当前对称默认 `[-0.008, 0.008]`：
     - `profit_factor`：`2.1445 -> 2.3884`
     - `sharpe`：`1.2386 -> 1.3521`
     - `max_drawdown`：`1.02% -> 0.86%`
   - 代价是：
     - `planned_trade_count`：`165 -> 129`
     - `total_return`：`7.84% -> 6.96%`

4. **因此当前最合理的理解是：**
   - 旗面 slope 不该像旧版那样允许到 `+0.008` 那么宽
   - 但也不该严格收成纯负
   - 一个更像“高质量 refinement”的区间，是：
     - `min_flag_channel_slope_pct_per_bar = -0.008`
     - `max_flag_channel_slope_pct_per_bar = 0.004`

当前阶段结论：

- 如果更看重：
  - `profit_factor`
  - `sharpe`
  - `max_drawdown`
  那本轮最值得继续用的是：`[-0.008, 0.004]`
- 如果更看重原始总收益和样本量，当前对称默认 `[-0.008, 0.008]` 仍然更激进
- 但“只做负斜率 flag”这条 thesis，在 `Dataframes/stock_price2.csv` 上**没有得到支持**

下一步计划：

- 先把这轮结论保留成一个明确分支口径：
  - 主版本：保留当前对称默认
  - 高质量备选版：`[-0.008, 0.004]`
- 如果后面继续细挖 slope，我更倾向于：
  - 围绕 `0.002 ~ 0.006` 这段小正斜率容忍区间再做一轮更细的网格
  - 而不是继续往“纯负斜率”方向收紧

## 第 14 轮：`Dataframes/csi_1000_stock_price2.csv` 多轮迭代研究

本轮做了什么：

- 这轮把研究对象从之前的 `stock_price.csv / stock_price2.csv` 切到：
  - `Dataframes/csi_1000_stock_price2.csv`
- 目标不是追求频率，而是按你要求去提高 `profit_factor / sharpe`
- 执行纪律固定为：
  - 每一轮只动一个参数
  - 每轮最多跑 3 个回测点
  - 每轮先总结，再决定下一轮

补充的小代码改动：

- 在 [bull_flag_continuation.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategies/bull_flag_continuation.py) 里把 `BullFlagStrategyConfig.universe` 放宽到支持 `csi1000`
- 这样这轮实验配置和数据标签可以对齐，不需要再用 `csi500` 假扮 `csi1000`

统一回测口径：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 股票池：`csi1000`
- 区间：`2020-01-02` 到 `2026-03-16`
- 资金口径：
  - `initial_capital=1,000,000`
  - `fixed_entry_notional=20,000`
  - `board_lot_size=100`

### Round 0：当前主线 baseline

参数：

- `max_flag_retrace_ratio = 0.30`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.008`
- `min_breakout_body_pct = 0.60`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`
- `max_peak_sma60_return_10 = 0.055`
- `tp1_fraction_of_target = 0.50`
- `trailing_stop_fraction_of_flagpole = 0.25`
- exit：`BullFlagTrailingAfterTp1Researcher`

结果：

- `planned_trade_count = 263`
- `trade_win_rate = 45.21%`
- `average_trade_return = 0.07%`
- `profit_factor = 1.0419`
- `sharpe = 0.0766`
- `max_drawdown = 5.24%`
- `total_return = 0.66%`

结论：

- 这版在 csi1000 上**不能直接用**
- 频率是有的，但质量太差，说明小盘股环境下必须把 flag 和 signal 收得更干净

归档输出：

- [csi1000_baseline_bull_flag_trailing.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_baseline_bull_flag_trailing.csv)

### Round 1：Flag 深度（`max_flag_retrace_ratio`）

grid：

- `0.20 / 0.25 / 0.30`

结果：

| `max_flag_retrace_ratio` | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | Sharpe | 最大回撤 | 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.20` | 59 | 42.37% | -0.12% | 1.0113 | 0.0146 | 2.20% | 0.04% |
| `0.25` | 119 | 47.86% | 1.06% | 1.3516 | 0.4136 | 1.94% | 2.45% |
| `0.30` | 263 | 45.21% | 0.07% | 1.0419 | 0.0766 | 5.24% | 0.66% |

结论：

- csi1000 上旗面不能太深
- `0.25` 明显优于 `0.30`
- `0.20` 又收得过头，频率和收益都掉得太多

归档输出：

- [csi1000_round1_flag_retrace_grid.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_round1_flag_retrace_grid.csv)

### Round 2：Flag slope 上界（固定 `retrace=0.25`）

grid：

- `max_flag_channel_slope_pct_per_bar = 0.0 / 0.004 / 0.008`
- 固定下界：
  - `min_flag_channel_slope_pct_per_bar = -0.008`

结果：

| slope 上界 | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | Sharpe | 最大回撤 | 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.000` | 48 | 54.17% | 2.61% | 2.0389 | 0.7345 | 0.55% | 2.49% |
| `0.004` | 96 | 49.47% | 1.62% | 1.5594 | 0.5933 | 1.81% | 2.93% |
| `0.008` | 119 | 47.86% | 1.06% | 1.3516 | 0.4136 | 1.94% | 2.45% |

结论：

- 这一轮和 csi500 的结论不同
- 在 csi1000 上，**纯负/横盘旗面更好**
- `slope_upper = 0` 明显提升了：
  - `profit_factor`
  - `sharpe`
  - `max_drawdown`
- 虽然 `0.004` 的总收益略高，但质量指标整体不如 `0`

归档输出：

- [csi1000_round2_slope_upper_grid.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_round2_slope_upper_grid.csv)

### Round 3：Signal K 实体（固定 `retrace=0.25`, `slope_upper=0`）

grid：

- `min_breakout_body_pct = 0.50 / 0.60 / 0.70`

结果：

| `min_breakout_body_pct` | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | Sharpe | 最大回撤 | 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.50` | 56 | 55.36% | 2.65% | 2.1749 | 0.8399 | 0.62% | 2.92% |
| `0.60` | 48 | 54.17% | 2.61% | 2.0389 | 0.7345 | 0.55% | 2.49% |
| `0.70` | 32 | 56.25% | 1.48% | 1.5729 | 0.3591 | 0.82% | 0.91% |

结论：

- csi1000 上 signal K 的实体也不宜过度苛刻
- `0.50` 比 `0.60/0.70` 更平衡
- `0.70` 样本太少，且单笔赚钱厚度反而下降

归档输出：

- [csi1000_round3_signal_body_grid.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_round3_signal_body_grid.csv)

### Round 4：退出方法（固定当前最优 entry）

固定 entry：

- `max_flag_retrace_ratio = 0.25`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.0`
- `min_breakout_body_pct = 0.50`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`
- `max_peak_sma60_return_10 = 0.055`

对比版本：

- `BullFlagTrailingAfterTp1Researcher`
- `BullFlagTrailingStopOnlyAfterTp1Researcher`
- `BullFlagStructureTrailAfterTp1Researcher`

结果：

| 退出版本 | 计划交易数 | 胜率 | 平均单笔收益 | Profit Factor | Sharpe | 最大回撤 | 总收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trailing_default` | 56 | 55.36% | 2.65% | 2.1749 | 0.8399 | 0.62% | 2.92% |
| `trailing_stop_only_after_tp1` | 56 | 55.36% | 2.51% | 2.1126 | 0.7712 | 0.62% | 2.76% |
| `structure_trail` | 56 | 50.00% | 3.17% | 2.2680 | 0.8679 | 0.87% | 3.45% |

结论：

- 在 csi1000 上，止盈/移动止损这层最优不是默认 trailing，而是 **structure trail**
- 它的特点是：
  - 胜率低一点
  - 但平均单笔收益更厚
  - `profit_factor` 和 `sharpe` 都是三者里最好
  - `total_return` 也最高
- `TP1` 后纯止损这次没有跑赢默认 trailing

归档输出：

- [csi1000_round4_exit_variants.csv](/C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_round4_exit_variants.csv)

### 额外说明

- 中间原本想补一轮 `max_breakout_upper_shadow_pct` 的 grid，但上一轮执行中断过一次
- 这次为了避免继续卡死，我没有再重复跑这一轮
- 因为前 4 轮已经足够形成清晰结论：
  - flag 要更浅
  - slope 要更纯负/横盘
  - body 不要太苛刻
  - exit 用 structure trail 更优

## 当前阶段总结（`Dataframes/csi_1000_stock_price2.csv`）

在 csi1000 上，当前最值得保留的 bull flag 配置是：

- `max_flag_retrace_ratio = 0.25`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.0`
- `min_breakout_body_pct = 0.50`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`
- `max_peak_sma60_return_10 = 0.055`
- exit：`BullFlagStructureTrailAfterTp1Researcher`

这版相对原始 baseline 的改善非常明显：

- `profit_factor`：`1.0419 -> 2.2680`
- `sharpe`：`0.0766 -> 0.8679`
- `max_drawdown`：`5.24% -> 0.87%`
- `total_return`：`0.66% -> 3.45%`

最后的判断：

1. **csi1000 可以做 bull flag，但不能直接沿用 csi500 的宽松主线。**
2. **小盘股上，flag 要更浅、形态要更纯，signal K 反而不需要过度完美。**
3. **退出层比我预期得更重要：**
   - csi1000 上结构性 trail 比默认 trailing 更适合把盈亏比拉起来

下一步计划：

- 如果后面继续深挖 csi1000，我优先会看两件事：
  1. `max_breakout_upper_shadow_pct` 的小范围 grid（例如 `0.25 / 0.35 / 0.45`）
  2. `structure_trail_lookback` 的小范围 grid（例如 `3 / 5 / 8`）

## 第 15 轮：`Dataframes/csi_1000_stock_price2.csv` baseline 亏损结构复盘

### 本轮做了什么

- 不再继续调参，转而专门复盘 csi1000 baseline 的亏损结构。
- 研究对象固定为：
  - 数据源：`Dataframes/csi_1000_stock_price2.csv`
  - entry / exit：最原始 baseline，也就是：
    - `max_flag_retrace_ratio = 0.30`
    - `min_flag_channel_slope_pct_per_bar = -0.008`
    - `max_flag_channel_slope_pct_per_bar = 0.008`
    - `min_breakout_body_pct = 0.60`
    - `max_breakout_upper_shadow_pct = 0.35`
    - `max_breakout_lower_shadow_pct = 0.50`
    - `max_peak_sma60_return_10 = 0.055`
    - exit：`BullFlagTrailingAfterTp1Researcher`
- 为了和回测指标严格一致，这次只统计 `2020-01-01` 到 `2026-03-16` 窗口内的 closed trades，不混入更早历史样本。

### 关键结果

- `closed_trade_count = 262`
- `loss_trade_count = 143`
- `loss_rate = 54.58%`

亏损单退出原因：

- `hard_stop = 134`
- `trailing_stop = 8`
- `take_profit = 1`

`TP1` 命中率对比：

- 亏损单 `tp1_reached_rate = 6.29%`
- 盈利单 `tp1_reached_rate = 98.32%`

这说明 baseline 最大的问题不是“TP1 后怎么拿”，而是：

- **大多数亏损单在到达 `TP1` 之前就已经失败了**
- **也就是 entry / setup 质量不过关，而不是动态止盈来不及救**

### baseline 亏损更像死在哪里

一句话总结：

- **baseline 主要死在“入场后很快走坏，直接打原始硬止损”**

这和后面几轮优化得到的方向是完全一致的：

- csi1000 先要解决的是 setup 纯度
- 不是先去堆更复杂的 trailing / stop-only 花样

### 特征上看到了什么

#### 1. 更深的 flag 明显更差

按 `flag_retrace_ratio` 三分桶：

- 最浅桶：`win_rate = 44.83%`，`mean_return = +0.89%`
- 中间桶：`win_rate = 48.86%`，`mean_return = -0.09%`
- 最深桶：`win_rate = 42.53%`，`mean_return = -0.25%`

结论：

- baseline 里 `0.30` 的 flag 深度放得太松了
- 这正是后面把 `max_flag_retrace_ratio` 收到 `0.25` 后质量明显改善的根因

#### 2. slope 越接近正，越差

按 `flag_upper_slope` 三分桶：

- 更负的桶：`win_rate = 47.13%`，`mean_return = +0.94%`
- 中间桶：`win_rate = 44.83%`，`mean_return = -0.16%`
- 更高/带正斜率的桶：`win_rate = 44.32%`，`mean_return = -0.23%`

结论：

- csi1000 的 bull flag 更像“要纯负或横盘偏负”，不适合放到正斜率
- 这和后面 slope grid 的最佳结果 `[-0.008, 0.0]` 一致

#### 3. signal K 不是越强越好

按 `signal_body_pct` 三分桶：

- 较低实体桶：`win_rate = 47.73%`，`mean_return = +0.81%`
- 中间实体桶：`win_rate = 42.53%`，`mean_return = -0.34%`
- 最高实体桶：`win_rate = 45.98%`，`mean_return = +0.07%`

结论：

- baseline 里并不是“大实体 breakout”自动更好
- 这解释了为什么后面 `min_breakout_body_pct = 0.50` 会优于 `0.60 / 0.70`

#### 4. 名义上的高 R 不代表更好

按 `reward_to_risk` 三分桶：

- 低 R 桶：`win_rate = 42.53%`，`mean_return = -1.96%`
- 中 R 桶：`win_rate = 55.17%`，`mean_return = +2.14%`
- 高 R 桶：`win_rate = 38.64%`，`mean_return = +0.37%`

结论：

- csi1000 baseline 里，超高的账面赔率很多是“看起来很美”
- 真正最好的是中等 R，不是越大越好
- 这说明仅靠 `reward_to_risk` 不能挑出最干净的 setup

#### 5. 过热背景也在拖后腿

按 `peak_sma60_return_10` 三分桶：

- 低热度桶：`mean_return = +0.10%`
- 中热度桶：`mean_return = +0.81%`
- 高热度桶：`mean_return = -0.34%`

结论：

- csi1000 上太热的 flag peak 更容易是假动作
- 中等热度比过热更健康

### 最差亏损样本的共性

最差几笔基本都有这些共同点：

- `exit_reason` 几乎都是 `hard_stop`
- `tp1_reached = False`
- `flag_retrace_ratio` 多数落在 `0.25 ~ 0.30`
- 有些 slope 明显为正，或者至少不够负
- signal K 肉眼看并不差，甚至实体还挺强

这很关键，因为它说明：

- baseline 的失败不是“信号 K 太丑”
- 而是 **很多 setup 看起来像 breakout，但本质不是高质量 continuation**

### 本轮结论

这轮 baseline 亏损复盘把前面多轮调参的方向基本坐实了：

1. **csi1000 的 baseline 主要死在 entry 前端，而不是 TP1 后管理。**
2. **最值钱的改进不是继续堆 exit complexity，而是先收紧 setup：**
   - flag 更浅
   - slope 更纯负
   - 不要迷信超强 body
   - 不要迷信超高 nominal R
3. **这也解释了为什么最后 csi1000 上是 `structure_trail` 胜出：**
   - 前端先提纯
   - 后端再用更适合小盘波动的结构性 trail 去保利润

### 下一步计划

- 如果后面继续深挖 csi1000 bull flag，我会优先看：
  1. `max_breakout_upper_shadow_pct` 的小范围 grid
  2. `structure_trail_lookback` / `structure_trail_buffer_pct` 的小范围 grid
- 但在当前阶段，关于 baseline 为什么差，已经有足够清晰的结论，不需要再继续扩样本做更大范围 brute-force。

## 2026-04-22：`narrow_state` 全历史漏斗诊断（`csi_1000_stock_price2.csv`）

本轮做了什么：

- 使用 [csi_1000_stock_price2.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/Dataframes/csi_1000_stock_price2.csv) 全历史数据
- 固定：
  - `left_trend_mode = "narrow_state"`
  - `narrow_trend_lookback_bars = 20`
  - 当前默认窄趋势参数：
    - `narrow_trend_max_bear_ratio = 0.25`
    - `narrow_trend_max_consecutive_bear_bars = 2`
    - `narrow_trend_min_ema20_above_ratio = 0.90`
    - `narrow_trend_max_upper_shadow_pct = 0.25`
    - `narrow_trend_min_run_bars = 1`
- 直接统计 `narrow_state -> flag -> breakout -> follow-through -> entry` 的漏斗，不再只看最终回测结果

结果：

- `raw_state_bars = 4350`
- `run_end_events = 3090`
- `flag_structured_rows = 156`
- `flag_retrace_ok_rows = 35`
- `flag_channel_ok_rows = 27`
- `bull_flag_candidate_rows = 2`
- `signal_candle_rows = 0`
- `follow_through_rows = 0`
- `entry_signal_rows = 0`

关键比例：

- `flag_structured / run_end = 5.05%`
- `bull_flag_candidate / flag_structured = 1.28%`
- `signal_candle / bull_flag_candidate = 0%`

耗时：

- 读 CSV：`8.242s`
- researcher 初始化：`515.542s`
- 总耗时：`523.784s`

结论：

- 之前关于“阴线比例放到 25% 后，`narrow_state` 会有 3000+ 个事件”的判断是对的。
- 现在的主要问题不是 `narrow_state` 本身没频率，而是 **`run-end -> 能形成合格 flag` 这一层几乎全部死掉了**。
- 也就是说，矛盾不在左侧状态频率，而在：
  - `narrow_state` 与当前 `flag` 定义的衔接方式
  - 以及当前 `flag` 本体对这种左侧入口过于苛刻
- 这也解释了为什么全历史完整回测最终是 `0` 笔交易：不是没有左侧趋势，而是左侧趋势几乎无法转化为合格 bull flag。

下一步计划：

- 优先研究 `narrow_state` 结束点与 `flag_start / flag_peak` 的衔接定义
- 其次再看 `flag` 本体是否需要为 `narrow_state` 模式单独放宽，而不是直接复用旧 `flagpole` 口径

### 补充：`run_end -> structured` 具体死因拆解

本轮做了什么：

- 在同一套 `narrow_state + N=20 + csi1000 全历史` 条件下
- 继续把 `run_end_events -> flag_structured_rows` 这一段拆成具体失败原因
- 同时把 `structured -> candidate / signal` 也拆开看

结果：

- 总体：
  - `run_end_events = 3090`
  - `structured_rows = 156`
  - `candidate_rows = 2`
  - `signal_rows = 0`

- `run_end -> structured` 失败原因：
  - `peak_not_in_bullish_stack = 1582`
  - `flagpole_bars_out_of_range = 1480`
  - `invalid_flagpole_low_or_peak = 10`
  - `flag_window_breaks_bullish_stack = 3`
  - `no_valid_flag_bars_window = 1`
  - `no_bar_after_state_end = 1`

- `structured -> next stage` 失败原因：
  - `fail_retrace = 7`
  - `fail_channel = 3`
  - `fail_shape = 1`
  - `candidate_no_breakout_signal = 1`

结论：

- 现在最大的瓶颈不是 `flag_retrace_ratio`、也不是 breakout K 本身。
- 真正的问题是：**新 `narrow_state` 入口仍然在复用旧 `flagpole` 前置门槛。**
- 具体有两层：
  1. 很多 `run_end` 本身不在 `bullish_stack` 里，直接在入口被杀掉。
  2. `narrow_state` 的 run 通常很短，但当前代码仍要求：
     - `flagpole_bars` 在 `5~20`
     - 这导致大量 run-end 在“还没进入 flag 本体”之前就被旧 `flagpole_bars` 规则过滤。
- 一旦真正进入 `structured` 阶段，后面的主要问题才轮到：
  - 回撤太深
  - channel 不合格
- 但这些已经是次级矛盾，因为前面两层先杀掉了绝大多数样本。

当前判断：

- 这版 `narrow_state` 失败，不是因为左侧窄趋势定义没用。
- 而是因为 **`narrow_state` 只是换了左侧入口，但 `record_setup()` 里仍然按旧 `flagpole` 的 bar 数和 stack 逻辑在裁它。**
- 如果后面继续改，第一优先级应该是：
  - 给 `narrow_state` 单独定义左侧 impulse 长度门槛
  - 而不是继续沿用旧 `min_flagpole_bars / max_flagpole_bars`

## 2026-04-22：独立窄趋势策略 baseline（`csi_1000_stock_price2.csv`）

本轮做了什么：

- 不改 notebook / archive，只在策略层完成拆分后，先跑一圈新独立策略：
  - `BullFlagNarrowTrendContinuationResearcher`
- 数据源明确使用：
  - [csi_1000_stock_price2.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/Dataframes/csi_1000_stock_price2.csv)
- baseline 参数口径：
  - `universe = "csi1000"`
  - `narrow_trend_lookback_bars = 20`
  - `narrow_trend_max_bear_ratio = 0.25`
  - `narrow_trend_max_consecutive_bear_bars = 2`
  - `narrow_trend_min_ema20_above_ratio = 0.90`
  - `narrow_trend_max_upper_shadow_pct = 0.25`
  - `narrow_trend_min_run_bars = 1`
- 其余 bull flag / 出场保持默认静态版，不叠动态 exit variant
- 回测参数延用前面的统一口径：
  - `initial_capital = 1,000,000`
  - `fixed_entry_notional = 20,000`
  - `board_lot_size = 100`

输出文件：

- baseline 汇总：
  - [csi1000_narrow_trend_baseline_summary.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_trend_baseline_summary.csv)
- 和旧 csi1000 baseline 对照：
  - [csi1000_narrow_trend_vs_old_baseline.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_trend_vs_old_baseline.csv)

结果：

- 新独立窄趋势 baseline：
  - `planned_trade_count = 96`
  - `entered_trade_count = 96`
  - `closed_trade_count = 96`
  - `trade_win_rate = 37.50%`
  - `average_trade_return = 1.2427%`
  - `profit_factor = 1.2726`
  - `sharpe = 0.2586`
  - `max_drawdown = 2.76%`
  - `total_return = 2.39%`

- 跑完一圈的耗时：
  - 读 CSV：`3.29s`
  - researcher：`400.65s`
  - backtester：`3.36s`
  - 总耗时：`407.31s`
  - 也就是大约 **6 分 47 秒**

和旧 `csi1000` baseline 对照：

- 旧 baseline（`BullFlagTrailingAfterTp1Researcher`）：
  - `planned_trade_count = 263`
  - `trade_win_rate = 45.21%`
  - `average_trade_return = 0.0739%`
  - `profit_factor = 1.0419`
  - `sharpe = 0.0766`
  - `max_drawdown = 5.24%`
  - `total_return = 0.66%`

结论：

- 新独立窄趋势 baseline 虽然频率明显更低：
  - `263 -> 96`
- 但质量已经明显提升：
  - `profit_factor: 1.04 -> 1.27`
  - `sharpe: 0.08 -> 0.26`
  - `max_drawdown: 5.24% -> 2.76%`
  - `total_return: 0.66% -> 2.39%`
- 这说明“把左侧 trend purity 单独抽出来”这条方向是对的。
- 但这版仍然只是 baseline，还远远不是最优：
  - 胜率还不高
  - `profit_factor` 也还没有到我们前面更成熟版本的水平
  - researcher 速度依然偏慢，主要时间还是花在特征和 signal 生成，不是 backtester

当前判断：

- 新独立窄趋势策略是**值得继续研究**的。
- 它至少已经证明：
  - 去掉旧 `bullish_stack` 入口约束
  - 把 `flagpole_start` 改成 `run_end` 回看最低 pivot low / 最低点
  这两件事合起来，不会把策略打坏，反而比旧 csi1000 baseline 更健康。

下一步计划：

- 先不急着改 exit。
- 下一轮优先做小范围 grid：
  1. `narrow_trend_lookback_bars = 12 / 16 / 20`
  2. `max_flag_retrace_ratio = 0.25 / 0.30 / 0.35`
  3. 视结果再决定要不要接动态 exit variant

## 2026-04-22：独立窄趋势策略漏斗分析（`csi_1000_stock_price2.csv`）

本轮做了什么：

- 继续使用新独立策略 baseline：
  - `BullFlagNarrowTrendContinuationResearcher`
- 数据源仍然是：
  - [csi_1000_stock_price2.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/Dataframes/csi_1000_stock_price2.csv)
- 参数口径和 baseline 保持一致：
  - `narrow_trend_lookback_bars = 20`
  - `narrow_trend_max_bear_ratio = 0.25`
  - `narrow_trend_max_consecutive_bear_bars = 2`
  - `narrow_trend_min_ema20_above_ratio = 0.90`
  - `narrow_trend_max_upper_shadow_pct = 0.25`
  - `narrow_trend_min_run_bars = 1`
- 这轮不再看收益，专门看：
  - `narrow_state -> flag -> breakout -> follow-through -> entry`
  这条链路到底死在哪一层

输出文件：

- run 级漏斗：
  - [csi1000_narrow_trend_funnel_runs.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_trend_funnel_runs.csv)
- 唯一行漏斗：
  - [csi1000_narrow_trend_funnel_unique_rows.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_trend_funnel_unique_rows.csv)
- run 级失败原因：
  - [csi1000_narrow_trend_funnel_failures.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_trend_funnel_failures.csv)

### Run 级漏斗

- `raw_state_bars = 4350`
- `run_end_events = 3090`
- `run_has_bar_after_state_end = 3089`
- `run_has_flag_window = 3088`
- `run_has_flagpole_anchor = 3088`
- `run_flagpole_geometry_ok = 3088`
- `run_flagpole_bars_ok = 3036`
- `run_flagpole_return_ok = 2764`
- `run_with_structured_flag_row = 2764`
- `run_with_candidate_row = 736`
- `run_with_breakout_row = 274`
- `run_with_follow_through_row = 202`
- `run_with_entry_row = 103`

关键比例：

- `run_end -> flagpole_return_ok = 2764 / 3090 = 89.4%`
- `flagpole_return_ok -> candidate = 736 / 2764 = 26.6%`
- `candidate -> breakout = 274 / 736 = 37.2%`
- `breakout -> follow_through = 202 / 274 = 73.7%`
- `follow_through -> entry = 103 / 202 = 51.0%`

### 唯一行漏斗

- `unique_structured_rows = 27238`
- `unique_candidate_rows = 3116`
- `unique_signal_candle_rows = 289`
- `unique_follow_through_rows = 215`
- `unique_entry_signal_rows = 103`
- `unique_entry_signal_executed_rows = 96`
- `unique_entry_signal_suppressed_rows = 7`

### 失败原因

run 级最主要失败原因是：

- `flagpole_return_too_small = 272`
- `flagpole_bars_out_of_range = 52`
- `no_bar_after_state_end = 1`
- `no_valid_flag_window = 1`

### 结论

这轮最重要的认识是：

- 现在**不是** `narrow_state` 本身没频率。
- 也**不是** `run_end -> flagpole` 这层在卡死样本。

和之前老的 `narrow_state` 接法不同，这次独立策略已经把左侧入口打通了：

- `3090` 个 run-end 里，
- 有 `2764` 个能走到 `flagpole_return_ok`

也就是说：

- **左侧入口已经基本不是主矛盾。**

现在真正掉频率最快的地方，已经变成了两层：

1. **`flagpole_return_ok -> candidate`**
   - `2764 -> 736`
   - 这里掉了约 `73%`
   - 说明真正的主瓶颈已经转移到：
     - `flag_retrace_ratio`
     - `flag_width_pct`
     - `flag channel slope`
   - 也就是 **flag 本体太严 / 和这类左侧入口不够匹配**

2. **`candidate -> breakout`**
   - `736 -> 274`
   - 这里又掉了约 `63%`
   - 说明第二个瓶颈是：
     - 当前 breakout 定义
     - `signal_quality_ok`
     - `close > projected_upper_line`
   - 对这类窄趋势入口来说依然比较苛刻

后面两层反而没有前面那么夸张：

- `breakout -> follow_through` 还保留了 `73.7%`
- `follow_through -> entry` 掉到 `103`，主要是：
  - `reward_to_risk`
  - `trend_environment_ok`
  在继续过滤
- 最后 `103 -> 96` 只是计划单层面对重叠信号做了压缩，不是核心问题

当前判断：

- 这次拆出来的新窄趋势策略已经证明：
  - 左侧入口方向是对的
  - 现在真正值得调的，不再是 `narrow_state` 自己
- 下一轮最值得动的是：
  1. `max_flag_retrace_ratio`
  2. `min_flag_channel_slope_pct_per_bar / max_flag_channel_slope_pct_per_bar`
  3. breakout bar 的质量门槛

一句话总结：

- **当前频率掉得最快的地方，不在左侧 trend detection，而在 flag 本体和 breakout 这两层。**

## 2026-04-23：独立窄趋势策略小范围 grid（`csi_1000_stock_price2.csv`）

本轮做了什么：

- 继续基于新独立策略 baseline：
  - `BullFlagNarrowTrendContinuationResearcher`
- 数据源仍然是：
  - [csi_1000_stock_price2.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/Dataframes/csi_1000_stock_price2.csv)
- 按“每轮只动一个参数、每轮不超过 3 个回测”的原则，连续跑了 3 轮：
  1. `max_flag_retrace_ratio`
  2. `max_flag_width_pct`
  3. `flag channel slope upper bound`

输出文件：

- Round 1：
  - [csi1000_narrow_round1_retrace_grid.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_round1_retrace_grid.csv)
- Round 2：
  - [csi1000_narrow_round2_width_grid.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_round2_width_grid.csv)
- Round 3：
  - [csi1000_narrow_round3_slope_grid.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_round3_slope_grid.csv)
- 汇总表：
  - [csi1000_narrow_flag_grid_summary.csv](C:/Users/Jay/GitRepo/codex_stock_pitch/strategy_archive/bull_flag_continuation/experiment_logs/outputs/csi1000_narrow_flag_grid_summary.csv)

### Round 1：`max_flag_retrace_ratio`

固定：

- `narrow_trend_lookback_bars = 20`
- `narrow_trend_max_bear_ratio = 0.25`
- `narrow_trend_min_run_bars = 1`
- 其余先用 baseline

结果：

- `0.25`
  - `planned_trade_count = 45`
  - `profit_factor = 1.9793`
  - `sharpe = 0.5654`
  - `max_drawdown = 0.93%`
  - `total_return = 3.29%`

- `0.30`
  - `planned_trade_count = 62`
  - `profit_factor = 1.7828`
  - `sharpe = 0.5262`
  - `max_drawdown = 1.35%`
  - `total_return = 3.86%`

- `0.40`（baseline）
  - `planned_trade_count = 96`
  - `profit_factor = 1.2726`
  - `sharpe = 0.2586`
  - `max_drawdown = 2.76%`
  - `total_return = 2.39%`

结论：

- **`flag_retrace_ratio` 是目前最值钱的前端参数。**
- 只要把 `0.40` 收到 `0.25 / 0.30`，质量立刻明显抬高。
- 如果更偏质量：
  - `0.25` 最好
- 如果更想保一点频率和总收益：
  - `0.30` 更平衡

### Round 2：`max_flag_width_pct`

这一轮固定：

- `max_flag_retrace_ratio = 0.25`

结果：

- `0.08`
  - `planned_trade_count = 27`
  - `profit_factor = 1.2881`
  - `sharpe = 0.1604`
  - `max_drawdown = 0.89%`
  - `total_return = 0.58%`

- `0.10`
  - `planned_trade_count = 36`
  - `profit_factor = 1.5673`
  - `sharpe = 0.3188`
  - `max_drawdown = 1.03%`
  - `total_return = 1.52%`

- `0.12`
  - `planned_trade_count = 45`
  - `profit_factor = 1.9793`
  - `sharpe = 0.5654`
  - `max_drawdown = 0.93%`
  - `total_return = 3.29%`

结论：

- **`flag_width_pct` 不是当前主矛盾。**
- 这一轮里，越收紧越差。
- 说明当前窄趋势策略里，`width` 已经不是最该继续压缩的地方。
- 至少在这条线上，`0.12` 仍然是更合理的默认值。

### Round 3：`flag channel slope`

这一轮固定：

- `max_flag_retrace_ratio = 0.25`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.008`

只扫上界：

- `upper = 0.0`
  - `planned_trade_count = 18`
  - `profit_factor = 3.4235`
  - `sharpe = 0.6144`
  - `max_drawdown = 0.41%`
  - `total_return = 1.64%`

- `upper = 0.004`
  - `planned_trade_count = 36`
  - `profit_factor = 1.8043`
  - `sharpe = 0.4404`
  - `max_drawdown = 0.94%`
  - `total_return = 2.15%`

- `upper = 0.008`
  - `planned_trade_count = 45`
  - `profit_factor = 1.9793`
  - `sharpe = 0.5654`
  - `max_drawdown = 0.93%`
  - `total_return = 3.29%`

结论：

- 这里没有出现“既提纯又保住收益”的完美点。
- `upper = 0.0` 的确把质量做得非常高：
  - `profit_factor` 直接上到 `3.42`
  - 回撤也最低
- 但代价是频率和总收益掉得很明显。
- `upper = 0.008` 反而给了更平衡的结果。

### 综合判断

这三轮串起来看，最清楚的结论是：

1. **真正最值钱的是 `flag_retrace_ratio`**
   - 这轮明确证明它是当前主提升点

2. **`flag_width_pct` 不是主矛盾**
   - 不需要继续收

3. **slope 可以提纯，但会明显吃掉频率**
   - `upper = 0.0` 更像一个高纯度低频版
   - `upper = 0.008` 更像实战平衡版

如果现在让我从这三轮里先定两个候选：

- 平衡版：
  - `max_flag_retrace_ratio = 0.25`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.008`
  - `max_flag_channel_slope_pct_per_bar = 0.008`

- 高纯度版：
  - `max_flag_retrace_ratio = 0.25`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.008`
  - `max_flag_channel_slope_pct_per_bar = 0.0`

当前更偏向的判断是：

- 如果你要继续把这条线做成能交易、还能保留一点样本数的版本，
  - **先用平衡版**
- 如果你要继续研究“最干净的窄趋势 bull flag 长什么样”，
  - **高纯度版值得保留作对照**

下一步计划：

- 先不要再动 `flag_width_pct`
- 如果继续优化，优先顺序会是：
  1. `max_flag_retrace_ratio` 附近再做一轮细扫（例如 `0.22 / 0.25 / 0.28`）
  2. breakout bar 质量参数
  3. 最后再看是否接动态 exit

## 2026-04-23 Round 4：把 slope 上界继续放大（csi1000 narrow trend）

### 本轮做了什么

基于上一轮已经确定的较优 flag 参数，继续只扫一个参数：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_retrace_ratio = 0.25`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.008`
- 只扫：
  - `max_flag_channel_slope_pct_per_bar = 0.012 / 0.016 / 0.020`

结果文件：

- `outputs/csi1000_narrow_round4_slope_expand_grid.csv`

### 关键结果

- `upper = 0.012`
  - `planned_trade_count = 49`
  - `trade_win_rate = 38.78%`
  - `average_trade_return = 2.49%`
  - `profit_factor = 1.5789`
  - `sharpe = 0.3734`
  - `max_drawdown = 1.07%`
  - `total_return = 2.44%`

- `upper = 0.016`
  - `planned_trade_count = 51`
  - `trade_win_rate = 41.18%`
  - `average_trade_return = 4.42%`
  - `profit_factor = 2.0381`
  - `sharpe = 0.6290`
  - `max_drawdown = 0.92%`
  - `total_return = 4.37%`

- `upper = 0.020`
  - `planned_trade_count = 51`
  - `trade_win_rate = 41.18%`
  - `average_trade_return = 4.42%`
  - `profit_factor = 2.0381`
  - `sharpe = 0.6290`
  - `max_drawdown = 0.92%`
  - `total_return = 4.37%`

### 结论

这一轮最重要的发现是：

- 对 csi1000 这种小盘环境，**slope 上界并不是越紧越好**。
- 上一轮我们以为 `0.008` 已经是比较平衡的点，但继续放宽到 `0.016` 之后：
  - 交易数从 `45` 提到 `51`
  - `profit_factor` 从 `1.9793` 提到 `2.0381`
  - `sharpe` 从 `0.5654` 提到 `0.6290`
  - `total_return` 从 `3.29%` 提到 `4.37%`
  - `max_drawdown` 还略有下降
- `0.020` 和 `0.016` 得到完全相同的结果，说明当前样本里真正有影响的 setup，大概已经在 `0.016` 之前全部被放进来了。

这支持一个很符合直觉的判断：

- **小盘股的 bull flag，旗面通道允许的上倾幅度可能确实要比 csi500 更宽一点。**
- 之前把 slope 压得过紧，可能把一部分“波动更大但仍然有效”的小盘趋势整理误杀了。

### 当前最优平衡候选

截至这一轮，csi1000 narrow trend 这条线里，当前最值得保留的平衡版候选是：

- `max_flag_retrace_ratio = 0.25`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.016`

这版对应：

- `planned_trade_count = 51`
- `profit_factor = 2.0381`
- `sharpe = 0.6290`
- `max_drawdown = 0.92%`
- `total_return = 4.37%`

### 下一步计划

如果继续沿这条线优化，优先级会变成：

1. 先固定新的 slope 平衡点 `0.016`
2. 在这个基础上，再回头细扫 `max_flag_retrace_ratio`
3. 最后再看 breakout bar 质量参数和 exit

## 2026-04-23 Round 5：同时放宽 retrace 和负 slope（csi1000 narrow trend）

### 本轮做了什么

这轮不是再单独扫一个参数，而是专门测试一个新的想法：

- 对小盘股来说，旗面可能不仅允许更高一点的正 slope 上界，
- 也可能允许：
  - **更深一点的回调**
  - **更负一点的下倾通道**

所以这轮固定：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_width_pct = 0.12`
  - `max_flag_channel_slope_pct_per_bar = 0.016`

同时测试：

- `max_flag_retrace_ratio = 0.30 / 0.35`
- `min_flag_channel_slope_pct_per_bar = -0.012 / -0.016`

结果文件：

- `outputs/csi1000_narrow_round5_retrace_min_slope_grid.csv`

### 关键结果

- `retrace = 0.30`, `min_slope = -0.012`
  - `planned_trade_count = 76`
  - `profit_factor = 1.8954`
  - `sharpe = 0.6295`
  - `max_drawdown = 1.62%`
  - `total_return = 5.55%`

- `retrace = 0.30`, `min_slope = -0.016`
  - `planned_trade_count = 81`
  - `profit_factor = 1.6091`
  - `sharpe = 0.4870`
  - `max_drawdown = 1.62%`
  - `total_return = 4.45%`

- `retrace = 0.35`, `min_slope = -0.012`
  - `planned_trade_count = 96`
  - `profit_factor = 1.6753`
  - `sharpe = 0.5596`
  - `max_drawdown = 2.44%`
  - `total_return = 5.70%`

- `retrace = 0.35`, `min_slope = -0.016`
  - `planned_trade_count = 101`
  - `profit_factor = 1.4816`
  - `sharpe = 0.4401`
  - `max_drawdown = 2.44%`
  - `total_return = 4.60%`

### 和当前平衡候选相比

当前上一轮的平衡候选是：

- `max_flag_retrace_ratio = 0.25`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.016`

它的结果是：

- `planned_trade_count = 51`
- `profit_factor = 2.0381`
- `sharpe = 0.6290`
- `max_drawdown = 0.92%`
- `total_return = 4.37%`

和它相比，这轮最值得看的组合是：

- `retrace = 0.30`
- `min_slope = -0.012`

它的特点是：

- 频率明显提高：`51 -> 76`
- `total_return` 提高：`4.37% -> 5.55%`
- `sharpe` 基本持平并略高：`0.6290 -> 0.6295`
- 但 `profit_factor` 下降：`2.0381 -> 1.8954`
- `max_drawdown` 也上升：`0.92% -> 1.62%`

### 结论

这轮说明：

- **同时放宽 retrace 和更负的 slope，下场不是完全变差。**
- 对 csi1000 这种小盘环境，这条路确实能带来：
  - 更多交易
  - 更高总收益
- 但代价也很清楚：
  - 纯度下降
  - 回撤上升

更具体地说：

- `min_slope` 放到 `-0.016` 基本都不值得
  - 四个组合里它都明显更差
- 真正有意思的是：
  - **`retrace = 0.30` + `min_slope = -0.012`**

所以当前可以保留两种不同取向的候选：

- **高纯度平衡版**
  - `retrace = 0.25`
  - `min_slope = -0.008`
  - `max_slope = 0.016`
  - 特点：`profit_factor` 更高、回撤更低

- **更进攻的交易版**
  - `retrace = 0.30`
  - `min_slope = -0.012`
  - `max_slope = 0.016`
  - 特点：频率更高、总收益更高、`sharpe` 仍然不差

### 当前判断

如果目标是：

- **尽量做成一条更像实盘可持续的纯化策略**
  - 仍然优先保留 `0.25 / -0.008 / 0.016`

- **想在 csi1000 上拿更多机会，同时不明显破坏 `sharpe`**
  - `0.30 / -0.012 / 0.016` 是目前最值得继续往下挖的一版

## 2026-04-23 Round 6：在更进攻版本上扫 breakout body（csi1000 narrow trend）

### 本轮做了什么

上一轮我们已经找到更进攻的候选：

- `max_flag_retrace_ratio = 0.30`
- `min_flag_channel_slope_pct_per_bar = -0.012`
- `max_flag_channel_slope_pct_per_bar = 0.016`

这一轮继续只扫一个参数，专门看 breakout bar 的实体要求：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_retrace_ratio = 0.30`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.012`
  - `max_flag_channel_slope_pct_per_bar = 0.016`
- 只扫：
  - `min_breakout_body_pct = 0.4 / 0.5 / 0.6`

结果文件：

- `outputs/csi1000_narrow_round6_body_grid.csv`

### 关键结果

- `body = 0.4`
  - `planned_trade_count = 85`
  - `profit_factor = 1.9885`
  - `sharpe = 0.6958`
  - `max_drawdown = 1.32%`
  - `total_return = 6.45%`

- `body = 0.5`
  - `planned_trade_count = 84`
  - `profit_factor = 1.9410`
  - `sharpe = 0.6644`
  - `max_drawdown = 1.62%`
  - `total_return = 6.14%`

- `body = 0.6`
  - `planned_trade_count = 76`
  - `profit_factor = 1.8954`
  - `sharpe = 0.6295`
  - `max_drawdown = 1.62%`
  - `total_return = 5.55%`

### 结论

这一轮非常清楚：

- 在 csi1000 这条更进攻的窄趋势版本里，
  - **signal K 实体要求不是越高越好**
- 相反，`min_breakout_body_pct = 0.4` 反而给出了最好的平衡：
  - 频率更高
  - `profit_factor` 更高
  - `sharpe` 更高
  - `max_drawdown` 更低
  - `total_return` 也最高

这说明：

- 小盘股的 breakout bar 波动更大，
- 如果把实体门槛压得太高，会把不少“虽然不完美、但后续仍然能走”的信号误杀掉。

### 当前最优进攻候选

截至这一轮，csi1000 narrow trend 这条线里，当前最值得继续往下挖的进攻版更新为：

- `max_flag_retrace_ratio = 0.30`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.012`
- `max_flag_channel_slope_pct_per_bar = 0.016`
- `min_breakout_body_pct = 0.4`

这版对应：

- `planned_trade_count = 85`
- `profit_factor = 1.9885`
- `sharpe = 0.6958`
- `max_drawdown = 1.32%`
- `total_return = 6.45%`

### 下一步计划

如果继续沿这条进攻版推进，优先级会是：

1. `max_breakout_upper_shadow_pct`
2. `max_breakout_lower_shadow_pct`
3. 等 dynamic exit 真正接到窄趋势策略之后，再比较 trailing / structure

## 2026-04-23 Round 7：上影线阈值（csi1000 narrow trend 进攻版）

### 本轮做了什么

在当前进攻版上只扫一个参数：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_retrace_ratio = 0.30`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.012`
  - `max_flag_channel_slope_pct_per_bar = 0.016`
  - `min_breakout_body_pct = 0.4`
- 只扫：
  - `max_breakout_upper_shadow_pct = 0.25 / 0.35 / 0.45`

结果文件：

- `outputs/csi1000_narrow_round7_upper_shadow_grid.csv`

### 关键结果

- `upper_shadow = 0.25`
  - `planned_trade_count = 85`
  - `profit_factor = 1.9885`
  - `sharpe = 0.6958`
  - `max_drawdown = 1.32%`
  - `total_return = 6.45%`

- `upper_shadow = 0.35`
  - `planned_trade_count = 95`
  - `profit_factor = 1.7872`
  - `sharpe = 0.6302`
  - `max_drawdown = 1.44%`
  - `total_return = 6.14%`

- `upper_shadow = 0.45`
  - `planned_trade_count = 100`
  - `profit_factor = 1.7831`
  - `sharpe = 0.6488`
  - `max_drawdown = 1.43%`
  - `total_return = 6.53%`

### 结论

这轮说明：

- 放宽上影线阈值确实能加频率，
- 但会明显拉低纯度。

比较起来：

- `0.45` 虽然给出最高交易数和略高一点的总收益，
- 但 `profit_factor` 和 `sharpe` 都不如 `0.25`，
- 所以当前更合理的判断是：
  - **上影线还是要严一点**
  - `max_breakout_upper_shadow_pct = 0.25` 更像当前 csi1000 进攻版的最佳点

## 2026-04-23 Round 8：下影线阈值（csi1000 narrow trend 进攻版）

### 本轮做了什么

在上一步确定的 `upper_shadow = 0.25` 基础上，再只扫一个参数：

- 固定：
  - `max_flag_retrace_ratio = 0.30`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.012`
  - `max_flag_channel_slope_pct_per_bar = 0.016`
  - `min_breakout_body_pct = 0.4`
  - `max_breakout_upper_shadow_pct = 0.25`
- 只扫：
  - `max_breakout_lower_shadow_pct = 0.35 / 0.50 / 0.65`

结果文件：

- `outputs/csi1000_narrow_round8_lower_shadow_grid.csv`

### 关键结果

- `lower_shadow = 0.35`
  - `planned_trade_count = 85`
  - `profit_factor = 1.9885`
  - `sharpe = 0.6958`
  - `max_drawdown = 1.32%`
  - `total_return = 6.45%`

- `lower_shadow = 0.50`
  - `planned_trade_count = 92`
  - `profit_factor = 2.0279`
  - `sharpe = 0.7106`
  - `max_drawdown = 1.31%`
  - `total_return = 7.09%`

- `lower_shadow = 0.65`
  - `planned_trade_count = 92`
  - `profit_factor = 2.0279`
  - `sharpe = 0.7106`
  - `max_drawdown = 1.31%`
  - `total_return = 7.09%`

### 结论

这轮和上影线正好相反：

- **下影线并不需要太严**
- 从 `0.35` 放宽到 `0.50` 之后：
  - 交易数提高
  - `profit_factor` 提高
  - `sharpe` 提高
  - `max_drawdown` 还略有改善
  - `total_return` 明显提高
- `0.65` 和 `0.50` 完全一样，说明当前样本里真正有用的信号已经在 `0.50` 时全部放进来了

### 当前最优进攻候选（更新）

截至目前，csi1000 narrow trend 这条线里，当前最值得继续推进的进攻版更新为：

- `max_flag_retrace_ratio = 0.30`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.012`
- `max_flag_channel_slope_pct_per_bar = 0.016`
- `min_breakout_body_pct = 0.4`
- `max_breakout_upper_shadow_pct = 0.25`
- `max_breakout_lower_shadow_pct = 0.50`

这版对应：

- `planned_trade_count = 92`
- `profit_factor = 2.0279`
- `sharpe = 0.7106`
- `max_drawdown = 1.31%`
- `total_return = 7.09%`

### 当前判断

这两轮把 signal K 这块的认识补完整了：

- **实体不需要太苛刻**：`0.4` 最好
- **上影线要严**：`0.25` 最好
- **下影线可以松**：`0.50` 最好

这很符合一个更像小盘股的 breakout 画像：

- 可以允许盘中有一定回踩，
- 但不能接受明显的冲高回落。

## 2026-04-24 Round 9：扫 `narrow_trend_lookback_bars = 10 / 15 / 30`

### 本轮做了什么

在当前最优进攻候选上，只扫左侧窄趋势状态机的窗口长度 `N`：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_retrace_ratio = 0.30`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.012`
  - `max_flag_channel_slope_pct_per_bar = 0.016`
  - `min_breakout_body_pct = 0.4`
  - `max_breakout_upper_shadow_pct = 0.25`
  - `max_breakout_lower_shadow_pct = 0.50`
- 只扫：
  - `narrow_trend_lookback_bars = 10 / 15 / 30`

结果文件：

- `outputs/csi1000_narrow_round9_lookback_grid.csv`

### 关键结果

- `N = 10`
  - `planned_trade_count = 441`
  - `profit_factor = 1.2831`
  - `sharpe = 0.4059`
  - `max_drawdown = 6.18%`
  - `total_return = 10.98%`

- `N = 15`
  - `planned_trade_count = 130`
  - `profit_factor = 1.9506`
  - `sharpe = 0.8499`
  - `max_drawdown = 1.86%`
  - `total_return = 9.69%`

- `N = 30`
  - `planned_trade_count = 11`
  - `profit_factor = 1.8099`
  - `sharpe = 0.2246`
  - `max_drawdown = 0.49%`
  - `total_return = 0.55%`

### 结论

这一轮非常清楚：

- `N = 10` 太短了
  - 会把大量局部加速段都识别成窄趋势
  - 频率直接爆发，但纯度明显下降
  - 回撤也被拉大

- `N = 30` 太长了
  - 会把窄趋势定义得过于严格
  - 样本几乎被杀光
  - 虽然回撤很低，但策略已经太稀了

- `N = 15` 是当前最像平衡点的
  - 交易数仍然足够
  - `profit_factor`、`sharpe`、回撤控制都明显优于 `N = 10`
  - 总收益也仍然很高

### 当前最优进攻候选（更新）

截至目前，csi1000 narrow trend 这条线里，当前最值得继续推进的进攻版更新为：

- `narrow_trend_lookback_bars = 15`
- `max_flag_retrace_ratio = 0.30`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.012`
- `max_flag_channel_slope_pct_per_bar = 0.016`
- `min_breakout_body_pct = 0.4`
- `max_breakout_upper_shadow_pct = 0.25`
- `max_breakout_lower_shadow_pct = 0.50`

这版对应：

- `planned_trade_count = 130`
- `profit_factor = 1.9506`
- `sharpe = 0.8499`
- `max_drawdown = 1.86%`
- `total_return = 9.69%`

### 当前判断

到这一步，左侧窄趋势状态机这块已经很清楚了：

- 太短的 `N` 会把太多局部加速误认成“干净趋势”
- 太长的 `N` 会把样本压得太狠
- **`N = 15` 在 csi1000 这条线上，目前是最合理的 compromise**

## 2026-04-24 Round 10：`N = 10~20` 细网格（csi1000 narrow trend）

### 本轮做了什么

在当前最优进攻候选上，把 `narrow_trend_lookback_bars` 做了一轮更细的扫描：

- 数据：`Dataframes/csi_1000_stock_price2.csv`
- 策略：`BullFlagNarrowTrendContinuationResearcher`
- 固定：
  - `max_flag_retrace_ratio = 0.30`
  - `max_flag_width_pct = 0.12`
  - `min_flag_channel_slope_pct_per_bar = -0.012`
  - `max_flag_channel_slope_pct_per_bar = 0.016`
  - `min_breakout_body_pct = 0.4`
  - `max_breakout_upper_shadow_pct = 0.25`
  - `max_breakout_lower_shadow_pct = 0.50`
- 只扫：
  - `narrow_trend_lookback_bars = 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

结果文件：

- `outputs/csi1000_narrow_round10_lookback_fine_grid.csv`

### 关键结果

- `N = 10`
  - `planned_trade_count = 441`
  - `profit_factor = 1.2831`
  - `sharpe = 0.4059`
  - `max_drawdown = 6.18%`
  - `total_return = 10.98%`

- `N = 11`
  - `planned_trade_count = 286`
  - `profit_factor = 1.4368`
  - `sharpe = 0.5712`
  - `max_drawdown = 4.42%`
  - `total_return = 10.78%`

- `N = 12`
  - `planned_trade_count = 446`
  - `profit_factor = 1.2985`
  - `sharpe = 0.4443`
  - `max_drawdown = 5.21%`
  - `total_return = 11.86%`

- `N = 13`
  - `planned_trade_count = 298`
  - `profit_factor = 1.2043`
  - `sharpe = 0.3161`
  - `max_drawdown = 4.55%`
  - `total_return = 5.78%`

- `N = 14`
  - `planned_trade_count = 204`
  - `profit_factor = 1.5206`
  - `sharpe = 0.6426`
  - `max_drawdown = 2.86%`
  - `total_return = 9.25%`

- `N = 15`
  - `planned_trade_count = 130`
  - `profit_factor = 1.9506`
  - `sharpe = 0.8499`
  - `max_drawdown = 1.86%`
  - `total_return = 9.69%`

- `N = 16`
  - `planned_trade_count = 203`
  - `profit_factor = 1.7814`
  - `sharpe = 0.8173`
  - `max_drawdown = 2.80%`
  - `total_return = 13.05%`

- `N = 17`
  - `planned_trade_count = 133`
  - `profit_factor = 2.1593`
  - `sharpe = 0.9257`
  - `max_drawdown = 1.71%`
  - `total_return = 11.50%`

- `N = 18`
  - `planned_trade_count = 87`
  - `profit_factor = 2.3491`
  - `sharpe = 0.8672`
  - `max_drawdown = 1.41%`
  - `total_return = 8.53%`

- `N = 19`
  - `planned_trade_count = 61`
  - `profit_factor = 2.7685`
  - `sharpe = 0.9149`
  - `max_drawdown = 1.13%`
  - `total_return = 7.34%`

- `N = 20`
  - `planned_trade_count = 92`
  - `profit_factor = 2.0279`
  - `sharpe = 0.7106`
  - `max_drawdown = 1.31%`
  - `total_return = 7.09%`

### 结论

这一轮最重要的发现是：

- `N` 对这条策略的影响**非常非线性**，
- 并不是简单的“越短越差、越长越好”或反过来。

更准确地说，当前出现了几种不同取向的局部最优：

1. **高频高收益但更脏**
   - `N = 10 / 12`
   - 交易数和总收益都很高
   - 但 `profit_factor`、`sharpe` 和回撤控制都明显差

2. **平衡型**
   - `N = 15 / 16 / 17`
   - 交易数还够
   - `profit_factor` 和 `sharpe` 明显更好
   - 其中 `N = 17` 的综合表现最突出

3. **高纯度低频型**
   - `N = 18 / 19`
   - `profit_factor` 和回撤非常漂亮
   - 但频率和总收益会继续下降

### 当前最值得保留的几个候选

如果按不同目标来选：

- **综合最优 / 当前主推**
  - `N = 17`
  - `planned_trade_count = 133`
  - `profit_factor = 2.1593`
  - `sharpe = 0.9257`
  - `max_drawdown = 1.71%`
  - `total_return = 11.50%`

- **最高总收益版**
  - `N = 16`
  - `total_return = 13.05%`
  - 但 `profit_factor` 和回撤不如 `N = 17`

- **最高纯度版**
  - `N = 19`
  - `profit_factor = 2.7685`
  - `max_drawdown = 1.13%`
  - 但频率只有 `61` 笔

### 当前判断

所以到这一步，我会把当前这条 csi1000 narrow trend 主线的默认候选，从 `N = 15` 更新为：

- **`N = 17`**

因为它比 `N = 15`：

- `profit_factor` 更高
- `sharpe` 更高
- `total_return` 更高
- 回撤还更低

这是目前最像“频率、质量、收益”三者都比较平衡的点。

## 2026-04-24 验证框架：时间切分 + 滚动窗口 + 邻域扰动

### 验证对象

先锁定当前主推参数做 out-of-sample 验证：

- `narrow_trend_lookback_bars = 17`
- `max_flag_retrace_ratio = 0.30`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.012`
- `max_flag_channel_slope_pct_per_bar = 0.016`
- `min_breakout_body_pct = 0.4`
- `max_breakout_upper_shadow_pct = 0.25`
- `max_breakout_lower_shadow_pct = 0.50`

数据：

- `Dataframes/csi_1000_stock_price2.csv`

输出文件：

- `outputs/csi1000_narrow_validation_time_splits.csv`
- `outputs/csi1000_narrow_validation_rolling_windows.csv`
- `outputs/csi1000_narrow_validation_n_neighbors.csv`
- `outputs/csi1000_narrow_validation_retrace_neighbors.csv`

### 1. 时间切分验证

切分方式：

- `train = 2015-01-05 ~ 2022-12-30`
- `validation = 2023-01-03 ~ 2024-12-31`
- `test = 2025-01-02 ~ 2026-04-21`

结果：

- `train`
  - `planned_trade_count = 97`
  - `profit_factor = 2.5662`
  - `sharpe = 1.1520`
  - `max_drawdown = 1.71%`
  - `total_return = 11.09%`

- `validation`
  - `planned_trade_count = 20`
  - `profit_factor = 0.7191`
  - `sharpe = -0.2607`
  - `max_drawdown = 1.59%`
  - `total_return = -0.48%`

- `test`
  - `planned_trade_count = 16`
  - `profit_factor = 1.7817`
  - `sharpe = 0.9530`
  - `max_drawdown = 0.64%`
  - `total_return = 0.90%`

### 2. 滚动窗口稳定性

结果：

- `2015-2017`
  - `profit_factor = 5.2041`
  - `sharpe = 1.7109`

- `2018-2020`
  - `profit_factor = 1.3968`
  - `sharpe = 0.4534`

- `2021-2023`
  - `profit_factor = 2.2086`
  - `sharpe = 0.8889`

- `2024-2026`
  - `profit_factor = 1.1493`
  - `sharpe = 0.1675`

### 3. 邻域扰动：`N = 16 / 17 / 18`

结果：

- `N = 16`
  - `validation`
    - `profit_factor = 0.7134`
    - `sharpe = -0.4153`
    - `total_return = -1.06%`
  - `test`
    - `profit_factor = 1.9601`
    - `sharpe = 1.1978`
    - `total_return = 1.37%`

- `N = 17`
  - `validation`
    - `profit_factor = 0.7191`
    - `sharpe = -0.2607`
    - `total_return = -0.48%`
  - `test`
    - `profit_factor = 1.7817`
    - `sharpe = 0.9530`
    - `total_return = 0.90%`

- `N = 18`
  - `validation`
    - `profit_factor = 0.1689`
    - `sharpe = -0.5990`
    - `total_return = -0.90%`
  - `test`
    - `profit_factor = 1.6180`
    - `sharpe = 0.5762`
    - `total_return = 0.52%`

### 4. 邻域扰动：`retrace = 0.28 / 0.30 / 0.32`

结果：

- `retrace = 0.28`
  - `validation`
    - `profit_factor = 0.6242`
    - `sharpe = -0.3542`
    - `total_return = -0.64%`
  - `test`
    - `profit_factor = 1.5053`
    - `sharpe = 0.6016`
    - `total_return = 0.52%`

- `retrace = 0.30`
  - `validation`
    - `profit_factor = 0.7191`
    - `sharpe = -0.2607`
    - `total_return = -0.48%`
  - `test`
    - `profit_factor = 1.7817`
    - `sharpe = 0.9530`
    - `total_return = 0.90%`

- `retrace = 0.32`
  - `validation`
    - `profit_factor = 0.7702`
    - `sharpe = -0.2310`
    - `total_return = -0.45%`
  - `test`
    - `profit_factor = 1.5127`
    - `sharpe = 0.6856`
    - `total_return = 0.69%`

### 综合结论

这套验证框架给出的判断非常清楚：

1. **有 overfit 风险，但不是“完全失真”的那种**
   - 因为 test 段依然是正的，而且 `profit_factor`、`sharpe` 都不差

2. **真正的问题是 regime dependence 很强**
   - `2023-2024` 这一段明显不适合这条策略
   - 而 `2025-2026` 又重新恢复

3. **当前参数不是单点尖峰**
   - `N = 16 / 17`
   - `retrace = 0.30 / 0.32`
   周围都有一定可行性
   - 这说明虽然有调参，但不是那种“一碰就碎”的极端点

4. **当前主推参数仍然可以保留**
   - 因为它在 full-history、test 段和邻域稳定性上都还算站得住
   - 但不能把它理解成“任何市场阶段都稳定赚钱”

### 当前判断

所以到这一步，我会把这条 csi1000 narrow trend 策略的状态定义为：

- **已形成可交易候选**
- **但具有明显市场阶段依赖**

下一步真正值得做的，不再是继续盲调 entry 参数，而是：

1. 加一个市场环境 / 情绪过滤
2. 或把这条策略限定在它更擅长的 regime 下使用

## 2026-04-23 当前最优平衡候选的漏斗（csi1000 narrow trend）

### 漏斗口径

当前漏斗使用的是这一版平衡候选：

- `narrow_trend_lookback_bars = 20`
- `narrow_trend_max_bear_ratio = 0.25`
- `narrow_trend_min_run_bars = 1`
- `max_flag_retrace_ratio = 0.25`
- `max_flag_width_pct = 0.12`
- `min_flag_channel_slope_pct_per_bar = -0.008`
- `max_flag_channel_slope_pct_per_bar = 0.016`

数据：

- `Dataframes/csi_1000_stock_price2.csv`

这里同时看两层：

- **run 级漏斗**：每段连续 `narrow_uptrend_state=True` 只取最后一根
- **row 级漏斗**：所有 candle 级别的信号统计

### run 级漏斗

- `run_end_events = 4350`
- `run_with_structured_row = 2764`
- `run_with_candidate_row = 457`
- `run_with_breakout_row = 138`
- `run_with_follow_through_row = 111`
- `run_with_entry_row = 53`
- `executed_trades = 51`

对应保留率：

- `run_end -> structured = 63.54%`
- `structured -> candidate = 16.53%`
- `candidate -> breakout = 30.20%`
- `breakout -> follow_through = 80.43%`
- `follow_through -> entry = 47.75%`
- `entry -> executed = 96.23%`

### row 级漏斗

- `structured_rows = 27238`
- `bull_flag_candidate = 1726`
- `breakout_candle = 174`
- `signal_candle = 174`
- `follow_through_confirmed = 133`
- `entry_signal = 55`
- `entry_signal_executed = 51`

### 结论

这一版最新平衡候选的漏斗已经比前面的旧版本清晰很多，主要问题集中在两层：

1. **`structured -> candidate`**
   - `2764 -> 457`
   - 这说明当前最主要的掉点，依然在 flag 本体本身：
     - `flag_retrace_ratio`
     - `flag_width_pct`
     - `flag channel slope`

2. **`follow_through -> entry`**
   - `111 -> 53`
   - 这层依然几乎是被 `reward_to_risk` 过滤掉的

具体看 row 级 follow-through：

- `follow_through_rows = 133`
- `reward_to_risk_ok_true = 55`
- `reward_to_risk_ok_false = 78`
- `trend_environment_ok_true = 133`
- `trend_environment_ok_false = 0`

这说明：

- 现在不是趋势环境在卡掉 follow-through
- 也不是次日无法成交
- **核心还是赔率不够**

### 当前判断

所以到这一步，当前 csi1000 narrow trend 这条线最该继续优化的地方，顺序会是：

1. `flag_retrace_ratio`
2. `flag channel slope`
3. breakout / follow-through 后的赔率结构

而不是再回头去怀疑 left trend 的 `narrow_state` 频率本身。
