# Gap Breakout Continuation 使用说明

## 策略一句话说明

这是一个**强势趋势中的向上缺口延续策略**。

核心逻辑是：

1. 先确认个股原本就处在多头趋势里。
2. 只挑真正有力度的向上跳空缺口，不做所有缺口。
3. 默认等 1 到 3 天确认，不追所有 gap day。
4. 入场后用**缺口 stop + MA10 趋势退出 + 持有上限**管理仓位。

## 当前实现范围

这版先落的是最小可行的价格结构版本，重点是把研究闭环跑通：

- 趋势过滤
- 缺口检测
- 缺口确认
- `trade_df`
- `backtester` 兼容
- `inspect_signal()` / `plot_signal_context()`
- `get_next_session_candidates()` / `monitor_positions()`

这版还没有做的增强项：

- 真正的全 A 拉数工作流
- 缺口按“突破 / 延续 / 衰竭”做更精细分类
- 事件 / 财报缺口标签
- 更复杂的动态止盈 / 分批止盈

## 当前默认规则

### 标的过滤

- `name` 里含 `ST` 的名字会被过滤
- 当天停牌或无有效成交的 bar 会被过滤
- 上市 bars 少于 `120` 不做
- `20` 日平均成交额必须大于 `1e8`

### 趋势过滤

默认要求：

- `close > SMA20 > SMA60`
- 过去 `40` 日涨幅大于 `10%`
- 最近 `20` 日里至少 `12` 天收盘在 `SMA20` 上方

### 缺口定义

默认要求：

- `low > prev_high`
- `gap_pct > 1.5%`
- `volume > 1.5 * avg_volume_20` 或 `turnover > 1.8 * avg_turnover_20`
- gap day 上影不能太长
- 缺口离 `SMA20` 不能太远，默认不超过 `12%`
- gap 前连续上涨天数不能太多，默认不超过 `6`

### 入场逻辑

默认 `entry_mode="confirm"`，也就是确认版：

- 先出现 gap day
- 接下来 `1~3` 天里不能回补缺口
- 收盘重新站上确认价位
- 默认确认价位是 `gap day close`
- 信号日次日开盘买入

如果你把 `entry_mode` 改成 `gap_day`，就会变成 gap day 当天触发，次日开盘买入。

### 出场逻辑

当前默认：

- `gap_stop`
  - 收盘跌破 `gap_day_low`
- `ma10_exit`
  - 收盘跌破 `SMA10`
- `time_stop`
  - 最长持有 `10` 个交易日

离场信号统一按**次日开盘执行**。

## 关键配置项

最常改的参数在 `GapBreakoutStrategyConfig`：

- `entry_mode`
  - `confirm` 或 `gap_day`
- `confirm_level_mode`
  - `gap_close` 或 `gap_midpoint`
- `min_gap_pct`
- `volume_multiple`
- `turnover_multiple`
- `confirm_window`
- `max_extension_from_sma20`
- `max_holding_days`

## 常用入口

主策略类：

- `GapBreakoutContinuationResearcher`

参数配置：

- `GapBreakoutStrategyConfig`

参数扫描：

- `run_gap_breakout_grid_search`

## 每日 notebook

归档目录里配了一个 daily notebook：

- `strategy_archive/gap_breakout_continuation/gap_breakout_continuation_daily.ipynb`

它会做这些事：

1. 读取本地价格表
2. 生成“下一交易日候选”
3. 读取 `open_positions.csv`
4. 给出现有持仓的 `hold / exit_next_open / data_issue`
5. 默认复盘一笔最值得看的候选

## open_positions.csv

模板文件：

- `strategy_archive/gap_breakout_continuation/open_positions.csv`

当前需要的最小字段：

- `ticker`
- `entry_date`
- `entry_price`
