# Bull Flag Continuation 使用说明

## 目录位置

本策略的归档目录在：

`strategy_archive/bull_flag_continuation/`

建议一起看的文件：

- `strategies/bull_flag_continuation.py`
- `strategies/bull_flag_exit_variants.py`
- `strategy_archive/bull_flag_continuation/bull_flag_continuation_daily.ipynb`
- `strategy_archive/bull_flag_continuation/bull_flag_trailing_exit_backtest.ipynb`
- `strategy_archive/bull_flag_continuation/open_positions.csv`
- `strategy_archive/bull_flag_continuation/experiment_logs/EXPERIMENT_LOG.md`

## 策略一句话说明

这是一个**多头环境里的标准牛旗延续策略**。

当前归档后的主版本不是静态止盈版，而是：

**bull flag entry + TP1 之后 trailing stop**

核心思路是：

1. 先要求股票仍然处在 `20 > 60 > 120` 的多头环境里。
2. 在这个背景里找一段强势 `flagpole`。
3. 再找一段浅而整齐的 `flag`。
4. 等价格收盘突破旗面上沿，并且下一根有 follow-through。
5. 入场后先给原始止损空间；一旦达到 `TP1`，就切换到动态 trailing stop 去锁利润。

## 当前主版本状态

这条线目前是整个项目里最成熟、最值得继续投入的版本。

当前推荐的主配置是：

### Entry 主配置

- `max_flag_retrace_ratio = 0.30`
- `min_breakout_body_pct = 0.60`
- `max_breakout_upper_shadow_pct = 0.35`
- `max_breakout_lower_shadow_pct = 0.50`
- `max_peak_sma60_return_10 = 0.055`

### Exit 主配置

- `tp1_fraction_of_target = 0.50`
- `trailing_stop_fraction_of_flagpole = 0.25`

在当前项目里，这套配置的意义是：

- 频率已经够了
- 主要工作转移到提高盈亏比
- 当前最有效的方向是“更聪明的退出”，而不是继续堆 entry 条件

## 这条策略在赚什么钱

它赚的是：

- 一段已经启动的多头趋势里
- 强势上涨后的短暂整理
- 随后再次向上突破并延续趋势的那一段钱

换句话说，它不是在赚横盘反弹的钱，也不是在赌底部反转，
而是在赚**强趋势中的二次推进**。

## 数据要求

策略输入必须是日线级别 `DataFrame`，并包含这些字段：

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

当前仓库里已经有现成的数据拉取函数：

- `get_csi500_member_prices()`
- `get_hs300_member_prices()`
- `get_next_trading_day()`

## 研究流程

### 静态 bull flag 研究器

`BullFlagContinuationResearcher` 的典型流程是：

1. `add_features()`
   生成 `bullish_stack`、pivot、flagpole、flag、signal K、背景过滤等特征。
2. `add_signals()`
   生成研究口径下的原始入场信号。
3. `add_research_outcomes()`
   按静态止损 / 静态止盈 / 时间止损去回放持仓结果。
4. `add_trade_df()`
   输出给 backtester 用的逐笔交易表。

### 动态退出主版本

当前归档主版本建议用：

- `BullFlagTrailingAfterTp1Researcher`

它保留 bull flag 的 entry 定义不变，只重写持仓后的退出管理。

当前已经实现过的退出族包括：

- 静态止盈 / 止损基线
- `TP1 -> 保本 stop`
- `TP1 -> trailing stop`
- `TP1 -> MA trail`
- `TP1 -> structure trail`
- `TP1 -> volume failure exit`
- `TP1 -> close retrace exit`

但从已有实证结果看，当前最值得保留的主线仍然是：

- `TP1 -> trailing stop`

## 当前主版本的入场逻辑

### 1. 多头背景

先要求：

- `sma_20 > sma_60 > sma_120`

这表示当前仍处在运行中的多头趋势环境里。

### 2. Flagpole

以一个已确认的 `pivot_high` 作为 `flag peak`，
再从这个高点往前回看一段窗口：

- 优先取最低 confirmed `pivot_low`
- 如果没有 confirmed `pivot_low`
  就退化成窗口最低价

由此得到：

- `flagpole_start_low`
- `flag_peak_high`
- `flagpole_length = flag_peak_high - flagpole_start_low`
- `flagpole_return`

### 3. Flag

在 `flag peak` 后面找一个短整理窗口，主要检查：

- 旗面长度
- 回撤深度
- 上下沿 slope
- 整体宽度

当前主版本最关键的过滤之一是：

- `max_flag_retrace_ratio = 0.30`

也就是旗面回撤不能太深。

### 4. Breakout Signal K

当前 signal K 需要同时满足：

- 收盘突破旗面上沿
- 收盘高于前高
- 实体不能太差
- 上下影不能太夸张

当前较优版本不是“极端完美的光头阳线”，而是：

- 实体够强
- 收盘真正站稳
- 允许一定正常影线

### 5. Follow-through

默认还需要下一根有 follow-through，
也就是 breakout 不是只冲一下马上失效。

## 当前主版本的退出逻辑

### 静态基础部分

入场后，仍然先沿用 bull flag 原本的：

- `signal_hard_stop_price`
- `signal_take_profit_price`

其中：

- 初始止损放在 `flag_low` 下方
- 最终止盈按 `measured_move_fraction * flagpole_length` 估算

### TP1

动态退出不会一进场就启用。

它先定义一个 `TP1`：

`tp1_price = entry_price + tp1_fraction_of_target * (final_target - entry_price)`

当前默认：

- `tp1_fraction_of_target = 0.50`

也就是先走到目标的一半。

### Trailing Stop

一旦达到 `TP1`，从**下一根 K**开始，保护止损切换成：

`active_stop = max(base_stop, highest_high_since_tp1 - trailing_stop_fraction_of_flagpole * flagpole_length)`

当前默认：

- `trailing_stop_fraction_of_flagpole = 0.25`

这表示 trailing 的距离按旗杆长度的 25% 来算。

### 为什么主推这一版

在当前项目里，这一版相对静态基线的主要优势是：

- `profit_factor` 更高
- `sharpe` 更高
- `max_drawdown` 更低
- 总收益只小幅回落，或者在不少版本里几乎持平

所以它更像“实战增强版”，而不是只在纸面上更复杂。

## 每日 notebook 的用途

`bull_flag_continuation_daily.ipynb` 主要做四件事：

1. 拉取最新指数成分股和最近一段时间的日线。
2. 用最新已收盘交易日生成“明天可入场”的 bull flag 候选。
3. 读取 `open_positions.csv`。
4. 按当前主版本的**动态退出规则**监控持仓，给出：
   - `hold`
   - `exit_next_open`
   - `exit_overdue`
   - `review`

这里和静态版本不同的地方是：

- notebook 里的持仓监控已经对齐到了动态 exit 口径
- 也就是说，如果你用的是 trailing 主版本，daily notebook 也会看 `TP1` 和当前 `active_protective_stop`

## notebook 里你每天最常改的参数

这些参数都在 notebook 开头的配置单元格里：

- `UNIVERSE`
  选 `csi500` 或 `hs300`

- `LOOKBACK_CALENDAR_DAYS`
  向前拉多久的数据

- `END_DATE`
  默认是今天，一般不用改

- `PAUSE_SECONDS`
  拉数节奏控制

- `MAX_CALLS_PER_MINUTE`
  Tushare 限频保护

- `ENTRY_PRICE_BASIS`
  生成明日候选时，计划入场价是用：
  - `follow_through_close`
  - `signal_close`

- `SAVE_OUTPUTS`
  是否自动把日报结果保存成 csv

## open_positions.csv 怎么填

这个文件放在：

`strategy_archive/bull_flag_continuation/open_positions.csv`

推荐列如下：

- `ticker`
  必填。六位股票代码，不带后缀。

- `entry_date`
  必填。实际入场日期。

- `entry_price`
  必填。你的真实入场价。

- `shares`
  可选。持仓股数。

- `signal_date`
  可选。对应的信号日。
  如果你填了，daily notebook 会优先按它去匹配策略信号。

- `note`
  可选。你自己的备注。

## notebook 会产出什么

如果 `SAVE_OUTPUTS=True`，它会自动创建：

`strategy_archive/bull_flag_continuation/outputs/`

并保存这些文件：

- `next_session_candidates_YYYYMMDD.csv`
- `position_monitor_YYYYMMDD.csv`
- `exit_alerts_YYYYMMDD.csv`
- `hold_watchlist_YYYYMMDD.csv`

## 实盘使用时要注意的几个口径差异

### 1. 明日候选是“收盘后计划”，不是实际成交

`get_next_session_candidates()` 用的是最新收盘信息，
止损止盈也是按计划参考价先估出来的。
第二天真实开盘可能有跳空。

### 2. 动态退出是日线语义，不是盘中逐笔语义

当前 trailing / MA / close-retrace 这些逻辑，都是按日线来解释：

- `TP1` 命中当天，不立即启用新的保护止损
- 新 stop 从下一根 bar 开始生效
- 收盘型退出信号都按“次日开盘执行”

所以它是一个保守且一致的日线口径，不等于真实盘中逐笔路径。

### 3. 当前 backtester 仍然是单次完整退出模型

这意味着：

- 动态全平没问题
- 分批止盈、卖一半再拿一半，当前不在主 pipeline 里

如果以后要做 partial exit，需要单独起一个 multi-leg 研究层。

## 最小使用示例

```python
from strategies.china_stock_data import get_csi500_member_prices, get_next_trading_day
from strategies.bull_flag_exit_variants import BullFlagDynamicExitConfig, BullFlagTrailingAfterTp1Researcher

price_df = get_csi500_member_prices(sd="2025-01-01", ed="2026-04-14", pause_seconds=0.0)

cfg = BullFlagDynamicExitConfig(
    universe="csi500",
    max_flag_retrace_ratio=0.30,
    min_breakout_body_pct=0.60,
    max_breakout_upper_shadow_pct=0.35,
    max_breakout_lower_shadow_pct=0.50,
    max_peak_sma60_return_10=0.055,
    tp1_fraction_of_target=0.50,
    trailing_stop_fraction_of_flagpole=0.25,
)

researcher = BullFlagTrailingAfterTp1Researcher(price_df, config=cfg)

as_of_date = price_df["date"].max()
next_trade_date = get_next_trading_day(as_of_date)

next_candidates = researcher.get_next_session_candidates(
    as_of_date=as_of_date,
    next_trade_date=next_trade_date,
)
```

## 这次归档里新增的内容

- 一份 bull flag 主版本中文说明
- 一个每日信号 notebook
- 一个 `open_positions.csv` 模板
- 一个已经对齐动态 exit 口径的 `monitor_positions()` 用法
- 一整套实验日志与输出表，方便继续迭代
