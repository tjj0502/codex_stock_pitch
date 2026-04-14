# Blue Chip Range Reversion 使用说明

## 目录位置

本策略的归档目录在：

`strategy_archive/blue_chip_range_reversion/`

建议一起看的文件：

- `blue_chip_range_reversion.py`
- `strategy_archive/blue_chip_range_reversion/blue_chip_range_reversion_daily.ipynb`
- `strategy_archive/blue_chip_range_reversion/open_positions.csv`

## 策略一句话说明

这是一个**蓝筹股区间均值回归**策略。

核心思路是：

1. 先筛出最近一段时间内明显在区间里来回波动、而不是强趋势单边走的股票。
2. 再等股价回到区间下沿附近。
3. 只有在下沿附近出现一定的反弹确认后，才把它列入次日可入场名单。
4. 入场后按固定止损、跌破区间、止盈、时间止损四套规则管理离场。

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

## 研究流程

`BlueChipRangeReversionResearcher` 的典型流程是：

1. `add_features()`
   生成区间、均线离散度、触边次数、反弹确认等特征。
2. `add_signals()`
   生成研究口径下的原始入场信号。这里默认信号出现在 `t` 日收盘后，真正入场发生在 `t+1` 开盘。
3. `add_research_outcomes()`
   按单票单持仓方式回放离场过程。
4. `add_trade_df()`
   输出逐笔交易表。

这次额外补了两个更适合日常使用的方法：

- `get_next_session_candidates()`
  用最新一个已收盘交易日直接生成“明天可入场”的股票清单，不再要求必须先知道 `t+1` 开盘价。
- `monitor_positions()`
  用你已经入场的股票表，按同一套策略规则给出继续持有、明日开盘离场、或已经过期应离场的提醒。

## RangeStrategyConfig 参数说明

下面是 `RangeStrategyConfig` 里每一个参数的具体作用。

### 标的范围

- `universe`
  可选 `csi500` 或 `hs300`。
  只是策略层面对样本空间的标记，最好和你实际拉取的数据源保持一致。

### 区间定义

- `range_window`
  计算区间上下沿时使用的滚动窗口天数。
  越大，区间越平滑、越慢；越小，区间越灵敏、但更容易被短期波动影响。

- `upper_quantile`
  上沿分位数。
  例如 `0.9` 表示用最近 `range_window` 天高价的 90% 分位来定义区间上沿。
  这么做比直接取最高点更稳，不容易被单日尖刺影响。

- `lower_quantile`
  下沿分位数。
  例如 `0.1` 表示用最近 `range_window` 天低价的 10% 分位来定义区间下沿。

### 区间振幅过滤

- `min_amplitude`
  区间最小振幅。
  计算方式大致是 `(range_upper - range_lower) / close`。
  太小表示波动不够，做均值回归没有空间。

- `max_amplitude`
  区间最大振幅。
  太大通常意味着波动已经过激，可能不是稳定区间，而是结构变坏或趋势转折。

### 中期走势过滤

- `min_return_60`
  最近 60 个交易日收益率下限。
  用来过滤掉过去 60 天跌得太狠的股票。
  如果你希望放宽“超跌反弹”类机会，可以把它调低到负值。

- `max_abs_return_60`
  最近 60 个交易日收益率上限。
  名字里有 `abs`，但当前实现实际是直接限制 `ret_60d <= max_abs_return_60`。
  主要目的是排除过去 60 天涨太猛、处在强趋势中的股票。

### 均线平滑过滤

- `ma_dispersion_window`
  用来衡量均线是否发散的均线窗口组合，默认是 `(20, 60, 120)`。
  策略会分别计算这些均线，然后看它们彼此之间是否分散得太开。

- `max_ma_dispersion`
  允许的最大均线离散度。
  越小，越偏向“横盘、均线缠绕”的股票；越大，允许更有趋势感的标的进来。

### 区间有效性过滤

- `touch_zone_pct`
  触边区域宽度。
  例如 `0.2` 表示把区间下方 20% 视作“靠近下沿”，把区间上方 20% 视作“靠近上沿”。
  这个参数用于统计股票在最近窗口内是否真的反复碰过上下沿。

- `min_lower_touches`
  最近 `range_window` 天里，至少多少次碰过下沿区域。
  太少说明并不是一个成熟的回摆区间。

- `min_upper_touches`
  最近 `range_window` 天里，至少多少次碰过上沿区域。
  这个和 `min_lower_touches` 一起确认“来回震荡”而不是单边。

### 入场过滤

- `entry_zone_threshold`
  入场时价格必须处在区间中的多低位置。
  例如 `0.2` 表示收盘时 `zone_position <= 0.2`，也就是收盘接近区间下沿。
  越小越保守，只接近最下沿才允许入场。

### 风险收益与离场规则

- `stop_loss_pct`
  固定止损比例。
  例如 `0.1` 表示入场后如果收盘价跌到入场价的 90% 或更低，就触发硬止损。

- `breakdown_buffer`
  跌破区间下沿时额外留出的缓冲带。
  例如 `0.03` 表示必须跌破 `range_lower * (1 - 0.03)` 才算有效跌破。
  这样可以少受“假跌破一下又拉回”的噪音影响。

- `breakdown_confirm_days`
  跌破区间下沿需要连续确认多少天才触发离场。
  例如默认 `2` 表示连续两天收盘都有效跌破下沿，才触发 `breakdown_stop`。

- `take_profit_r_multiple`
  止盈按几倍风险来设。
  当前策略里，理论止盈会取下面两者的较小值：
  1. 区间上沿；
  2. `入场价 * (1 + stop_loss_pct * take_profit_r_multiple)`。

- `max_holding_days`
  最长持有交易日数。
  到了这个天数仍未触发别的离场规则，就触发 `time_stop`。

### 离场开关

- `enable_hard_stop`
  是否启用固定止损。

- `enable_breakdown_stop`
  是否启用跌破区间下沿止损。

- `enable_take_profit`
  是否启用止盈。

- `enable_time_stop`
  是否启用时间止损。

这四个开关至少要保留一个为 `True`。

## 信号是怎么形成的

候选股票必须同时满足这些条件：

1. `range_candidate = True`
   也就是通过了区间振幅、60 日收益、均线离散度、上下沿触碰次数这些过滤。
2. `entry_zone_ok = True`
   收盘位置足够靠近区间下沿。
3. `expected_upside_ok = True`
   当前位置到区间上沿的预期空间，至少能覆盖策略设定的风险收益要求。
4. `rebound_confirmed = True`
   反弹确认成立。

其中反弹确认要求：

- 当日 `close > open`
- 当日不是 `inside bar`
- 并且满足下面两种之一：
  - 是 `outside bar`
  - 或者 `close > sma_5` 且 `close > 前一日 high`

## 离场是怎么触发的

持仓建立后，策略会每天收盘检查：

1. `hard_stop`
   收盘价跌破固定止损价。
2. `breakdown_stop`
   连续若干天有效跌破区间下沿。
3. `take_profit`
   收盘价达到止盈目标。
4. `time_stop`
   持仓时间达到上限。

一旦某天收盘触发离场，研究口径与 notebook 提醒口径都按**下一交易日开盘离场**处理。

## 每日 notebook 的用途

`blue_chip_range_reversion_daily.ipynb` 主要做四件事：

1. 拉取最新成分股和最新日线。
2. 用最新已收盘日生成“明天可入场”名单。
3. 读取 `open_positions.csv`。
4. 对你已入场股票给出：
   - `hold`
   - `exit_next_open`
   - `exit_overdue`
   - `data_issue`

## notebook 里你每天最常改的参数

这些参数都在 notebook 开头的配置单元格里：

- `UNIVERSE`
  选 `csi500` 或 `hs300`。

- `LOOKBACK_CALENDAR_DAYS`
  向前拉多久的数据。建议至少覆盖 120 日均线窗口，再多留一点缓冲。

- `END_DATE`
  默认是今天。一般不用改。

- `PAUSE_SECONDS`
  每只股票拉数之间的暂停时间。Tushare 限频紧时可以调大。

- `MAX_CALLS_PER_MINUTE`
  每分钟最大请求数上限。用来尽量避开 Tushare 的限频。

- `ENTRY_PRICE_BASIS`
  生成“明天候选”时，参考哪个价格来估算止损止盈。
  默认是 `close`，意思是先用今天收盘价做计划价。
  这只是计划口径，不等于你明天真实成交价。

- `SAVE_OUTPUTS`
  是否把日报结果自动保存成 csv。

## open_positions.csv 怎么填

这个文件放在：

`strategy_archive/blue_chip_range_reversion/open_positions.csv`

推荐列如下：

- `ticker`
  必填。六位股票代码，不带后缀。

- `entry_date`
  必填。实际入场日期，格式建议 `YYYY-MM-DD`。

- `entry_price`
  必填。你的实际入场价格。

- `shares`
  可选。持仓股数。填了以后 notebook 会顺带算浮盈亏金额。

- `signal_date`
  可选。触发买点的信号日期。
  如果不填，系统会默认认为信号日在 `entry_date` 的前一个交易日。

- `note`
  可选。你自己的备注。

## notebook 会产出什么

如果 `SAVE_OUTPUTS=True`，它会在运行时自动创建：

`strategy_archive/blue_chip_range_reversion/outputs/`

并保存这些文件：

- `next_session_candidates_YYYYMMDD.csv`
- `position_monitor_YYYYMMDD.csv`
- `exit_alerts_YYYYMMDD.csv`

## 实盘使用时要注意的两个口径差异

### 1. 次日候选是“收盘后计划”，不是“实际成交”

`get_next_session_candidates()` 是按最新收盘信息判断“明天是否值得看”，
止损止盈也是按配置好的参考价先估出来的。
真正明天开盘成交价和今天收盘价可能有跳空差异。

### 2. 持仓监控默认按策略原始规则提醒

`monitor_positions()` 会尽量按策略研究逻辑来给提醒。
如果你实际交易时做过人工干预，比如：

- 不是次日开盘买的
- 中途加仓减仓
- 手动把止损价改过

那 notebook 的提醒更适合作为“策略视角提醒”，不一定等于你人工交易后的真实风险管理结果。

## 最小使用示例

```python
from china_stock_data import get_csi500_member_prices, get_next_trading_day
from blue_chip_range_reversion import BlueChipRangeReversionResearcher, RangeStrategyConfig

price_df = get_csi500_member_prices(sd="2025-01-01", ed="2026-04-14", pause_seconds=0.0)
cfg = RangeStrategyConfig(universe="csi500")
researcher = BlueChipRangeReversionResearcher(price_df, config=cfg)

as_of_date = price_df["date"].max()
next_trade_date = get_next_trading_day(as_of_date)

next_candidates = researcher.get_next_session_candidates(
    as_of_date=as_of_date,
    next_trade_date=next_trade_date,
)
```

## 这次归档里新增的内容

- 一份参数说明和使用说明
- 一个每日可跑 notebook
- 一个 `open_positions.csv` 模板
- 两个实盘辅助接口：
  - `get_next_session_candidates()`
  - `monitor_positions()`
