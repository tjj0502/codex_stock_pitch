# A股蓝筹震荡均值回归策略 v1

策略源码当前位于 [blue_chip_range_reversion.py](/C:/Users/Jay/GitRepo/codex_stock_pitch/blue_chip_range_reversion.py)。

## 定位

- 这是一个**信号研究类**，不是组合层资金回测器。
- 目标是研究蓝筹/指数成分股在震荡区间下沿附近的低吸机会。
- 当前实现类名是 `BlueChipRangeReversionResearcher`。

## 数据入口

- 通用成分股入口：
  - `get_index_constituents(index_code, ...)`
  - `get_index_member_prices(index_code, ...)`
- 便捷 wrapper：
  - `get_csi500_member_prices(...)`
  - `get_hs300_member_prices(...)`
- 当前价格数据使用 `qfq` 前复权。
- 当前成分股采用**最新快照回看历史**，因此历史研究存在 survivorship bias。

## 源码阅读顺序

如果你要顺着代码读，建议按下面顺序：

1. 先看 `get_index_constituents(...)`
   - 它负责拿指数最新成分股快照，并把 `index_code`、`universe`、`constituent_history_mode` 这些元信息写进 `attrs`
2. 再看 `get_index_member_prices(...)`
   - 它负责逐只股票抓取 `qfq` 日线，并统一整理成项目内通用的 `stock_candle_df` schema
3. 再看 `RangeStrategyConfig`
   - 它定义了这套策略用到的全部阈值
4. 再看 `BlueChipRangeReversionResearcher.add_features()`
   - 这里定义“什么样的股票算震荡股”
5. 再看 `BlueChipRangeReversionResearcher.add_signals()`
   - 这里定义“什么时候发出低吸信号”
6. 最后看 `BlueChipRangeReversionResearcher.add_research_outcomes()`
   - 这里是单标的事件状态机，负责把信号变成研究结果

## 代码执行流程

研究类每一步的职责是固定的：

### 1. 初始化

- 构造函数先校验 `stock_candle_df` 必备字段是否齐全
- 然后统一转换日期和数值类型
- 再按 `date/ticker` 排序，并保留原始 `attrs`

### 2. `add_features()`

这一步只做“描述”，不做交易决策：

- 先算 `ret_60d` 和 `sma_5/20/60/120`
- 再用均线最大值和最小值的差定义 `ma_dispersion`
- 再用滚动分位数定义 `range_upper` 和 `range_lower`
- 再把当前价格投影到区间内，得到 `zone_position`
- 再统计最近窗口内接近上沿和下沿的次数
- 最后组合出 `range_candidate`

### 3. `add_signals()`

这一步把“描述特征”变成“原始信号”：

- 信号时间是 `t` 日收盘后
- 执行价格默认是 `t+1` 日开盘价，所以这里会先写好 `entry_date_next` 和 `entry_open_next`
- 再判断是否在区间下沿附近
- 再判断上方空间是否够到 `2R`
- 再判断 3 个反弹条件是否至少满足 2 个
- 全部满足后，才把 `entry_signal=True`

### 4. `add_research_outcomes()`

这一步是事件回放核心：

- 对每只股票单独处理
- 找到一条 `entry_signal=True` 的信号
- 假设在下一根 K 线开盘价买入
- 从买入当天开始逐日看收盘价
- 只要命中硬止损、结构止损、止盈、时间止损中的任意一个，就在下一交易日开盘价退出
- 如果持仓期间又出现新的同向信号，不重复开仓，只记成 `entry_signal_suppressed=True`
- 最后把这次交易的退出原因、持有天数、收益、MFE、MAE 写回原信号行

## 配置对象

使用 `RangeStrategyConfig` 管理参数，默认值对应 v1 规则：

- `universe="csi500"`
- `range_window=120`
- `upper_quantile=0.9`
- `lower_quantile=0.1`
- `min_amplitude=0.20`
- `max_amplitude=0.45`
- `max_abs_return_60=0.15`
- `ma_dispersion_window=(20, 60, 120)`
- `max_ma_dispersion=0.08`
- `touch_zone_pct=0.20`
- `min_lower_touches=2`
- `min_upper_touches=2`
- `entry_zone_threshold=0.20`
- `stop_loss_pct=0.10`
- `breakdown_buffer=0.03`
- `breakdown_confirm_days=2`
- `take_profit_r_multiple=2.0`
- `max_holding_days=20`

## 特征

`add_features()` 会追加以下核心字段：

- `ret_60d`
- `sma_5`, `sma_20`, `sma_60`, `sma_120`
- `ma_dispersion`
- `range_upper`, `range_lower`, `range_mid`
- `range_width`, `range_amplitude`
- `zone_position`
- `lower_touch_count`, `upper_touch_count`
- `expected_upside_to_upper`
- `close_gt_open`, `close_gt_prev_close`, `close_gt_sma_5`
- `rebound_confirm_count`
- `range_candidate`

其中：

- 区间上沿/下沿分别来自 `high` / `low` 的滚动分位数。
- `zone_position = (close - lower) / (upper - lower)`，若区间宽度为 0，则回退到 `0.5`。
- `range_candidate` 要同时满足：
  - `abs(ret_60d) <= max_abs_return_60`
  - `range_amplitude` 在 `[min_amplitude, max_amplitude]`
  - `ma_dispersion <= max_ma_dispersion`
  - 下沿触达次数和上沿触达次数都达到阈值

## 入场信号

`add_signals()` 会追加：

- `entry_date_next`
- `entry_open_next`
- `entry_zone_ok`
- `expected_upside_ok`
- `rebound_confirmed`
- `signal_take_profit_price`
- `signal_hard_stop_price`
- `entry_signal`

规则固定为：

- 当日收盘后判断
- 次日开盘执行
- 需要同时满足：
  - `range_candidate`
  - `zone_position <= entry_zone_threshold`
  - `(range_upper - close) / close >= 2R`
  - 三个反弹条件里至少满足两个：
    - `close > open`
    - `close > close[t-1]`
    - `close > sma_5`

## 事件研究输出

`add_research_outcomes()` 会做单标的事件回放，并追加：

- `entry_signal_executed`
- `entry_signal_suppressed`
- `exit_signal_date`
- `exit_date_next`
- `exit_open_next`
- `exit_reason`
- `holding_days`
- `realized_open_to_open_return`
- `max_favorable_excursion`
- `max_adverse_excursion`

退出优先级：

1. `hard_stop`
2. `breakdown_stop`
3. `take_profit`
4. `time_stop`

如果样本结束前还没退出，则标记为 `open_position`。

持仓期间若再次出现新的同向 `entry_signal`，会标记为 `entry_signal_suppressed=True`，但不会重复开仓。

## 检查接口

- `get_candidates(as_of_date=None)`
  - 返回某个信号日满足入场条件的候选股
- `inspect_signal(ticker, signal_date)`
  - 返回结构化检查结果
- `plot_signal_context(ticker, signal_date)`
  - 用 Plotly 展示 K 线、区间线、入场/出场点和条件命中情况

## 当前限制

- 只做价格行为版，不含基本面过滤
- 不含指数 regime 过滤
- 不做分批建仓
- 不做组合资金管理
- 不解决历史成分股变化导致的 survivorship bias
