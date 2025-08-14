import pandas as pd
import numpy as np
import talib

#通道型布林带信号：最新价突破布林带上界发出做空信号，突破下界发出做多信号，最新价由上到下突破中轨，则平空；最新价由下到上突破中轨，则平多
def bollinger_r(price, window, num_std_upper, num_std_lower, shift_for_exec = 0):
    """
    布林带信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, 布林带移动平均窗口
    :param num_std_upper: float, 上轨的标准差倍数
    :param num_std_lower: float, 下轨的标准差倍数
    :return: pandas Series, 信号值（1: 开多, -1: 开空, 2: 平多, -2: 平空, 0: 无操作）
    """
    close = price["close"].astype(float)
    upperband, middleband, lowerband = talib.BBANDS(
        close,
        timeperiod=window,
        nbdevup=num_std_upper,
        nbdevdn=num_std_lower,
        matype=0
    )
    upperband = upperband.shift(shift_for_exec)
    middleband = middleband.shift(shift_for_exec)
    lowerband = lowerband.shift(shift_for_exec)

    signal = pd.Series(0, index=close.index)
    current_position = 0

    for i in range(window, len(close)):
        # 跳过NaN值
        if pd.isna(upperband.iloc[i]) or pd.isna(middleband.iloc[i]) or pd.isna(lowerband.iloc[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if close.iloc[i] < middleband.iloc[i] and close.iloc[i - 1] >= middleband.iloc[i - 1]: # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if close.iloc[i] > middleband.iloc[i] and close.iloc[i - 1] <= middleband.iloc[i - 1]: # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑（仅在无仓位时）
        if current_position == 0:
            if close.iloc[i] > upperband.iloc[i]: # 开空
                signal.iloc[i] = -1
                current_position = -1
            elif close.iloc[i] < lowerband.iloc[i]: # 开多
                signal.iloc[i] = 1
                current_position = 1

    return signal

# ###################### 动量 ######################
#动量型布林带信号：最新价突破布林带上界发出做多信号，突破下界发出做空信号，最新价由上到下突破中轨，则平多；最新价由下到上突破中轨，则平空
def bollinger_mom(price, window, num_std_upper, num_std_lower,shift_for_exec = 0):
    """
    动量布林带信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, 布林带移动平均窗口
    :param num_std_upper: float, 上轨的标准差倍数
    :param num_std_lower: float, 下轨的标准差倍数
    :return: pandas Series, 信号值（1: 开多, -1: 开空, 2: 平多, -2: 平空, 0: 无操作）
    """
    close = price["close"].astype(float)
    upperband, middleband, lowerband = talib.BBANDS(
        close,
        timeperiod=window,
        nbdevup=num_std_upper,
        nbdevdn=num_std_lower,
        matype=0
    )
    upperband = upperband.shift(shift_for_exec)
    middleband = middleband.shift(shift_for_exec)
    lowerband = lowerband.shift(shift_for_exec)

    signal = pd.Series(0, index=close.index)
    current_position = 0

    for i in range(window, len(close)):
        # 跳过NaN值
        if pd.isna(upperband.iloc[i]) or pd.isna(middleband.iloc[i]) or pd.isna(lowerband.iloc[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if close.iloc[i] > middleband.iloc[i] and close.iloc[i - 1] <= middleband.iloc[i - 1]: # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if close.iloc[i] < middleband.iloc[i] and close.iloc[i - 1] >= middleband.iloc[i - 1]: # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑（仅在无仓位时）
        if current_position == 0:
            if close.iloc[i] > upperband.iloc[i]: # 开多
                signal.iloc[i] = 1
                current_position = 1
            elif close.iloc[i] < lowerband.iloc[i]: # 开空
                signal.iloc[i] = -1
                current_position = -1

    return signal

#动量型 ROC 信号：过去 x 天上涨则发出做多信号，过去 x 天下跌则发出做空信号
def roc_mom(price, window):
    """
    ROC 动量信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, ROC 计算的回看周期（过去 x 天）
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 无操作）
    """
    close = price["close"]
    roc = talib.ROC(close, timeperiod=window)
    signal = pd.Series(0, index=close.index)

    current_position = 0  # 追踪当前仓位

    for i in range(window, len(close)):
        # 检测反向信号强制平仓
        if current_position == 1 and roc.iloc[i] <= 0:
            signal.iloc[i] = 0
            current_position = 0
        elif current_position == -1 and roc.iloc[i] >= 0:
            signal.iloc[i] = 0
            current_position = 0

        # 开新仓逻辑
        if roc[i] > 0 and current_position != 1:
            signal.iloc[i] = 1
            current_position = 1
        elif roc[i] < 0 and current_position != -1:
            signal.iloc[i] = -1
            current_position = -1
        else:
            # 仓位不变的情况（传递前值）
            signal.iloc[i] = signal.iloc[i - 1] if i > window else 0

    return signal

#动量型连续上涨天数信号：连续上涨 x 天则发出做多信号，连续下跌 x 天则发出做空信号，过去 x 天涨跌天数相同则平仓
def continuous_signal(price, window):
    """
    计算连续上涨天数的信号
    :param price: pandas Series, 股票收盘价
    :param window: int, 回看窗口大小
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 平仓）
    """
    # 计算每天的涨跌方向 (1 表示上涨, -1 表示下跌)
    close = price["close"]
    daily_change = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算过去 x 天内的上涨天数和下跌天数
    up_days = daily_change.rolling(window).apply(lambda x: sum(x > 0), raw=True)
    down_days = daily_change.rolling(window).apply(lambda x: sum(x < 0), raw=True)

    # 初始化信号
    signal = pd.Series(0, index=close.index)
    current_position = 0
    # 信号生成
    for i in range(window, len(close)):
        # 提取第 i 天的 up_days 和 down_days 值
        up_days_i = up_days.iloc[i]
        down_days_i = down_days.iloc[i]

        # 如果 up_days 或 down_days 是 NaN，跳过
        if pd.isna(up_days_i) or pd.isna(down_days_i):
            continue

        # 平仓逻辑
        if current_position == 1:
            if up_days_i == down_days_i:  # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == -1:
            if down_days_i == up_days_i:  # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑
        if current_position == 0 and up_days_i == window:  # 开多
            signal.iloc[i] = 1
            current_position = 1
        elif current_position == 0 and down_days_i == window:  # 开空
            signal.iloc[i] = -1
            current_position = -1

    return signal

# ###################### 均线 ######################

def calculate_ma_talib(close, window, ma_type):
    """支持多种均线类型的统一接口"""
    ma_type = ma_type.upper()

    calc_map = {
        "DOUBLEMA": lambda: talib.DEMA(close, window),
        "WMA": lambda: talib.WMA(close, window),
        "EXPWMA": lambda: talib.EMA(close, window),
        "KAUFMAN": lambda: talib.KAMA(close, window),
        "MIDPOINT": lambda: talib.MIDPOINT(close, window)
    }

    if ma_type not in calc_map:
        raise ValueError(f"不支持的均线类型: {ma_type}")

    return calc_map[ma_type]()

#均线型若干指标的信号：短均线上穿长均线（或下穿）则发出做多信号（或做空信号）
def generate_ma_signal(price, short_window, long_window, ma_type, fastlimit=0.1, slowlimit=0.6, vfactor=1):
    """
    改进版均线交叉信号生成
    :param price: pandas DataFrame, 价格序列
    :param short_window: int, 短期参数(对于Hilbert变换表示偏移周期)
    :param long_window: int, 长期参数(仅传统均线使用)
    :param ma_type: str, 均线类型 ("DoubleMA", "WMA", "EXPWMA", "Hilbert_Transform",
                "Kaufman", "MESA_Adaptive", "MidPoint", "TRIX")
    """
    # Hilbert变换特殊处理
    if short_window > long_window:
        return pd.Series(0, index=price.index)
    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)
    if ma_type == "Hilbert_Transform":
        short_ma = talib.HT_TRENDLINE(close)
        long_ma = short_ma.shift(short_window)  # 使用short_window作为偏移量
    elif ma_type == "MESA_Adaptive":
        short_ma, long_ma = talib.MAMA(close, fastlimit, slowlimit)
    elif ma_type == "TRIX":
        short_ma = talib.T3(close, short_window, vfactor)
        long_ma = talib.T3(close, long_window, vfactor)
    elif ma_type == "MIDPRICE":
        short_ma = talib.MIDPRICE(high, low, short_window)
        long_ma = talib.MIDPRICE(high, low, long_window)
    else:
        short_ma = calculate_ma_talib(close, short_window, ma_type)
        long_ma = calculate_ma_talib(close, long_window, ma_type)

    # 信号生成逻辑
    signal = pd.Series(0, index=close.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓
    valid = short_ma.notna() & long_ma.notna()

    for i in range(1, len(close)):  # 从1开始比较交叉
        if valid.iloc[i] and valid.iloc[i - 1]:
            prev_short, curr_short = short_ma.iloc[i - 1], short_ma.iloc[i]
            prev_long, curr_long = long_ma.iloc[i - 1], long_ma.iloc[i]

            # 精确检测交叉
            cross_up = (curr_short > curr_long) and (prev_short <= prev_long)
            cross_down = (curr_short < curr_long) and (prev_short >= prev_long)

            if current_position == 0:
                if cross_up:
                    current_position = 1
                elif cross_down:
                    current_position = -1
            else:
                if current_position == 1 and cross_down:
                    current_position = -1
                elif current_position == -1 and cross_up:
                    current_position = 1

        # 继承前一日仓位（若当日无操作）
        signal.iloc[i] = current_position

    return signal

#均线型 MACD 均线信号：DIF、DEA、MACD 均为正则做多，DIF、DEA、MACD 均为负则做空
def macd_signal(price, short_window=12, long_window=26, signalperiod=9):
    """
    MACD三线策略信号生成
    :param price: 价格序列
    :param short_window: 快线周期(默认12)
    :param long_window: 慢线周期(默认26)
    :param signalperiod: 信号周期(默认9)
    :return: 交易信号（1: 做多, -1: 做空, 0: 无操作）
    """
    # 计算MACD指标
    if short_window > long_window:
        return pd.Series(0, index=price.index)
    close = price["close"]
    dif, dea, macd = talib.MACD(close,
                                fastperiod=short_window,
                                slowperiod=long_window,
                                signalperiod=signalperiod)

    signal = pd.Series(0, index=close.index)
    current_position = 0  # 0: 空仓，1: 多仓，-1:空仓
    valid = dif.notna() & dea.notna() & macd.notna()

    for i in range(len(close)):
        if valid.iloc[i]:
            dif_val = dif.iloc[i]
            dea_val = dea.iloc[i]
            macd_val = macd.iloc[i]

            long_cond = (dif_val > 0) and (dea_val > 0) and (macd_val > 0)
            short_cond = (dif_val < 0) and (dea_val < 0) and (macd_val < 0)

            if current_position == 0:
                if long_cond:
                    current_position = 1
                elif short_cond:
                    current_position = -1
            else:
                if current_position == 1 and short_cond:
                    current_position = -1
                elif current_position == -1 and long_cond:
                    current_position = 1

        signal.iloc[i] = current_position

    return signal


# ###################### 反转 ######################
#反转型 ROC 信号：过去x天上涨则发出做空信号，过去x天下跌则发出做多信号
def roc_r(price, window):
    """
    计算 ROC 指标的信号，过去x天上涨则发出做空信号，过去x天下跌则发出做多信号
    :param price: pandas Series, 股票收盘价
    :param window: int, 回看窗口大小
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 无操作）
    """
    # 使用TA-Lib计算ROC
    close = price["close"]
    roc = talib.ROC(close, timeperiod=window)
    signal = pd.Series(0, index=close.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(close)):
        if pd.notna(roc.iloc[i]):
            if current_position == 0:
                # 无持仓时检测开仓信号
                if roc.iloc[i] > 0:
                    current_position = -1  # ROC上涨做空
                elif roc.iloc[i] < 0:
                    current_position = 1  # ROC下跌做多
            else:
                # 持仓时检测平仓信号
                if (current_position == 1 and roc.iloc[i] > 0) or \
                        (current_position == -1 and roc.iloc[i] < 0):
                    current_position = 0  # 遇反向信号平仓

        # 记录当日最终仓位
        signal.iloc[i] = current_position

    return signal

def mom_r(price,period,threshold):
    """
    反转型动量信号：过去 period 天涨幅超过 threshold% 做空，
    跌幅超过 threshold% 做多，其它情况保持空仓（0）。

    :param price: pandas Series，通常是收盘价序列
    :param period: 回看窗口（天数）
    :param threshold: 百分比阈值（如 5 表示 5%）
    :return: pandas Series，信号（1: 开多, -1: 开空, 0: 无操作）
    """
    # 计算 period 天前的价格
    close = price["close"]
    prev = close.shift(period)
    # 计算百分比变化
    pct_chg = (close / prev - 1) * 100

    # 根据阈值生成信号
    signal = pd.Series(0, index=close.index)
    signal[pct_chg >  threshold] = -1  # 反转做空
    signal[pct_chg < -threshold] =  1  # 反转做多

    return signal
#反转型 RSI 信号：过去 x 天 RSI 值突破上限则发出做空信号，过去 x 天 RSI 值突破下限则发出做多信号，RSI 值重新回到中轨，则平仓。
def rsi_r(price, window=14, lower=30, middle=50):
    """
    RSI逆向策略信号生成
    :param price: 价格序列
    :param window: RSI计算周期(默认14)
    :param lower: 超卖阈值(默认30)
    :param middle: 平仓阈值(默认50)
    :return: 交易信号（1: 做多, -1: 做空, 0: 无操作）
    """
    upper = 100-lower
    close = price["close"]
    # 计算RSI指标
    rsi = talib.RSI(close, timeperiod=window)

    # 初始化信号序列
    signal = pd.Series(0, index=close.index)
    position = 0  # 持仓状态 0:无仓 1:多仓 -1:空仓

    for i in range(window, len(close)):
        if pd.isna(rsi.iloc[i]):
            continue
        if position == 1:
            if rsi.iloc[i] >= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        elif position == -1:
            if rsi.iloc[i] <= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position

        # 开仓逻辑
        if position == 0:
            if rsi.iloc[i] > upper:
                signal.iloc[i] = -1
                position = -1
            elif rsi.iloc[i] < lower:
                signal.iloc[i] = 1
                position = 1

    return signal

#反转型 CMO 信号：过去 x 天 CMO 值突破上限则发出做空信号，过去 x 天 CMO 值突破下限则发出做多信号，CMO 值重回中轨则平仓。
def cmo_r(price, window=14, upper=50, middle=0):
    """
    CMO逆向策略信号生成
    :param price: 价格序列
    :param window: CMO计算周期(默认14)
    :param upper: 超买阈值(默认50)
    :param middle: 平仓阈值(默认0)
    :return: 交易信号（1: 做多, -1: 做空, 0: 无操作）
    """
    # 计算CMO指标
    lower = -upper
    close = price["close"]
    cmo = talib.CMO(close, timeperiod=window)

    # 初始化信号序列
    signal = pd.Series(0, index=close.index)
    position = 0  # 持仓状态 0:无仓 1:多仓 -1:空仓

    for i in range(window, len(close)):
        if pd.isna(cmo.iloc[i]):
            continue
        if position == 1:
            if cmo.iloc[i] >= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        elif position == -1:
            if cmo.iloc[i] <= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        # 开仓逻辑
        if position == 0:
            if cmo.iloc[i] > upper:
                signal.iloc[i] = -1
                position = -1
            elif cmo.iloc[i] < lower:
                signal.iloc[i] = 1
                position = 1

    return signal

#反转型 连续上涨天数 信号：连续上涨 x 天则发出做空信号，连续下跌 x 天则发出做多信号，过去 x 天涨跌天数相同则平仓。
def continuous_r(price, window):
    """
    计算连续上涨天数的信号
    :param price: pandas Series, 股票收盘价
    :param window: int, 回看窗口大小
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 平仓）
    """
    # 计算每天的涨跌方向 (1 表示上涨, -1 表示下跌)
    close = price["close"]
    daily_change = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算过去 x 天内的上涨天数和下跌天数
    up_days = daily_change.rolling(window).apply(lambda x: sum(x > 0), raw=True)
    down_days = daily_change.rolling(window).apply(lambda x: sum(x < 0), raw=True)

    # 初始化信号
    signal = pd.Series(0, index=close.index)

    # 信号生成
    current_position = 0
    for i in range(window, len(close)):
        # 提取第 i 天的 up_days 和 down_days 值
        up_days_i = up_days.iloc[i]
        down_days_i = down_days.iloc[i]

        # 如果 up_days 或 down_days 是 NaN，跳过
        if pd.isna(up_days_i) or pd.isna(down_days_i):
            continue

        # 平仓逻辑
        if current_position == -1:  # 当前持有空仓
            if up_days_i == down_days_i:  # 达到平仓条件
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:  # 当前持有多仓
            if up_days_i == down_days_i:  # 达到平仓条件
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑
        if current_position == 0:  # 当前无仓位
            if up_days_i == window:  # 开空
                signal.iloc[i] = -1
                current_position = -1
            elif down_days_i == window:  # 开多
                signal.iloc[i] = 1
                current_position = 1

    return signal

#反转型 百分位信号：最新价百分位达到历史高点则做空，最新价百分位达到历史低点则做多。
def quantile_signal(price, window, shift_for_exec = 1):
    """
    百分位信号生成 (Quantile Signal)
    :param price: pandas Series, 股票收盘价
    :param window: int, 滑动窗口大小，用于计算历史高点和低点的百分位
    :param shift_for_exec: int, 用于计算百分位的偏移量
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 无信号）
    """
    close = price["close"]
    rolling_max = close.rolling(window).max().shift(shift_for_exec)
    rolling_min = close.rolling(window).min().shift(shift_for_exec)
    quantile = (close - rolling_min) / (rolling_max - rolling_min + 1e-8)  # 防零除

    signal = pd.Series(0, index=close.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(close)):
        if pd.notna(quantile.iloc[i]):
            if current_position == 0:
                # 空仓状态下检测开仓信号
                if quantile.iloc[i] >= 1:
                    current_position = -1
                elif quantile.iloc[i] <= 0:
                    current_position = 1
            else:
                # 持仓状态下检测平仓信号
                if (current_position == 1 and quantile.iloc[i] >= 1) or \
                        (current_position == -1 and quantile.iloc[i] <= 0):
                    current_position = 0

        # 记录当日最终仓位
        signal.iloc[i] = current_position

    return signal

#==========补充信号==========
#布林带+ATR 震荡（有中轴板）
def bollinger_atr_mom(price:pd.DataFrame,
                    window:int = 20,
                    atr_mult_upper:int = 2,
                    atr_mult_lower:int = 2
):
    """
    基于“布林带(中轨=SMA)+ATR带宽”的动量持仓信号：
    - 收盘价 > 上轨：开多（持仓= +1）
    - 收盘价 < 下轨：开空（持仓= -1）
    - 由上到下 跌破 中轨：平多（持仓 -> 0）
    - 由下到上 突破 中轨：平空（持仓 -> 0）

    参数
    ----
    price : 必含列 ["high","low","close"]
    window : 中轨 SMA 的窗口，ATR 的窗口默认与 window 相同
    atr_mult_upper / atr_mult_lower : 上/下轨的 ATR 倍数

    返回
    ----
    pd.Series: 位置序列（1=多，-1=空，0=空仓）
    """
    if not {"high","low","close"}.issubset(price.columns):
        raise ValueError("price 必须包含列：'high','low','close'")
    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)

    middleband = talib.SMA(close, timeperiod=window)
    atr = talib.ATR(high, low, close, timeperiod=window)

    upperband = middleband + atr_mult_upper * atr
    lowerband = middleband - atr_mult_lower * atr

    signal = pd.Series(0, index=close.index,dtype=int)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(close)):
        # 跳过NaN值
        if pd.isna(upperband.iloc[i]) or pd.isna(middleband.iloc[i]) or pd.isna(lowerband.iloc[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if close.iloc[i] > middleband.iloc[i] and close.iloc[i - 1] <= middleband.iloc[i - 1]:  # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if close.iloc[i] < middleband.iloc[i] and close.iloc[i - 1] >= middleband.iloc[i - 1]:  # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑（仅在无仓位时）
        if current_position == 0:
            if close.iloc[i] > upperband.iloc[i]:  # 开多
                signal.iloc[i] = 1
                current_position = 1
            elif close.iloc[i] < lowerband.iloc[i]:  # 开空
                signal.iloc[i] = -1
                current_position = -1

    return signal

#海龟交易法则：最新价超过唐安奇通道上轨后发出做多信号 1，低于唐安奇通道下轨发出做空信号-1
def turtle_trading(price:pd.DataFrame,
                window:int = 20,
                shift_for_exec:int = 1):
    """
    海龟交易法则（基于唐安奇通道）的动量持仓信号：
    - 收盘价 > 上轨：开多（持仓= +1）
    - 收盘价 < 下轨：开空（持仓= -1）
    - 由上到下跌破中轨：平多（持仓 -> 0）
    - 由下到上突破中轨：平空（持仓 -> 0）

    参数
    ----
    price : 必含列 ["high","low","close"]
    window : 唐安奇通道的回溯窗口
    shift_for_exec : 执行信号的回溯窗口

    返回
    ----
    pd.Series: 位置序列（1=多，-1=空，0=空仓）
    """
    if not {"high", "low", "close"}.issubset(price.columns):
        raise ValueError("price 必须包含列：'high','low','close'")

    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)

    # 唐安奇通道
    upperband = high.rolling(window=window, min_periods=window).max().shift(shift_for_exec)
    lowerband = low.rolling(window=window, min_periods=window).min().shift(shift_for_exec)
    middleband = (upperband + lowerband) / 2

    signal = pd.Series(0, index=close.index, dtype=int)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(close)):
        if pd.isna(upperband.iloc[i]) or pd.isna(lowerband.iloc[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if close.iloc[i] > middleband.iloc[i] and close.iloc[i - 1] <= middleband.iloc[i - 1]:
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if close.iloc[i] < middleband.iloc[i] and close.iloc[i - 1] >= middleband.iloc[i - 1]:
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑
        if current_position == 0:
            if close.iloc[i] > upperband.iloc[i]:
                signal.iloc[i] = 1
                current_position = 1
            elif close.iloc[i] < lowerband.iloc[i]:
                signal.iloc[i] = -1
                current_position = -1

    return signal

#抛物线策略：最新价上穿抛物线发出做多信号 1，最新价下穿抛物线发出做空信号-1
def sar(price:pd.DataFrame,
        acceleration:float = 0.2):
    """
    抛物线 SAR 策略信号（事件信号）
    - 最新收盘价上穿 SAR → 做多信号 +1
    - 最新收盘价下穿 SAR → 做空信号 -1

    参数
    ----
    price : 必含列 ["high","low","close"]
    acceleration : 加速因子步长（AF step），常见 0.02

    返回
    ----
    pd.Series: 事件信号
    """
    required = {"high","low","close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    high  = price["high"].astype(float)
    low   = price["low"].astype(float)
    close = price["close"].astype(float)

    sar = talib.SAR(high, low, acceleration=acceleration, maximum=0.2)

    # 生成持仓信号
    signal = pd.Series(0, index=close.index, dtype=int)
    position = 0  # 当前持仓状态：0 无仓, 1 多, -1 空

    for i in range(len(close)):
        if pd.isna(sar.iloc[i]):
            signal.iloc[i] = position
            continue

        if close.iloc[i] > sar.iloc[i]:
            position = 1
        elif close.iloc[i] < sar.iloc[i]:
            position = -1

        signal.iloc[i] = position

    return signal
#日内动量策略：当日内动量大于 threshold 的时候做多，反之做空
def intramom(price:pd.DataFrame,
            window:int = 20,
            threshold:float = 1):
    """
    日内动量策略（返回持仓信号）
    - 指标: IM = ((high + low)/2) / open
    - 信号: MA(IM, timeperiod) > threshold → 多(+1)，否则空(-1)

    参数
    ----
    price : 必含列 ["open","high","low","close"]
    timeperiod : int, 移动平均窗口
    threshold : float, 阈值，默认 0.03

    返回
    ----
    pd.Series: 持仓信号（+1/-1，窗口未满为0）
    """
    required = {"high","low","close","open"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    open_p = price["open"].astype(float)
    high  = price["high"].astype(float)
    low   = price["low"].astype(float)
    close = price["close"].astype(float)

    mom = ((high + low) / 2) / open_p
    mom_ma = mom.rolling(window=window, min_periods=window).mean()

    # 生成持仓信号
    signal = pd.Series(0, index=close.index, dtype=int)
    position = 0  # 当前持仓状态：0 无仓, 1 多, -1 空

    for i in range(len(close)):
        if pd.isna(mom_ma.iloc[i]):
            signal.iloc[i] = position
            continue

        if mom_ma.iloc[i] > threshold:
            position = 1
        elif mom_ma.iloc[i] < threshold:
            position = -1

        signal.iloc[i] = position

    return signal

#日内震幅动量
def stds(price: pd.DataFrame,
                    short_window: int = 10,
                    long_window: int = 30,
                    method: str = "signed_range",   # "signed_range" | "stoch_pos" | "combo"
                    norm_window: int = 20           # combo 用于 H 的标准化窗口
                    ) -> pd.Series:
    """
    日内震幅动量（STDS）双均线信号：
    1) 先由 (H,L,C) 构造 STDS 序列
    2) 对 STDS 做短/长移动平均
    3) 短上穿长 → 多(+1)，下穿 → 空(-1)，其他保持最近持仓；窗口未满为 0

    参数
    ----
    price : 必含列 ["open","high","low","close"]
    short_window, long_window : STDS 的短/长均线窗口
    method : STDS 构造方式
        - "signed_range": STDS = sign(C) * H
        - "stoch_pos"   : STDS = 2 * ((logC - logL) / H) - 1
        - "combo"       : 0.5* standardized(sign(C)*H) + 0.5*stoch_pos
    norm_window : combo 中 H 的标准化窗口

    返回
    ----
    pd.Series: 持仓信号（+1/-1/0）
    """
    required = {"open","high","low","close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    if short_window > long_window:
        return pd.Series(0, index=price.index)

    o = price["open"].astype(float)
    h = price["high"].astype(float)
    l = price["low"].astype(float)
    c = price["close"].astype(float)

    # 对数价，避免非正值导致的 log 报错
    eps = 1e-12
    log_o = np.log(np.clip(o, eps, None))
    log_h = np.log(np.clip(h, eps, None))
    log_l = np.log(np.clip(l, eps, None))
    log_c = np.log(np.clip(c, eps, None))

    H = (log_h - log_l)            # 当日对数振幅
    L = (log_l - log_o)            # 开盘->最低
    C = (log_c - log_o)            # 开盘->收盘

    method = method.lower()
    if method == "signed_range":
        stds = np.sign(C) * H
    elif method == "stoch_pos":
        rng = H.replace(0.0, np.nan)
        pos = (log_c - log_l) / rng          # [0,1]
        stds = 2.0 * pos - 1.0               # [-1,1]
    elif method == "combo":
        # 组件1：带方向的区间强度（做标准化）
        z = (np.sign(C) * H)
        z_std = z.rolling(norm_window, min_periods=norm_window).std()
        comp1 = z / (z_std.replace(0.0, np.nan))
        # 组件2：收盘位置分数
        rng = H.replace(0.0, np.nan)
        pos = (log_c - log_l) / rng
        comp2 = 2.0 * pos - 1.0
        stds = 0.5 * comp1 + 0.5 * comp2
    else:
        raise ValueError("method must be one of: 'signed_range', 'stoch_pos', 'combo'")

    # 对 STDS 做短/长均线
    short = stds.rolling(short_window, min_periods=short_window).mean()
    long  = stds.rolling(long_window,  min_periods=long_window).mean()

    # 生成持仓：金叉→+1，死叉→-1；其余沿用；窗口未满为 0
    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0

    for i in range(len(signal)):
        if pd.isna(short.iloc[i]) or pd.isna(long.iloc[i]):
            signal.iloc[i] = position
            continue

        if i > 0:
            cross_up   = (short.iloc[i] >  long.iloc[i]) and (short.iloc[i-1] <= long.iloc[i-1])
            cross_down = (short.iloc[i] <  long.iloc[i]) and (short.iloc[i-1] >= long.iloc[i-1])

            if cross_up:
                position = 1
            elif cross_down:
                position = -1

        signal.iloc[i] = position

    return signal

#顺势指标：当 CCI 大于 threshold，看空 (-1);小于 -threshold，看多 (+1);其他情况，保持上一次持仓
def cci(price:pd.DataFrame,
        window:int = 14,
        threshold:float = 100):
    """
    基于 CCI 的顺势/反转交易信号（返回持仓信号）
    - CCI > threshold → 看空 (-1)
    - CCI < -threshold → 看多 (+1)
    - 其他情况 → 保持上一次仓位

    参数
    ----
    price : DataFrame, 必含 ["high","low","close"]
    timeperiod : int, CCI 计算窗口
    threshold : float, 阈值（常用 100）

    返回
    ----
    pd.Series: 持仓信号（+1/-1，窗口未满为0）
    """
    required = {"high", "low", "close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)

    # 计算 CCI
    cci_val = talib.CCI(high, low, close, timeperiod=window)

    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0  # 当前持仓状态：0 无仓，1 多，-1 空

    for i in range(len(close)):
        if pd.isna(cci_val.iloc[i]):
            signal.iloc[i] = position
            continue

        if cci_val.iloc[i] > threshold:
            position = -1  # 看空
        elif cci_val.iloc[i] < -threshold:
            position = 1   # 看多

        signal.iloc[i] = position

    return signal

#随机指标：当 KDJ 大于100-threshold，发出看空信号；当 KDJ 小于 threshold，发出看多信号
def kdj(price: pd.DataFrame,
        fastk_period: int = 9,
        fastd_period: int = 3,
        threshold: float = 20.0,
        line: str = "k",          # "k" | "d" | "j" | "mean"（K/D均值）
        shift_for_exec: int = 0   # 0=当日收盘执行；1=次日开盘执行（防前视）
       ) -> pd.Series:
    """
    随机指标(KDJ)反转信号（返回持仓信号）
    - 指标值 > 100 - threshold → 看空(-1)
    - 指标值 < threshold       → 看多(+1)
    - 其他保持上一次仓位；窗口未满为 0

    参数
    ----
    price : 必含列 ["high","low","close"]
    fastk_period : %K 的周期
    fastd_period : %D 的平滑周期
    threshold : 阈值（常用 20；超买/超卖区为 100-20=80 与 20）
    line : 用哪条线判定：K / D / J(=3K-2D) / mean(K,D)
    shift_for_exec : 若=1，则把用于判定的序列右移1根，适配次日开盘执行
    """
    required = {"high", "low", "close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)

    # talib STOCHF 返回 fastk(%K) 与 fastd(%D)，范围通常 0~100
    k, d = talib.STOCHF(high, low, close,
                        fastk_period=fastk_period,
                        fastd_period=fastd_period,
                        fastd_matype=0)

    line = line.lower()
    if line == "k":
        ind = k
    elif line == "d":
        ind = d
    elif line == "j":
        ind = 3 * k - 2 * d
    elif line == "mean":
        ind = (k + d) / 2.0
    else:
        raise ValueError("line 必须是 'k' | 'd' | 'j' | 'mean' 之一")

    # 执行对齐：次日开盘执行时右移，避免用到将要交易这根的当日信息
    if shift_for_exec:
        ind = ind.shift(shift_for_exec)

    upper = 100.0 - threshold
    lower = threshold

    signal = pd.Series(0, index=close.index, dtype=int)
    position = 0  # 0=空仓, 1=多, -1=空

    for i in range(len(close)):
        val = ind.iloc[i]
        if pd.isna(val):
            signal.iloc[i] = position
            continue

        if val > upper:       # 超买区 → 反转看空
            position = -1
        elif val < lower:     # 超卖区 → 反转看多
            position = 1
        # else: 保持原持仓

        signal.iloc[i] = position

    return signal

#终极震荡指标
def ultosc_contrarian(price: pd.DataFrame,
                      t1: int = 7,
                      t2: int = 14,
                      t3: int = 28,
                      threshold: float = 30.0,
                      shift_for_exec: int = 0) -> pd.Series:
    """
    终极震荡指标 ULTOSC 反转信号（返回持仓信号）
    - ULTOSC > 100 - threshold → 看空 (-1)
    - ULTOSC < threshold       → 看多 (+1)
    - 其余保持上一持仓；窗口未满为 0

    参数
    ----
    price : 必含列 ["high","low","close"]
    t1,t2,t3 : ULTOSC 的三个周期（常用 7,14,28）
    threshold : 阈值（常用 30 → 超买70/超卖30）
    shift_for_exec : 执行对齐（0=当日收盘执行；1=次日开盘执行，防前视）

    返回
    ----
    pd.Series: 持仓信号（+1/-1/0）
    """
    required = {"high","low","close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    high  = price["high"].astype(float)
    low   = price["low"].astype(float)
    close = price["close"].astype(float)

    # 计算 ULTOSC（范围通常 0~100）
    ult = talib.ULTOSC(high, low, close, timeperiod1=t1, timeperiod2=t2, timeperiod3=t3)

    # 次日开盘执行时，右移一根以避免用到将要交易这根的数据
    if shift_for_exec:
        ult = ult.shift(shift_for_exec)

    upper = 100.0 - threshold
    lower = threshold

    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0  # 0=空仓, 1=多, -1=空

    for i in range(len(close)):
        val = ult.iloc[i]
        if pd.isna(val):
            signal.iloc[i] = position
            continue

        if val > upper:       # 超买 → 反转看空
            position = -1
        elif val < lower:     # 超卖 → 反转看多
            position = 1
        # 否则保持原持仓

        signal.iloc[i] = position

    return signal

#威廉指数
def williams_r_contrarian(price: pd.DataFrame,
                          timeperiod: int = 14,
                          threshold: float = 20.0,
                          shift_for_exec: int = 0) -> pd.Series:
    """
    威廉指标 Williams %R 反转信号（返回持仓信号）
    - WILLR > 100 - threshold → 看空 (-1)
    - WILLR < threshold       → 看多 (+1)
    - 其余保持上一持仓；窗口未满为 0

    参数
    ----
    price : 必含列 ["high","low","close"]
    timeperiod : int，威廉指标周期
    threshold : float，阈值（常用 20 → 对应超买 -20 / 超卖 -80）
    shift_for_exec : 执行对齐（0=当日收盘执行，1=次日开盘执行，防前视）

    返回
    ----
    pd.Series: 持仓信号（+1/-1/0）
    """
    required = {"high", "low", "close"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)

    # 计算 Williams %R（范围 -100 ~ 0）
    willr = talib.WILLR(high, low, close, timeperiod=timeperiod)

    # 如果次日执行，右移一根，避免用到未来数据
    if shift_for_exec:
        willr = willr.shift(shift_for_exec)

    # 将 %R 转为 0 ~ 100 区间（方便和 threshold 规则统一）
    willr_pos = 100 + willr

    upper = 100.0 - threshold
    lower = threshold

    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0

    for i in range(len(close)):
        val = willr_pos.iloc[i]
        if pd.isna(val):
            signal.iloc[i] = position
            continue

        if val > upper:       # 超买区 → 反转做空
            position = -1
        elif val < lower:     # 超卖区 → 反转做多
            position = 1
        # 其他保持原仓位

        signal.iloc[i] = position

    return signal
#嘉庆指标
def chaikin(price: pd.DataFrame,
                       fastperiod: int = 3,
                       slowperiod: int = 10,
                       volume_method: str = "sub") -> pd.Series:
    """
    基于 Chaikin Oscillator (ADOSC) 的交易信号：
    - ADOSC > 0 → 做多 (+1)
    - ADOSC < 0 → 做空 (-1)
    - ADOSC = 0 → 保持原仓位

    参数
    ----
    price : 必含列 ["high","low","close","volume"]
    fastperiod : int，快线周期
    slowperiod : int，慢线周期
    shift_for_exec : 执行对齐（0=当日收盘执行，1=次日执行，防前视）
    volume_method : str，成交量处理方式
        - sub: 相减
        - abs: 相减后取绝对值
        - max: 取最大值
        - min: 取最小值

    返回
    ----
    pd.Series: 持仓信号（+1/-1/0）
    """
    required = {"high", "low", "close",f"volume_{volume_method}"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    if fastperiod > slowperiod:
        return pd.Series(0, index=price.index)

    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)
    volume = price[f"volume_{volume_method}"].astype(float)

    adosc = talib.ADOSC(high, low, close, volume,
                        fastperiod=fastperiod,
                        slowperiod=slowperiod)

    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0

    for i in range(len(close)):
        val = adosc.iloc[i]
        if pd.isna(val):
            signal.iloc[i] = position
            continue

        if val > 0:
            position = 1
        elif val < 0:
            position = -1
        # 0 值保持原仓位

        signal.iloc[i] = position

    return signal

#成交量加权均线
def vwma_signal(price: pd.DataFrame,
                timeperiod: int = 20,
                volume_method: str = "sub",
                shift_for_exec: int = 0) -> pd.Series:
    """
    基于成交量加权移动均线（VWMA）的交易信号：
    - 收盘价上穿 VWMA → 做多 (+1)
    - 收盘价下穿 VWMA → 做空 (-1)
    - 其余保持原仓位

    参数
    ----
    price : 必含列 ["close", "volume"]
    timeperiod : int，均线窗口
    shift_for_exec : 执行对齐（0=当日收盘执行，1=次日执行）

    返回
    ----
    pd.Series: 持仓信号（+1/-1/0）
    """
    required = {"close", f"volume_{volume_method}"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")

    close = price["close"].astype(float)
    volume = price[f"volume_{volume_method}"].astype(float)

    # 计算成交量加权移动均线
    vwma = (close * volume).rolling(window=timeperiod).sum() / volume.rolling(window=timeperiod).sum()

    if shift_for_exec:
        vwma = vwma.shift(shift_for_exec)

    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0

    for i in range(1, len(close)):
        if pd.isna(vwma.iloc[i]) or pd.isna(vwma.iloc[i - 1]):
            signal.iloc[i] = position
            continue

        # 上穿均线 → 开多
        if close.iloc[i] > vwma.iloc[i] and close.iloc[i - 1] <= vwma.iloc[i - 1]:
            position = 1
        # 下穿均线 → 开空
        elif close.iloc[i] < vwma.iloc[i] and close.iloc[i - 1] >= vwma.iloc[i - 1]:
            position = -1

        signal.iloc[i] = position

    return signal

#能量潮指标
def obv_ma_signal(price: pd.DataFrame,
                  short_window: int = 10,
                  long_window: int = 30,
                  ma: str = "SMA",            # "SMA" 或 "EMA"
                  volume_method: str = "sub"
                 ) -> pd.Series:
    """
    能量潮(OBV)均线交叉信号（返回持仓信号）
    - 计算：OBV = talib.OBV(close, volume)
    - 短均线上穿长均线 → 做多(+1)
    - 短均线下穿长均线 → 做空(-1)
    - 其他保持上一持仓；窗口未满为 0
    """
    required = {"close", f"volume_{volume_method}"}
    if not required.issubset(price.columns):
        raise ValueError(f"price 必须包含列：{required}")
    if not price.index.is_monotonic_increasing:
        price = price.sort_index()
    if short_window > long_window:
        raise ValueError("短均线窗口必须小于等于长均线窗口")

    close  = price["close"].astype(float)
    volume = price[f"volume_{volume_method}"].astype(float)

    # 1) 计算 OBV（累积的量价指标）
    obv = talib.OBV(close, volume).astype(float)

    # 2) 对 OBV 做短/长均线
    ma_up = ma.upper()
    if ma_up == "EMA":
        short = obv.ewm(span=short_window, adjust=False).mean()
        long  = obv.ewm(span=long_window,  adjust=False).mean()
    else:  # SMA
        short = obv.rolling(short_window, min_periods=short_window).mean()
        long  = obv.rolling(long_window,  min_periods=long_window).mean()

    # 3) 生成持仓信号：精确检测金叉/死叉，其他沿用
    signal = pd.Series(0, index=price.index, dtype=int)
    position = 0
    for i in range(1, len(signal)):
        if pd.isna(short.iloc[i]) or pd.isna(long.iloc[i]) or pd.isna(short.iloc[i-1]) or pd.isna(long.iloc[i-1]):
            signal.iloc[i] = position
            continue

        cross_up   = (short.iloc[i] >  long.iloc[i]) and (short.iloc[i-1] <= long.iloc[i-1])
        cross_down = (short.iloc[i] <  long.iloc[i]) and (short.iloc[i-1] >= long.iloc[i-1])

        if cross_up:
            position = 1
        elif cross_down:
            position = -1

        signal.iloc[i] = position

    return signal

#量价相关性
def obv_price_corr(price: pd.DataFrame,
        timeperiod: int = 20,
        volume_method: str = "sub") -> pd.Series:
    """
    量价相关性信号：
    - 先计算 OBV
    - 计算 OBV 与收盘价的滚动 timeperiod 日相关系数
    - 相关系数在 [0.4, 1] → 信号=  1
      相关系数在 (-0.4, 0.4) → 信号= 0
      相关系数在 [-1, -0.4] → 信号= -1

    参数
    ----
    price : pd.DataFrame, 必须包含 "close","volume"
    timeperiod : 滚动相关系数的窗口

    返回
    ----
    pd.Series: 信号（1=多, -1=空, 0=观望）
    """
    if not {"close", f"volume_{volume_method}"}.issubset(price.columns):
        raise ValueError("price 必须包含列：'close',f'volume_{volume_method}'")
    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    close  = price["close"].astype(float)
    volume = price[f"volume_{volume_method}"].astype(float)

    # 1) 计算 OBV
    obv = talib.OBV(close, volume)

    # 2) 滚动相关系数
    corr = close.rolling(timeperiod).corr(obv)

    # 3) 信号映射
    signal = pd.Series(0, index=close.index, dtype=int)
    signal[(corr >= 0.85) & (corr <= 1.0)]  = 1
    signal[(corr <  0.8) & (corr >= -1.0)] = -1

    return signal

#成交量加权价格仅限

#==========月度数据处理===========
# 聚合为月度数据 - 为每个月计算价格涨跌情况

# 聚合为月度数据 - 为每个月计算价格涨跌情况
def aggregate_to_monthly_price_change(df):
    """
    为每个月计算多空组合的净值涨跌情况
    :param df: 多空组合后的净值
    :return: 月度涨跌数据
    """
    # 为日频数据添加年月信息
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # 按年月分组
    monthly_grouped = df.groupby(['year', 'month'])

    monthly_data = []
    for (year, month), group in monthly_grouped:
        # 按日期排序
        group = group.sort_values('date')

        # 计算月度价格变化
        first_price = group['close'].iloc[0]
        last_price = group['close'].iloc[-1]
        monthly_return = (last_price / first_price) - 1

        # 判断价格是涨还是跌
        price_up = monthly_return > 0

        monthly_data.append({
            'year': year,
            'month': month,
            'return': monthly_return,
            'price_up': price_up,  # True表示价格上涨，False表示价格下跌或不变
            'last_date': group['date'].iloc[-1]
        })

    return pd.DataFrame(monthly_data)

# 函数：计算到某年某月为止的历史价格涨跌胜率

def calculate_price_up_win_rate(df, current_year, current_month):
    # 只包含当前月之前的历史数据
    mask = ((df['year'] <= current_year) | ((df['year'] == current_year) & (df['month'] <= current_month)))
    historical_data = df[mask].copy()
    # 按月份分组计算价格上涨的胜率
    monthly_win_rates = {}
    for month in range(1, 13):
        month_data = historical_data[historical_data['month'] == month]
        if len(month_data) > 0:
            wins = month_data['price_up'].sum()
            total = len(month_data)
            win_rate = wins / total if total > 0 else 0.5
            monthly_win_rates[month] = win_rate
        else:
            monthly_win_rates[month] = 0.5  # 如果没有历史数据，默认为50%
    return monthly_win_rates