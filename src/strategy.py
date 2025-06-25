# -*- coding: utf-8 -*-

"""
@Time: 2024/12/25 14:14

@Author: Dayuan Shen

@File: strategy.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from iFinDPy import *
# from WindPy import w
from datetime import datetime
import talib
# from sympy.physics.units import current


###################### 基差 ######################



###################### 通道 ######################
#通道型布林带信号：最新价突破布林带上界发出做空信号，突破下界发出做多信号，最新价由上到下突破中轨，则平空；最新价由下到上突破中轨，则平多
def bollinger_r(price, window, num_std_upper, num_std_lower):
    """
    布林带信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, 布林带移动平均窗口
    :param num_std_upper: float, 上轨的标准差倍数
    :param num_std_lower: float, 下轨的标准差倍数
    :return: pandas Series, 信号值（1: 开多, -1: 开空, 2: 平多, -2: 平空, 0: 无操作）
    """
    upperband, middleband, lowerband = talib.BBANDS(
        price,
        timeperiod=window,
        nbdevup=num_std_upper,
        nbdevdn=num_std_lower,
        matype=0
    )

    signal = pd.Series(0, index=price.index)
    current_position = 0

    for i in range(window, len(price)):
        # 跳过NaN值
        if pd.isna(upperband[i]) or pd.isna(middleband[i]) or pd.isna(lowerband[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if price[i] < middleband[i] and price[i - 1] >= middleband[i - 1]: # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if price[i] > middleband[i] and price[i - 1] <= middleband[i - 1]: # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑（仅在无仓位时）
        if current_position == 0:
            if price[i] > upperband[i]: # 开空
                signal.iloc[i] = -1
                current_position = -1
            elif price[i] < lowerband[i]: # 开多
                signal.iloc[i] = 1
                current_position = 1

    return signal

# ###################### 动量 ######################
#动量型布林带信号：最新价突破布林带上界发出做多信号，突破下界发出做空信号，最新价由上到下突破中轨，则平多；最新价由下到上突破中轨，则平空
def bollinger_MOM(price, window, num_std_upper, num_std_lower):
    """
    动量布林带信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, 布林带移动平均窗口
    :param num_std_upper: float, 上轨的标准差倍数
    :param num_std_lower: float, 下轨的标准差倍数
    :return: pandas Series, 信号值（1: 开多, -1: 开空, 2: 平多, -2: 平空, 0: 无操作）
    """
    upperband, middleband, lowerband = talib.BBANDS(
        price,
        timeperiod=window,
        nbdevup=num_std_upper,
        nbdevdn=num_std_lower,
        matype=0
    )

    signal = pd.Series(0, index=price.index)
    current_position = 0

    for i in range(window, len(price)):
        # 跳过NaN值
        if pd.isna(upperband[i]) or pd.isna(middleband[i]) or pd.isna(lowerband[i]):
            continue

        # 平仓逻辑优先
        if current_position == -1:
            if price[i] > middleband[i] and price[i - 1] <= middleband[i - 1]: # 平空
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position
        elif current_position == 1:
            if price[i] < middleband[i] and price[i - 1] >= middleband[i - 1]: # 平多
                signal.iloc[i] = 0
                current_position = 0
            else:
                signal.iloc[i] = current_position

        # 开仓逻辑（仅在无仓位时）
        if current_position == 0:
            if price[i] > upperband[i]: # 开多
                signal.iloc[i] = 1
                current_position = 1
            elif price[i] < lowerband[i]: # 开空
                signal.iloc[i] = -1
                current_position = -1

    return signal

#动量型 ROC 信号：过去 x 天上涨则发出做多信号，过去 x 天下跌则发出做空信号
def roc_MOM(price, window):
    """
    ROC 动量信号生成
    :param price: pandas Series, 股票收盘价
    :param window: int, ROC 计算的回看周期（过去 x 天）
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 无操作）
    """
    roc = talib.ROC(price, timeperiod=window)
    signal = pd.Series(0, index=price.index)

    current_position = 0  # 追踪当前仓位

    for i in range(window, len(price)):
        # 检测反向信号强制平仓
        if current_position == 1 and roc[i] <= 0:
            signal.iloc[i] = 0
            current_position = 0
        elif current_position == -1 and roc[i] >= 0:
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
    daily_change = price.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算过去 x 天内的上涨天数和下跌天数
    up_days = daily_change.rolling(window).apply(lambda x: sum(x > 0), raw=True)
    down_days = daily_change.rolling(window).apply(lambda x: sum(x < 0), raw=True)

    # 初始化信号
    signal = pd.Series(0, index=price.index)
    current_position = 0
    # 信号生成
    for i in range(window, len(price)):
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

def calculate_ma_talib(price, window, ma_type):
    """支持多种均线类型的统一接口"""
    ma_type = ma_type.upper()

    calc_map = {
        "DOUBLEMA": lambda: talib.DEMA(price, window),
        "WMA": lambda: talib.WMA(price, window),
        "EXPWMA": lambda: talib.EMA(price, window),
        "KAUFMAN": lambda: talib.KAMA(price, window),
        "MIDPOINT": lambda: talib.MIDPOINT(price, window)
    }

    if ma_type not in calc_map:
        raise ValueError(f"不支持的均线类型: {ma_type}")

    return calc_map[ma_type]()

#均线型若干指标的信号：短均线上穿长均线（或下穿）则发出做多信号（或做空信号）
def generate_ma_signal(price, short_window, long_window, ma_type, fastlimit=0.1, slowlimit=0.6, vfactor=1):
    """
    改进版均线交叉信号生成
    :param price: pandas Series, 价格序列
    :param short_window: int, 短期参数(对于Hilbert变换表示偏移周期)
    :param long_window: int, 长期参数(仅传统均线使用)
    :param ma_type: str, 均线类型 ("DoubleMA", "WMA", "EXPWMA", "Hilbert_Transform",
                "Kaufman", "MESA_Adaptive", "MidPoint", "TRIX")
    """
    # Hilbert变换特殊处理
    if ma_type == "Hilbert_Transform":
        short_ma = talib.HT_TRENDLINE(price)
        long_ma = short_ma.shift(short_window)  # 使用short_window作为偏移量
    elif ma_type == "MESA_Adaptive":
        short_ma, long_ma = talib.MAMA(price, fastlimit, slowlimit)
    elif ma_type == "TRIX":
        short_ma = talib.TRIX(price, short_window)
        long_ma = talib.TRIX(price, long_window)
    else:
        short_ma = calculate_ma_talib(price, short_window, ma_type)
        long_ma = calculate_ma_talib(price, long_window, ma_type)

    # 信号生成逻辑
    signal = pd.Series(0, index=price.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓
    valid = short_ma.notna() & long_ma.notna()

    for i in range(1, len(price)):  # 从1开始比较交叉
        if valid[i] and valid[i - 1]:
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
    dif, dea, macd = talib.MACD(price,
                                fastperiod=short_window,
                                slowperiod=long_window,
                                signalperiod=signalperiod)

    signal = pd.Series(0, index=price.index)
    current_position = 0  # 0: 空仓，1: 多仓，-1:空仓
    valid = dif.notna() & dea.notna() & macd.notna()

    for i in range(len(price)):
        if valid[i]:
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
    roc = talib.ROC(price, timeperiod=window)
    signal = pd.Series(0, index=price.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(price)):
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

#反转型 RSI 信号：过去 x 天 RSI 值突破上限则发出做空信号，过去 x 天 RSI 值突破下限则发出做多信号，RSI 值重新回到中轨，则平仓。
def rsi_r(price, window=14, upper=70, lower=30, middle=50):
    """
    RSI逆向策略信号生成
    :param price: 价格序列
    :param window: RSI计算周期(默认14)
    :param upper: 超买阈值(默认70)
    :param lower: 超卖阈值(默认30)
    :param middle: 平仓阈值(默认50)
    :return: 交易信号（1: 做多, -1: 做空, 0: 无操作）
    """
    # 计算RSI指标
    rsi = talib.RSI(price, timeperiod=window)

    # 初始化信号序列
    signal = pd.Series(0, index=price.index)
    position = 0  # 持仓状态 0:无仓 1:多仓 -1:空仓

    for i in range(window, len(price)):
        if pd.isna(rsi[i]):
            continue
        if position == 1:
            if rsi[i] >= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        elif position == -1:
            if rsi[i] <= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position

        # 开仓逻辑
        if position == 0:
            if rsi[i] > upper:
                signal.iloc[i] = -1
                position = -1
            elif rsi[i] < lower:
                signal.iloc[i] = 1
                position = 1


    return signal

#反转型 CMO 信号：过去 x 天 CMO 值突破上限则发出做空信号，过去 x 天 CMO 值突破下限则发出做多信号，CMO 值重回中轨则平仓。
def cmo_r(price, window=14, upper=50, lower=-50, middle=0):
    """
    CMO逆向策略信号生成
    :param price: 价格序列
    :param window: CMO计算周期(默认14)
    :param upper: 超买阈值(默认50)
    :param lower: 超卖阈值(默认-50)
    :param middle: 平仓阈值(默认0)
    :return: 交易信号（1: 做多, -1: 做空, 0: 无操作）
    """
    # 计算CMO指标
    cmo = talib.CMO(price, timeperiod=window)

    # 初始化信号序列
    signal = pd.Series(0, index=price.index)
    position = 0  # 持仓状态 0:无仓 1:多仓 -1:空仓

    for i in range(window, len(price)):
        if pd.isna(cmo[i]):
            continue
        if position == 1:
            if cmo[i] >= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        elif position == -1:
            if cmo[i] <= middle:
                signal.iloc[i] = 0
                position = 0
            else:
                signal.iloc[i] = position
        # 开仓逻辑
        if position == 0:
            if cmo[i] > upper:
                signal.iloc[i] = -1
                position = -1
            elif cmo[i] < lower:
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
    daily_change = price.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算过去 x 天内的上涨天数和下跌天数
    up_days = daily_change.rolling(window).apply(lambda x: sum(x > 0), raw=True)
    down_days = daily_change.rolling(window).apply(lambda x: sum(x < 0), raw=True)

    # 初始化信号
    signal = pd.Series(0, index=price.index)

    # 信号生成
    current_position = 0
    for i in range(window, len(price)):
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
def quantile_signal(price, window):
    """
    百分位信号生成 (Quantile Signal)
    :param price: pandas Series, 股票收盘价
    :param window: int, 滑动窗口大小，用于计算历史高点和低点的百分位
    :return: pandas Series, 信号值（1: 做多, -1: 做空, 0: 无信号）
    """
    rolling_max = price.rolling(window).max()
    rolling_min = price.rolling(window).min()
    quantile = (price - rolling_min) / (rolling_max - rolling_min + 1e-8)  # 防零除

    signal = pd.Series(0, index=price.index)
    current_position = 0  # 0-空仓 1-多仓 -1-空仓

    for i in range(len(price)):
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

######################月份季节性信号######################
# 聚合为月度数据 - 为每个月计算价格涨跌情况
def aggregate_to_monthly_price_change(df):
    """
    为每个月计算多空组合的净值涨跌情况
    :param df: 多空组合后的净值
    :return: 月度涨跌数据
    """
    # 为日频数据添加年月信息
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
    # 筛选历史数据（不包括当前月及之后的数据）
    mask = ((df['year'] <= current_year) & (df['month'] <= current_month))
    historical_data = df[mask].copy()

    # 按月份分组计算价格上涨的胜率
    monthly_win_rates = {}
    for month in range(1, 13):
        month_data = historical_data[historical_data['month'] == month]
        if len(month_data) > 0:
            # 计算该月价格上涨的比例作为胜率
            wins = sum(month_data['price_up'])
            total = len(month_data)
            win_rate = wins / total if total > 0 else 0.5
            monthly_win_rates[month] = win_rate
        else:
            monthly_win_rates[month] = 0.5  # 如果没有历史数据，默认为50%

    return monthly_win_rates


# 计算季节性信号和权重
def calculate_seasonal_signals(monthly_data, start_year, end_year):
    results = []

    # 从起始年份到结束年份，计算每月的信号
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 跳过未来数据
            current_date = datetime(year, month, 1)
            if current_date > datetime.now():
                continue

            # 计算该月的历史价格上涨胜率
            win_rates = calculate_price_up_win_rate(monthly_data, year, month)
            current_win_rate = win_rates[month]

            # 计算偏离基准的程度
            deviation = current_win_rate - 0.5

            # 确定信号方向和权重
            signal_direction = 1 if deviation > 0 else -1  # 1表示多头，-1表示空头
            weight = abs(deviation)  # 权重为偏离的绝对值

            results.append({
                'year': year,
                'month': month,
                'win_rate': current_win_rate,
                'deviation': deviation,
                'signal': signal_direction,
                'weight': weight
            })

    return pd.DataFrame(results)


if __name__ == "__main__":

    pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    for pair in pairs:
        daily_data = pd.read_csv("../data/{}_signal.csv".format(pair), usecols=['date', 'open', 'high', 'low', 'close'])
        monthly_data = aggregate_to_monthly_price_change(daily_data)
        # 生成信号
        start_year = 2010  # 从2010年开始生成信号
        end_year = 2024  # 到2023年结束
        signals = calculate_seasonal_signals(monthly_data, start_year, end_year)
        signals[["month", "win_rate", "signal", "weight"]].tail(12).to_csv("../tests/season_signal_{}.csv".format(pair), index=False)
    # print(signals.tail(12))
    # THS_iFinDLogin('ghyjsxs207', '505933')
    # w.start()
    # # 股指数据
    # start_date = "20100101"
    # end_date = "20250216"
    # index_data = THS_HQ('000016.SH,000300.SH,000852.SH,000905.SH', 'close;changeRatio', '', start_date, end_date).data
    # index_data.to_csv("index_data_{}_{}.csv".format(start_date, end_date), index=False)
    # #
    # # # 基差信号
    # start = "2023-12-14"
    # end = "2025-02-16"
    # IC = w.wsd("IC.CFE", "anal_basisannualyield,ltdate_new,open,pct_chg", start, end, "", usedf=True)[1]
    # IF = w.wsd("IF.CFE", "anal_basisannualyield,ltdate_new,open,pct_chg", start, end, "", usedf=True)[1]
    # IH = w.wsd("IH.CFE", "anal_basisannualyield,ltdate_new,open,pct_chg", start, end, "", usedf=True)[1]
    # IM = w.wsd("IM.CFE", "anal_basisannualyield,ltdate_new,open,pct_chg", start, end, "", usedf=True)[1]
    # IC.columns = ['IC_' + col for col in IC.columns]
    # IF.columns = ['IF_' + col for col in IF.columns]
    # IH.columns = ['IH_' + col for col in IH.columns]
    # IM.columns = ['IM_' + col for col in IM.columns]
    # df_futures = pd.concat([IC, IF, IH, IM], axis=1)
    # df_futures["IFIH"] = df_futures["IF_ANAL_BASISANNUALYIELD"] - df_futures["IH_ANAL_BASISANNUALYIELD"]
    # df_futures["ICIF"] = df_futures["IC_ANAL_BASISANNUALYIELD"] - df_futures["IF_ANAL_BASISANNUALYIELD"]
    # df_futures["IMIC"] = df_futures["IM_ANAL_BASISANNUALYIELD"] - df_futures["IC_ANAL_BASISANNUALYIELD"]
    # df_futures["IMIH"] = df_futures["IM_ANAL_BASISANNUALYIELD"] - df_futures["IH_ANAL_BASISANNUALYIELD"]
    # df_futures.to_csv("futures_data_{}_{}.csv".format(start, end), index=True)
    #
    # df_futures = pd.read_csv("futures_data_{}_{}.csv".format(start, end), index_col=0)
    # df_futures.index = pd.to_datetime(df_futures.index)
    #
    # index_data = pd.read_csv("index_data_{}_{}.csv".format(start_date, end_date))
    # df_index_close = pd.pivot(index_data, index="time", columns="thscode", values="close")
    # # df_index_return1 = df_index_close.pct_change(periods=1)
    # df_index_return = pd.pivot(index_data, index="time", columns="thscode", values="changeRatio") / 100
    # df_index_return.index = pd.to_datetime(df_index_return.index)
    #
    # columns = df_index_return.columns
    # for col in columns:
    #     # 计算每个标的的滚动60日波动率
    #     df_index_return[f'{col}_60d_volatility'] = df_index_return[col].rolling(window=60).std()
    #
    # df_vol = df_index_return.copy(deep=True)
    # df_vol.rename(columns=lambda x: x.replace('000016.SH', 'IH').replace('000905.SH', 'IC').replace('000300.SH', 'IF').replace('000852.SH', 'IM'), inplace=True)
    # pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    # for pair in pairs:
    #     # 计算多头标的权重
    #     df_vol["{}_{}_weight".format(pair, pair[:2])] = 2 * df_vol["{}_60d_volatility".format(pair[2:])] / (
    #             df_vol["{}_60d_volatility".format(pair[:2])] + df_vol["{}_60d_volatility".format(pair[2:])])
    #     # 计算空头标的权重
    #     df_vol["{}_{}_weight".format(pair, pair[2:])] = 2 * df_vol["{}_60d_volatility".format(pair[:2])] / (
    #             df_vol["{}_60d_volatility".format(pair[:2])] + df_vol["{}_60d_volatility".format(pair[2:])])
    #     # 计算收益率60日相关性
    #     df_vol["{}_60d_corr".format(pair)] = df_vol["{}".format(pair[:2])].rolling(window=60).corr(df_vol["{}".format(pair[2:])])
    #
    # df_basis_signal = pd.merge(df_vol, df_futures, left_index=True, right_index=True)
    # for pair in pairs:
    #     # 计算基差信号
    #     df_basis_signal["signal_{}".format(pair)] = (df_basis_signal["{}_{}_weight".format(pair, pair[:2])] * \
    #                                                  df_basis_signal["{}_ANAL_BASISANNUALYIELD".format(pair[:2])] - df_basis_signal["{}_{}_weight".format(pair, pair[2:])] * \
    #                                                  df_basis_signal["{}_ANAL_BASISANNUALYIELD".format(pair[2:])]) / 100
    #     # 如果收益率60日相关性小于0.7，则空仓
    #     df_basis_signal["signal_{}_position".format(pair)] = np.where(
    #         df_basis_signal["{}_60d_corr".format(pair)] >= 0.7,
    #         -df_basis_signal["signal_{}".format(pair)] * 10,
    #         0)
    #
    # df_basis_signal["date"] = pd.to_datetime(df_basis_signal.index)
    # df_basis_signal = df_basis_signal[['date', 'signal_IFIH', 'signal_IFIH_position',
    #                                    'signal_ICIF', 'signal_ICIF_position', 'signal_IMIC',
    #                                    'signal_IMIC_position', 'signal_IMIH', 'signal_IMIH_position']]
    #
    # df_basis_signal.to_csv("basis_signal.csv", index=False)
    #
    # df_futures["IFIH_futures"] = (df_futures["IF_PCT_CHG"] - df_futures["IH_PCT_CHG"]) / 100
    # df_futures["ICIF_futures"] = (df_futures["IC_PCT_CHG"] - df_futures["IF_PCT_CHG"]) / 100
    # df_futures["IMIC_futures"] = (df_futures["IM_PCT_CHG"] - df_futures["IC_PCT_CHG"]) / 100
    # df_futures["IMIH_futures"] = (df_futures["IM_PCT_CHG"] - df_futures["IH_PCT_CHG"]) / 100
    #
    # df_futures["IFIH_futures_nv"] = (1 + df_futures["IFIH_futures"]).cumprod()
    # df_futures["ICIF_futures_nv"] = (1 + df_futures["ICIF_futures"]).cumprod()
    # df_futures["IMIC_futures_nv"] = (1 + df_futures["IMIC_futures"]).cumprod()
    # df_futures["IMIH_futures_nv"] = (1 + df_futures["IMIH_futures"]).cumprod()
    #
    # df_futures_nv = df_futures[["IFIH_futures_nv", "ICIF_futures_nv", "IMIC_futures_nv", "IMIH_futures_nv"]]
    # df_futures_nv = df_futures_nv.div(df_futures_nv.iloc[0])
    #
    # df_futures_nv.index = pd.to_datetime(df_futures.index)
    # # 股指期货多空组合的净值
    # df_futures_nv.to_csv("futures_nv_data_{}_{}.csv".format(start, end), index=True)
    # df_index_return["IF_60_std"]
    #
    # df_index_return["IFIH_index"] = df_index_return["000300.SH"] - df_index_return["000016.SH"]
    # df_index_return["ICIF_index"] = df_index_return["000905.SH"] - df_index_return["000300.SH"]
    # df_index_return["IMIC_index"] = df_index_return["000852.SH"] - df_index_return["000905.SH"]
    # df_index_return["IMIH_index"] = df_index_return["000852.SH"] - df_index_return["000016.SH"]
    #
    # df_index_return["IFIH_index_nv"] = (1 + df_index_return["IFIH_index"]).cumprod()
    # df_index_return["ICIF_index_nv"] = (1 + df_index_return["ICIF_index"]).cumprod()
    # df_index_return["IMIC_index_nv"] = (1 + df_index_return["IMIC_index"]).cumprod()
    # df_index_return["IMIH_index_nv"] = (1 + df_index_return["IMIH_index"]).cumprod()
    #
    # df_index_nv = df_index_return[["IFIH_index_nv", "ICIF_index_nv", "IMIC_index_nv", "IMIH_index_nv"]]
    # df_index_nv = df_index_nv.div(df_index_nv.iloc[0])
    # df_index_nv.index = pd.to_datetime(df_index_return.index)
    # df_index_nv.to_csv("index_nv_data_{}_{}.csv".format(start_date, end_date), index=True)
    #
    #
    #
    # plt.figure(figsize=(12, 6))  # 设置图表大小
    # for col in df_index_nv.columns:
    #     plt.plot(df_index_nv.index, df_index_nv[col], label=col)  # 为每列画线
    #
    # # 添加图例、标题和坐标轴标签
    # plt.legend(title="Columns", loc="upper left")
    # plt.title("Normalized DataFrame Plot (Starting at 1)")
    # plt.xlabel("Index")
    # plt.ylabel("Normalized Value")
    # plt.show()
    #
    # print()
