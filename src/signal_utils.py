import pandas as pd
import talib

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
        short_ma = talib.T3(price, short_window, vfactor)
        long_ma = talib.T3(price, long_window, vfactor)
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


# 聚合为月度数据 - 为每个月计算价格涨跌情况

def aggregate_to_monthly_price_change(df):
    """
    为每个月计算多空组合的净值涨跌情况
    :param df: 多空组合后的净值
    :return: 月度涨跌数据
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    # 用groupby+agg优化
    monthly = df.groupby(['year', 'month']).agg(
        first_price=('close', 'first'),
        last_price=('close', 'last'),
        last_date=('time', 'last')
    ).reset_index()
    monthly['return'] = (monthly['last_price'] / monthly['first_price']) - 1
    monthly['price_up'] = monthly['return'] > 0
    return monthly[['year', 'month', 'return', 'price_up', 'last_date']]

# 函数：计算到某年某月为止的历史价格涨跌胜率

def calculate_price_up_win_rate(df, current_year, current_month):
    # 只包含当前月之前的历史数据
    mask = ((df['year'] < current_year) | ((df['year'] == current_year) & (df['month'] < current_month)))
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