import numpy as np
from data_utils import *
from signal_utils import *
from config import *
from datetime import datetime
from typing import Dict
#====================== 基差 ======================
# 计算基差因子信号
def calculate_basis_signal(
        df_index:pd.DataFrame,
        df_futures:pd.DataFrame,
        pair: str,
        window: int=60,
        corr_threshold:float=0.7,
        leverage:float=10
        )->pd.DataFrame:
    """
    对单个多空组合（pair，如'IFIH'），计算基差信号及仓位。

    参数:
    df_index: pd.DataFrame，包含指数数据，包括时间和指数收盘价以及指数收益。
    df_futures: pd.DataFrame，包含期货数据，包括时间和期货收盘价等。
    pair: str，多空组合的代码，如'IFIH'。
    window: int，计算滚动波动率的窗口大小和相关性的窗口大小，默认为60。
    corr_threshold: float，当收益率相关性低于此阈值时强制平仓，默认为0.7。
    leverage: float，信号乘数，默认为10。

    返回:
    pd.DataFrame，包含时间、基差信号和仓位。
        包含三列：
        - date                  (datetime)
        - signal_{pair}         (float) 基差信号
        - signal_{pair}_position(float) 最终仓位信号
    """
    if search_file_recursive(SIGNAL_DATA_PATH, f'{pair}_raw_basis_signal.csv'):
        df_out = pd.read_csv(os.path.join(SIGNAL_DATA_PATH, f'{SIGNAL_DATA_PATH}/{pair}_raw_basis_signal.csv'))
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.set_index("date", inplace=True)
        return df_out
    # 计算滚动波动率
    sym_long, sym_short = pair[:2], pair[2:]
    vol_long = df_index[f"{sym_long}_udy_changeRatio"].rolling(window).std()
    vol_short = df_index[f"{sym_short}_udy_changeRatio"].rolling(window).std()

    # 计算当日权重
    weight_long = 2*vol_short / (vol_long + vol_short)
    weight_short = 2*vol_long / (vol_long + vol_short)

    # 计算收益率相关性
    corr = df_index[f"{sym_long}_udy_changeRatio"].rolling(window).corr(df_index[f"{sym_short}_udy_changeRatio"])

    # 计算基差信号
    basis_long = df_futures[f"{sym_long}_ANAL_BASISANNUALYIELD"]/100
    basis_short = df_futures[f"{sym_short}_ANAL_BASISANNUALYIELD"]/100
    raw_signal = weight_long * basis_long - weight_short * basis_short

    signal = raw_signal.reindex(df_futures.index)
    corr_aligned = corr.reindex(df_futures.index)
    # 仓位计算
    position = (signal * (-leverage)).where(corr_aligned >= corr_threshold, 0)

    # 组装输出
    df_out = pd.DataFrame({
        'date':                    df_futures.index,
        f"signal_{pair}":          signal.values,
        f"signal_{pair}_position": position.values
    }).set_index("date")
    # 返回结果
    df_out.to_csv(f'{SIGNAL_DATA_PATH}/{pair}_raw_basis_signal.csv',index=True)
    return df_out

def generate_all_basis_signal(index_data, futures_data, pairs:list=AVAILABLE_PAIRS):#暂时没有用到
    all_signals=[]
    for pair in pairs:
        df_sig = calculate_basis_signal(index_data, futures_data, pair)
        all_signals.append(df_sig.set_index('date'))
    df_all_sig = pd.concat(all_signals, axis=1).reset_index()
    return df_all_sig

#====================== 因子信号 ======================
# 计算单个组合的技术因子信号
def calculate_technical_signal(
        index_nv_df: pd.DataFrame,
        pair: str,
        cal_col: str = "close",
) -> pd.DataFrame:
    """
    根据指定列计算技术因子信号，并生成综合仓位信号

    输入：
    df: pd.DataFrame, 包含股票高开低收数据的 DataFrame
    pair: str, 要计算技术因子的多空组合
    cal_col: str, 要计算技术因子的列名称

    输出：
    df_signal: pd.DataFrame, 包含指定技术因子信号和综合仓位信号的 DataFrame，其中 position_signal是多个因子的简单平均
    """
    if search_file_recursive(SIGNAL_DATA_PATH, f'{pair}_raw_technical_signal.csv'):
        df_out = pd.read_csv(os.path.join(SIGNAL_DATA_PATH, f'{SIGNAL_DATA_PATH}/{pair}_raw_technical_signal.csv'))
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.set_index("date", inplace=True)
        return df_out
    df = generate_ohlc(index_nv_df, pair).copy()
    price = df[cal_col]
    # 1) 从配置里取当前 pair 应跑的因子；若不存在，则用 ALL_FACTORS
    factors = PAIR_FACTORS.get(pair, ALL_FACTORS)
    # 2）按配置循环调用，每次生成一列“{因子名}_signal”
    signal_cols = []
    for name,(func,params) in factors.items():
        col = f"{name}_signal"
        df[col] = func(price, **params)
        signal_cols.append(col)
    # 3）综合仓位信号
    df["position_signal"] = df[signal_cols].mean(axis=1)
    df.to_csv(f"{SIGNAL_DATA_PATH}/{pair}_raw_technical_signal.csv", index=True)
    return df

def generate_all_technical_signals(
        pairs: list = AVAILABLE_PAIRS,
        index_nv_df: pd.DataFrame = None,
)-> Dict[str, pd.DataFrame]:
    """
    批量为每个 pair 生成技术因子信号，并将结果存入字典返回。

    参数:
        pairs: list of str, 可用的多空组合列表
        index_nv_df: pd.DataFrame, 包含净值序列的 DataFrame，用于生成 OHLC 数据

    返回:
        signals_dict: Dict[str, pd.DataFrame]
            - key: pair 名称
            - value: 对应 pair 的技术因子信号 DataFrame
    """
    signals_dict: Dict[str, pd.DataFrame] = {}

    for pair in pairs:
        # 计算单个组合的技术因子信号
        signal_df = calculate_technical_signal(index_nv_df, pair)
        # 存入字典，键为 pair，值为对应的 DataFrame
        signals_dict[pair] = signal_df

    return signals_dict

#===================季节性信号===================
# 计算季节性信号和权重
def calculate_seasonal_signal(index_nv_df, pair, start_year=2010, end_year=datetime.now().year-1):
    if search_file_recursive(SIGNAL_DATA_PATH, f'{pair}_raw_seasonal_signal.csv'):
        df_out = pd.read_csv(os.path.join(SIGNAL_DATA_PATH, f'{SIGNAL_DATA_PATH}/{pair}_raw_seasonal_signal.csv'))
        return df_out
    daily_data = generate_ohlc(index_nv_df, pair)
    monthly_data = aggregate_to_monthly_price_change(daily_data)
    results = []
    from datetime import datetime
    now = datetime.now()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 跳过未来数据（只到本月）
            if (year > now.year) or (year == now.year and month > now.month):
                continue
            # 计算该月的历史价格上涨胜率
            win_rates = calculate_price_up_win_rate(monthly_data, year, month)
            current_win_rate = win_rates[month]
            deviation = current_win_rate - 0.5
            signal_direction = 1 if deviation > 0 else -1  # 1表示多头，-1表示空头
            weight = abs(deviation)
            results.append({
                'year': year,
                'month': month,
                'win_rate': current_win_rate,
                'deviation': deviation,
                'signal': signal_direction,
                'weight': weight
            })
    results_df = pd.DataFrame(results)
    df_out = results_df[["month", "win_rate", "signal", "weight"]].tail(12)
    df_out.to_csv(f"{SIGNAL_DATA_PATH}/{pair}_raw_seasonal_signal.csv",index=False)
    return df_out

def generate_all_seasonal_signal(
            index_nv_df: pd.DataFrame,
            pairs: list = AVAILABLE_PAIRS,
            start_year: int = 2010,
            end_year: int = datetime.now().year - 1
    ) -> Dict[str, pd.DataFrame]:
    """
    批量为每个 pair 生成季节性因子信号，并返回一个字典：
    - key: pair 名称
    - value: 对应 pair 的季节性信号 DataFrame

    参数:
        index_nv_df: pd.DataFrame, 包含净值序列数据，用于生成日度 OHLC
        pairs: list of str, 可用的多空组合列表
        start_year: int, 信号生成的起始年份
        end_year: int, 信号生成的结束年份（不包括今年）
    返回:
        signals_dict: Dict[str, pd.DataFrame]
    """
    signals_dict: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        # 生成信号
        signals = calculate_seasonal_signal(index_nv_df, pair, start_year, end_year)
        signals_dict[pair] = signals
    return signals_dict

#===================信号提取函数===================
def extract_signals(futures_nv_df: pd.DataFrame, pair: str,signals_df: pd.DataFrame,strategy_name: str) -> pd.DataFrame:
    """
    根据不同策略，把 generate_ohlc 输出的 OHLC 与信号做匹配。

    参数:
    futures_nv_df: pd.DataFrame，原始期货净值，至少含用于 generate_ohlc 的字段
    pair: str，品种名，例如 "IFIH"
    signals_df: pd.DataFrame or dict，
        - basis/technical: 包含 date + signal_{pair}_position 列的 DataFrame
        - season: 包含 month, signal, weight 列的 DataFrame
    strategy_name: str，"basis","seasonal","technical"
    返回:
    pd.DataFrame，含 [date, symbol, open, high, low, close, position_signal]
    """
    if search_file_recursive(SIGNAL_DATA_PATH, f'{pair}_{strategy_name}_signal.csv'):
        df_out = pd.read_csv(os.path.join(SIGNAL_DATA_PATH, f'{SIGNAL_DATA_PATH}/{pair}_{strategy_name}_signal.csv'))
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.set_index("date", inplace=True)
        return df_out
    # 1) 生成 OHLC 框架
    data_df = generate_ohlc(futures_nv_df, pair).copy()

    df = data_df.copy()
    if strategy_name == "basis":
        raw = signals_df.copy()
        col = f"signal_{pair}_position"
        df = df.join(raw[[col]].rename(columns={col: "position_signal"}),how="left")
    elif strategy_name == "technical":
        raw = signals_df.copy()
        col = f"position_signal"
        df = df.join(raw[[col]].rename(columns={col: "position_signal"}),how="left")
    elif strategy_name == "seasonal":
        raw = signals_df.copy()
        raw['position_signal'] = raw['signal']*raw['weight']
        month_map = raw.set_index('month')['position_signal'].to_dict()
        df['month'] = df.index.month
        df['position_signal'] = df['month'].map(month_map)
        df.drop('month', axis=1, inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    df["symbol"] = pair
    df = df.reset_index()
    df_out = df[['date','symbol','open','high','low','close','position_signal']]
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.set_index("date", inplace=True)
    df_out.to_csv(f"{SIGNAL_DATA_PATH}/{pair}_{strategy_name}_signal.csv",index=True)
    return df_out

def extract_concat_signals(
        futures_nv_df: pd.DataFrame,
        pair: str,
        basis_raw_df: pd.DataFrame,
        seasonal_raw_df: pd.DataFrame,
        technical_raw_df: pd.DataFrame
)->pd.DataFrame:
    """
    为 concat 策略生成 OHLC+离散化仓位信号

    参数：
    futures_nv_df: pd.DataFrame，原始期货净值，至少含用于 generate_ohlc 的字段
    pair: str，品种名，例如 "IFIH"
    basis_raw_df: pd.DataFrame，包含 date + signal_{pair}_position 列的 DataFrame
    seasonal_raw_df: pd.DataFrame，包含 month, signal, weight 列的 DataFrame
    technical_raw_df: pd.DataFrame，包含 date + position_signal 列的 DataFrame
    返回:
    pd.DataFrame，含 [date, symbol, open, high, low, close, position_signal]
    """
    if search_file_recursive(SIGNAL_DATA_PATH, f'{pair}_concat_signal.csv'):
        df_out = pd.read_csv(os.path.join(SIGNAL_DATA_PATH, f'{SIGNAL_DATA_PATH}/{pair}_concat_signal.csv'))
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.set_index("date", inplace=True)
        return df_out
    # 1) 生成 OHLC
    df = generate_ohlc(futures_nv_df, pair).copy()
    # 2) 计算三路信号序列
    # -- basis
    b = basis_raw_df.copy()
    col_b = f"signal_{pair}_position"
    df['basis_signal'] = b[col_b]

    # -- technical
    t = technical_raw_df.copy()
    col_t = f"position_signal"
    df['technical_signal'] = t[col_t]

    # -- seasonal
    s = seasonal_raw_df.copy()
    # 先把 signal*weight 变成 position
    s['position_signal'] = s['signal'] * s['weight']
    month_map = s.set_index('month')['position_signal'].to_dict()
    df['seasonal_signal'] = df.index.month.map(month_map)

    # 3) 计算连续仓位 Position0
    df['Position0'] = (
        (0.2 * df['basis_signal'] + 0.8 * df['technical_signal'])
        * (1 - df['seasonal_signal'].abs())
    ) + df['seasonal_signal']

    # 4) 离散化到最终 position_signal
    conditions = [
        df['Position0'] >= 1,
        df['Position0'].between(0.5, 1, inclusive='left'),
        df['Position0'].between(0.2, 0.5, inclusive='left'),
        df['Position0'].between(-0.2, 0.2),
        df['Position0'].between(-0.5, -0.2, inclusive='right'),
        df['Position0'].between(-1, -0.5, inclusive='right'),
        df['Position0'] <= -1,
    ]
    values = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]
    df['position_signal'] = np.select(conditions, values)
    df['symbol'] = pair
    df.to_csv(f"{SIGNAL_DATA_PATH}/{pair}_concat_signal.csv",index=True)
    return df