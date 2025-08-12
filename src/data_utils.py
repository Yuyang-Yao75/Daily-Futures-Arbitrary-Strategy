import os
import pandas as pd
from datetime import datetime
from config import CODE_MAP,AVAILABLE_PAIRS,RAW_DATA_PATH,RESULT_PATH,INDEX_DATA,FUTURES_DATA
# from iFinDPy import *#todo
# from WindPy import w#todo

#====================== 原始数据读取======================
# def ths_login():#todo
#     ret = THS_iFinDLogin('ghyjsxs207', '505933')
#     print(ret)
#     if ret != 0:
#         raise  RuntimeError("登陆失败")
#     print("登陆成功")

def search_file_recursive(directory, filename):
    """
    在指定目录及其子目录中搜索文件

    Args:
        directory: 要搜索的目录路径
        filename: 要查找的文件名（包含扩展名）

    Returns:
        str: 如果找到文件，返回1；如果未找到，返回0
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return True
    return False

def pivot_index_data(df, code_map):#todo
    """
    将长格式的指数日度数据转成宽格式。

    参数
    ----
    df : pandas.DataFrame
        包含列 ['time', 'thscode', 'close', 'changeRatio'] 的长表，直接从 iFind 中下载即可。，
        time 已经是 datetime 或可解析为 datetime 的字符串。
    code_map : dict
        从指数代码到你想要前缀的映射，例如
            {
            '000300.SH': 'IF',
            '000016.SH': 'IH',
            '000905.SH': 'IC',
            '000852.SH': 'IM',
            }

    返回
    ----
    pandas.DataFrame
        宽表，列依次为
        time,
        IF_udy_close, IF_udy_changeRatio,
        IH_udy_close, IH_udy_changeRatio,
        IC_udy_close, IC_udy_changeRatio,
        IM_udy_close, IM_udy_changeRatio
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df[df['thscode'].isin(code_map.keys())]

    wide = df.pivot(index='time',
                    columns='thscode',
                    values=['close', 'changeRatio'])
    wide.columns = [
        f"{code_map[code]}_udy_{fld}"
        for fld, code in wide.columns
    ]
    wide = wide.reset_index().sort_values('time')

    return wide


# #定义获取股指数据的函数#todo
# def get_stock_index_data(stock_index, start_date, end_date):
#     """
#     获取股指数据，如果本地存在则直接读取，否则从API获取并保存
#
#     Args:
#         stock_index: 股票指数代码，如'000300.SH'
#         start_date: 开始日期，datetime格式
#         end_date: 结束日期，datetime格式
#
#     Returns:
#         pd.DataFrame: 包含指数数据的DataFrame
#
#     Raises:
#         ValueError: 当API调用失败时抛出
#     """
#     filename = f"index_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
#     filepath = INDEX_DATA
#     #判断数据是否存在
#     if search_file_recursive(filepath, filename):
#         index_data = pd.read_csv(filepath)
#         index_data["time"] = pd.to_datetime(index_data["time"])
#         index_data.set_index("time", inplace=True)
#         return index_data
#
#     ths_login()
#     data_result=THS_HQ(stock_index, 'close;changeRatio', '', start_date, end_date)
#     if data_result.errorcode != 0:
#         print('error:{}'.format(data_result.errmsg))
#         raise ValueError(f"指数数据调取失败")
#     else:
#         index_data = data_result.data
#         index_data["changeRatio"] = index_data["changeRatio"]/100
#         index_data = pivot_index_data(data_result.data, CODE_MAP)
#         index_data.to_csv(f"{filepath}/{filename}", index=False)
#         index_data.rename(columns={'time': 'date'}, inplace=True)
#         index_data["date"] = pd.to_datetime(index_data["date"])
#         index_data.set_index("date", inplace=True)
#         return index_data
#
# #定义获取期货数据的函数#todo
# def get_futures_data(start_date, end_date):
#     """
#     获取期货数据：如果本地已有缓存则加载，否则通过 Wind API 拉取并保存
#
#     Args:
#         start_date (datetime): 数据开始日期
#         end_date (datetime): 数据结束日期
#
#     Returns:
#         pd.DataFrame: 拉取并加工后的期货数据，索引为 time 列
#     """
#     filename = f"futures_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
#     filepath = FUTURES_DATA
#
#     # 判断数据是否存在
#     if search_file_recursive(filepath, filename):
#         futures_data = pd.read_csv(filepath)
#         futures_data["time"] = pd.to_datetime(futures_data["time"])
#         futures_data.set_index("time", inplace=True)
#         return futures_data
#
#     # 否则通过 Wind API 拉取
#     w.start()  # 如需登录，解除注释
#     codes = {'IC': 'IC.CFE', 'IF': 'IF.CFE', 'IH': 'IH.CFE', 'IM': 'IM.CFE'}
#     frames = []
#     for prefix, code in codes.items():
#         data = w.wsd(
#             code,
#             'anal_basisannualyield,ltdate_new,open,pct_chg',
#             start_date.strftime('%Y-%m-%d'),
#             end_date.strftime('%Y-%m-%d'),
#             '',
#             usedf=True
#         )[1]
#         data['PCT_CHG']=data['PCT_CHG']/100
#         data.columns = [f"{prefix}_{col}" for col in data.columns]
#         frames.append(data)
#
#     df_futures = pd.concat(frames, axis=1)
#     # 计算各配对基差信号（年化基差差值）
#     df_futures['IFIH'] = (
#         df_futures['IF_anal_basisannualyield'] - df_futures['IH_anal_basisannualyield']
#     ) / 100
#     df_futures['ICIF'] = (
#         df_futures['IC_anal_basisannualyield'] - df_futures['IF_anal_basisannualyield']
#     ) / 100
#     df_futures['IMIC'] = (
#         df_futures['IM_anal_basisannualyield'] - df_futures['IC_anal_basisannualyield']
#     ) / 100
#     df_futures['IMIH'] = (
#         df_futures['IM_anal_basisannualyield'] - df_futures['IH_anal_basisannualyield']
#     ) / 100
#
#     # 保存到本地缓存
#     df_futures.to_csv(filepath, index=False)
#     # 设置时间索引并返回
#     df_futures.rename(columns={'ltdate_new': 'date'}, inplace=True)
#     df_futures['date'] = pd.to_datetime(df_futures['date'])
#     df_futures.set_index('date', inplace=True)
#     return df_futures

#====================原始数据预处理====================
#获取净值数据
def get_nv_data(df_underlying: pd.DataFrame, cal_type: str) -> pd.DataFrame:
    if search_file_recursive(RAW_DATA_PATH, f"{cal_type}_nv_data.csv"):
        df_nv = pd.read_csv(os.path.join(RAW_DATA_PATH, f"{cal_type}_nv_data.csv"))
        df_nv["date"] = pd.to_datetime(df_nv["date"])
        df_nv.set_index("date", inplace=True)
        return df_nv
    else:
        df_nv = calculate_nv_data(df_underlying, cal_type)
        return df_nv
#计算净值
def calculate_nv_data(df_underlying: pd.DataFrame,
                    cal_type: str,
                    code_map: dict = CODE_MAP,
                    available_pairs: list = AVAILABLE_PAIRS) -> pd.DataFrame:
    """
    输入：
        df_underlying: 包含 'time' 列 和对应后缀的原始 Returns 数据
        type: 'index' 或 'futures'
    返回：
        df_nv: index，由各配对组合的净值序列组成，列名形如 'IFIH_index_nv' 或 'IFIH_futures_nv'
    """
    # 根据 type 选列后缀 & 文件名前缀
    if cal_type == 'index':
        return_cols = {v: f"{v}_udy_changeRatio" for v in code_map.values()}
        file_prefix = 'index_nv_data'
    elif cal_type == 'futures':
        return_cols = {v: f"{v}_PCT_CHG" for v in code_map.values()}
        file_prefix = 'futures_nv_data'
    else:
        raise ValueError("type 必须是 'index' 或 'futures'")

    # 时间列转 datetime 并设为索引
    df = df_underlying.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif df.index.name == "date":
        pass
    else:
        raise KeyError("No 'date' column or index found in DataFrame.")

    # 存放所有组合净值
    df_nv = pd.DataFrame(index=df.index)

    for pair in available_pairs:
        first, second = pair[:2], pair[2:]
        col_f, col_s = return_cols[first], return_cols[second]
        # 累积净值
        cum_nv = (1 + (df[col_f] - df[col_s])).cumprod()
        # 归一化到 1
        df_nv[f"{pair}_{cal_type}_nv"] = cum_nv / cum_nv.iloc[0]

    out_path = os.path.join(RAW_DATA_PATH,
                            f"{file_prefix}.csv")
    df_nv.to_csv(out_path, index=True)
    return df_nv

# 生成高开低收格式数据
def generate_ohlc(df_nv:pd.DataFrame,
                prefix:str,
                open_offset: float = -0.01,
                high_offset: float = +0.01,
                low_offset: float  = -0.02,)->pd.DataFrame:
    """
    从净值数据中提取以 {prefix}_ 前缀命名的列，生成 OHLC 表格。
    规则：
    - 必须存在 {prefix}_close
    - 若 {prefix}_open / _high / _low 不存在，则按给定 offset 由 close 推出

    参数
    ----
    df_nv : pd.DataFrame
        以时间为索引或包含 'date' 列的净值表。
    prefix : str
        符号前缀，例如 'IFIH'。
    open_offset, high_offset, low_offset : float
        当缺失对应列时，用 close 加（或减）该偏移量生成。

    返回
    ----
    pd.DataFrame
        以日期为索引，包含 ['close','open','high','low'] 的表格。
    """
    close_col = f"{prefix}_close"
    if close_col not in df_nv.columns:
        raise ValueError(f"缺少必要列：{close_col}")

    vol_col = f"{prefix}_volume"
    if vol_col not in df_nv.columns:
        raise ValueError(f"缺少必要列：{vol_col}")

    # 索引处理：优先 DatetimeIndex，其次 'date' 列
    if isinstance(df_nv.index, pd.DatetimeIndex):
        date_index = df_nv.index
    elif "date" in df_nv.columns:
        date_index = pd.to_datetime(df_nv["date"])
        if date_index.isna().any():
            raise ValueError("date 列存在非法日期，无法转换为 DatetimeIndex")
    else:
        raise ValueError("无法获取日期索引：需要 DatetimeIndex 或 'date' 列")

    out = pd.DataFrame(index=date_index.copy())
    out.index.name = "date"

    out["close"] = df_nv[close_col].to_numpy()

    open_col = f"{prefix}_open"
    high_col = f"{prefix}_high"
    low_col = f"{prefix}_low"

    out["open"] = (df_nv[open_col].to_numpy()
                if open_col in df_nv.columns
                else out["close"] + open_offset)
    out["high"] = (df_nv[high_col].to_numpy()
                if high_col in df_nv.columns
                else out["close"] + high_offset)
    out["low"] = (df_nv[low_col].to_numpy()
                if low_col in df_nv.columns
                else out["close"] + low_offset)
    out["volume"]=df_nv[vol_col].to_numpy()

    # 可选兜底，确保 OHLC 合理关系
    # out["high"] = out[["high", "open", "close"]].max(axis=1)
    # out["low"] = out[["low", "open", "close"]].min(axis=1)

    return out.sort_index()

#======================提取回测数据=========================
def get_concat_nv_data(available_pairs:str = AVAILABLE_PAIRS,
                    result_path:str = RESULT_PATH):
    """
    读取所有品种的 '{pair}_concat_portfolio.csv'，提取 market_value 计算净值，
    并合并到一个 DataFrame 中。

    参数
    ----
    available_pairs : list of str
        所有品种代码列表，比如 ["ICIF", "IFIH", ...]
    result_path : str
        存放 CSV 文件的目录

    返回
    ----
    pd.DataFrame
        包含列： date, {pair1}_nv, {pair2}_nv, ...
    """
    dfs = []
    for pair in available_pairs:
        fn = f"{pair}_concat_portfolio.csv"
        fp = os.path.join(result_path, fn)
        df = pd.read_csv(fp,parse_dates=["date"])
        df.sort_values("date", inplace=True)

        df[f"{pair}_nv"] = df["market_value"]/df["market_value"].iloc[0]
        df_pair = df[['date',f"{pair}_nv"]].set_index('date')
        dfs.append(df_pair)

    combined = pd.concat(dfs,axis = 1)
    combined.reset_index(inplace = True)

    return combined