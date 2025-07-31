import os
import pandas as pd
from datetime import datetime
from config import CODE_MAP,AVAILABLE_PAIRS,RAW_DATA_PATH
# from iFinDPy import *
# from WindPy import w

#====================== 原始数据读取======================
# def ths_login():
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


# #定义获取股指数据的函数
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
#         index_data.to_csv(filename, index=False)
#         index_data["time"] = pd.to_datetime(index_data["time"])
#         index_data.set_index("time", inplace=True)
#         return index_data
#
# #定义获取期货数据的函数
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
#     # w.start()  # 如需登录，解除注释
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
#     df_futures['time'] = pd.to_datetime(df_futures['time'])
#     df_futures.set_index('time', inplace=True)
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
def generate_ohlc(df_nv:pd.DataFrame,prefix:str)->pd.DataFrame:#todo
    """
    从净值数据中提取以指定前缀开头的列，并生成 OHLC 表格。

    参数：
    df_nv (pd.DataFrame)：以时间索引的净值表，包含如 'IFIH_index_nv'、'IFIH_futures_nv' 等
    prefix (str)        ：要模糊搜索的列前缀，例如 'IFIH'

    返回：
    pd.DataFrame：包含 ['date','close','open','high','low'] 的新表格
    """
    # 在列名中进行模糊搜索
    matched_cols = [col for col in df_nv.columns if col.startswith(prefix)]
    if not matched_cols:
        raise ValueError(f"没有找到以 '{prefix}' 开头的列")
    # 如果有多个匹配，默认使用第一个
    target_col = matched_cols[0]
    # 生成 OHLC 表格
    df = df_nv[[target_col]].copy().reset_index()
    df.rename(columns={target_col: 'close'}, inplace=True)
    df['open']=df['close']-0.01
    df['high']=df['close']+0.01
    df['low']=df['close']-0.02
    df['date']=pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)
    return df