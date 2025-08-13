import os
import datetime
from itertools import combinations
from collections import OrderedDict
from glob import glob
from signal_utils import *
import json
#todo 基差策略和季节性策略仍有部分参数为直接设定
# 策略品种设定
STOCK_INDEX='000016.SH,000300.SH,000852.SH,000905.SH'
# 日期设定：可自由设定回测时间
INDEX_START_DATE=datetime.datetime(2010, 1, 1)
START_DATE=datetime.datetime(2023, 12, 14)
END_DATE=datetime.datetime(2025, 6, 25)
INITIAL_CASH=10_000
BARS_PER_YEAR=244
CODE_MAP = {
    '000852.SH': 'IM',  # 中证1000指数 -> IM期货
    '000905.SH': 'IC',  # 中证500指数 -> IC期货
    '000300.SH': 'IF',  # 沪深300指数 -> IF期货
    '000016.SH': 'IH',  # 上证50指数 -> IH期货
}
AVAILABLE_PAIRS = [f"{a}{b}" for a, b in combinations(list(CODE_MAP.values()), 2)]
AVAILABLE_STRATEGY=['basis','seasonal','technical','concat']

# 文件路径相关设置
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "..",'data'))
RESULT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "..", 'results'))

RAW_DATA_PATH=os.path.join(DATA_PATH, 'raw')
SIGNAL_DATA_PATH=os.path.join(DATA_PATH, 'signal')
FACTOR_SELECTION_PATH=os.path.join(RESULT_PATH, 'factor_selection')
# 原始数据文件路径：正常使用时将下方注释取消并删除现有 INDEX_DATA 部分
FUTURES_DATA = os.path.join(RAW_DATA_PATH, f'futures_data_{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}.csv')
# INDEX_DATA = os.path.join(RAW_DATA_PATH, f'index_data_{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}.csv')
INDEX_DATA = os.path.join(RAW_DATA_PATH, f'index_data_{20100101}_{END_DATE.strftime("%Y%m%d")}.csv')

# 因子参数设定
FUNC_MAP = {
    'bollinger_r': bollinger_r,
    'bollinger_MOM': bollinger_mom,
    'DoubleMA': generate_ma_signal,
    'EXPWMA': generate_ma_signal,
    'Hilbert_Transform': generate_ma_signal,
    'Kaufman': generate_ma_signal,
    'MESA_Adaptive': generate_ma_signal,
    'MIDPOINT': generate_ma_signal,
    'TRIX': generate_ma_signal,
    'WMA': generate_ma_signal,
    'MIDPRICE': generate_ma_signal,
    'MACD': macd_signal,
    'ROC_R': roc_r,
    'ROC_MOM': roc_mom,
    'MOM_r': mom_r,
    "RSI": rsi_r,
    'CMO': cmo_r,
    'Quantile': quantile_signal,
    'continuous': continuous_signal,
    'continuous_r': continuous_r,
    'bollinger_atr_mom':bollinger_atr_mom,
    'turtle_trading':turtle_trading,
    'sar':sar,
    'intramom':intramom,
    'stds':stds,
    'cci':cci
}

PARAMS_MAP = {
    'bollinger_r': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120], "num_std_upper": [1, 1.5, 2],
                    "num_std_lower": [1, 1.5, 2]},
    'bollinger_MOM': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120], "num_std_upper": [1, 1.5, 2],
                    "num_std_lower": [1, 1.5, 2]},
    'DoubleMA': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
                "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['DoubleMA']},
    'EXPWMA': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
            "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['EXPWMA']},
    'Hilbert_Transform': {"short_window": [1, 2, 3, 4, 5, 10, 20, 60, 120, 240], "long_window": [1],
                        'ma_type': ['Hilbert_Transform']},
    'Kaufman': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
                "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['Kaufman']},
    'MESA_Adaptive': {"short_window": [3], "long_window": [20], 'ma_type': ['MESA_Adaptive'],
                    "fastlimit": [0.1, 0.3, 0.5, 0.7, 0.9], "slowlimit": [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]},
    'MIDPOINT': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
                "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['MidPoint']},
    'TRIX': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
            "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['TRIX'],
            'vfactor': [0.6, 0.8, 1]},
    'WMA': {"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
            "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['WMA']},
    'MIDPRICE':{"short_window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200],
            "long_window": [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200], 'ma_type': ['MIDPRICE']},
    'MACD': {"short_window": [7.8, 9, 10, 12, 13], "long_window": [24, 26, 30, 34, 40], "signalperiod": [6, 9, 12]},
    'ROC_R': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200]},
    'ROC_MOM': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200]},
    'MOM_r': {"period": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200], "threshold": [5, 10, 15, 20, 25, 30]},
    'RSI': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200], "lower": [5, 10, 15, 20, 25, 30],
            "middle": [50]},
    'CMO': {"window": [3, 5, 10, 20, 30, 40, 50, 60, 120, 140, 160, 200], "upper": [50, 55, 60, 65, 70, 75, 80, 85],
            "middle": [0]},
    'Quantile': {"window": [20, 60, 90, 100, 150, 200, 250, 500]},
    'continuous': {"window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]},
    'continuous_r': {"window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]},
    'bollinger_atr_mom':{"window":[3,5,10,20,30,40,50,60,120],"atr_mult_upper":[1,1.5,2],"atr_mult_lower":[1,1.5,2]},
    'turtle_trading':{"window":[2,3,5,10,20,30,40,50,60,70,80,90,100,150,200,250]},
    'sar':{"acceleration":[0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5]},
    'intramom':{"threshold":[0.97,0.98,0.99,1,1.01,1.02,1.03]},
    'stds':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],"long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200],"method":["signed_range","stoch_pos","combo"],"norm_window":[5,10,20,30,40,50,60]},
    'cci':{"window":[3,5,10,20,30,40,50,60,120,140,160,200],"threshold":[80,100,120,140,160,180,200]}
}

PAIR_FACTORS = {}
for pair in AVAILABLE_PAIRS:
    # 在 FACTOR_SELECTION_PATH 下查找以 "{pair}_selected_factors" 开头的文件
    pattern = os.path.join(FACTOR_SELECTION_PATH, f"{pair}_selected_factors*")
    matches = glob(pattern)
    if not matches:
        # 如果没有找到文件，跳过
        continue
    # 只取第一个文件
    filepath = matches[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    od = OrderedDict()
    for factor_name, info in data.items():
        func_name = info.get('func', None)
        params = info.get('params', {})
        func = FUNC_MAP.get(func_name)
        if func is None:
            # 如果没在 FUNC_MAP 中找到对应函数，也可以尝试 globals() 获取
            func = globals().get(func_name)
        if not callable(func):
            # 如果依然没找到合法函数，跳过
            continue
        od[factor_name] = (func, params)
        PAIR_FACTORS[pair] = od



