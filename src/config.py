import os
import datetime
from itertools import combinations
from collections import OrderedDict
from signal_utils import (
    bollinger_r,     # 通道型布林带策略
    bollinger_MOM,   # 动量型布林带策略
    roc_MOM,         # ROC 动量策略
    continuous_signal,# 连续上涨/下跌天数策略
    generate_ma_signal,# 多种均线交叉策略
    macd_signal,     # MACD 三线策略
    rsi_r,           # RSI 反转策略
    cmo_r,           # CMO 反转策略
    roc_r,           # ROC 反转策略
    continuous_r,    # 反转型连续天数策略
    quantile_signal, # 百分位反转策略
)
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
# 原始数据文件路径：正常使用时将下方注释取消并删除现有 INDEX_DATA 部分
FUTURES_DATA = os.path.join(RAW_DATA_PATH, f'futures_data_{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}.csv')
# INDEX_DATA = os.path.join(RAW_DATA_PATH, f'index_data_{START_DATE.strftime("%Y%m%d")}_{END_DATE.strftime("%Y%m%d")}.csv')
INDEX_DATA = os.path.join(RAW_DATA_PATH, f'index_data_{20100101}_{END_DATE.strftime("%Y%m%d")}.csv')

# 因子参数设定
#默认所有因子及其参数
ALL_FACTORS = OrderedDict([
    # key: 因子名称；value: (函数, {参数名: 参数值})
    ('bollinger_r',   (bollinger_r,   {'window': 20, 'num_std_upper': 3, 'num_std_lower': 3})),
    ('bollinger_MOM', (bollinger_MOM, {'window': 20, 'num_std_upper': 1, 'num_std_lower': 1})),
    ('roc_MOM',       (roc_MOM,       {'window': 10})),
    ('continuous',    (continuous_signal, {'window': 10})),
    ('WMA',           (generate_ma_signal, {'short_window': 5,  'long_window': 20, 'ma_type': 'WMA'})),
    ('EXPWMA',        (generate_ma_signal, {'short_window': 5,  'long_window': 10, 'ma_type': 'EXPWMA'})),
    ('Kaufman',       (generate_ma_signal, {'short_window': 3,  'long_window': 5,  'ma_type': 'Kaufman'})),
    ('MESA_Adaptive', (generate_ma_signal, {'short_window': 3,  'long_window': 20, 'ma_type': 'MESA_Adaptive',
                                        'fastlimit': 0.1,  'slowlimit': 0.6})),
    ('TRIX',          (generate_ma_signal, {'short_window': 5,  'long_window': 20, 'ma_type': 'TRIX','vfactor':0.7})),
    ('MACD',          (macd_signal,   {'short_window': 7,  'long_window': 14, 'signalperiod': 6})),
    ('roc_r',         (roc_r,         {'window': 20})),
    ('RSI',           (rsi_r,         {'window': 20, 'upper': 70, 'lower': 30, 'middle': 50})),
    ('CMO',           (cmo_r,         {'window': 20, 'upper': 50, 'lower': -50, 'middle': 0})),
    ('continuous_r',  (continuous_r,  {'window': 10})),
    ('quantile',      (quantile_signal, {'window': 10})),
])

#针对特定组合，覆盖要跑的因子 & 参数
PAIR_FACTORS = {
    # IFIH: 多沪深300、空上证50
    'IFIH': OrderedDict([
        ('CMO',           (cmo_r,              {'window':20, 'upper':60,  'lower':-60, 'middle':0})),
        ('RSI',           (rsi_r,              {'window':20, 'upper':80,  'lower':20,  'middle':50})),
        ('continuous',    (continuous_signal,  {'window':10})),
        ('MESA_Adaptive', (generate_ma_signal, {'short_window':3, 'long_window':20, 'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.1, 'slowlimit':0.6})),
        ('TRIX',          (generate_ma_signal, {'short_window':5, 'long_window':20, 'ma_type':'TRIX'})),
        ('bollinger_r',   (bollinger_r,        {'window':20,'num_std_upper':3,'num_std_lower':3})),
        ('EXPWMA',        (generate_ma_signal, {'short_window':5,'long_window':10,'ma_type':'EXPWMA'})),
        ('roc_MOM',       (roc_MOM,            {'window':10})),
        ('WMA',           (generate_ma_signal, {'short_window':5,'long_window':20,'ma_type':'WMA','vfactor':1})),
        ('Kaufman',       (generate_ma_signal, {'short_window':3,'long_window':5,'ma_type':'Kaufman'})),
        ('MACD',          (macd_signal,        {'short_window':7,'long_window':14,'signalperiod':6})),
    ]),
    # ICIF: 多中证500、空沪深300
    'ICIF': OrderedDict([
        ('Kaufman',       (generate_ma_signal, {'short_window':3,  'long_window':5,  'ma_type':'Kaufman'})),
        ('TRIX_1',        (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'TRIX', 'vfactor':1})),
        ('TRIX_2',        (generate_ma_signal, {'short_window':5,  'long_window':20, 'ma_type':'TRIX', 'vfactor':1})),
        ('MESA_Adaptive1',(generate_ma_signal, {'short_window':3,  'long_window':20,  'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.3,  'slowlimit':0.4})),
        ('MESA_Adaptive2',(generate_ma_signal, {'short_window':3,  'long_window':20,  'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.1,  'slowlimit':0.6})),
        ('DoubleMA',      (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'DoubleMA'})),
        ('MidPoint',      (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'MidPoint'})),
        ('WMA',           (generate_ma_signal, {'short_window':5,  'long_window':20, 'ma_type':'WMA'})),
        ('bollinger_MOM', (bollinger_MOM,      {'window':20,'num_std_upper':1,'num_std_lower':3})),
        ('CMO',           (cmo_r,              {'window':40,'upper':40,'lower':-40,'middle':0})),
        ('RSI',           (rsi_r,              {'window':40,'upper':70,'lower':30,'middle':50})),
    ]),
    # IMIC: 多中证1000、空中证500
    'IMIC': OrderedDict([
        ('MESA_Adaptive1',(generate_ma_signal, {'short_window':3,  'long_window':20,  'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.5,  'slowlimit':0.03})),
        ('MESA_Adaptive2',(generate_ma_signal, {'short_window':3,  'long_window':20,  'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.5,  'slowlimit':0.4})),
        ('TRIX_1',        (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'TRIX', 'vfactor':0.6})),
        ('TRIX_2',        (generate_ma_signal, {'short_window':3,  'long_window':5,  'ma_type':'TRIX', 'vfactor':0.1})),
        ('DoubleMA',      (generate_ma_signal, {'short_window':10, 'long_window':20, 'ma_type':'DoubleMA'})),
        ('Kaufman',       (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'Kaufman'})),
        ('EXPWMA',        (generate_ma_signal, {'short_window':5,  'long_window':20, 'ma_type':'EXPWMA'})),
        ('MidPoint',      (generate_ma_signal, {'short_window':5,  'long_window':10, 'ma_type':'MidPoint'})),
        ('MACD',          (macd_signal,        {'short_window':7,  'long_window':14, 'signalperiod':5})),
        ('continuous',    (continuous_signal,  {'window':3})),
        ('roc_MOM',       (roc_MOM,            {'window':10})),
        ('bollinger_MOM', (bollinger_MOM,      {'window':20,'num_std_upper':1,'num_std_lower':1})),
    ]),
    # IMIH: 多中证1000、空上证50
    'IMIH': OrderedDict([
        ('TRIX_1',        (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'TRIX', 'vfactor':1})),
        ('TRIX_2',        (generate_ma_signal, {'short_window':5,  'long_window':10, 'ma_type':'TRIX', 'vfactor':0.6})),
        ('WMA_1',         (generate_ma_signal, {'short_window':10, 'long_window':20, 'ma_type':'WMA'})),
        ('WMA_2',         (generate_ma_signal, {'short_window':5,  'long_window':20, 'ma_type':'WMA'})),
        ('DoubleMA',      (generate_ma_signal, {'short_window':3,  'long_window':20, 'ma_type':'DoubleMA'})),
        ('Kaufman',       (generate_ma_signal, {'short_window':3,  'long_window':5,  'ma_type':'Kaufman'})),
        ('CMO',           (cmo_r,              {'window':40,'upper':40,'lower':-40,'middle':0})),
        ('RSI',           (rsi_r,              {'window':40,'upper':70,'lower':30,'middle':50})),
        ('MESA_Adaptive', (generate_ma_signal, {'short_window':3,  'long_window':20,  'ma_type':'MESA_Adaptive',
                                            'fastlimit':0.3,  'slowlimit':0.2})),
        ('WMA_3',         (generate_ma_signal, {'short_window':3,  'long_window':40, 'ma_type':'WMA'})),
        ('roc_MOM',       (roc_MOM,            {'window':10})),
    ]),
}


