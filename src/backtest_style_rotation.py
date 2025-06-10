import pandas as pd
import pybroker
from pybroker.data import DataSource
import numpy as np
from pybroker import Strategy, StrategyConfig
from sqlalchemy.util import symbol
from tqdm import tqdm


# 添加数据源
class CSVDataSource(DataSource):

    def __init__(self, csv_file='IFIH_signal.csv'):
        super().__init__()
        self.csv_file = csv_file
        # Register custom columns in the CSV.
        pybroker.register_columns('position_signal')

    def _fetch_data(self, symbols, start_date, end_date, _timeframe, _adjust):
        df = pd.read_csv(self.csv_file)
        df = df[df['symbol'].isin(symbols)]
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# 定义策略
# 最终仓位调整信号
def position_signal_trade(ctx):

    if not ctx.long_pos() and not ctx.short_pos():
        target_pos = ctx.position_signal[-1]
        if  target_pos > 0:
            target_shares = ctx.calc_target_shares(target_pos)
            ctx.buy_shares = target_shares
        elif target_pos < 0:
            target_shares = ctx.calc_target_shares(-target_pos)
            ctx.sell_shares = target_shares
    # 持有多头时
    elif ctx.long_pos():
        pos = ctx.long_pos()
        # 目标仓位
        target_pos = ctx.position_signal[-1]
        # 计算仓位份额
        target_shares = ctx.calc_target_shares(abs(target_pos)) # 空仓仓位配置有问题，需要更改
        # 目标仓位大于0
        if target_pos > 0:
            if target_shares > pos.shares:
                # 加仓
                ctx.buy_shares = target_shares - pos.shares
            elif target_shares < pos.shares:
                # 减仓
                ctx.sell_shares = pos.shares - target_shares
        # 目标仓位小于0
        elif target_pos < 0:
            # 空仓
            ctx.sell_shares = pos.shares + target_shares
        else:
            # 平多
            ctx.sell_all_shares()

    elif ctx.short_pos():
        pos = ctx.short_pos()
        target_pos = ctx.position_signal[-1]  # 目标仓位
        target_shares = ctx.calc_target_shares(abs(target_pos))  # 计算仓位份额
        if target_pos < 0:
            if target_shares < pos.shares:
                ctx.buy_shares = pos.shares - target_shares
            elif target_shares > pos.shares:
                ctx.sell_shares = target_shares - pos.shares
        elif target_pos > 0:
            ctx.buy_shares = target_shares + pos.shares
        else:
            ctx.cover_all_shares()



if __name__ == '__main__':

    # 这里是回测的代码，symbol为["IFIH", "ICIF", "IMIC", "IMIH"]
    # "_basis_signal.csv"（基差信号）可以更换为
    # "_concat_signal.csv"（最终组合信号）,
    # "_season_signal.csv"（季节性轮动信号）,
    # "_futures_signal.csv"（股指期货动量信号）,
    # "_signal.csv"（指数动量信号）
    symbol = "IMIH"
    csv_data_source = CSVDataSource(csv_file=symbol + "_basis_signal.csv")
    # 初始资金10000元，每年交易日期设置为252，最后一个交易日bar平仓
    config = StrategyConfig(initial_cash=10_000, bars_per_year=252, exit_on_last_bar=True)
    strategy = Strategy(csv_data_source, start_date='12/1/2023', end_date='3/1/2025', config=config)
    # 添加交易策略position_signal_trade和标的symbol
    strategy.add_execution(position_signal_trade, symbol)
    result = strategy.backtest()
    # 回测的结果指标
    # '_trades.csv'(回测期间交易信息)
    # '_metrics.csv'(回测的结果各类指标)
    # ‘_positions.csv'（回测期间的每日持仓）
    # '_portfolio.csv'(资产组合的信息)
    result.trades.to_csv(symbol + '_basis_trades.csv')
    result.metrics_df.to_csv(symbol + '_basis_metrics.csv')
    result.positions.to_csv(symbol + '_basis_positions.csv')
    result.portfolio.to_csv(symbol + '_basis_portfolio.csv')


    #############回测数据预处理##############
    ################指数动量信号整合#####################
    # from strategy import *
    #
    # # df = pd.read_csv('index_nv_data_20100101_20250216.csv', usecols=["time", "IFIH_index_nv"])
    # # df = pd.read_csv('index_nv_data_20100101_20250216.csv', usecols=["time", "ICIF_index_nv"])
    # # df = pd.read_csv('index_nv_data_20100101_20250216.csv', usecols=["time", "IMIC_index_nv"])
    # # df = pd.read_csv('index_nv_data_20100101_20250216.csv', usecols=["time", "IMIH_index_nv"])
    # df = pd.read_csv("IFIH_futures_signal.csv")
    # # df = df.rename(columns={"time": "date", "IMIH_index_nv": "close"})
    # # df["symbol"] = "IFIH"
    # # df["symbol"] = "ICIF"
    # # df["symbol"] = "IMIC"
    # # df["symbol"] = "IMIH"
    # # df["open"] = df["close"] - 0.01
    # # df["high"] = df["close"] + 0.01
    # # df["low"] = df["close"] - 0.02
    # # df = df[["date", "symbol", "open", "high", "low", "close"]]
    # # df['date'] = pd.to_datetime(df['date'])
    # # df = df[df["date"] >= "2016-1-1"].reset_index(drop=True)
    #
    # df["bollinger_r_signal"] = bollinger_r(price=df["close"], window=20, num_std_upper=3, num_std_lower=3)
    # # df["bollinger_MOM_signal"] = bollinger_MOM(price=df["close"], window=20, num_std_upper=1, num_std_lower=1)
    # df["roc_MOM_signal"] = roc_MOM(price=df["close"], window=10)
    # df["continuous_MOM_signal"] = continuous_signal(price=df["close"], window=10)
    # # df["continuous_r_signal"] = continuous_r(price=df["close"], window=10)
    # # df["quantile_signal"] = quantile_signal(price=df["close"], window=10)
    # #
    # # df["DoubleMa_signal"] = generate_ma_signal(price=df["close"], short_window=3, long_window=20, ma_type="DoubleMA")
    # df["WMA_signal"] = generate_ma_signal(price=df["close"], short_window=5, long_window=20, ma_type="WMA")
    # df["EXPWMA_signal"] = generate_ma_signal(price=df["close"], short_window=5, long_window=10, ma_type="EXPWMA")
    # # df["Hilbert_Transform_signal"] = generate_ma_signal(price=df["close"], short_window=3, long_window=20, ma_type="Hilbert_Transform")
    # df["Kaufman_signal"] = generate_ma_signal(price=df["close"], short_window=3, long_window=5, ma_type="Kaufman")
    # df["MESA_Adaptive_signal"] = generate_ma_signal(price=df["close"], short_window=3, long_window=20, ma_type="MESA_Adaptive", fastlimit=0.1, slowlimit=0.6)
    # # df["MidPoint_signal"] = generate_ma_signal(price=df["close"], short_window=5, long_window=10, ma_type="MidPoint")
    # df["TRIX_signal"] = generate_ma_signal(price=df["close"], short_window=5, long_window=20, ma_type="TRIX")
    # df["MACD_signal"] = macd_signal(price=df["close"], short_window=7, long_window=14, signalperiod=6)
    # #
    # df["CMO_signal"] = cmo_r(price=df["close"], window=20)
    # df["RSI_signal"] = rsi_r(price=df["close"], window=20)
    # # df["ROC_r_signal"] = roc_r(price=df["close"], window=20)
    #
    # df["position_signal"] = (df["bollinger_r_signal"]+df["roc_MOM_signal"]+df["continuous_MOM_signal"]+\
    #                          df["WMA_signal"]+df["EXPWMA_signal"]+df["Kaufman_signal"]+df["MESA_Adaptive_signal"]+\
    #                          df["TRIX_signal"]+df["MACD_signal"]+df["CMO_signal"]+df["RSI_signal"]) / 11
    # # df["position_signal"] = (df["bollinger_MOM_signal"]+df["DoubleMa_signal"]+df["WMA_signal"]+\
    # #                          df["Kaufman_signal"]+df["MESA_Adaptive_signal"]+df["TRIX_signal"]+df["CMO_signal"]+\
    # #                          df["RSI_signal"]) / 8
    #
    # # df["position_signal"] = (df["roc_MOM_signal"]+df["DoubleMa_signal"]+df["WMA_signal"]+\
    # #                          df["Kaufman_signal"]+df["MESA_Adaptive_signal"]+df["TRIX_signal"]+df["CMO_signal"]+\
    # #                          df["RSI_signal"]) / 8
    # # df.to_csv("IMIH_signal.csv", index=False)
    # # df.to_csv("IMIC_signal.csv", index=False)
    # # df.to_csv("ICIF_signal.csv", index=False)
    # df.to_csv("IFIH_futures_signal.csv", index=False)

    ####################期货多空组合仓位信息整合######################

    # pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    # for pair in pairs:
    #     df_signal = pd.read_csv(pair+ '_signal.csv')
    #     df_futures_nv = pd.read_csv('futures_nv_data_2023-12-14_2025-02-16.csv')
    #     df_futures_nv.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    #
    #     df_signal.index = pd.to_datetime(df_signal["date"])
    #     df_futures_nv.index = pd.to_datetime(df_futures_nv["date"])
    #     df_futures_merge = pd.merge(df_futures_nv, df_signal, how='left', left_index=True, right_index=True)
    #     df_futures_signal = df_futures_merge[["date_x", "{}_futures_nv".format(pair), "position_signal"]]
    #     df_futures_signal.rename(columns={"date_x": "date",
    #                                       "{}_futures_nv".format(pair): "close"}, inplace=True)
    #     df_futures_signal["symbol"] = pair
    #     df_futures_signal["open"] = df_futures_signal["close"] - 0.01
    #     df_futures_signal["high"] = df_futures_signal["close"] + 0.01
    #     df_futures_signal["low"] = df_futures_signal["close"] - 0.02
    #     df_futures_signal = df_futures_signal[["date", "symbol", "open", "high", "low", "close", "position_signal"]]
    #     df_futures_signal['date'] = pd.to_datetime(df_futures_signal['date'])
    #     df_futures_signal.to_csv("{}_futures_signal.csv".format(pair), index=False)

    ##################基差仓位信号整合#######################
    # 基差仓位信息
    # basis_signal = pd.read_csv("basis_signal.csv")
    # pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    # for pair in pairs:
    #     # 期货价格信息
    #     df_futures = pd.read_csv("{}_futures_signal.csv".format(pair))
    #     df_basis_signal = pd.merge(basis_signal, df_futures, how="left", on="date")
    #     df_basis_signal = df_basis_signal[['date', 'symbol', 'open', 'high', 'low', 'close', 'signal_{}_position'.format(pair)]]
    #     df_basis_signal.rename(columns={'signal_{}_position'.format(pair): 'position_signal'}, inplace=True)
    #     df_basis_signal.to_csv("{}_basis_signal.csv".format(pair), index=False)
    ######################组合季节效应轮动的仓位信号##############################
    # pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    #
    # for pair in pairs:
    #     df_season_signal = pd.read_csv("season_signal_{}.csv".format(pair))
    #     df_season_signal["position_signal"] = df_season_signal["signal"] * df_season_signal["weight"]
    #     signal_dict = df_season_signal.set_index('month')['position_signal'].to_dict()
    #     df_price = pd.read_csv("{}_signal.csv".format(pair))
    #     df_price["date"] = pd.to_datetime(df_price["date"])
    #     df_price["month"] = df_price["date"].dt.month
    #     df_price['season_signal'] = df_price['month'].map(signal_dict)
    #     df_price = df_price[['date', 'symbol', 'open', 'high', 'low', 'close', 'season_signal']]
    #     df_price.rename(columns={"season_signal": "position_signal"}, inplace=True)
    #     df_price.to_csv("{}_season_signal.csv".format(pair), index=False)
        # print(df_price.head())
        # break
    #######################合成最终信号###########################
    # pairs = ["IFIH", "ICIF", "IMIC", "IMIH"]
    # for pair in pairs:
    #     df_basis_signal = pd.read_csv("{}_basis_signal.csv".format(pair))
    #     df_basis_signal["date"] = pd.to_datetime(df_basis_signal["date"])
    #     start_date = df_basis_signal["date"].min()
    #     end_date = df_basis_signal["date"].max()
    #     df_basis_signal.set_index(['date'], inplace=True)
    #     df_basis_signal.rename(columns={"position_signal": "basis_signal"}, inplace=True)
    #
    #     df_mom_signal = pd.read_csv("{}_signal.csv".format(pair))
    #     df_mom_signal["date"] = pd.to_datetime(df_mom_signal["date"])
    #     df_mom_signal = df_mom_signal[(df_mom_signal["date"] >= start_date) & (df_mom_signal["date"] <= end_date)]
    #     df_mom_signal.set_index('date', inplace=True)
    #     df_mom_signal.rename(columns={"position_signal": "mom_signal"}, inplace=True)
    #
    #     df_season_signal = pd.read_csv("{}_season_signal.csv".format(pair))
    #     df_season_signal["date"] = pd.to_datetime(df_season_signal["date"])
    #     df_season_signal = df_season_signal[(df_season_signal["date"] >= start_date) & (df_season_signal["date"] <= end_date)]
    #     df_season_signal.set_index('date', inplace=True)
    #     df_season_signal.rename(columns={"position_signal": "season_signal"}, inplace=True)
    #
    #     df_basis_signal_columns = set(df_basis_signal.columns)
    #     df_mom_signal = df_mom_signal[[col for col in df_mom_signal.columns if col not in df_basis_signal_columns]]
    #     df_season_signal = df_season_signal[[col for col in df_season_signal.columns if col not in df_basis_signal_columns]]
    #     df_concat = pd.concat([df_basis_signal, df_mom_signal, df_season_signal], axis=1)
    #     df_concat = df_concat[["symbol", 'open', 'high', 'low', 'close', "basis_signal", "mom_signal", "season_signal"]]
    #     df_concat.loc[:, "Position0"] = (df_concat.loc[:, "basis_signal"] * 0.2 + df_concat.loc[:, "mom_signal"] * 0.8) * \
    #                                  (1 - abs(df_concat.loc[:, "season_signal"])) + df_concat.loc[:, "season_signal"]
    #
    #     # 定义条件和对应的值
    #     conditions = [
    #         df_concat['Position0'] >= 1,
    #         (df_concat['Position0'] >= 0.5) & (df_concat['Position0'] < 1),
    #         (df_concat['Position0'] >= 0.2) & (df_concat['Position0'] < 0.5),
    #         (df_concat['Position0'] > -0.2) & (df_concat['Position0'] < 0.2),
    #         (df_concat['Position0'] >= -0.5) & (df_concat['Position0'] < -0.2),
    #         (df_concat['Position0'] >= -1) & (df_concat['Position0'] < -0.5),
    #         df_concat['Position0'] <= -1
    #     ]
    #
    #     values = [1.5, 1, 0.5, 0, -0.5, -1, -1.5]
    #
    #     # 应用条件创建新列
    #     df_concat['position_signal'] = np.select(conditions, values)
    #     df_concat.to_csv("{}_concat_signal.csv".format(pair), index=True)