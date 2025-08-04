from pybroker import Strategy, StrategyConfig
from strategy_utils import *
from backtest_utils import *
from picture_utils import *
from config import INITIAL_CASH, BARS_PER_YEAR, INDEX_START_DATE, START_DATE, END_DATE, RESULT_PATH, AVAILABLE_STRATEGY, AVAILABLE_PAIRS

def run_cross_section_strategy():
    futures_data = pd.read_csv(FUTURES_DATA)
    futures_data['date'] = pd.to_datetime(futures_data['date'])
    futures_data.set_index('date', inplace=True)

    symbols = ["IC", "IH", "IM", "IF"]
    for symbol in symbols:
        pct_chg_col = f"{symbol}_PCT_CHG"
        vol_col = f"{symbol}_vol"
        futures_data[vol_col] = futures_data[pct_chg_col].rolling(window=60).std()

    futures_data = futures_data.dropna()

    inv = {sym: 1 / futures_data[f'{sym}_vol'] for sym in symbols}
    inv_df = pd.DataFrame(inv)

    norm_inv = inv_df.div(inv_df.sum(axis=1), axis=0)

    for symbol in symbols:
        yr_basis_col = f'{symbol}_ANAL_BASISANNUALYIELD'
        futures_data[f'{symbol}_adjusted_yrbasis'] = norm_inv[symbol] * futures_data[yr_basis_col]

    yrb_cols = [f'{sym}_ANAL_BASISANNUALYIELD' for sym in symbols]
    open_cols = [f'{sym}_OPEN' for sym in symbols]
    adj_cols = [f'{sym}_adjusted_yrbasis' for sym in symbols]

    futures_data.reset_index(inplace=True)
    # 为每个 symbol 生成一张小表，然后 concat
    frames = []
    for sym in symbols:
        tmp = futures_data[['date',
                            f'{sym}_OPEN',
                            f'{sym}_ANAL_BASISANNUALYIELD',
                            f'{sym}_adjusted_yrbasis']].copy()
        tmp['symbol'] = sym
        tmp = tmp.rename(columns={
            f'{sym}_OPEN': 'open',
            f'{sym}_ANAL_BASISANNUALYIELD': 'yrbasis',
            f'{sym}_adjusted_yrbasis': 'adjusted_yrbasis'
        })
        frames.append(tmp)

    df_final = pd.concat(frames, ignore_index=True)

    # 重排一下列顺序
    df_final = df_final[['date', 'symbol', 'open', 'yrbasis', 'adjusted_yrbasis']]

def run_strategy(symbol="IMIH",strategy_name="basis"):
    """
    执行指定量化策略的完整流程，包括数据加载、因子计算、信号提取、回测执行及结果保存。

    参数:
    ----------
    symbol : str, 默认 "IMIH"
        表示组合对的代码，如 IMIH、IFIC 等，用于确定要回测的品种组合。

    strategy_name : str, 默认 "basis"
        策略名称，可选：
            - "basis"：基差策略，仅使用基差信号；
            - "seasonal"：季节性策略，仅使用季节性信号；
            - "technical"：技术指标策略，仅使用技术指标信号；
            - "concat"：融合策略，同时考虑基差、季节性、技术指标。
    """
    # Step 1:加载数据
    # 获取指数数据
    # index_data = get_stock_index_data(STOCK_INDEX, INDEX_START_DATE, END_DATE)#todo
    index_data = pd.read_csv(INDEX_DATA)
    index_data['date'] = pd.to_datetime(index_data['date'])
    index_data.set_index('date', inplace=True)

    # 获取期货数据
    # futures_data = get_futures_data(STOCK_INDEX, START_DATE, END_DATE)#todo
    futures_data = pd.read_csv(FUTURES_DATA)
    futures_data['date'] = pd.to_datetime(futures_data['date'])
    futures_data.set_index('date', inplace=True)

    # 计算指数和期货的累计净值数据
    index_nv_data=get_nv_data(index_data,"index")
    futures_nv_data=get_nv_data(futures_data,"futures")


    # Step 2:因子计算
    if strategy_name == "concat":
        basis_raw_df = calculate_basis_signal(index_data, futures_data, symbol)
        seasonal_raw_df = calculate_seasonal_signal(index_nv_data, symbol)
        technical_raw_df = calculate_technical_signal(index_nv_data, symbol)
        raw_signal = extract_concat_signals(futures_nv_data,symbol,basis_raw_df,seasonal_raw_df,technical_raw_df)
    elif strategy_name == "basis":
        raw_signal = calculate_basis_signal(index_data, futures_data, symbol)
    elif strategy_name == "seasonal":
        raw_signal = calculate_seasonal_signal(index_nv_data, symbol)
    elif strategy_name == "technical":
        raw_signal = calculate_technical_signal(index_nv_data, symbol)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    processed_signal = extract_signals(futures_nv_data,symbol,raw_signal,strategy_name)
    signal_path = f"{RESULT_PATH}/{symbol}_{strategy_name}_signal.csv"

    # Step 3:执行策略与回测
    csv_data_source = CSVDataSource(csv_file=signal_path)
    # 初始资金10000元，每年交易日期设置为252，最后一个交易日bar平仓
    config = StrategyConfig(initial_cash=INITIAL_CASH, bars_per_year=BARS_PER_YEAR, exit_on_last_bar=True)
    strategy = Strategy(csv_data_source, start_date=START_DATE.strftime("%m/%d/%Y"), end_date=END_DATE.strftime("%m/%d/%Y"), config=config)
    # 添加交易策略position_signal_trade和标的symbol
    strategy.add_execution(position_signal_trade, symbol)
    result = strategy.backtest()
    # 回测的结果指标
    # '_trades.csv'(回测期间交易信息)
    result.trades.to_csv(f'{RESULT_PATH}/{symbol}_{strategy_name}_trades.csv')
    # '_metrics.csv'(回测的结果各类指标)
    result.metrics_df.to_csv(f'{RESULT_PATH}/{symbol}_{strategy_name}_metrics.csv')
    # ‘_positions.csv'（回测期间的每日持仓）
    result.positions.to_csv(f'{RESULT_PATH}/{symbol}_{strategy_name}_positions.csv')
    # '_portfolio.csv'(资产组合的信息)
    result.portfolio.to_csv(f'{RESULT_PATH}/{symbol}_{strategy_name}_portfolio.csv')

    plot_trade_nv(f'{symbol}_{strategy_name}_portfolio.csv', f'{symbol}_{strategy_name}_signal.csv')

if __name__ == "__main__":
    # #========= 遍历回测 ==========
    # for symbol in AVAILABLE_PAIRS:
    #     for strategy_name in AVAILABLE_STRATEGY:
    #         print(f"...正在对{symbol}执行{strategy_name}策略")
    #         run_strategy(symbol=symbol,strategy_name=strategy_name)

    #========= 整合绘图 ==========#todo 后续可以融合到 run_strategy 中
    for symbol in AVAILABLE_PAIRS:
        portfolio_files = {
            "合成信号_仓位调整": f"{symbol}_concat_portfolio.csv",
            "basis_position": f"{symbol}_basis_portfolio.csv",
            "tech_position": f"{symbol}_technical_portfolio.csv",
            "season_position": f"{symbol}_seasonal_portfolio.csv",
        }
        plot_multiple_strategies(portfolio_files)