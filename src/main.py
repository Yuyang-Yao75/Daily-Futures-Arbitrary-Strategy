from pybroker import Strategy, StrategyConfig
from strategy_utils import *
from backtest_utils import *
from picture_utils import *

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
            - "season"：季节性策略，仅使用季节性信号；
            - "technical"：技术指标策略，仅使用技术指标信号；
            - "concat"：融合策略，同时考虑基差、季节性、技术指标。
    """
    # Step 1:加载数据
    # 获取指数数据
    # index_data = get_stock_index_data(STOCK_INDEX, START_DATE, END_DATE)
    index_data = pd.read_csv(INDEX_DATA)
    index_data['date'] = pd.to_datetime(index_data['date'])
    index_data.set_index('date', inplace=True)

    # 获取期货数据
    # future_data = get_futures_data(STOCK_INDEX, START_DATE, END_DATE)
    futures_data = pd.read_csv(FUTURES_DATA)
    futures_data['date'] = pd.to_datetime(futures_data['date'])
    futures_data.set_index('date', inplace=True)

    # 计算指数和期货的累计净值数据
    index_nv_data=get_nv_data(index_data,"index")
    futures_nv_data=get_nv_data(futures_data,"futures")


    # Step 2:因子计算
    if strategy_name == "concat":
        basis_raw_df = calculate_basis_signal(index_data, futures_data, symbol)
        season_raw_df = calculate_seasonal_signal(index_nv_data, symbol)
        technical_raw_df = calculate_technical_signal(index_nv_data, symbol)
        raw_signal = extract_concat_signals(futures_nv_data,symbol,basis_raw_df,season_raw_df,technical_raw_df)

    if strategy_name == "basis":
        raw_signal = calculate_basis_signal(index_data, futures_data, symbol)
    elif strategy_name == "season":
        raw_signal = calculate_seasonal_signal(index_nv_data, symbol)
    elif strategy_name == "technical":
        raw_signal = calculate_technical_signal(index_nv_data, symbol)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")

    processed_signal = extract_signals(futures_nv_data,symbol,raw_signal,strategy_name)
    signal_path = f"{SIGNAL_DATA_PATH}/{symbol}_{strategy_name}_signal.csv"

    # Step 3:执行策略与回测
    csv_data_source = CSVDataSource(csv_file=signal_path)
    # 初始资金10000元，每年交易日期设置为252，最后一个交易日bar平仓
    config = StrategyConfig(initial_cash=10_000, bars_per_year=252, exit_on_last_bar=True)
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
    run_strategy(symbol="IMIH",strategy_name="basis")