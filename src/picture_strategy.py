import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['simsun'] # 显示中文字体
def plot_trade_nv(portfolio_csv, position_csv):
    """
    根据给定的 portfolio 和 position 数据绘制净值曲线和仓位信号图。

    参数：
    portfolio_csv (str): 包含 market_value 数据的 CSV 文件路径。
    position_csv (str): 包含 position_signal 数据的 CSV 文件路径。

    返回：
    None
    """
    matplotlib.rcParams.update({'font.size': 16})

    # 读取 CSV 文件
    portfolio_df = pd.read_csv(portfolio_csv)
    position_df = pd.read_csv(position_csv)
    position_df = position_df[position_df['date'] >= '2016-01-01']
    # 确保 time 列是日期格式
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    position_df['date'] = pd.to_datetime(position_df['date'])

    # 计算 market_value 的净值（相对第一行）
    portfolio_df['market_value_nv'] = portfolio_df['market_value'] / portfolio_df['market_value'].iloc[0]

    # 创建画布和双坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    # 绘制 market_value 净值曲线（红色折线）
    ax1.plot(portfolio_df['date'], portfolio_df['market_value_nv'], color='red', label='Market Value (NV)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Market Value (NV)', color='red')
    ax1.tick_params(axis='y', colors='red')
    # ax1.set_ylim(0.95, 1.9)

    # 绘制 position_signal 柱状图
    ax2.bar(position_df['date'], position_df['position_signal'], color='lightblue', label='Position Signal',
            alpha=0.6)
    ax2.set_ylabel('Position Signal', color='black')
    ax2.tick_params(axis='y', colors='black')
    # ax2.set_ylim(-0.5, 0.5)

    # 添加纵坐标网格线
    # ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    # 设置标题和图例
    plt.title(portfolio_csv[:4] + " concat Backtest Result")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    fig.tight_layout()

    # 保存图像到指定文件
    plt.savefig(portfolio_csv[:4]+ "_concat_curve.png", dpi=300)
    plt.show()

def plot_concat_nv(*args):
    result = []
    for arg in args:
        df = pd.read_csv(arg)
        df['market_value_nv_{}'.format(arg[:4])] = df['market_value'] / df['market_value'].iloc[0]
        df.set_index('date', inplace=True)
        result.append(df[["market_value_nv_{}".format(arg[:4])]])
    df_concat = pd.concat(result, axis=1)
    df_concat["concat_nv"] = df_concat["market_value_nv_IFIH"] * 0.25 + df_concat["market_value_nv_ICIF"] * 0.25 + \
        df_concat["market_value_nv_IMIC"] * 0.25 + df_concat["market_value_nv_IMIH"] * 0.25
    df_concat["date"] = pd.to_datetime(df_concat.index)
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制净值曲线
    ax.plot(df_concat["date"], df_concat["concat_nv"], linewidth=2, color='red')

    # 设置标题和标签
    plt.title("组合净值曲线", fontsize=16)
    plt.ylabel("净值", fontsize=14)
    plt.xlabel('日期', fontsize=14)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 自动旋转日期标签以避免重叠
    plt.gcf().autofmt_xdate()

    # 设置y轴从1或最小值开始
    y_min = min(1.0, df_concat["concat_nv"].min() * 0.95)
    y_max = df_concat["concat_nv"].max() * 1.05
    plt.ylim(y_min, y_max)

    # 添加图例
    plt.legend(["组合净值"], fontsize=14)

    # 紧凑布局
    plt.tight_layout()
    plt.savefig("concat_nv_curve.png", dpi=300)
    plt.show()





if __name__ == '__main__':

    # 这部分代码是画图用的，'IFIH_portfolio.csv'是IFIH的指数动量回测结果，'IFIH_signal.csv'是IFIH的指数动量策略信号
    # plot_trade_nv('IFIH_portfolio.csv', 'IFIH_signal.csv')
    # plot_trade_nv('ICIF_portfolio.csv', 'ICIF_signal.csv')
    # plot_trade_nv('IMIC_portfolio.csv', 'IMIC_signal.csv')
    # plot_trade_nv('IMIH_portfolio.csv', 'IMIH_signal.csv')

    # 'IFIH_futures_portfolio.csv'是IFIH的期货动量回测结果，'IFIH_futures_signal.csv'是IFIH的期货动量策略信号
    # plot_trade_nv('IFIH_futures_portfolio.csv', 'IFIH_futures_signal.csv')
    # plot_trade_nv('ICIF_futures_portfolio.csv', 'ICIF_futures_signal.csv')
    # plot_trade_nv('IMIC_futures_portfolio.csv', 'IMIC_futures_signal.csv')
    # plot_trade_nv('IMIH_futures_portfolio.csv', 'IMIH_futures_signal.csv')

    # 'IFIH_basis_portfolio.csv'是IFIH的期货基差回测结果，'IFIH_basis_signal.csv'是IFIH的期货基差策略信号
    # plot_trade_nv('IFIH_basis_portfolio.csv', 'IFIH_basis_signal.csv')
    # plot_trade_nv('ICIF_basis_portfolio.csv', 'ICIF_basis_signal.csv')
    # plot_trade_nv('IMIC_basis_portfolio.csv', 'IMIC_basis_signal.csv')
    # plot_trade_nv('IMIH_futures_portfolio.csv', 'IMIH_futures_signal.csv')

    # 'IFIH_season_portfolio.csv'是IFIH的期货季节性轮动回测结果，'IFIH_season_signal.csv'是IFIH的期货季节性轮动策略信号
    # plot_trade_nv('IFIH_season_portfolio.csv', 'IFIH_season_signal.csv')
    # plot_trade_nv('ICIF_season_portfolio.csv', 'ICIF_season_signal.csv')
    # plot_trade_nv('IMIC_season_portfolio.csv', 'IMIC_season_signal.csv')
    # plot_trade_nv('IMIH_season_portfolio.csv', 'IMIH_season_signal.csv')

    # 'IFIH_concat_portfolio.csv'是IFIH期货组合基差、动量和季节性轮动的回测结果，'IFIH_concat_signal.csv'是IFIH期货组合基差、动量和季节性轮动策略信号
    # plot_trade_nv('IFIH_concat_portfolio.csv', 'IFIH_concat_signal.csv')
    # plot_trade_nv('ICIF_concat_portfolio.csv', 'ICIF_concat_signal.csv')
    # plot_trade_nv('IMIC_concat_portfolio.csv', 'IMIC_concat_signal.csv')
    # plot_trade_nv('IMIH_concat_portfolio.csv', 'IMIH_concat_signal.csv')

    # 绘制最后四个组合品种的净值曲线
    plot_concat_nv('IFIH_concat_portfolio.csv', 'ICIF_concat_portfolio.csv', 'IMIC_concat_portfolio.csv','IMIH_concat_portfolio.csv')