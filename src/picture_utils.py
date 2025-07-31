import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from config import RESULT_PATH,SIGNAL_DATA_PATH
# plt.rcParams['font.sans-serif'] = ['simsun']  # 中文字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FZKai-Z03S']
mpl.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size': 16})

def plot_trade_nv(portfolio_csv, position_csv):
    """
    根据给定的 portfolio 和 position 数据绘制净值曲线和仓位信号图。
    在一张图里：
    - 左轴：净值曲线（红色折线，归一化到首行）
    - 右轴：仓位信号（浅蓝色柱状图）
    并且：
    - 对 position 信号做了从 2016-01-01 起的筛选
    - 在 y=0 处画了灰色虚线
    - 自动保存成 “XXXX_concat_curve.png”

    参数：
    portfolio_csv (str): 包含 market_value 数据的 CSV 文件路径。
    position_csv (str): 包含 position_signal 数据的 CSV 文件路径。

    返回：
    None
    """
    matplotlib.rcParams.update({'font.size': 16})
    # 提取第一个下划线前的部分（symbol）
    symbol = portfolio_csv.split('_')[0]  # 'IFIH'
    # 提取第一个和第二个下划线之间的部分（strategy）
    strategy = portfolio_csv.split('_')[1]  # 'futures'
    # 读取 CSV 文件
    portfolio_df = pd.read_csv(f"{RESULT_PATH}/{portfolio_csv}")
    position_df = pd.read_csv(f"{SIGNAL_DATA_PATH}/{position_csv}")
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
    plt.title(symbol + " " + strategy + " Backtest Result")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    fig.tight_layout()

    # 保存图像到指定文件
    plt.savefig(f"RESULT_PATH/{symbol}_{strategy}_curve.png", dpi=300)
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