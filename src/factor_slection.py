"""用于因素选择和参数优化的工具。
此模块提供了辅助函数，用于在不同的参数组合下对技术因素进行网格搜索，对每个配置进行 pybroker 回测，并保留具有稳健夏普比率的配置。
所选因素的结果以 JSON 格式保存，以便能够将其在配置文件中重复使用。
"""

import json
import os
from collections import OrderedDict
from itertools import product
from typing import Callable, Dict, Iterable, List, Any

import pandas as pd
from pybroker import Strategy, StrategyConfig

from backtest_utils import CSVDataSource, position_signal_trade
from config import (
    BARS_PER_YEAR,
    END_DATE,
    INITIAL_CASH,
    START_DATE,
    FACTOR_SELECTION_PATH
)
from data_utils import generate_ohlc, get_nv_data  # type: ignore


def _grid(param_space: Dict[str, Iterable]) -> List[Dict]:
    """Expand a parameter grid into a list of combinations."""
    print("正在生成参数组合。")
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    combos: List[Dict] = []
    for values in product(*param_space.values()):
        combos.append(dict(zip(keys, values)))
    return combos

def _run_backtest(signal_df: pd.DataFrame, symbol: str, tag: str) -> pd.DataFrame:
    """Run pybroker backtest on a prepared signal DataFrame."""
    print(f"开始回测 {symbol} {tag}。")
    os.makedirs(FACTOR_SELECTION_PATH, exist_ok=True)
    csv_path = os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_{tag}_signal.csv")
    signal_df.to_csv(csv_path, index=True)

    data_source = CSVDataSource(csv_file=csv_path)
    config = StrategyConfig(
        initial_cash=INITIAL_CASH,
        bars_per_year=BARS_PER_YEAR,
        exit_on_last_bar=True,
    )
    strategy = Strategy(
        data_source,
        start_date=START_DATE.strftime("%m/%d/%Y"),
        end_date=END_DATE.strftime("%m/%d/%Y"),
        config=config,
    )
    strategy.add_execution(position_signal_trade, symbol)
    result = strategy.backtest()

    result.trades.to_csv(os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_{tag}_trades.csv"))
    result.metrics_df.to_csv(os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_{tag}_metrics.csv"))
    return result.metrics_df


def _extract_metric(metrics_df: pd.DataFrame,metric:str) -> float:
    """提取指定指标的值。"""
    print(f"正在提取 {metric} 指标。")
    metric = metric.lower()
    if "name" in metrics_df.columns:
        row = metrics_df[metrics_df["name"].str.lower() == metric]
        if not row.empty:
            return float(row["value"].iloc[0])
    if metric in metrics_df.columns:
        return float(metrics_df[metric].iloc[0])
    if metric in metrics_df.index:
        return float(metrics_df.loc[metric])
    print(f"无法找到 {metric} 指标。")
    return float("nan")


def select_factors(
    index_nv_df: pd.DataFrame,
    symbol: str,
    factor_funcs: Dict[str, Callable[..., pd.Series]],
    param_spaces: Dict[str, Dict[str, Iterable]],
    metrics_to_optimize: List[str]=None,
    metric_thresholds: Dict[str, float]=None,
) -> OrderedDict[str,Any]:
    """执行网格搜索优化和因子筛选。两轮筛选：先基于平均指标筛选因子，再选出每个因子的最优参数。

    参数
    ----------
    index_nv_df : pd.DataFrame
        所有指数的净值数据，必须按日期建立索引。
    symbol : str
        交易对符号，例如 "IFIH"。
    factor_funcs : dict
        因子名称到其计算函数的映射。
    param_spaces : dict
        因子名称到其参数搜索空间的映射。
    sharpe_threshold : float, 可选
        保留因子的最小平均夏普比率，默认为0.5。

    返回值
    -------
    OrderedDict
        选中的因子，格式为 {因子名: (函数, 最佳参数)}。
    """
    print(f"\n{'='*50}")
    print(f"开始因子选择流程 - 交易对: {symbol}")
    # 初始化默认参数
    if metrics_to_optimize is None:
        metrics_to_optimize = ["sharpe"]
    if metric_thresholds is None:
        metric_thresholds = {"sharpe": 0.5}
        
    print(f"待测试因子数量: {len(factor_funcs)}")
    print(f"优化目标指标: {', '.join(metrics_to_optimize)}")
    print(f"指标阈值要求: {metric_thresholds}")
    os.makedirs(FACTOR_SELECTION_PATH, exist_ok=True)
    # 1. 收集所有因子、参数和指标
    records: List[Dict[str, Any]] = []
    print("\n" + "=" * 20 + " 开始网格搜索 " + "=" * 20)
    for name, func in factor_funcs.items():
        grid = _grid(param_spaces.get(name, {}))
        param_count = len(grid)
        print(f"\n处理因子: {name} (共 {param_count} 组参数组合)")
        for i, params_kwargs in enumerate(grid, 1):
            print(f"\n参数组合 {i}/{param_count}: {params_kwargs}")
            # 生成信号并回测
            ohlc = generate_ohlc(index_nv_df, symbol)
            ohlc["position_signal"] = func(ohlc, **params_kwargs)
            ohlc["symbol"] = symbol
            df_signal = ohlc[["open", "high", "low", "close", "position_signal"]].copy()
            df_signal["symbol"] = symbol
            df_signal = df_signal.reset_index()
            df_signal = df_signal[["date", "symbol", "open", "high", "low", "close", "position_signal"]]

            tag = f"{name}_" + "_".join(f"{k}{v}" for k, v in params_kwargs.items())
            print(f"执行回测: {tag}")
            metrics_df = _run_backtest(df_signal.set_index("date"), symbol, tag)
            # 提取所有目标指标
            metric_values = {
                m:_extract_metric(metrics_df, m) for m in metrics_to_optimize
            }
            print(f"指标结果: {metric_values}")
            # 记录一条结果
            records.append({
                "name": name,
                "func": func,
                "params": params_kwargs,
                **metric_values
            })
    print("\n" + "=" * 20 + " 分析结果 " + "=" * 20)
    results_df = pd.DataFrame(records)
    selected: "OrderedDict[str, Any]" = OrderedDict()
    # 2. 平均指标筛选 #todo 后续改进筛选标准时需重构该部分代码
    primary = metrics_to_optimize[0]
    max_df = results_df.groupby("name")[primary].max().reset_index()
    good = max_df[max_df[primary] > metric_thresholds.get(primary, float("-inf"))]["name"]
    print(f"\n初始因子数量: {len(factor_funcs)}")
    print(f"通过{primary}阈值({metric_thresholds.get(primary)})的因子数量: {len(good)}")
    # 3. 按因子筛选最优参数
    print("\n" + "=" * 20 + " 选择最优参数 " + "=" * 20)
    for name in good:
        sub = results_df[results_df["name"] == name]
        best = sub.sort_values(primary, ascending=False).iloc[0]

        print(f"\n因子: {name}")
        print(f"最佳参数: {best['params']}")
        print(f"最佳指标: {{", end="")
        for m in metrics_to_optimize:
            print(f" {m}: {best[m]:.4f}", end="")
        print(" }")

        selected[name] = {
            "func": best["func"],
            "best_params": best["params"],
            "best_metrics": {m: best[m] for m in metrics_to_optimize}
        }

    # （可选）保存中间结果
    save_path = os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_grid_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n保存详细回测结果到: {save_path}")

    # 最后把选中结果写成 JSON
    json_ready = {
        k: {
            "func": v["func"].__name__,
            "params": v["best_params"],
            "metrics": v["best_metrics"],
        }
        for k, v in selected.items()
    }
    json_path = os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_selected_factors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)

    print(f"保存选中因子到: {json_path}")
    print(f"\n{'=' * 50}")
    print(f"因子选择完成! 共选中 {len(selected)} 个因子")
    print("=" * 50 + "\n")

    return selected


if __name__ == "__main__":  # pragma: no cover - usage example
    from config import INDEX_DATA

    index_df = pd.read_csv(INDEX_DATA, parse_dates=["date"]).set_index("date")
    index_nv = get_nv_data(index_df, "index")

    from signal_utils import *

    funcs = {
        'bollinger_r': bollinger_r,
        'bollinger_MOM': bollinger_mom,
        'DoubleMA':generate_ma_signal,
        'EXPWMA':generate_ma_signal,
        'Hilbert_Transform':generate_ma_signal,
        'Kaufman':generate_ma_signal,
        'MESA_Adaptive': generate_ma_signal,
        'MIDPOINT':generate_ma_signal,
        'TRIX':generate_ma_signal,
        'WMA':generate_ma_signal,
        'MACD': macd_signal,
        'ROC_R': roc_r,
        'ROC_MOM': roc_mom,
        'MOM_r': mom_r,
        "RSI": rsi_r,
        'CMO': cmo_r,
        'Quantile': quantile_signal,
        'continuous': continuous_signal,
        'continuous_r': continuous_r
    }
    params = {
        'bollinger_r': {"window": [3,5,10,20,30,40,50,60,120], "num_std_upper": [1,1.5,2], "num_std_lower": [1,1.5,2]},
        'bollinger_MOM': {"window": [3,5,10,20,30,40,50,60,120], "num_std_upper": [1,1.5,2], "num_std_lower": [1,1.5,2]},
        'DoubleMA':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200], 'ma_type':['DoubleMA']},
        'EXPWMA':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200], 'ma_type': ['EXPWMA']},
        'Hilbert_Transform':{"short_window":[1,2,3,4,5,10,20,60,120,240], 'ma_type': ['Hilbert_Transform']},
        'Kaufman':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200],  'ma_type': ['Kaufman']},
        'MESA_Adaptive': {"short_window": [3], "long_window": [20], 'ma_type': ['MESA_Adaptive'],
                        "fastlimit": [0.1, 0.3, 0.5, 0.7, 0.9], "slowlimit": [0.01, 0.03, 0.05, 0.1, 0.2,0.4,0.6,0.8]},
        'MIDPOINT':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200],  'ma_type': ['MidPoint']},
        'TRIX':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200],  'ma_type': ['TRIX'], 'vfactor': [0.6,0.8,1]},
        'WMA':{"short_window":[3,5,10,20,30,40,50,60,120,140,160,200],  "long_window":[5,10,20,30,40,50,60,80,100,120,140,160,200],  'ma_type': ['WMA']},
        'MACD':{"short_window":[7.8,9,10,12,13],"long_window":[24,26,30,34,40],"signalperiod":[6,9,12]},
        'ROC_R':{"window": [3,5,10,20,30,40,50,60,120,140,160,200]},
        'ROC_MOM':{"window": [3,5,10,20,30,40,50,60,120,140,160,200]},
        'MOM_r':{"period": [3,5,10,20,30,40,50,60,120,140,160,200],"threshold": [5,10,15,20,25,30]},
        'RSI': {"window": [3,5,10,20,30,40,50,60,120,140,160,200],"lower": [5,10,15,20,25,30], "middle": [50]},
        'CMO': {"window": [3,5,10,20,30,40,50,60,120,140,160,200],"upper": [50,55,60,65,70,75,80,85], "middle": [0]},
        'Quantile': {"window": [20,60,90,100,150,200,250,500]},
        'continuous': {"window": [1,2,3,4,5,6,7,8,9,10,15,20]},
        'continuous_r': {"window": [1,2,3,4,5,6,7,8,9,10,15,20]}
    }

    select_factors(index_nv, "IFIH", funcs, params)