"""用于因素选择和参数优化的工具。
此模块提供了辅助函数，用于在不同的参数组合下对技术因素进行网格搜索，对每个配置进行 pybroker 回测，并保留具有稳健夏普比率的配置。
所选因素的结果以 JSON 格式保存，以便能够将其在配置文件中重复使用。
"""

import json
import os
from collections import OrderedDict
from itertools import product
from typing import Callable, Dict, Iterable, List, Tuple, Any

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
    metrics_to_optimize: List[str]=["sharpe"],
    metric_thresholds: Dict[str, float]={"sharpe": 0.5},
) -> OrderedDict[str,Any]:
    """执行网格搜索优化和因子筛选。

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
    os.makedirs(FACTOR_SELECTION_PATH, exist_ok=True)
    selected: "OrderedDict[str, Any]" = OrderedDict()

    for name, func in factor_funcs.items():
        grid = _grid(param_spaces.get(name, {}))
        results:List[Dict[str, Any]] = []

        for params_kwargs in grid:
            # (1) 生成信号并回测
            ohlc = generate_ohlc(index_nv_df, symbol)
            ohlc["position_signal"] = func(ohlc["close"], **params_kwargs)
            ohlc["symbol"] = symbol
            df_signal = ohlc[["open", "high", "low", "close", "position_signal"]].copy()
            df_signal["symbol"] = symbol
            df_signal = df_signal.reset_index()
            df_signal = df_signal[["date", "symbol", "open", "high", "low", "close", "position_signal"]]

            tag = f"{name}_" + "_".join(f"{k}{v}" for k, v in params_kwargs.items())
            metrics_df = _run_backtest(df_signal.set_index("date"), symbol, tag)
            # (2) 提取所有目标指标
            metric_values = {
                m:_extract_metric(metrics_df, m) for m in metrics_to_optimize
            }
            # (3) 记录一条结果
            results.append({
                "params":params_kwargs,
                **metric_values
            })
        # (4) 组织成 DataFrame，方便筛选
        results_df = pd.DataFrame(results)

        # (5) 应用阈值过滤
        mask = pd.Series(True, index=results_df.index)
        for m,th in metric_thresholds.items():
            mask &= results_df[m] >= th
        filtered = results_df[mask]
        if filtered.empty:
            continue

        # (6) 多指标排序：按第一个指标降序，也可以自定义权重
        best = filtered.sort_values(by=metrics_to_optimize, ascending=False).iloc[0]
        selected[name] = {
            "func": func,
            "best_params": best["params"],
            "best_metrics": {m: best[m] for m in metrics_to_optimize}
        }

        # （可选）保存中间结果
        results_df.to_csv(os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_{name}_grid_results.csv"), index=False)

    # 最后把选中结果写成 JSON
    json_ready = {
        k: {
            "func": v["func"].__name__,
            "params": v["best_params"],
            "metrics": v["best_metrics"],
        }
        for k, v in selected.items()
    }
    with open(os.path.join(FACTOR_SELECTION_PATH, f"{symbol}_selected_factors.json"), "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)

    return selected


if __name__ == "__main__":  # pragma: no cover - usage example
    from config import INDEX_DATA

    index_df = pd.read_csv(INDEX_DATA, parse_dates=["date"]).set_index("date")
    index_nv = get_nv_data(index_df, "index")

    from signal_utils import rsi_r

    funcs = {"RSI": rsi_r}
    params = {"RSI": {"window": [14, 20], "upper": [70], "lower": [30], "middle": [50]}}

    select_factors(index_nv, "IFIH", funcs, params)