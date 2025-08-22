# Copyright (c) 2025 Yuyang Yao
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import pandas as pd
import pybroker
from pybroker.data import DataSource

class CSVDataSource(DataSource):

    def __init__(self, csv_file=None, df=None):
        super().__init__()
        self.csv_file = csv_file
        self.df = df
        # Register custom columns in the CSV.
        pybroker.register_columns('position_signal')

    def _fetch_data(self, symbols, start_date, end_date, _timeframe, _adjust):
        if self.df is not None:
            self.df = self.df.reset_index(drop=False)
            return self.df
        else:
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