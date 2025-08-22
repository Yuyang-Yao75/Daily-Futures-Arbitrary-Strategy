# Cross-Species Index Futures Arbitrage Strategy: Replication and Optimization

[![EN](https://img.shields.io/badge/lang-English-blue.svg)](README.en.md)
[![CN](https://img.shields.io/badge/lang-中文-red.svg)](./README.md)

> **License**
> The source code in this repository is released under the **PolyForm Noncommercial License 1.0.0**, strictly for learning, research, and non-commercial evaluation purposes.
> **Any commercial usage requires a separate license.** For commercial inquiries, please contact: [yyyao75@163.com](mailto:yyyao75@163.com).
> See [LICENSE](./LICENSE) and [COMMERCIAL.md](./COMMERCIAL.md) for details.

---

## 📘 Project Overview

This project is a replication and optimization of the research report **“Index Futures Arbitrage Strategy Series IV: Basis, Momentum, and Seasonality Characteristics of Cross-Species Arbitrage”** by the **Dongzheng Futures Quantitative Research Team**.

The goal is to build a framework with:

* **Clear Strategy Logic**: fully integrating basis, momentum, and seasonality signals;
* **Modular Architecture**: extensible and reusable research framework;
* **Engineering Implementation**: end-to-end workflow from data loading to backtesting.

---

## ✨ Core Features

* **Cross-Species Arbitrage Signal System**

  * **Basis Signal**: captures convergence opportunities based on annualized futures–spot basis spreads;
  * **Momentum Signal**: applies technical indicators (moving averages, channel breakouts, trend strength, etc.) to extract large-cap vs. small-cap rotation trends;
  * **Seasonality Signal**: enhances signals during specific windows using historical statistics (monthly win rates, holiday effects).

* **Modular Architecture**

  * `data_utils`: data loading and preprocessing (basis, spread, volatility calculations);
  * `signal_utils`: library of basis, momentum, and seasonality signals;
  * `strategy_utils`: multi-factor signal integration and position decision-making;
  * `backtest_utils`: lightweight backtesting engine and performance analytics;
  * `factor_selection`: factor filtering and parameter optimization (supports grid search & JSON reuse);
  * `picture_utils`: visualization of strategy NAV and signals.

* **Flexible Configuration & Automated Execution**

  * Centralized configuration via `config.py` (symbols, backtest period, parameter space);
  * One-command execution with `run_strategy`: data → signals → backtest → results.

* **Intuitive Performance Analysis**

  * NAV curves for single/multiple strategies, position signal charts;
  * Factor selection results for reporting and fast iteration.

---

## 📈 Strategy Logic

Strategy returns mainly come from three sources:

1. **Basis Convergence (Structural Returns)**

   * As contracts approach maturity, futures and spot basis converges;
   * Larger annualized basis spreads imply greater arbitrage opportunities and higher target exposure.

2. **Cross-Species Spread Momentum (Trend Returns)**

   * Large vs. small-cap rotation and style shifts often display “slow variable” characteristics;
   * Momentum signals capture these trends, with moderate win rate but favorable payoff ratio.

3. **Seasonality Premium (Time Effect Returns)**

   * Post-Spring Festival small-cap outperformance is near 100%, with other monthly patterns also significant;
   * Historical monthly win rates are used to weight signals, improving position stability.

> **Signal Integration**
> Basis Signal (20%) + Momentum Signal (80%) → Initial Signal
>
> Initial Signal × Seasonality Weighting → Final Trading Signal (discretized to positions 0, ±0.5, ±1, ±1.5)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run factor selection
python src/factor_selection.py

# Run main strategy pipeline
python src/main.py
```

> ⚠️ If using Wind / iFind data sources, ensure local environment is properly configured.

---

## 🧠 Strategy Insights & Advantages

* **Multi-Factor Driven**: Basis (structure) × Momentum (timing) × Seasonality (stability)
* **Cross-Species Arbitrage**: captures spread deviations via correlation & cointegration
* **Execution Decoupling**: signal generation outputs target positions; execution layer manages rebalancing & risk control
* **Risk Management**: capital utilization, exposure limits, slippage/fee simulation
* **Extensibility**: supports factor expansion, parameter optimization, and portfolio configuration

---

## 🔧 Limitations & Future Work

* High-frequency data integration (minute-level index data) not yet complete
* Cross-sectional strategy replication and portfolio optimization still under development
* Transaction cost/slippage models not fully integrated → requires improvement for live trading feasibility
* Potential to extend into multi-strategy portfolios with commodity arbitrage and options strategies

---

## 📞 Contact / Acknowledgements

* Thanks to the **Dongzheng Futures Quantitative Research Team** for their research materials and methodology.
* Some research results and data are not fully uploaded — please reach out via email for inquiries.
* For questions or collaboration, contact: **[yyyao75@163.com](mailto:yyyao75@163.com)**

