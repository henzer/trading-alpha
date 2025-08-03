# Trading Alpha 📈

**Institutional Trading Framework** targeting **20-30% annual returns**

A comprehensive Python framework for building, backtesting, and optimizing institutional-level trading strategies using vectorbt and riskfolio-lib.

## 🎯 Strategy Focus

**High-Alpha Multi-Factor Long/Short Equity**
- Target Return: 20-30% annual
- Max Drawdown: 15-25%
- Sharpe Ratio: 1.0-1.6
- Market Beta: Low correlation (0.1-0.3)

## 🛠️ Core Technologies

- **[vectorbt](https://vectorbt.dev/)**: High-performance backtesting
- **[riskfolio-lib](https://riskfolio-lib.readthedocs.io/)**: Portfolio optimization
- **[yfinance](https://github.com/ranaroussi/yfinance)**: Market data
- **[optuna](https://optuna.org/)**: Hyperparameter optimization

## 🚀 Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd trading-alpha

# Setup environment
./activate_env.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 📊 Framework Architecture

```
trading-alpha/
├── core/                   # Base strategy classes
├── data/                   # Yahoo Finance integration
├── strategies/             # Trading strategies
├── backtesting/           # Vectorbt engine
├── optimization/          # Riskfolio + Optuna
├── reporting/             # Dashboards & analysis
└── utils/                 # Configuration & helpers
```

## 💼 Strategy Implementation

### High-Alpha Multi-Factor Long/Short
- **Universe**: Russell 2000 small caps
- **Factors**: Momentum (40%), Quality (25%), Value (20%), Growth (15%)
- **Leverage**: 1.6x (160% long, 60% short)
- **Rebalance**: Weekly for momentum capture
- **Risk Management**: Sector neutral, volatility targeting

## 📈 Expected Performance

| Metric | Target Range |
|--------|-------------|
| Annual Return | 20-30% |
| Sharpe Ratio | 1.0-1.6 |
| Max Drawdown | 15-25% |
| Win Rate | 55-60% |
| Market Beta | 0.1-0.3 |

## 🧮 Optimization Pipeline

1. **Hyperparameter Tuning** (Optuna)
   - Lookback periods
   - Factor weights
   - Rebalance frequency
   
2. **Portfolio Optimization** (Riskfolio)
   - Risk-adjusted returns
   - Correlation analysis
   - Drawdown minimization

3. **Walk-Forward Validation**
   - Out-of-sample testing
   - Regime robustness
   - Transaction cost analysis

## 📊 Backtesting Features

- **Vectorized execution** for speed
- **Transaction costs** modeling
- **Slippage** simulation
- **Position sizing** controls
- **Risk limits** enforcement

## 🔍 Risk Management

- **Sector neutral** construction
- **Volatility targeting** (15-20% annual)
- **Drawdown limits** (stop at -20%)
- **Correlation monitoring**
- **Regime detection** filters

## 📋 Usage Example

```python
from trading_alpha import HighAlphaMultiFactor, VectorbtEngine, RiskfolioOptimizer

# Initialize strategy
strategy = HighAlphaMultiFactor(
    universe="RUSSELL_2000",
    leverage=1.6,
    rebalance_freq="weekly"
)

# Backtest
engine = VectorbtEngine()
results = engine.run(strategy, start_date="2020-01-01")

# Optimize
optimizer = RiskfolioOptimizer()
optimal_weights = optimizer.optimize(results)

# Analyze
print(f"Annual Return: {results.annual_return:.1%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.1%}")
```

## 🔬 Research & Development

### Implemented Strategies
- [x] High-Alpha Multi-Factor Long/Short
- [ ] Cross-Asset Momentum
- [ ] Statistical Arbitrage
- [ ] Volatility Trading

### Optimization Methods
- [x] Bayesian Optimization (Optuna)
- [x] Portfolio Optimization (Riskfolio)
- [ ] Walk-Forward Analysis
- [ ] Monte Carlo Simulation

## 📚 References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns.

## ⚠️ Disclaimer

This framework is for educational and research purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. Please consult with financial professionals before implementing any trading strategies.

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for institutional-level trading**