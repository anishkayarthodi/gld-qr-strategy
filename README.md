# GLD Quantitative Trading Strategy

A systematic, interpretable trading strategy for GLD designed to capture
medium-term trends while controlling drawdowns via volatility-adjusted exposure.

## Strategy Overview

This strategy combines multiple signals to generate daily exposure recommendations:

1. **Trend Following (Primary Signal)**: Uses fast/slow rolling slope of log prices to identify timing + regime
2. **Volatility Adjustment**: Scales exposure inversely with realized volatility for risk control
3. **Ulcer (Drawdown) Filter**: Reduces exposure when drawdowns persist (drawdown “pain” rises)
4. **Mean Reversion Modifier**: Optional RSI timing tweak (disabled in current config)
5. **Smoothing**: Applies exponential smoothing to prevent whipsawing

### Key Features
- Continuous exposure scaling (no binary buy/sell)
- Daily close-to-close execution
- Leverage constrained to [-1.0, +1.5]
- Volatility targeting for consistent risk exposure

## Objectives
- Calmar Ratio ≥ 2.0 (2011–2019)
- Robust to ±10% parameter shocks
- Time-in-market ≥ 20%
- No lookahead bias

## Repository Structure

```
gld-qr-strategy/
├── configs/
│   └── default.yaml        # Strategy parameters (all tunable values)
├── data/
│   └── processed/
│       └── gld.csv         # GLD price data (2011-2019)
├── notebooks/
│   └── analysis.ipynb      # Interactive analysis notebook
├── reports/                # Generated reports and plots
├── src/
│   ├── data.py            # Data loading and preprocessing
│   ├── indicators.py      # Technical indicator calculations
│   ├── signals.py         # Signal generation logic
│   ├── backtest.py        # Backtesting engine
│   ├── metrics.py         # Performance metrics (Calmar, Sharpe, etc.)
│   └── run.py             # Main entry point
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd gld-qr-strategy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### Basic Run
```bash
python src/run.py
```

This will:
1. Load configuration from `configs/default.yaml`
2. Load and preprocess GLD data
3. Calculate technical indicators
4. Generate trading signals
5. Run the backtest (2011-2019)
6. Display performance metrics
7. Generate and save plots

### Command Line Options
```bash
# Run without saving results
python src/run.py --no-save

# Run without generating plots
python src/run.py --no-plots

# Run robustness analysis (±10% parameter shocks)
python src/run.py --robustness

# Use a custom config file
python src/run.py --config configs/custom.yaml
```

### Jupyter Notebook
For interactive analysis:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Configuration

All strategy parameters are in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend.lookback_days` | 60 | Days for trend calculation |
| `volatility.lookback_days` | 20 | Days for volatility calculation |
| `volatility.target_volatility` | 0.15 | Target annualized volatility |
| `mean_reversion.rsi_period` | 14 | RSI calculation period |
| `exposure.max_long` | 1.5 | Maximum long leverage |
| `exposure.max_short` | -1.0 | Maximum short leverage |

## Strategy Logic (Detailed)

### 1. Trend Signal
```
slope = linear_regression_slope(log_prices, window=60)
slope_zscore = (slope - rolling_mean(slope)) / rolling_std(slope)
trend_signal = tanh(slope_zscore * 0.5) * 2
```

### 2. Volatility Adjustment
```
realized_vol = std(log_returns, window=20) * sqrt(252)
vol_scale = min(target_vol / realized_vol, 2.0)
adjusted_signal = trend_signal * vol_scale

# Additional reduction in high-vol regime
if vol_percentile > 80:
    adjusted_signal *= 0.5
```

### 3. Mean Reversion Modifier
```
if RSI < 30:  # Oversold
    adjusted_signal += 0.3
if RSI > 70:  # Overbought
    adjusted_signal -= 0.3
```

### 4. Final Processing
```
smoothed_signal = EMA(adjusted_signal, alpha=0.3)
exposure = clip(smoothed_signal, -1.0, +1.5)
execution_signal = shift(exposure, 1)  # Avoid lookahead
```

## Performance Metrics

The strategy is evaluated on:

- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: CAGR / |Max Drawdown| (target ≥ 2.0)
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Time in Market**: % of days with |exposure| > 0.1 (target ≥ 20%)
- **Win Rate**: % of profitable days
- **Profit Factor**: Gross profit / Gross loss

## Robustness Testing

Run `python src/run.py --robustness` to test strategy stability with ±10% parameter shocks on:
- Trend lookback window
- Volatility lookback window
- RSI period

A robust strategy should maintain Calmar ≥ 2.0 across all parameter variations.

## Files Description

### `src/data.py`
- Loads GLD CSV data
- Parses dates and handles data cleaning
- Calculates log prices and log returns
- Provides data summary statistics

### `src/indicators.py`
- **Rolling Slope**: Linear regression slope of log prices
- **Slope Z-Score**: Normalized trend strength
- **Realized Volatility**: Rolling standard deviation of returns
- **Volatility Percentile**: Historical ranking of current volatility
- **RSI**: Relative Strength Index for mean reversion

### `src/signals.py`
- Combines indicators into raw signal
- Applies volatility adjustment
- Applies mean reversion modifier
- Enforces leverage constraints
- Smooths signal to prevent whipsawing
- Shifts signal to avoid lookahead bias

### `src/backtest.py`
- Simulates trading with the generated signals
- Calculates strategy returns: `exposure × asset_return - costs`
- Tracks equity curve and drawdowns
- Compares against buy-and-hold benchmark

### `src/metrics.py`
- Calculates all performance metrics
- Generates formatted reports
- Checks challenge requirements

### `src/run.py`
- Main orchestration script
- Loads config, runs pipeline, displays results
- Generates plots and saves outputs
- Supports robustness testing mode

## License

MIT
