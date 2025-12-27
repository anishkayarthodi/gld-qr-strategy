# GLD Quantitative Trading Strategy

A systematic, interpretable trading strategy for GLD designed to capture
medium-term trends while controlling drawdowns via volatility-adjusted exposure.

## Strategy Overview
- Trend-following via rolling slope of log prices
- Volatility regime filtering using realized volatility
- Continuous exposure scaling (no binary buy/sell)
- Daily close-to-close execution
- Leverage constrained to [-1.0, +1.5]

## Objectives
- Calmar Ratio ≥ 2.0 (2011–2019)
- Robust to ±10% parameter shocks
- Time-in-market ≥ 20%
- No lookahead bias

## Repository Structure
- `src/` — strategy logic and backtest engine
- `configs/` — parameter configurations
- `data/processed/` — cleaned GLD data
- `reports/` — research memo and figures
- `notebooks/` — exploratory analysis

## How to Run
Coming soon.
