"""
=============================================================================
PERFORMANCE METRICS MODULE
=============================================================================
This module calculates all performance metrics for evaluating the strategy.

WHY METRICS MATTER:
-------------------
A strategy might make money, but is it GOOD? Metrics help us answer:
- Is the return worth the risk?
- How bad could losses get?
- Is the strategy consistent or just lucky?

KEY METRICS FOR THIS CHALLENGE:
-------------------------------
1. CALMAR RATIO (the main target: ≥ 2.0)
   - CAGR / Maximum Drawdown
   - Measures return per unit of worst-case risk

2. SHARPE RATIO
   - (Return - Risk-free rate) / Volatility
   - Measures return per unit of volatility

3. MAXIMUM DRAWDOWN
   - Worst peak-to-trough decline
   - Measures the biggest loss from a peak

4. TIME IN MARKET
   - % of days with non-zero exposure
   - Must be ≥ 20% per the challenge rules

5. CAGR (Compound Annual Growth Rate)
   - Annualized return accounting for compounding

=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculate_cagr(
    equity_curve: pd.Series,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    WHAT IS CAGR?
    -------------
    CAGR is the annualized rate of return that would give you the same 
    final value if you compounded at that rate every year.
    
    It's different from simple average return:
    - Year 1: +50%
    - Year 2: -33%
    - Simple average: (50 - 33) / 2 = 8.5%
    - But actual: $100 → $150 → $100 = 0% total!
    - CAGR: 0%
    
    THE MATH:
    ---------
    If you start with V₀ and end with V_T after T years:
    
    CAGR = (V_T / V₀)^(1/T) - 1
    
    This is the constant annual growth rate that would turn V₀ into V_T.
    
    EXAMPLE:
    If you turn $100,000 into $200,000 over 5 years:
    CAGR = (200000/100000)^(1/5) - 1 = 2^0.2 - 1 = 14.87%
    
    This means if you grew at exactly 14.87% every year for 5 years,
    you'd turn $100k into $200k.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time
    trading_days_per_year : int
        Number of trading days per year (typically 252)
        
    Returns
    -------
    float
        CAGR as a decimal (0.15 = 15%)
    """
    
    # Get start and end values
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    
    # Calculate number of years
    n_days = len(equity_curve)
    n_years = n_days / trading_days_per_year
    
    # CAGR formula
    # (end/start)^(1/years) - 1
    if start_value <= 0 or end_value <= 0:
        return 0.0
    
    cagr = (end_value / start_value) ** (1 / n_years) - 1
    
    return cagr


def calculate_max_drawdown(drawdown_series: pd.Series) -> float:
    """
    Get the maximum drawdown from a drawdown series.
    
    THE MATH:
    ---------
    Max Drawdown = min(all drawdowns)
    
    Since drawdowns are negative (or zero), the minimum is the worst one.
    
    Parameters
    ----------
    drawdown_series : pd.Series
        Series of drawdown values (should be ≤ 0)
        
    Returns
    -------
    float
        Maximum drawdown as a decimal (e.g., -0.30 for 30% drawdown)
    """
    
    return drawdown_series.min()


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float
) -> float:
    """
    Calculate the Calmar Ratio.
    
    THIS IS THE KEY METRIC FOR THE CHALLENGE!
    
    WHAT IS CALMAR RATIO?
    ---------------------
    Calmar Ratio = CAGR / |Maximum Drawdown|
    
    It measures how much return you get per unit of worst-case risk.
    
    INTERPRETATION:
    - Calmar = 2.0: For every 1% of max drawdown, you earned 2% annually
    - If your max drawdown was 15%, your CAGR should be at least 30%
    
    WHY CALMAR?
    -----------
    Unlike Sharpe (which uses volatility), Calmar focuses on the WORST LOSS.
    
    Two strategies with the same Sharpe might have very different Calmars:
    - Strategy A: Steady small losses, never big drops, Calmar = 3.0
    - Strategy B: Occasional huge drawdowns, Calmar = 0.5
    
    For most investors, Strategy A is better even if returns are similar,
    because you can actually HOLD it through tough times.
    
    THE CHALLENGE REQUIRES CALMAR ≥ 2.0
    
    Parameters
    ----------
    cagr : float
        Compound Annual Growth Rate (as decimal)
    max_drawdown : float
        Maximum drawdown (as negative decimal, e.g., -0.30)
        
    Returns
    -------
    float
        Calmar Ratio
    """
    
    # Max drawdown should be negative, we use absolute value
    if max_drawdown == 0:
        return np.inf if cagr > 0 else 0.0
    
    calmar = cagr / abs(max_drawdown)
    
    return calmar


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe Ratio.
    
    WHAT IS SHARPE RATIO?
    ---------------------
    Sharpe = (Return - Risk-free rate) / Volatility
    
    It measures excess return per unit of risk (volatility).
    
    THE LOGIC:
    ----------
    If you can get 2% risk-free from treasury bonds, and a strategy returns 
    12% with 20% volatility:
    
    Sharpe = (12% - 2%) / 20% = 0.5
    
    For every 1% of volatility you accept, you get 0.5% of excess return.
    
    INTERPRETATION:
    - Sharpe < 0: You're losing money (or doing worse than risk-free)
    - Sharpe 0-1: Mediocre risk-adjusted returns
    - Sharpe 1-2: Good risk-adjusted returns
    - Sharpe > 2: Excellent (but be suspicious of overfitting!)
    
    ANNUALIZATION:
    - Returns: Multiply daily mean by 252
    - Volatility: Multiply daily std by sqrt(252)
    - Or just calculate on annual numbers
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns (as decimals)
    risk_free_rate : float
        Annual risk-free rate (as decimal, 0.02 = 2%)
    trading_days_per_year : int
        Trading days per year
        
    Returns
    -------
    float
        Sharpe Ratio (annualized)
    """
    
    # Drop NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Annualized return
    annual_return = returns.mean() * trading_days_per_year
    
    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(trading_days_per_year)
    
    if annual_vol == 0:
        return 0.0
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate the Sortino Ratio.
    
    WHAT IS SORTINO RATIO?
    ----------------------
    Like Sharpe, but uses DOWNSIDE volatility instead of total volatility.
    
    Sortino = (Return - Risk-free rate) / Downside Volatility
    
    WHY SORTINO?
    ------------
    Sharpe penalizes ALL volatility, including upside volatility.
    But investors don't mind upside volatility (making more than expected)!
    
    Sortino only penalizes downside volatility (losing more than expected).
    
    EXAMPLE:
    Two strategies with same Sharpe:
    - Strategy A: Returns vary from -5% to +10%
    - Strategy B: Returns vary from -2% to +7%
    
    Strategy B has higher Sortino because its downside is smaller,
    even though total volatility might be similar.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns (as decimals)
    risk_free_rate : float
        Annual risk-free rate
    trading_days_per_year : int
        Trading days per year
        
    Returns
    -------
    float
        Sortino Ratio (annualized)
    """
    
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Annualized return
    annual_return = returns.mean() * trading_days_per_year
    
    # Downside returns only (negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if annual_return > risk_free_rate else 0.0
    
    # Downside volatility (annualized)
    downside_vol = downside_returns.std() * np.sqrt(trading_days_per_year)
    
    if downside_vol == 0:
        return np.inf if annual_return > risk_free_rate else 0.0
    
    # Sortino ratio
    sortino = (annual_return - risk_free_rate) / downside_vol
    
    return sortino


def calculate_time_in_market(
    exposure: pd.Series,
    min_exposure: float = 0.1
) -> float:
    """
    Calculate the percentage of time the strategy is actively trading.
    
    THE CHALLENGE REQUIRES ≥ 20% TIME IN MARKET
    
    WHAT IS TIME IN MARKET?
    -----------------------
    The percentage of days where we have a non-trivial position.
    
    We define "in market" as |exposure| > min_exposure.
    
    WHY THIS METRIC?
    ----------------
    A strategy that says "stay out of the market 99% of the time" might 
    have great metrics when it does trade, but it's not useful.
    
    We want a strategy that's actively trading a significant portion of time.
    
    Parameters
    ----------
    exposure : pd.Series
        Daily exposure values
    min_exposure : float
        Minimum |exposure| to count as "in market"
        
    Returns
    -------
    float
        Percentage of time in market (0-100)
    """
    
    exposure = exposure.dropna()
    
    if len(exposure) == 0:
        return 0.0
    
    # Count days where |exposure| > threshold
    in_market = (abs(exposure) > min_exposure).sum()
    
    # Calculate percentage
    pct_in_market = (in_market / len(exposure)) * 100
    
    return pct_in_market


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate the percentage of positive return days.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Win rate as percentage (0-100)
    """
    
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    wins = (returns > 0).sum()
    return (wins / len(returns)) * 100


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate the profit factor.
    
    WHAT IS PROFIT FACTOR?
    ----------------------
    Profit Factor = Sum of Winning Trades / |Sum of Losing Trades|
    
    It tells you how many dollars you make for every dollar you lose.
    
    INTERPRETATION:
    - PF < 1: You lose more than you win (bad!)
    - PF = 1: Break even
    - PF = 1.5: You make $1.50 for every $1 lost (decent)
    - PF > 2: You make $2+ for every $1 lost (good)
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Profit factor
    """
    
    returns = returns.dropna()
    
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_average_trade(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate average winning and losing trade statistics.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    dict
        Dictionary with average win, average loss, and ratio
    """
    
    returns = returns.dropna()
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    
    # Win/loss ratio (risk/reward)
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    return {
        'avg_win': avg_win * 100,        # as percentage
        'avg_loss': avg_loss * 100,      # as percentage
        'win_loss_ratio': win_loss_ratio
    }


def calculate_all_metrics(
    backtest_results: Dict,
    risk_free_rate: float = 0.02,
    min_active_exposure: float = 0.1
) -> Dict:
    """
    Calculate all performance metrics from backtest results.
    
    This is the main function that computes everything.
    
    Parameters
    ----------
    backtest_results : dict
        Results from run_backtest() containing 'data' DataFrame
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation
    min_active_exposure : float
        Minimum exposure to count as "in market"
        
    Returns
    -------
    dict
        Comprehensive metrics dictionary
    """
    
    df = backtest_results['data']
    
    # Get valid data (non-NaN returns)
    valid_mask = df['strategy_return'].notna()
    valid_df = df[valid_mask]
    
    # =========================================================================
    # RETURN METRICS
    # =========================================================================
    
    # CAGR
    cagr = calculate_cagr(valid_df['equity'])
    bh_cagr = calculate_cagr(valid_df['bh_equity'])
    
    # Total return
    total_return = (valid_df['equity'].iloc[-1] / valid_df['equity'].iloc[0] - 1)
    bh_total_return = (valid_df['bh_equity'].iloc[-1] / valid_df['bh_equity'].iloc[0] - 1)
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    # Maximum drawdown
    max_dd = calculate_max_drawdown(valid_df['drawdown'])
    bh_max_dd = calculate_max_drawdown(valid_df['bh_drawdown'])
    
    # Volatility (annualized)
    volatility = valid_df['strategy_return'].std() * np.sqrt(252)
    bh_volatility = valid_df['pct_return'].std() * np.sqrt(252)
    
    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================
    
    # Calmar Ratio (THE KEY METRIC!)
    calmar = calculate_calmar_ratio(cagr, max_dd)
    bh_calmar = calculate_calmar_ratio(bh_cagr, bh_max_dd)
    
    # Sharpe Ratio
    sharpe = calculate_sharpe_ratio(valid_df['strategy_return'], risk_free_rate)
    bh_sharpe = calculate_sharpe_ratio(valid_df['pct_return'], risk_free_rate)
    
    # Sortino Ratio
    sortino = calculate_sortino_ratio(valid_df['strategy_return'], risk_free_rate)
    
    # =========================================================================
    # TRADING METRICS
    # =========================================================================
    
    # Time in market
    time_in_market = calculate_time_in_market(valid_df['exposure'], min_active_exposure)
    
    # Win rate
    win_rate = calculate_win_rate(valid_df['strategy_return'])
    
    # Profit factor
    profit_factor = calculate_profit_factor(valid_df['strategy_return'])
    
    # Average trade
    avg_trade = calculate_average_trade(valid_df['strategy_return'])
    
    # =========================================================================
    # COMPILE RESULTS
    # =========================================================================
    
    metrics = {
        # Date info
        'start_date': str(valid_df.index.min().date()),
        'end_date': str(valid_df.index.max().date()),
        'trading_days': len(valid_df),
        
        # Strategy performance
        'strategy': {
            'cagr': cagr * 100,
            'total_return': total_return * 100,
            'max_drawdown': max_dd * 100,
            'volatility': volatility * 100,
            'calmar_ratio': calmar,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
        },
        
        # Buy and hold comparison
        'buy_and_hold': {
            'cagr': bh_cagr * 100,
            'total_return': bh_total_return * 100,
            'max_drawdown': bh_max_dd * 100,
            'volatility': bh_volatility * 100,
            'calmar_ratio': bh_calmar,
            'sharpe_ratio': bh_sharpe,
        },
        
        # Trading statistics
        'trading': {
            'time_in_market': time_in_market,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_trade['avg_win'],
            'avg_loss': avg_trade['avg_loss'],
            'win_loss_ratio': avg_trade['win_loss_ratio'],
        },
        
        # Challenge requirements check
        'challenge_check': {
            'calmar_meets_target': calmar >= 2.0,
            'time_in_market_meets_target': time_in_market >= 20,
        }
    }
    
    return metrics


def print_metrics_report(metrics: Dict) -> None:
    """
    Print a formatted metrics report.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from calculate_all_metrics()
    """
    
    print("\n" + "="*70)
    print("                    PERFORMANCE METRICS REPORT")
    print("="*70)
    
    print(f"\n📅 Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"   Trading Days: {metrics['trading_days']}")
    
    # Strategy vs Buy & Hold comparison table
    print("\n" + "-"*70)
    print("                        STRATEGY    |    BUY & HOLD")
    print("-"*70)
    
    s = metrics['strategy']
    bh = metrics['buy_and_hold']
    
    print(f"CAGR:                   {s['cagr']:8.2f}%   |    {bh['cagr']:8.2f}%")
    print(f"Total Return:           {s['total_return']:8.2f}%   |    {bh['total_return']:8.2f}%")
    print(f"Max Drawdown:           {s['max_drawdown']:8.2f}%   |    {bh['max_drawdown']:8.2f}%")
    print(f"Volatility:             {s['volatility']:8.2f}%   |    {bh['volatility']:8.2f}%")
    print(f"Calmar Ratio:           {s['calmar_ratio']:8.2f}    |    {bh['calmar_ratio']:8.2f}")
    print(f"Sharpe Ratio:           {s['sharpe_ratio']:8.2f}    |    {bh['sharpe_ratio']:8.2f}")
    
    print("\n" + "-"*70)
    print("TRADING STATISTICS")
    print("-"*70)
    
    t = metrics['trading']
    print(f"Time in Market:         {t['time_in_market']:8.2f}%")
    print(f"Win Rate:               {t['win_rate']:8.2f}%")
    print(f"Profit Factor:          {t['profit_factor']:8.2f}")
    print(f"Avg Winning Day:        {t['avg_win']:8.4f}%")
    print(f"Avg Losing Day:         {t['avg_loss']:8.4f}%")
    print(f"Win/Loss Ratio:         {t['win_loss_ratio']:8.2f}")
    
    print("\n" + "-"*70)
    print("CHALLENGE REQUIREMENTS CHECK")
    print("-"*70)
    
    check = metrics['challenge_check']
    calmar_status = "✅ PASS" if check['calmar_meets_target'] else "❌ FAIL"
    tim_status = "✅ PASS" if check['time_in_market_meets_target'] else "❌ FAIL"
    
    print(f"Calmar Ratio ≥ 2.0:     {calmar_status}  (Actual: {s['calmar_ratio']:.2f})")
    print(f"Time in Market ≥ 20%:   {tim_status}  (Actual: {t['time_in_market']:.2f}%)")
    
    print("\n" + "="*70)


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def run_robustness_test(
    df: pd.DataFrame,
    base_params: Dict,
    param_to_shock: str,
    shock_pct: float = 0.10,
    run_backtest_func=None,
    generate_signals_func=None,
    calculate_indicators_func=None
) -> Dict:
    """
    Test robustness by shocking a parameter by ±10%.
    
    THE IDEA:
    ---------
    If changing a parameter by 10% causes the strategy to fall apart,
    we might be overfitting to that specific value.
    
    A robust strategy should perform reasonably well even with slightly
    different parameter values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Base data
    base_params : dict
        Base parameter values
    param_to_shock : str
        Parameter name to shock (e.g., 'trend_lookback')
    shock_pct : float
        Percentage to shock (0.10 = 10%)
    run_backtest_func : callable
        Backtest function
    generate_signals_func : callable
        Signal generation function
    calculate_indicators_func : callable
        Indicator calculation function
        
    Returns
    -------
    dict
        Results for base, +shock, and -shock cases
    """
    
    # This function would need to be implemented with the full pipeline
    # For now, return placeholder
    
    return {
        'parameter': param_to_shock,
        'base_value': base_params.get(param_to_shock),
        'shock_pct': shock_pct,
        'results': {
            'base': None,
            'up_shock': None,
            'down_shock': None
        }
    }


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Import required modules for testing
    from data import load_gld_data, calculate_price_returns
    from indicators import calculate_all_indicators
    from signals import generate_signals
    from backtest import run_backtest
    
    # Load and prepare data
    print("Loading data...")
    df = load_gld_data("data/processed/gld.csv")
    df = calculate_price_returns(df)
    
    print("\nCalculating indicators...")
    df = calculate_all_indicators(df)
    
    print("\nGenerating signals...")
    df = generate_signals(df)
    
    # Run backtest
    results = run_backtest(
        df,
        transaction_cost=0.001,
        initial_capital=100000,
        start_date='2011-01-01',
        end_date='2019-12-31'
    )
    
    # Calculate metrics
    metrics = calculate_all_metrics(results)
    
    # Print report
    print_metrics_report(metrics)

