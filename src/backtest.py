"""
=============================================================================
BACKTESTING ENGINE MODULE
=============================================================================
This module simulates trading the strategy on historical data to evaluate
its performance.

WHAT IS BACKTESTING?
--------------------
Backtesting is "playing back" history with your trading rules to see how 
you would have performed. It answers: "If I had used this strategy from 
2011-2019, what would have happened to my money?"

THE SIMULATION:
---------------
For each trading day:
1. We have a SIGNAL from yesterday (our desired exposure)
2. We observe today's RETURN (how much GLD moved)
3. Our STRATEGY RETURN = signal × GLD return

EXAMPLE:
If yesterday's signal was +0.8 (80% long) and GLD went up 2% today:
Strategy return = 0.8 × 2% = 1.6%

If yesterday's signal was -0.5 (50% short) and GLD went up 2% today:
Strategy return = -0.5 × 2% = -1.0% (we lost because we were short)

KEY ASSUMPTIONS:
----------------
1. We can execute at the closing price (reasonable for liquid ETFs like GLD)
2. No slippage beyond our transaction cost estimate
3. We can short GLD (yes, ETFs can be shorted)
4. We can use leverage up to 1.5x (available through margin accounts)

TRANSACTION COSTS:
------------------
Every time we change our position, we pay a cost:
- Bid-ask spread (buying at higher price, selling at lower)
- Commissions (broker fees)
- Market impact (our trades move the price slightly)

We model this as: cost = |position_change| × transaction_cost_rate

=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def calculate_strategy_returns(
    asset_returns: pd.Series,
    exposure: pd.Series,
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    """
    Calculate strategy returns from asset returns and exposure signals.
    
    THE CORE BACKTEST CALCULATION:
    ------------------------------
    strategy_return[t] = exposure[t] × asset_return[t] - transaction_cost[t]
    
    Where:
    - exposure[t] is the position we ENTERED at the start of day t
      (decided based on day t-1 information, already shifted in signals.py)
    - asset_return[t] is GLD's return on day t
    - transaction_cost[t] is the cost of changing our position
    
    WHY EXPOSURE × RETURN?
    ----------------------
    If we're 100% long (exposure=1.0) and GLD goes up 2%:
    - We make 2% on our capital
    - strategy_return = 1.0 × 2% = 2%
    
    If we're 50% long (exposure=0.5) and GLD goes up 2%:
    - We make 2% on half our capital, half is in cash
    - strategy_return = 0.5 × 2% = 1%
    
    If we're 150% long (exposure=1.5) and GLD goes up 2%:
    - We borrowed 50% extra to buy more gold
    - strategy_return = 1.5 × 2% = 3%
    
    If we're 100% short (exposure=-1.0) and GLD goes up 2%:
    - We bet against gold and lost
    - strategy_return = -1.0 × 2% = -2%
    
    TRANSACTION COSTS:
    ------------------
    When we change positions, we incur costs.
    Cost = |exposure[t] - exposure[t-1]| × transaction_cost_rate
    
    Example:
    - Yesterday: exposure = 0.8 (80% long)
    - Today: exposure = -0.2 (20% short)
    - Position change: |-0.2 - 0.8| = 1.0 (we had to sell 100% of portfolio)
    - If transaction_cost_rate = 0.001 (10 bps):
    - Cost = 1.0 × 0.001 = 0.1% of portfolio
    
    Parameters
    ----------
    asset_returns : pd.Series
        Daily returns of the underlying asset (GLD)
    exposure : pd.Series
        Daily exposure signal (already shifted for execution)
    transaction_cost : float
        Transaction cost as a fraction (0.001 = 10 basis points = 0.1%)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - asset_return: The underlying GLD return
        - exposure: Our position
        - position_change: How much we changed our position
        - transaction_costs: Cost of changing position
        - gross_return: Return before costs (exposure × asset_return)
        - strategy_return: Net return after costs
    """
    
    # Create results DataFrame
    results = pd.DataFrame(index=asset_returns.index)
    
    # Store inputs
    results['asset_return'] = asset_returns
    results['exposure'] = exposure
    
    # Calculate position changes (for transaction costs)
    # .diff() gives us today's exposure minus yesterday's exposure
    results['position_change'] = exposure.diff().abs()
    
    # First day has no previous exposure, assume we start flat
    results['position_change'] = results['position_change'].fillna(abs(exposure))
    
    # Calculate transaction costs
    results['transaction_costs'] = results['position_change'] * transaction_cost
    
    # Calculate gross returns (before transaction costs)
    # This is the core: our return = our exposure × the asset's return
    results['gross_return'] = exposure * asset_returns
    
    # Calculate net returns (after transaction costs)
    results['strategy_return'] = results['gross_return'] - results['transaction_costs']
    
    return results


def calculate_equity_curve(
    strategy_returns: pd.Series,
    initial_capital: float = 100000
) -> pd.Series:
    """
    Calculate the equity curve (portfolio value over time).
    
    THE CONCEPT:
    ------------
    If you started with $100,000 and applied the strategy, how much would 
    you have on each day?
    
    THE MATH:
    ---------
    Equity[t] = Equity[t-1] × (1 + return[t])
    
    Or equivalently:
    Equity[t] = Initial × (1 + return[1]) × (1 + return[2]) × ... × (1 + return[t])
    
    This is COMPOUNDING: each day's return is applied to your current balance,
    not just the initial amount.
    
    EXAMPLE:
    Day 1: Start with $100,000, return = +2%
           End of Day 1: $100,000 × 1.02 = $102,000
           
    Day 2: Start with $102,000, return = -1%
           End of Day 2: $102,000 × 0.99 = $100,980
           
    Day 3: Start with $100,980, return = +3%
           End of Day 3: $100,980 × 1.03 = $104,009.40
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns (as decimals, e.g., 0.02 for 2%)
    initial_capital : float
        Starting capital
        
    Returns
    -------
    pd.Series
        Portfolio value over time
    """
    
    # (1 + return) is the growth factor for each day
    growth_factors = 1 + strategy_returns.fillna(0)
    
    # Cumulative product gives compounded value
    # .cumprod() multiplies all factors up to that point
    cumulative_growth = growth_factors.cumprod()
    
    # Multiply by initial capital
    equity_curve = initial_capital * cumulative_growth
    
    return equity_curve


def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdowns from the equity curve.
    
    WHAT IS DRAWDOWN?
    -----------------
    Drawdown measures how much you've lost from your peak (highest point).
    It answers: "If I had sold at the best possible time, how much money 
    have I 'given back' since then?"
    
    THE MATH:
    ---------
    For each day t:
    - Peak[t] = max(Equity[1], Equity[2], ..., Equity[t])  (highest so far)
    - Drawdown[t] = (Equity[t] - Peak[t]) / Peak[t]        (% below peak)
    
    Drawdown is always ≤ 0 (you're always at or below your peak).
    
    EXAMPLE:
    Day 1: Equity = $100,000, Peak = $100,000, DD = 0%
    Day 2: Equity = $110,000, Peak = $110,000, DD = 0%    (new high!)
    Day 3: Equity = $105,000, Peak = $110,000, DD = -4.5% (below peak)
    Day 4: Equity = $95,000, Peak = $110,000, DD = -13.6% (deeper)
    Day 5: Equity = $115,000, Peak = $115,000, DD = 0%    (new high!)
    
    MAXIMUM DRAWDOWN:
    -----------------
    The worst (most negative) drawdown over the entire period.
    In the example above, max drawdown = -13.6%
    
    This is crucial because:
    1. It measures your worst pain point
    2. It's used in risk-adjusted metrics (Calmar ratio = Return / MaxDD)
    3. It determines if you'd have been able to stick with the strategy
       (imagine seeing your $100k drop to $86.4k - would you panic sell?)
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - peak: Running maximum of equity
        - drawdown: Current drawdown (as decimal, e.g., -0.136 for -13.6%)
    """
    
    results = pd.DataFrame(index=equity_curve.index)
    
    # Running maximum (the peak so far)
    # .cummax() gives the maximum value up to and including each point
    results['peak'] = equity_curve.cummax()
    
    # Drawdown = (current - peak) / peak
    # Will be 0 when at peak, negative when below peak
    results['drawdown'] = (equity_curve - results['peak']) / results['peak']
    
    return results


def _strategy_returns_with_drawdown_scaling(
    asset_returns: pd.Series,
    exposure: pd.Series,
    transaction_cost: float,
    initial_capital: float,
    drawdown_threshold: float,
    drawdown_scale_min: float,
    max_drawdown_days: Optional[int] = None,
    time_stop_scale: float = 0.5,
    max_short: float = -1.0,
    max_long: float = 1.5
) -> pd.DataFrame:
    """
    Compute strategy returns with drawdown-based exposure scaling.
    When equity drawdown exceeds threshold, scale down exposure to limit further loss.
    Optionally: after X consecutive days in drawdown, apply time_stop_scale (flatten exposure).
    No lookahead: scale at t uses equity/peak through t-1.
    """
    # Linear scale: at dd=0 scale=1, at dd=-threshold scale=1, at dd=-threshold and below scale ramps to scale_min
    def scale_from_dd(dd: float) -> float:
        if dd >= -drawdown_threshold:
            return 1.0
        depth = min(-dd - drawdown_threshold, drawdown_threshold)
        return 1.0 + (drawdown_scale_min - 1.0) * (depth / drawdown_threshold)

    n = len(asset_returns)
    strategy_return = np.nan * np.ones(n)
    effective_exposure = np.nan * np.ones(n)
    equity = initial_capital
    peak = initial_capital
    prev_eff_exp = 0.0
    consecutive_drawdown_days = 0

    for i in range(n):
        if pd.isna(exposure.iloc[i]) or pd.isna(asset_returns.iloc[i]):
            strategy_return[i] = np.nan
            effective_exposure[i] = exposure.iloc[i] if isinstance(exposure.iloc[i], (int, float)) else np.nan
            continue
        raw_exp = float(exposure.iloc[i])
        ret = float(asset_returns.iloc[i])
        dd_prev = (equity - peak) / peak if peak > 0 else 0.0
        in_drawdown = equity < peak
        if in_drawdown:
            consecutive_drawdown_days += 1
        else:
            consecutive_drawdown_days = 0
        scale = scale_from_dd(dd_prev)
        if max_drawdown_days is not None and consecutive_drawdown_days >= max_drawdown_days:
            scale *= time_stop_scale
        eff_exp = raw_exp * scale
        eff_exp = np.clip(eff_exp, max_short, max_long)
        position_change = abs(eff_exp - prev_eff_exp)
        cost = position_change * transaction_cost
        gross = eff_exp * ret
        strategy_return[i] = gross - cost
        effective_exposure[i] = eff_exp
        equity = equity * (1 + strategy_return[i])
        peak = max(peak, equity)
        prev_eff_exp = eff_exp

    results = pd.DataFrame(index=asset_returns.index)
    results['asset_return'] = asset_returns.values
    results['exposure'] = effective_exposure
    eff_series = pd.Series(effective_exposure, index=asset_returns.index)
    results['position_change'] = eff_series.diff().abs()
    results.loc[results.index[0], 'position_change'] = abs(eff_series.iloc[0])
    results['transaction_costs'] = results['position_change'] * transaction_cost
    results['gross_return'] = results['exposure'].values * asset_returns.values
    results['strategy_return'] = results['gross_return'] - results['transaction_costs']
    return results


def run_backtest(
    df: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 100000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    drawdown_threshold: Optional[float] = None,
    drawdown_scale_min: Optional[float] = None,
    max_drawdown_days: Optional[int] = None,
    time_stop_scale: float = 0.5,
    max_short: float = -1.0,
    max_long: float = 1.5
) -> Dict:
    """
    Run a complete backtest and return results.
    
    This is the main backtesting function that ties everything together.
    
    PROCESS:
    1. Filter to date range (if specified)
    2. Calculate strategy returns from exposure and asset returns
    3. Calculate equity curve (compounded portfolio value)
    4. Calculate drawdowns
    5. Return comprehensive results
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'pct_return' and 'exposure' columns
    transaction_cost : float
        Transaction cost per trade (as fraction)
    initial_capital : float
        Starting capital for equity curve
    start_date : str, optional
        Start date for backtest (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for backtest (format: 'YYYY-MM-DD')
    drawdown_threshold : float, optional
        When equity drawdown exceeds this (e.g. 0.10 = 10%), scale exposure down to limit loss.
        Improves Calmar by capping max drawdown. Used with drawdown_scale_min.
    drawdown_scale_min : float, optional
        Minimum exposure multiplier in drawdown (e.g. 0.3 = scale to 30% of signal).
    max_drawdown_days : int, optional
        Time stop: after this many consecutive days in drawdown, scale exposure by time_stop_scale.
    time_stop_scale : float
        Exposure multiplier when max_drawdown_days exceeded (e.g. 0.5 = cut in half, 0 = flat).
    max_short : float
        Maximum short exposure (for clipping; e.g. -0.75 for asymmetric sizing).
    max_long : float
        Maximum long exposure (for clipping).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'data': Full DataFrame with all backtest columns
        - 'returns': DataFrame with return calculations
        - 'equity_curve': Portfolio value over time
        - 'drawdowns': Drawdown calculations
        - 'summary': Summary statistics
    """
    
    df = df.copy()
    
    # =========================================================================
    # STEP 1: Filter to date range
    # =========================================================================
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    print(f"\nRunning backtest from {df.index.min().date()} to {df.index.max().date()}")
    print(f"  - Trading days: {len(df)}")
    print(f"  - Transaction cost: {transaction_cost*100:.2f}%")
    print(f"  - Initial capital: ${initial_capital:,.0f}")
    
    # =========================================================================
    # STEP 2: Calculate strategy returns
    # =========================================================================
    # We use pct_return (simple returns) for strategy return calculation.
    # BUG FIX: Previously used log_return, but log returns are NOT scalable
    # by leverage. For exposure != 1, the correct formula is:
    #   strategy_return = exposure * simple_return
    # NOT: exposure * log_return
    # Since the equity curve compounds via (1 + r).cumprod(), the returns
    # fed in must be simple (arithmetic) returns.
    # Exposure is already shifted (from signals.py)
    # Optionally scale exposure down when in drawdown to improve Calmar (limit max DD)
    
    if drawdown_threshold is not None and drawdown_scale_min is not None:
        returns_df = _strategy_returns_with_drawdown_scaling(
            asset_returns=df['pct_return'],
            exposure=df['exposure'],
            transaction_cost=transaction_cost,
            initial_capital=initial_capital,
            drawdown_threshold=drawdown_threshold,
            drawdown_scale_min=drawdown_scale_min,
            max_drawdown_days=max_drawdown_days,
            time_stop_scale=time_stop_scale,
            max_short=max_short,
            max_long=max_long
        )
    else:
        returns_df = calculate_strategy_returns(
            asset_returns=df['pct_return'],
            exposure=df['exposure'],
            transaction_cost=transaction_cost
        )
    
    # Add returns to main DataFrame
    for col in returns_df.columns:
        df[col] = returns_df[col]
    
    # =========================================================================
    # STEP 3: Calculate equity curve
    # =========================================================================
    df['equity'] = calculate_equity_curve(
        df['strategy_return'],
        initial_capital=initial_capital
    )
    
    # Also calculate buy-and-hold equity for comparison (using simple returns)
    df['bh_equity'] = calculate_equity_curve(
        df['pct_return'],
        initial_capital=initial_capital
    )
    
    # =========================================================================
    # STEP 4: Calculate drawdowns
    # =========================================================================
    dd_df = calculate_drawdowns(df['equity'])
    df['peak'] = dd_df['peak']
    df['drawdown'] = dd_df['drawdown']
    
    # Buy-and-hold drawdowns
    bh_dd_df = calculate_drawdowns(df['bh_equity'])
    df['bh_drawdown'] = bh_dd_df['drawdown']
    
    # =========================================================================
    # STEP 5: Generate summary
    # =========================================================================
    # Only consider days where we have valid returns
    valid_mask = df['strategy_return'].notna()
    valid_df = df[valid_mask]
    
    summary = {
        'start_date': str(valid_df.index.min().date()),
        'end_date': str(valid_df.index.max().date()),
        'trading_days': len(valid_df),
        'initial_capital': initial_capital,
        'final_value': valid_df['equity'].iloc[-1],
        'total_return': (valid_df['equity'].iloc[-1] / initial_capital - 1) * 100,
        'max_drawdown': valid_df['drawdown'].min() * 100,  # Convert to %
        'total_transaction_costs': valid_df['transaction_costs'].sum() * initial_capital,
        
        # Buy and hold comparison
        'bh_final_value': valid_df['bh_equity'].iloc[-1],
        'bh_total_return': (valid_df['bh_equity'].iloc[-1] / initial_capital - 1) * 100,
        'bh_max_drawdown': valid_df['bh_drawdown'].min() * 100,
    }
    
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"\n📊 STRATEGY PERFORMANCE:")
    print(f"   Final Value:     ${summary['final_value']:,.2f}")
    print(f"   Total Return:    {summary['total_return']:.2f}%")
    print(f"   Max Drawdown:    {summary['max_drawdown']:.2f}%")
    print(f"   Trans. Costs:    ${summary['total_transaction_costs']:,.2f}")
    
    print(f"\n📈 BUY & HOLD COMPARISON:")
    print(f"   Final Value:     ${summary['bh_final_value']:,.2f}")
    print(f"   Total Return:    {summary['bh_total_return']:.2f}%")
    print(f"   Max Drawdown:    {summary['bh_max_drawdown']:.2f}%")
    
    excess_return = summary['total_return'] - summary['bh_total_return']
    print(f"\n✨ ALPHA (Excess Return): {excess_return:.2f}%")
    print(f"{'='*60}")
    
    return {
        'data': df,
        'summary': summary
    }


def calculate_rolling_performance(
    df: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    This helps visualize how the strategy performed over different periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with strategy_return column
    window : int
        Rolling window in days (252 = 1 year)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling metrics
    """
    
    results = pd.DataFrame(index=df.index)
    
    # Rolling returns (annualized)
    results['rolling_return'] = df['strategy_return'].rolling(window).sum() * (252 / window)
    
    # Rolling volatility (annualized)
    results['rolling_vol'] = df['strategy_return'].rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe (assuming 0 risk-free rate for simplicity)
    results['rolling_sharpe'] = results['rolling_return'] / results['rolling_vol']
    
    return results


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Import required modules for testing
    from data import load_gld_data, calculate_price_returns
    from indicators import calculate_all_indicators
    from signals import generate_signals
    
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
    
    # Show sample of results
    print("\nSample of backtest data:")
    cols = ['close', 'exposure', 'asset_return', 'strategy_return', 'equity', 'drawdown']
    print(results['data'][cols].dropna().tail(10))

