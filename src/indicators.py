"""
=============================================================================
TECHNICAL INDICATORS MODULE
=============================================================================
This module calculates all the technical indicators used in our trading strategy.

WHAT ARE TECHNICAL INDICATORS?
------------------------------
Technical indicators are mathematical calculations based on price, volume, or 
other market data. They help us identify:
- TRENDS: Is the price going up, down, or sideways?
- MOMENTUM: How strong is the current move?
- VOLATILITY: How much is the price swinging?
- MEAN REVERSION: Is the price stretched too far and likely to snap back?

OUR INDICATORS:
1. Rolling Slope of Log Prices (Trend)
2. Moving Average Crossover (Trend)
2. Realized Volatility (Risk)
3. RSI - Relative Strength Index (Mean Reversion)
4. Volatility Trend (Is volatility rising or falling?)

=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


# =============================================================================
# TREND INDICATORS
# =============================================================================

def calculate_rolling_slope(
    series: pd.Series,
    lookback: int = 60,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate the rolling slope of a series using linear regression.
    
    THIS IS THE CORE TREND INDICATOR.
    
    THE INTUITION:
    --------------
    Imagine plotting the last 60 days of log prices on a graph.
    Now draw the "best fit" straight line through those points.
    The SLOPE of that line tells you the trend:
    
    - Positive slope → Price trending UP → Bullish
    - Negative slope → Price trending DOWN → Bearish
    - Near-zero slope → Price going sideways → Neutral
    
    THE MATH:
    ---------
    Linear regression finds the line y = mx + b that minimizes the sum of 
    squared errors (distance from each point to the line).
    
    For our case:
    - y = log prices
    - x = day number (0, 1, 2, ..., 59)
    - m = slope (this is what we want!)
    - b = intercept (we ignore this)
    
    The slope 'm' represents the average daily change in log price.
    
    If m = 0.001, that means log price increases by 0.001 per day on average,
    which corresponds to about 0.1% per day or ~25% per year.
    
    WHY LOG PRICES?
    ---------------
    Using log prices instead of raw prices has benefits:
    1. The slope represents PERCENTAGE change, not dollar change
    2. A $1 move at $100 is treated the same as a $1.50 move at $150
    3. Makes the slope comparable across different price levels
    
    Parameters
    ----------
    series : pd.Series
        The time series to calculate slope for (typically log prices)
    lookback : int
        Number of periods to include in each regression
    annualize : bool
        If True, multiply slope by 252 to annualize (252 trading days/year)
        
    Returns
    -------
    pd.Series
        Rolling slope values (NaN for first lookback-1 periods)
    """
    
    def calc_slope(y):
        """
        Calculate slope for a single window.
        
        We use scipy.stats.linregress which returns:
        - slope: The slope of the regression line
        - intercept: The y-intercept
        - r_value: Correlation coefficient (how well the line fits)
        - p_value: Statistical significance
        - std_err: Standard error of the slope estimate
        
        We only care about the slope.
        """
        if len(y) < 2:
            return np.nan
        
        # Handle NaN values
        if np.isnan(y).any():
            return np.nan
        
        # x is just the day number: 0, 1, 2, ..., lookback-1
        x = np.arange(len(y))
        
        # Perform linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return slope
        except:
            return np.nan
    
    # Apply the slope calculation to each rolling window
    # .rolling(lookback) creates windows of size 'lookback'
    # .apply(calc_slope) applies our function to each window
    slopes = series.rolling(window=lookback, min_periods=lookback).apply(
        calc_slope, raw=True
    )
    
    # Annualize if requested
    # Multiply by 252 to convert daily slope to annual slope
    if annualize:
        slopes = slopes * 252
    
    return slopes


def calculate_slope_zscore(
    slopes: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Convert raw slopes to z-scores for normalization.
    
    WHY Z-SCORES?
    -------------
    Raw slopes are hard to interpret:
    - Is a slope of 0.05 "big" or "small"?
    - It depends on what slopes looked like historically!
    
    Z-score tells us how many standard deviations the current slope is 
    from its historical mean:
    
    z = (slope - mean(slope)) / std(slope)
    
    Interpretation:
    - z = +2: Current slope is 2 std devs above average → Strong uptrend
    - z = 0: Current slope is at average → Normal trend
    - z = -2: Current slope is 2 std devs below average → Strong downtrend
    
    This normalization makes our signal:
    1. Comparable across time (works in both 2011 and 2019)
    2. Bounded (typically between -3 and +3)
    3. Interpretable (z > 1 means "above average trend")
    
    Parameters
    ----------
    slopes : pd.Series
        Raw slope values
    lookback : int
        Window for calculating mean and std (252 = 1 year)
        
    Returns
    -------
    pd.Series
        Z-score normalized slopes
    """
    
    # Calculate rolling mean of slopes
    rolling_mean = slopes.rolling(window=lookback, min_periods=lookback//2).mean()
    
    # Calculate rolling standard deviation of slopes
    rolling_std = slopes.rolling(window=lookback, min_periods=lookback//2).std()
    
    # Z-score: (value - mean) / std
    # Adding a small epsilon to avoid division by zero
    zscore = (slopes - rolling_mean) / (rolling_std + 1e-10)
    
    return zscore


def calculate_sma(
    prices: pd.Series,
    lookback: int
) -> pd.Series:
    """
    Calculate a simple moving average (SMA).

    SMA = mean of prices over the last N days.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., close)
    lookback : int
        Window length for the SMA

    Returns
    -------
    pd.Series
        SMA values
    """

    return prices.rolling(window=lookback, min_periods=lookback).mean()


def calculate_drawdown_series(
    prices: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Calculate rolling drawdown from a lookback-window peak.

    Drawdown is the percentage drop from the rolling peak:
        drawdown_t = (price_t - rolling_max_t) / rolling_max_t

    Values are 0 at new highs and negative during drawdowns.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., close)
    lookback : int
        Rolling window for the peak calculation

    Returns
    -------
    pd.Series
        Rolling drawdown series (negative values)
    """

    rolling_peak = prices.rolling(window=lookback, min_periods=lookback).max()
    drawdown = (prices - rolling_peak) / rolling_peak
    return drawdown


def calculate_ulcer_index(
    prices: pd.Series,
    lookback: int = 14
) -> pd.Series:
    """
    Calculate the Ulcer Index (UI), a drawdown-based risk metric.

    WHAT IS ULCER INDEX?
    --------------------
    UI measures the depth and duration of drawdowns.
    It is the square root of the mean of squared drawdowns:

        UI = sqrt( mean( drawdown^2 ) )

    Unlike volatility, UI only penalizes downside movement (pain).

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., close)
    lookback : int
        Rolling window for the UI calculation

    Returns
    -------
    pd.Series
        Ulcer Index values (positive, higher = worse drawdown)
    """

    drawdown = calculate_drawdown_series(prices, lookback=lookback)
    ulcer = drawdown.pow(2).rolling(window=lookback, min_periods=lookback).mean().pow(0.5)
    return ulcer


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def calculate_realized_volatility(
    log_returns: pd.Series,
    lookback: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate realized (historical) volatility from log returns.
    
    WHAT IS VOLATILITY?
    -------------------
    Volatility measures how much prices swing up and down.
    
    Think of it as the "riskiness" of an asset:
    - High volatility: Prices swing wildly (±5% daily swings)
    - Low volatility: Prices are stable (±0.5% daily swings)
    
    THE MATH:
    ---------
    Realized volatility = standard deviation of returns
    
    For log returns r_1, r_2, ..., r_n:
    
    volatility = sqrt( (1/(n-1)) * sum((r_i - mean(r))^2) )
    
    This is just the standard deviation formula you learned in statistics!
    
    WHY ANNUALIZE?
    --------------
    Daily volatility is tiny (usually 0.5% to 2%).
    Annualized volatility is more intuitive (10% to 30%).
    
    To annualize: multiply by sqrt(252)
    
    Why sqrt(252)? Because:
    - Variance adds: annual_variance = 252 * daily_variance
    - Volatility is sqrt(variance): annual_vol = sqrt(252) * daily_vol
    
    EXAMPLE:
    --------
    If daily volatility is 1% (0.01), annualized volatility is:
    0.01 * sqrt(252) = 0.01 * 15.87 = 15.87%
    
    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns
    lookback : int
        Number of days in the rolling window
    annualize : bool
        If True, multiply by sqrt(252) to annualize
        
    Returns
    -------
    pd.Series
        Realized volatility values
    """
    
    # Calculate rolling standard deviation of returns
    vol = log_returns.rolling(window=lookback, min_periods=lookback).std()
    
    # Annualize if requested
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_volatility_percentile(
    volatility: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Calculate the percentile rank of current volatility.
    
    WHY PERCENTILES?
    ----------------
    Instead of asking "is volatility 20%?" we ask 
    "where does 20% rank compared to the last year?"
    
    If 20% is in the 90th percentile, it means volatility is higher than 
    90% of days in the past year → This is VERY high volatility!
    
    If 20% is in the 30th percentile, it means volatility is lower than 
    70% of days in the past year → This is relatively low volatility.
    
    Percentiles automatically adjust for changing volatility regimes.
    
    Parameters
    ----------
    volatility : pd.Series
        Volatility values
    lookback : int
        Window for calculating percentile (252 = 1 year)
        
    Returns
    -------
    pd.Series
        Percentile rank (0 to 100)
    """
    
    def percentile_rank(x):
        """Calculate percentile of the last value within the window."""
        if len(x) < 2:
            return np.nan
        # How many values are less than the current value?
        return (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
    
    percentile = volatility.rolling(window=lookback, min_periods=lookback//2).apply(
        percentile_rank, raw=True
    )
    
    return percentile


def calculate_volatility_trend(
    volatility: pd.Series,
    lookback: int = 10
) -> pd.Series:
    """
    Calculate if volatility is trending up or down.
    
    THE INTUITION:
    --------------
    Not just "is volatility high?" but "is volatility INCREASING?"
    
    Rising volatility = Market becoming unstable = Reduce exposure
    Falling volatility = Market calming down = Can increase exposure
    
    We calculate this as the slope of volatility over the last N days.
    
    Parameters
    ----------
    volatility : pd.Series
        Volatility values
    lookback : int
        Window for calculating trend
        
    Returns
    -------
    pd.Series
        Volatility trend (positive = rising, negative = falling)
    """
    
    # Use the same slope calculation as for prices
    vol_slope = calculate_rolling_slope(volatility, lookback=lookback, annualize=False)
    
    # Normalize by dividing by average volatility
    avg_vol = volatility.rolling(window=lookback, min_periods=lookback).mean()
    vol_trend = vol_slope / (avg_vol + 1e-10)
    
    return vol_trend


# =============================================================================
# MEAN REVERSION INDICATORS
# =============================================================================

def calculate_rsi(
    prices: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    WHAT IS RSI?
    ------------
    RSI measures the speed and magnitude of recent price changes.
    It oscillates between 0 and 100.
    
    - RSI > 70: "Overbought" - price rose too fast, might pull back
    - RSI < 30: "Oversold" - price fell too fast, might bounce
    - RSI = 50: Neutral
    
    THE MATH:
    ---------
    RSI = 100 - 100 / (1 + RS)
    
    where RS = Average Gain / Average Loss
    
    Step by step:
    1. Calculate daily price changes
    2. Separate into gains (positive changes) and losses (negative changes)
    3. Calculate average gain and average loss over the period
    4. RS = avg_gain / avg_loss
    5. RSI = 100 - 100/(1 + RS)
    
    EXAMPLE:
    --------
    If avg_gain = 2% and avg_loss = 1%:
    RS = 2/1 = 2
    RSI = 100 - 100/(1+2) = 100 - 33.3 = 66.7 (slightly bullish)
    
    If avg_gain = 1% and avg_loss = 2%:
    RS = 1/2 = 0.5
    RSI = 100 - 100/(1+0.5) = 100 - 66.7 = 33.3 (slightly bearish)
    
    WHY USE RSI?
    ------------
    RSI helps identify when price has moved "too far, too fast" and might reverse.
    We use it as a MODIFIER to our trend signal, not as the primary signal.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices)
    period : int
        Lookback period for RSI calculation
        
    Returns
    -------
    pd.Series
        RSI values (0 to 100)
    """
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0  # Keep only positive changes
    losses[losses > 0] = 0  # Keep only negative changes
    losses = abs(losses)  # Make losses positive for calculation
    
    # Calculate average gains and losses using exponential moving average
    # This is the "smoothed" RSI, which is standard
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    
    # Calculate RS (Relative Strength)
    # Add small epsilon to avoid division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore_returns(
    log_returns: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Calculate z-score of returns for mean reversion signals.
    
    THE INTUITION:
    --------------
    Similar to RSI, but using statistical z-scores.
    
    If today's return has z-score of -2, it means today was 2 standard 
    deviations below average - a relatively rare bad day that might reverse.
    
    This is a more statistically rigorous approach to mean reversion.
    
    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns
    lookback : int
        Window for calculating mean and std
        
    Returns
    -------
    pd.Series
        Z-score of returns
    """
    
    rolling_mean = log_returns.rolling(window=lookback, min_periods=lookback//2).mean()
    rolling_std = log_returns.rolling(window=lookback, min_periods=lookback//2).std()
    
    zscore = (log_returns - rolling_mean) / (rolling_std + 1e-10)
    
    return zscore


# =============================================================================
# COMPOSITE INDICATOR CALCULATION
# =============================================================================

def calculate_all_indicators(
    df: pd.DataFrame,
    trend_lookback: int = 60,
    trend_fast_lookback: int = 20,
    trend_slow_lookback: int = 120,
    trend_zscore_lookback: int = 252,
    sma_fast_lookback: int = 20,
    sma_slow_lookback: int = 100,
    vol_lookback: int = 20,
    vol_percentile_lookback: int = 252,
    vol_trend_lookback: int = 10,
    rsi_period: int = 14,
    returns_zscore_lookback: int = 20,
    ulcer_lookback: int = 14
) -> pd.DataFrame:
    """
    Calculate all indicators and add them to the DataFrame.
    
    This is the main function that orchestrates all indicator calculations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'log_close', 'log_return', 'close' columns
    trend_lookback : int
        Lookback for trend slope calculation
    trend_zscore_lookback : int
        Lookback for slope z-score normalization
    vol_lookback : int
        Lookback for volatility calculation
    vol_percentile_lookback : int
        Lookback for volatility percentile calculation
    vol_trend_lookback : int
        Lookback for volatility trend calculation
    rsi_period : int
        Period for RSI calculation
    returns_zscore_lookback : int
        Lookback for returns z-score calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all indicators added as new columns
    """
    
    df = df.copy()
    
    print("Calculating indicators...")
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    print(f"  - Rolling slope (lookback={trend_lookback})")
    df['slope'] = calculate_rolling_slope(
        df['log_close'], 
        lookback=trend_lookback,
        annualize=True
    )
    
    print(f"  - Slope z-score (lookback={trend_zscore_lookback})")
    df['slope_zscore'] = calculate_slope_zscore(
        df['slope'],
        lookback=trend_zscore_lookback
    )

    print(f"  - Fast slope (lookback={trend_fast_lookback})")
    df['slope_fast'] = calculate_rolling_slope(
        df['log_close'],
        lookback=trend_fast_lookback,
        annualize=True
    )

    print(f"  - Slow slope (lookback={trend_slow_lookback})")
    df['slope_slow'] = calculate_rolling_slope(
        df['log_close'],
        lookback=trend_slow_lookback,
        annualize=True
    )

    print(f"  - Fast slope z-score (lookback={trend_zscore_lookback})")
    df['slope_fast_zscore'] = calculate_slope_zscore(
        df['slope_fast'],
        lookback=trend_zscore_lookback
    )

    print(f"  - Slow slope z-score (lookback={trend_zscore_lookback})")
    df['slope_slow_zscore'] = calculate_slope_zscore(
        df['slope_slow'],
        lookback=trend_zscore_lookback
    )

    # =========================================================================
    # MOVING AVERAGE CROSSOVER (ALTERNATE TREND MODEL)
    # =========================================================================
    print(f"  - SMA fast (lookback={sma_fast_lookback})")
    df['sma_fast'] = calculate_sma(df['close'], lookback=sma_fast_lookback)

    print(f"  - SMA slow (lookback={sma_slow_lookback})")
    df['sma_slow'] = calculate_sma(df['close'], lookback=sma_slow_lookback)
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    print(f"  - Realized volatility (lookback={vol_lookback})")
    df['volatility'] = calculate_realized_volatility(
        df['log_return'],
        lookback=vol_lookback,
        annualize=True
    )
    
    print(f"  - Volatility percentile (lookback={vol_percentile_lookback})")
    df['vol_percentile'] = calculate_volatility_percentile(
        df['volatility'],
        lookback=vol_percentile_lookback
    )
    
    print(f"  - Volatility trend (lookback={vol_trend_lookback})")
    df['vol_trend'] = calculate_volatility_trend(
        df['volatility'],
        lookback=vol_trend_lookback
    )
    
    # =========================================================================
    # MEAN REVERSION INDICATORS
    # =========================================================================
    print(f"  - RSI (period={rsi_period})")
    df['rsi'] = calculate_rsi(
        df['close'],
        period=rsi_period
    )
    
    print(f"  - Returns z-score (lookback={returns_zscore_lookback})")
    df['returns_zscore'] = calculate_zscore_returns(
        df['log_return'],
        lookback=returns_zscore_lookback
    )

    print(f"  - Ulcer Index (lookback={ulcer_lookback})")
    df['ulcer_index'] = calculate_ulcer_index(
        df['close'],
        lookback=ulcer_lookback
    )
    
    # =========================================================================
    # COUNT NON-NULL VALUES
    # =========================================================================
    non_null_counts = df[
        [
            'slope', 'slope_zscore', 'slope_fast', 'slope_slow',
            'slope_fast_zscore', 'slope_slow_zscore',
            'sma_fast', 'sma_slow',
            'volatility', 'vol_percentile', 'vol_trend',
            'rsi', 'returns_zscore', 'ulcer_index'
        ]
    ].notna().sum()
    
    print(f"\nIndicator coverage (non-null values):")
    for col, count in non_null_counts.items():
        print(f"  - {col}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")
    
    return df


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Import data module for testing
    from data import load_gld_data, calculate_price_returns
    
    # Load data
    df = load_gld_data("data/processed/gld.csv")
    df = calculate_price_returns(df)
    
    # Calculate all indicators
    df = calculate_all_indicators(df)
    
    # Show sample of indicators
    print("\nSample of indicators:")
    print(df[['close', 'slope', 'slope_zscore', 'volatility', 'rsi']].dropna().head(10))
    
    # Show statistics
    print("\nIndicator statistics:")
    print(df[['slope', 'slope_zscore', 'volatility', 'vol_percentile', 'rsi']].describe())

