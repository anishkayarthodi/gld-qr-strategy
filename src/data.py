"""
=============================================================================
DATA LOADING AND PREPROCESSING MODULE
=============================================================================
This module handles loading raw GLD price data and transforming it into a 
clean format ready for analysis.

KEY CONCEPTS EXPLAINED:
-----------------------

1. LOG PRICES vs RAW PRICES
   ---------------------------
   Instead of using raw prices ($134, $138, etc.), we convert to log prices.
   
   WHY? Consider two scenarios:
   - Stock goes from $100 to $110 to $100: +10%, then -9.09%
   - Stock goes from $100 to $90 to $100: -10%, then +11.11%
   
   These percentage returns are ASYMMETRIC (not equal in magnitude).
   
   Log returns are SYMMETRIC:
   - $100 to $110: ln(110/100) = +0.0953
   - $110 to $100: ln(100/110) = -0.0953  (exactly opposite!)
   
   This symmetry makes mathematical analysis much cleaner.
   
2. LOG RETURNS vs PERCENTAGE RETURNS
   -----------------------------------
   Log return = ln(Price_t / Price_t-1) = ln(Price_t) - ln(Price_t-1)
   
   PROPERTIES:
   - Additive over time: Total return = sum of daily log returns
   - Approximately equal to % return for small moves
   - More normally distributed (important for statistics)
   
3. NO LOOKAHEAD BIAS
   -------------------
   When calculating signals for day T, we can ONLY use data from day T-1 
   and earlier. We NEVER use day T's data to make decisions for day T.
   
   This module ensures data is sorted chronologically to enable proper
   shifting in downstream calculations.

=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_gld_data(
    file_path: str,
    date_column: str = "Date",
    price_column: str = "Close",
    volume_column: str = "Volume"
) -> pd.DataFrame:
    """
    Load GLD price data from CSV and perform initial preprocessing.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing GLD data
    date_column : str
        Name of the column containing dates
    price_column : str
        Name of the column containing prices (usually 'Close')
    volume_column : str
        Name of the column containing volume
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        - date: datetime index
        - open, high, low, close: OHLC prices
        - volume: trading volume
        - log_close: natural log of close price
        - log_return: daily log return (ln(close_t / close_t-1))
        
    Notes
    -----
    The returned DataFrame is sorted by date in ASCENDING order (oldest first).
    This is critical for time-series analysis and avoiding lookahead bias.
    """
    
    # =========================================================================
    # STEP 1: Load the raw CSV file
    # =========================================================================
    # pandas.read_csv() reads the file into a DataFrame (think: Excel spreadsheet)
    # 
    # The raw data looks like:
    # Date       | Open   | High   | Low    | Close  | Change | %Change  | Volume
    # 1/3/11     | 138.67 | 139    | 137.88 | 138    | -0.72  | -0.52%   | 11510100
    
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")
    
    # =========================================================================
    # STEP 2: Parse dates properly
    # =========================================================================
    # The dates are in format "1/3/11" which means January 3, 2011
    # We need to convert this string to a proper datetime object
    #
    # WHY DATETIME?
    # - Enables date arithmetic (how many days between two dates?)
    # - Enables filtering by date range
    # - Ensures proper chronological sorting
    
    df[date_column] = pd.to_datetime(df[date_column], format='%m/%d/%y')
    
    # =========================================================================
    # STEP 3: Sort by date (oldest first)
    # =========================================================================
    # CRITICAL: Time series analysis requires chronological order
    # 
    # If data is sorted wrong, our calculations will be garbage:
    # - Returns would be calculated wrong
    # - Trends would be backwards
    # - Lookahead bias would be introduced
    
    df = df.sort_values(date_column).reset_index(drop=True)
    
    print(f"After sorting - Date range: {df[date_column].iloc[0]} to {df[date_column].iloc[-1]}")
    
    # =========================================================================
    # STEP 4: Create clean column names (lowercase)
    # =========================================================================
    # Standardize column names for consistency
    
    df = df.rename(columns={
        date_column: 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        price_column: 'close',
        volume_column: 'volume'
    })
    
    # =========================================================================
    # STEP 5: Clean the volume column
    # =========================================================================
    # Volume might have commas or be stored as string
    # Convert to numeric, replacing any errors with NaN
    
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'].astype(str).str.replace(',', ''), errors='coerce')
    
    # =========================================================================
    # STEP 6: Calculate LOG PRICES
    # =========================================================================
    # log_close = ln(close_price)
    #
    # MATHEMATICAL FOUNDATION:
    # ------------------------
    # The natural logarithm (ln) converts multiplicative relationships to additive:
    # 
    # If price goes: $100 → $110 → $121
    # Multiplicative: 1.10 × 1.10 = 1.21 (10% each day, 21% total)
    # 
    # Log prices: ln(100)=4.605, ln(110)=4.700, ln(121)=4.796
    # Log returns: 0.095 + 0.095 = 0.191 (just add them!)
    #
    # This additivity makes regression and trend analysis much easier.
    
    df['log_close'] = np.log(df['close'])
    
    # =========================================================================
    # STEP 7: Calculate LOG RETURNS
    # =========================================================================
    # log_return_t = ln(close_t / close_t-1) = log_close_t - log_close_t-1
    #
    # The .diff() method calculates the difference from the previous row:
    # diff[i] = log_close[i] - log_close[i-1]
    #
    # The first row will have NaN because there's no "previous day" to compare to.
    
    df['log_return'] = df['log_close'].diff()
    
    # =========================================================================
    # STEP 8: Set date as index
    # =========================================================================
    # Using date as the index makes time-series operations easier:
    # - df.loc['2015-01-01':'2015-12-31'] to filter by date range
    # - Automatic alignment when combining multiple series
    
    df = df.set_index('date')
    
    # =========================================================================
    # STEP 9: Select only the columns we need
    # =========================================================================
    # Drop unnecessary columns like 'Change' and '%Change' 
    # (we calculate our own returns)
    
    columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'log_close', 'log_return']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # =========================================================================
    # STEP 10: Basic data quality checks
    # =========================================================================
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nWARNING: Missing values detected:")
        print(missing[missing > 0])
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        print(f"\nWARNING: Duplicate dates detected!")
        df = df[~df.index.duplicated(keep='first')]
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Number of trading days: {len(df)}")
    
    return df


def filter_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific date range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    start_date : str, optional
        Start date (inclusive) in format 'YYYY-MM-DD'
    end_date : str, optional
        End date (inclusive) in format 'YYYY-MM-DD'
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
        
    Examples
    --------
    >>> df_2015 = filter_date_range(df, '2015-01-01', '2015-12-31')
    >>> df_from_2017 = filter_date_range(df, start_date='2017-01-01')
    """
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    print(f"Filtered to {len(df)} rows: {df.index.min()} to {df.index.max()}")
    
    return df


def calculate_price_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional return metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'log_return' columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - pct_return: Simple percentage return (close_t / close_t-1 - 1)
        - cum_log_return: Cumulative log return from start
        - cum_pct_return: Cumulative percentage return (for equity curve)
        
    Notes
    -----
    DIFFERENCE BETWEEN LOG AND PERCENTAGE RETURNS:
    
    Percentage return: r = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1
    Log return: R = ln(P_t / P_{t-1})
    
    Relationship: R = ln(1 + r)
    
    For small returns (|r| < 10%), they're approximately equal.
    For larger returns, they diverge:
    - 100% gain: r=1.0, R=0.693
    - 50% loss: r=-0.5, R=-0.693
    """
    
    df = df.copy()
    
    # Simple percentage return
    df['pct_return'] = df['close'].pct_change()
    
    # Cumulative log return (just sum them up)
    df['cum_log_return'] = df['log_return'].cumsum()
    
    # Cumulative percentage return (compound them)
    # (1 + r1) * (1 + r2) * ... - 1
    df['cum_pct_return'] = (1 + df['pct_return']).cumprod() - 1
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and return data
        
    Returns
    -------
    dict
        Summary statistics including:
        - Date range
        - Price statistics
        - Return statistics
        - Data quality metrics
    """
    
    summary = {
        'date_range': {
            'start': str(df.index.min().date()),
            'end': str(df.index.max().date()),
            'trading_days': len(df)
        },
        'price': {
            'start_price': df['close'].iloc[0],
            'end_price': df['close'].iloc[-1],
            'min_price': df['close'].min(),
            'max_price': df['close'].max(),
            'total_return_pct': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        },
        'returns': {
            'mean_daily_return': df['log_return'].mean() * 100,  # in percent
            'std_daily_return': df['log_return'].std() * 100,    # in percent
            'annualized_return': df['log_return'].mean() * 252 * 100,
            'annualized_volatility': df['log_return'].std() * np.sqrt(252) * 100,
            'min_daily_return': df['log_return'].min() * 100,
            'max_daily_return': df['log_return'].max() * 100
        },
        'data_quality': {
            'missing_values': df.isnull().sum().sum(),
            'zero_volume_days': (df['volume'] == 0).sum() if 'volume' in df.columns else 0
        }
    }
    
    return summary


def print_data_summary(summary: dict) -> None:
    """Pretty print the data summary."""
    
    print("\n" + "="*60)
    print("GLD DATA SUMMARY")
    print("="*60)
    
    print(f"\n📅 DATE RANGE:")
    print(f"   Start: {summary['date_range']['start']}")
    print(f"   End:   {summary['date_range']['end']}")
    print(f"   Trading Days: {summary['date_range']['trading_days']}")
    
    print(f"\n💰 PRICE:")
    print(f"   Start Price:  ${summary['price']['start_price']:.2f}")
    print(f"   End Price:    ${summary['price']['end_price']:.2f}")
    print(f"   Min Price:    ${summary['price']['min_price']:.2f}")
    print(f"   Max Price:    ${summary['price']['max_price']:.2f}")
    print(f"   Total Return: {summary['price']['total_return_pct']:.2f}%")
    
    print(f"\n📊 RETURNS:")
    print(f"   Mean Daily Return:     {summary['returns']['mean_daily_return']:.4f}%")
    print(f"   Daily Volatility:      {summary['returns']['std_daily_return']:.4f}%")
    print(f"   Annualized Return:     {summary['returns']['annualized_return']:.2f}%")
    print(f"   Annualized Volatility: {summary['returns']['annualized_volatility']:.2f}%")
    print(f"   Worst Day:             {summary['returns']['min_daily_return']:.2f}%")
    print(f"   Best Day:              {summary['returns']['max_daily_return']:.2f}%")
    
    print(f"\n✅ DATA QUALITY:")
    print(f"   Missing Values:    {summary['data_quality']['missing_values']}")
    print(f"   Zero Volume Days:  {summary['data_quality']['zero_volume_days']}")
    
    print("="*60)


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Test the data loading
    df = load_gld_data("data/processed/gld.csv")
    df = calculate_price_returns(df)
    summary = get_data_summary(df)
    print_data_summary(summary)
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Show last few rows
    print("\nLast 5 rows:")
    print(df.tail())

