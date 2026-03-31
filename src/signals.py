"""
=============================================================================
SIGNAL GENERATION MODULE
=============================================================================
This module converts technical indicators into trading signals (desired exposure).

THE CORE CONCEPT:
-----------------
We have calculated various indicators:
- slope_zscore: How strong is the trend? (+2 = strong uptrend, -2 = strong downtrend)
- volatility: How risky is the market right now?
- rsi: Is the market overbought/oversold?
- vol_trend: Is volatility rising or falling?

Now we need to COMBINE these into a single number: the EXPOSURE.

EXPOSURE means:
- +1.5 = Go 150% long (use 50% leverage to buy more gold)
- +1.0 = Go 100% long (fully invested in gold)
- +0.5 = Go 50% long (half invested, half cash)
- 0.0 = Stay in cash (no position)
- -0.5 = Go 50% short (bet against gold with half your capital)
- -1.0 = Go 100% short (fully short gold)

THE SIGNAL LOGIC:
-----------------
Our signal combines multiple components:

1. BASE SIGNAL (from trend):
   - Strong uptrend (slope_zscore > 1) → positive exposure
   - Strong downtrend (slope_zscore < -1) → negative exposure
   - Weak trend (slope_zscore near 0) → low exposure

2. VOLATILITY ADJUSTMENT:
   - High volatility → reduce exposure (protect from large losses)
   - Low volatility → can increase exposure
   
3. MEAN REVERSION MODIFIER:
   - Extreme RSI readings adjust the signal slightly
   - RSI < 30 (oversold) → add a bit to long signal
   - RSI > 70 (overbought) → reduce long signal a bit

4. FINAL CONSTRAINTS:
   - Clip to [-1.0, +1.5] as required
   - Apply smoothing to prevent whipsawing

AVOIDING LOOKAHEAD BIAS:
------------------------
CRITICAL: The signal for day T must only use information from day T-1 and earlier!

When we calculate exposure for day T:
- We SHIFT the signal by 1 day
- Signal[T] = function of indicators[T-1]

This means we're always trading on YESTERDAY's information, which is realistic.
In practice, we look at the close on Monday, make our decision, and trade at 
Tuesday's close.

=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_trend_signal(
    slope_zscore: pd.Series,
    min_threshold: float = 0.3
) -> pd.Series:
    """
    Generate the base trend-following signal from slope z-score.
    
    THE LOGIC:
    ----------
    We want to map the slope z-score to an exposure level.
    
    - slope_zscore = +2 → We're in a strong uptrend → Go long
    - slope_zscore = 0 → No clear trend → Stay neutral
    - slope_zscore = -2 → We're in a strong downtrend → Go short
    
    We use a simple approach: the signal IS the z-score, with some modifications:
    
    1. Apply a minimum threshold: If |zscore| < threshold, signal = 0
       This prevents trading in weak/choppy trends.
    
    2. Apply tanh transformation to bound the signal:
       tanh(x) maps any number to the range (-1, +1)
       This prevents extreme signals from outlier z-scores.
    
    THE MATH:
    ---------
    raw_signal = slope_zscore if |slope_zscore| > threshold else 0
    bounded_signal = tanh(raw_signal * 0.5) * 2
    
    The 0.5 and 2 are scaling factors to get reasonable exposure levels:
    - zscore = 1 → tanh(0.5) * 2 ≈ 0.92 → ~92% exposure
    - zscore = 2 → tanh(1.0) * 2 ≈ 1.52 → ~152% exposure (will be clipped)
    - zscore = 0.5 → tanh(0.25) * 2 ≈ 0.49 → ~49% exposure
    
    Parameters
    ----------
    slope_zscore : pd.Series
        Z-score of the trend slope
    min_threshold : float
        Minimum |zscore| to generate a non-zero signal
        
    Returns
    -------
    pd.Series
        Trend signal (roughly in range -1.5 to +1.5)
    """
    
    # Apply minimum threshold
    # If the trend is too weak, don't trade
    signal = slope_zscore.copy()
    signal[abs(signal) < min_threshold] = 0
    
    # Apply tanh transformation to bound the signal
    # tanh(x) smoothly maps any number to (-1, +1)
    # We scale it to get exposures in our desired range
    bounded_signal = np.tanh(signal * 0.5) * 2
    
    return bounded_signal


def generate_sma_crossover_signal(
    sma_fast: pd.Series,
    sma_slow: pd.Series,
    min_threshold: float = 0.0
) -> pd.Series:
    """
    Generate a trend signal from SMA crossover.

    THE LOGIC:
    ----------
    - If fast SMA > slow SMA => uptrend => long
    - If fast SMA < slow SMA => downtrend => short

    We use the normalized difference to create a smooth signal:
        diff = (sma_fast - sma_slow) / sma_slow

    Parameters
    ----------
    sma_fast : pd.Series
        Fast SMA
    sma_slow : pd.Series
        Slow SMA
    min_threshold : float
        Minimum absolute diff to trigger a non-zero signal

    Returns
    -------
    pd.Series
        Trend signal (roughly in range -1.5 to +1.5)
    """

    diff = (sma_fast - sma_slow) / (sma_slow + 1e-10)
    diff[diff.abs() < min_threshold] = 0
    return np.tanh(diff * 10) * 2


def generate_multi_timeframe_signal(
    fast_zscore: pd.Series,
    medium_zscore: pd.Series,
    fast_weight: float = 0.5,
    medium_weight: float = 0.5,
    min_threshold: float = 0.1
) -> pd.Series:
    """
    Blend trend signals from multiple timeframes for diversification.

    WHY BLEND TIMEFRAMES?
    ---------------------
    Different lookback periods capture different market dynamics:
    - Fast (20-day): Responsive, catches reversals early, more false signals
    - Medium (60-day): Smoother, fewer whipsaws, slower to react

    Blending reduces drawdowns because the two signals are imperfectly
    correlated -- when one whipsaws, the other may hold steady.

    Parameters
    ----------
    fast_zscore : pd.Series
        Z-score from the fast slope (e.g., 20-day)
    medium_zscore : pd.Series
        Z-score from the medium slope (e.g., 60-day)
    fast_weight : float
        Weight on the fast signal (0 to 1)
    medium_weight : float
        Weight on the medium signal (0 to 1)
    min_threshold : float
        Minimum |zscore| to generate non-zero signal (per timeframe)

    Returns
    -------
    pd.Series
        Blended trend signal
    """
    fast_signal = generate_trend_signal(fast_zscore, min_threshold=min_threshold)
    medium_signal = generate_trend_signal(medium_zscore, min_threshold=min_threshold)

    blended = fast_weight * fast_signal + medium_weight * medium_signal
    return blended


def apply_rebalance_threshold(
    signal: pd.Series,
    threshold: float = 0.05
) -> pd.Series:
    """
    Only change position when signal differs from current by more than threshold.

    WHY A REBALANCE THRESHOLD?
    --------------------------
    EMA smoothing changes exposure by a small amount EVERY day, which means
    you pay transaction costs every day. Over 9 years, this adds up to $6-9k
    on a $100k portfolio -- a massive drag on returns.

    A rebalance threshold creates a "dead band": if the new signal is within
    ±threshold of the current position, DON'T TRADE. Only rebalance when
    conviction has shifted meaningfully.

    Example with threshold=0.1:
        Raw signal: 0.50, 0.52, 0.48, 0.51, 0.30, 0.25
        Result:     0.50, 0.50, 0.50, 0.50, 0.30, 0.25
        (Days 2-4: no trade because change < 0.1, saving 3 days of costs)

    Parameters
    ----------
    signal : pd.Series
        Signal after smoothing
    threshold : float
        Minimum change to trigger rebalance (0 = disabled)

    Returns
    -------
    pd.Series
        Signal with dead-band rebalancing applied
    """
    if threshold <= 0:
        return signal

    values = signal.values.copy()
    current = 0.0

    for i in range(len(values)):
        if np.isnan(values[i]):
            continue
        if abs(values[i] - current) > threshold:
            current = values[i]
        else:
            values[i] = current

    return pd.Series(values, index=signal.index)


def apply_trend_regime_filter(
    signal: pd.Series,
    slow_slope_zscore: pd.Series,
    regime_threshold: float = 0.2,
    long_regime_threshold: Optional[float] = None,
    short_regime_threshold: Optional[float] = None,
    weak_regime_scale: float = 0.0
) -> pd.Series:
    """
    Filter trades based on a slow (long-term) trend regime.

    THE IDEA:
    ---------
    We only want to take fast-trend signals in the direction of the
    long-term regime to reduce whipsaws.

    - If slow trend is clearly positive, allow only long signals
    - If slow trend is clearly negative, allow only short signals
    - If slow trend is weak/unclear, go to cash

    ASYMMETRY (capture more upside):
    We use a lower bar for "long regime" than "short regime" so we don't
    stay flat for months after a bottom while the 180d slope catches up.
    long_regime_threshold < short_regime_threshold => go long earlier.

    Parameters
    ----------
    signal : pd.Series
        Fast trend signal
    slow_slope_zscore : pd.Series
        Slow trend z-score (regime indicator)
    regime_threshold : float
        Used for both if long/short thresholds not set
    long_regime_threshold : float, optional
        Min z-score to allow long signals (default: regime_threshold)
    short_regime_threshold : float, optional
        Min |z-score| to allow short signals (default: regime_threshold)
    weak_regime_scale : float
        Multiplier for signals in weak/unclear regime.
        0.0 = zero out (old behavior), 0.3 = keep 30% of signal.
        Setting > 0 prevents the regime filter from completely killing
        signals, which avoids the "filter stacking" problem.

    Returns
    -------
    pd.Series
        Regime-filtered signal
    """
    long_thresh = long_regime_threshold if long_regime_threshold is not None else regime_threshold
    short_thresh = short_regime_threshold if short_regime_threshold is not None else regime_threshold

    filtered = signal.copy()

    # Long regime: allow longs when slow trend is at least long_thresh (e.g. 0.15)
    long_regime = slow_slope_zscore >= long_thresh
    # Short regime: allow shorts only when slow trend is clearly negative (e.g. -0.3)
    short_regime = slow_slope_zscore <= -short_thresh
    # Weak regime: between -short_thresh and +long_thresh => scale down
    weak_regime = ~long_regime & ~short_regime
    filtered[weak_regime] = filtered[weak_regime] * weak_regime_scale

    filtered[long_regime] = filtered[long_regime].clip(lower=0)
    filtered[short_regime] = filtered[short_regime].clip(upper=0)

    return filtered


def apply_volatility_adjustment(
    signal: pd.Series,
    volatility: pd.Series,
    vol_percentile: pd.Series,
    target_vol: float = 0.15,
    high_vol_percentile: float = 80,
    high_vol_reduction: float = 0.5,
    high_vol_reduction_long: Optional[float] = None,
    target_vol_high_regime: Optional[float] = None
) -> pd.Series:
    """
    Adjust signal based on volatility regime.
    
    THE LOGIC:
    ----------
    We want to REDUCE exposure when volatility is high because:
    1. High volatility means larger potential losses
    2. Trends are less reliable during turbulent times
    3. We want to protect our capital
    
    Two approaches combined:
    
    1. VOLATILITY TARGETING:
       Scale exposure so that position_vol ≈ target_vol
       
       If gold's volatility is 30% and our target is 15%:
       We should only have 50% exposure (15/30 = 0.5)
       
       If gold's volatility is 10% and our target is 15%:
       We could have 150% exposure (15/10 = 1.5)
    
    2. REGIME-BASED REDUCTION:
       If volatility is in the top 20% historically, 
       apply an additional reduction factor.
       
       This is a "circuit breaker" for extreme turbulence.
    
    Parameters
    ----------
    signal : pd.Series
        Raw signal before volatility adjustment
    volatility : pd.Series
        Realized volatility (annualized)
    vol_percentile : pd.Series
        Percentile rank of current volatility
    target_vol : float
        Target portfolio volatility (annualized)
    high_vol_percentile : float
        Percentile above which we consider "high vol" regime
    high_vol_reduction : float
        Multiplier for exposure in high-vol regime when short (0.5 = cut by half)
    high_vol_reduction_long : float, optional
        Multiplier when long in high-vol regime (default: same as high_vol_reduction)
        
    Returns
    -------
    pd.Series
        Volatility-adjusted signal
    """
    
    # =========================================================================
    # STEP 1: Volatility targeting (regime-based: lower target in high vol)
    # =========================================================================
    # Scale factor = target_vol / current_vol
    # In high-vol regime, use target_vol_high_regime (smaller positions)
    high_vol_mask = vol_percentile > high_vol_percentile
    effective_target = pd.Series(target_vol, index=volatility.index)
    if target_vol_high_regime is not None:
        effective_target = effective_target.where(~high_vol_mask, target_vol_high_regime)
    vol_scale = effective_target / (volatility + 1e-10)
    
    # Cap the scale to prevent extreme leverage in very low vol periods
    vol_scale = vol_scale.clip(upper=2.0)
    
    # Apply volatility scaling to signal
    adjusted_signal = signal * vol_scale
    
    # =========================================================================
    # STEP 2: High volatility regime reduction
    # =========================================================================
    # Asymmetry: cut shorts more in high vol, cut longs less so we participate in rebounds
    vol_red_long = high_vol_reduction_long if high_vol_reduction_long is not None else high_vol_reduction
    long_vol = high_vol_mask & (adjusted_signal > 0)
    short_vol = high_vol_mask & (adjusted_signal <= 0)
    adjusted_signal[long_vol] = adjusted_signal[long_vol] * vol_red_long
    adjusted_signal[short_vol] = adjusted_signal[short_vol] * high_vol_reduction
    return adjusted_signal


def apply_volatility_momentum_adjustment(
    signal: pd.Series,
    vol_trend: pd.Series,
    reduction_threshold: float = 0.5
) -> pd.Series:
    """
    Adjust signal based on volatility momentum (is vol rising or falling?).
    
    THE LOGIC:
    ----------
    Rising volatility = Market becoming unstable = Be cautious
    Falling volatility = Market calming = Can be more aggressive
    
    If volatility is rapidly INCREASING (vol_trend > threshold):
    - Reduce exposure as a defensive measure
    - The market is becoming more dangerous
    
    If volatility is DECREASING:
    - No adjustment (keep current signal)
    - Market is stabilizing
    
    Parameters
    ----------
    signal : pd.Series
        Signal after volatility adjustment
    vol_trend : pd.Series
        Trend of volatility (positive = rising)
    reduction_threshold : float
        Vol trend above this triggers reduction
        
    Returns
    -------
    pd.Series
        Signal adjusted for volatility momentum
    """
    
    adjusted_signal = signal.copy()
    
    # If vol is rising fast, reduce exposure
    rising_vol_mask = vol_trend > reduction_threshold
    adjusted_signal[rising_vol_mask] = adjusted_signal[rising_vol_mask] * 0.5
    
    return adjusted_signal


def apply_mean_reversion_modifier(
    signal: pd.Series,
    rsi: pd.Series,
    oversold_threshold: float = 30,
    overbought_threshold: float = 70,
    adjustment: float = 0.3
) -> pd.Series:
    """
    Modify signal based on RSI mean reversion.
    
    THE LOGIC:
    ----------
    RSI captures short-term overbought/oversold conditions.
    
    Even in an uptrend, if RSI is very high (overbought), we might see a 
    short-term pullback. We don't want to go maximum long right before a dip.
    
    Similarly, in a downtrend, if RSI is very low (oversold), we might see 
    a bounce. We don't want to go maximum short right before a bounce.
    
    MODIFICATION RULES:
    - If RSI < oversold (30): Add 'adjustment' to signal (expect bounce)
    - If RSI > overbought (70): Subtract 'adjustment' from signal (expect dip)
    - Otherwise: No change
    
    This is a SMALL modification (±0.3), not a signal reversal.
    It's meant to improve entry timing, not override the trend.
    
    Parameters
    ----------
    signal : pd.Series
        Signal after volatility adjustments
    rsi : pd.Series
        RSI values (0-100)
    oversold_threshold : float
        RSI below this is oversold
    overbought_threshold : float
        RSI above this is overbought
    adjustment : float
        Amount to add/subtract at extreme RSI
        
    Returns
    -------
    pd.Series
        Signal with mean reversion modification
    """
    
    adjusted_signal = signal.copy()
    
    # Oversold: Add to long signal (expect bounce)
    oversold_mask = rsi < oversold_threshold
    adjusted_signal[oversold_mask] = adjusted_signal[oversold_mask] + adjustment
    
    # Overbought: Reduce long signal (expect pullback)
    overbought_mask = rsi > overbought_threshold
    adjusted_signal[overbought_mask] = adjusted_signal[overbought_mask] - adjustment
    
    return adjusted_signal


def apply_ulcer_filter(
    signal: pd.Series,
    ulcer_index: pd.Series,
    threshold: float = 0.02,
    reduction: float = 0.5,
    reduction_when_long: Optional[float] = None
) -> pd.Series:
    """
    Reduce exposure when drawdowns are persistent (high Ulcer Index).

    THE LOGIC:
    ----------
    Ulcer Index measures downside "pain". When it is elevated, we reduce
    risk because the market is unstable or trending against us.

    ASYMMETRY (capture more upside):
    After a drawdown, UI stays high just when gold starts to recover. If we
    cut all exposure we miss the rebound. So we apply full reduction only
    when SHORT; when long, use reduction_when_long (e.g. 1.0 = no cut).

    Parameters
    ----------
    signal : pd.Series
        Signal before ulcer adjustment
    ulcer_index : pd.Series
        Ulcer Index values
    threshold : float
        UI level above which we reduce exposure
    reduction : float
        Exposure multiplier when UI is high (applied when short or symmetric)
    reduction_when_long : float, optional
        Multiplier when UI high and signal is long (default: same as reduction)

    Returns
    -------
    pd.Series
        Ulcer-adjusted signal
    """
    if reduction_when_long is None:
        reduction_when_long = reduction
    adjusted = signal.copy()
    high_ulcer = ulcer_index > threshold
    # When long, use milder (or no) reduction so we don't cut participation in rebounds
    long_mask = high_ulcer & (adjusted > 0)
    short_mask = high_ulcer & (adjusted <= 0)
    adjusted[long_mask] = adjusted[long_mask] * reduction_when_long
    adjusted[short_mask] = adjusted[short_mask] * reduction
    return adjusted


def compute_risk_scalar(
    signal: pd.Series,
    vol_percentile: pd.Series,
    ulcer_index: pd.Series,
    high_vol_percentile: float = 90,
    high_vol_reduction: float = 0.7,
    high_vol_reduction_long: float = 0.9,
    ulcer_threshold: float = 0.02,
    ulcer_reduction: float = 0.5,
    ulcer_reduction_when_long: float = 1.0,
) -> pd.Series:
    """
    Combine defensive risk filters into a SINGLE scalar using min().

    THE PROBLEM WITH STACKING:
    --------------------------
    If you apply 3 filters sequentially, each reducing by 30-50%:
        signal * 0.7 * 0.5 * 0.5 = signal * 0.175  (82.5% reduction!)

    With min(), only the most pessimistic filter matters:
        min(0.7, 0.5, 0.5) = 0.5  (50% reduction)

    This prevents the multiplicative crush that kills exposure.

    Parameters
    ----------
    signal : pd.Series
        Current signal (used to determine long/short direction)
    vol_percentile : pd.Series
        Percentile rank of current volatility
    ulcer_index : pd.Series
        Ulcer Index values
    high_vol_percentile : float
        Percentile above which we consider "high vol"
    high_vol_reduction : float
        Multiplier when short in high-vol regime
    high_vol_reduction_long : float
        Multiplier when long in high-vol regime
    ulcer_threshold : float
        UI level above which we reduce exposure
    ulcer_reduction : float
        Multiplier when short and UI is high
    ulcer_reduction_when_long : float
        Multiplier when long and UI is high

    Returns
    -------
    pd.Series
        Risk scalar in [0, 1] to multiply signal by
    """
    is_long = signal > 0

    # High-vol regime scalar
    high_vol = vol_percentile > high_vol_percentile
    vol_s = pd.Series(1.0, index=signal.index)
    vol_s[high_vol & is_long] = high_vol_reduction_long
    vol_s[high_vol & ~is_long] = high_vol_reduction

    # Ulcer (drawdown pain) scalar
    high_ulcer = ulcer_index > ulcer_threshold
    ulcer_s = pd.Series(1.0, index=signal.index)
    ulcer_s[high_ulcer & is_long] = ulcer_reduction_when_long
    ulcer_s[high_ulcer & ~is_long] = ulcer_reduction

    # Take the MIN (most cautious single view), NOT the product
    risk_scalar = pd.DataFrame({'vol': vol_s, 'ulcer': ulcer_s}).min(axis=1)

    return risk_scalar


def apply_exposure_constraints(
    signal: pd.Series,
    max_long: float = 1.5,
    max_short: float = -1.0
) -> pd.Series:
    """
    Apply leverage constraints to the signal.
    
    THE RULES:
    ----------
    Per the challenge requirements:
    - Maximum long exposure: 1.5 (150% long, meaning 50% leverage)
    - Maximum short exposure: -1.0 (100% short)
    
    We use numpy.clip to enforce these bounds.
    
    Parameters
    ----------
    signal : pd.Series
        Raw signal (can be any value)
    max_long : float
        Maximum positive exposure
    max_short : float
        Maximum negative exposure (should be negative)
        
    Returns
    -------
    pd.Series
        Constrained signal within [max_short, max_long]
    """
    
    return signal.clip(lower=max_short, upper=max_long)


def smooth_signal(
    signal: pd.Series,
    alpha: float = 0.3
) -> pd.Series:
    """
    Apply exponential smoothing to the signal to reduce whipsawing.
    
    WHY SMOOTH?
    -----------
    Without smoothing, the signal might flip rapidly:
    Day 1: +0.8 (long)
    Day 2: -0.2 (short)
    Day 3: +0.5 (long)
    
    This causes:
    1. High transaction costs (buying and selling frequently)
    2. Potential losses from bid-ask spread
    3. Possible execution issues
    
    Smoothing makes transitions gradual:
    Day 1: +0.8 (long)
    Day 2: +0.5 (reduced long)
    Day 3: +0.5 (maintained)
    
    THE MATH:
    ---------
    Exponential Moving Average (EMA):
    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]
    
    - alpha = 1: No smoothing, smoothed = raw
    - alpha = 0.5: Half current, half previous
    - alpha = 0.1: Very smooth, slow to change
    
    We use alpha = 0.3, which means the signal responds moderately fast
    but doesn't flip-flop on noise.
    
    Parameters
    ----------
    signal : pd.Series
        Raw signal
    alpha : float
        Smoothing factor (0 < alpha <= 1)
        Higher = more responsive, Lower = smoother
        
    Returns
    -------
    pd.Series
        Smoothed signal
    """
    
    # ewm = Exponential Weighted Moving average
    # span = 2/alpha - 1 (conversion between alpha and span)
    return signal.ewm(alpha=alpha, adjust=False).mean()


def shift_signal_for_execution(
    signal: pd.Series,
    shift_days: int = 1
) -> pd.Series:
    """
    Shift signal to avoid lookahead bias.
    
    THIS IS CRITICAL FOR REALISTIC BACKTESTING!
    
    THE PROBLEM:
    ------------
    If we calculate signal[T] using data[T] and then trade at close[T],
    we're using information we wouldn't have in real life!
    
    In reality:
    - We see data up to close[T]
    - We make our decision overnight
    - We execute at close[T+1]
    
    THE SOLUTION:
    -------------
    Shift the signal by 1 day:
    - execution_signal[T] = computed_signal[T-1]
    
    This means:
    - On day T, we execute based on what we calculated using day T-1's data
    - The first day has NaN (no signal to execute)
    
    Parameters
    ----------
    signal : pd.Series
        Computed signal (uses data up to that day)
    shift_days : int
        Number of days to shift (1 for daily trading)
        
    Returns
    -------
    pd.Series
        Shifted signal for execution
    """
    
    return signal.shift(shift_days)


def generate_signals(
    df: pd.DataFrame,
    min_slope_threshold: float = 0.3,
    slow_regime_threshold: float = 0.2,
    long_regime_threshold: Optional[float] = None,
    short_regime_threshold: Optional[float] = None,
    weak_regime_scale: float = 0.0,
    trend_model: str = "slope",
    sma_diff_threshold: float = 0.0,
    target_vol: float = 0.15,
    target_vol_high_regime: Optional[float] = None,
    high_vol_percentile: float = 80,
    high_vol_reduction: float = 0.5,
    high_vol_reduction_long: Optional[float] = None,
    vol_trend_reduction_threshold: float = 0.5,
    oversold_threshold: float = 30,
    overbought_threshold: float = 70,
    rsi_adjustment: float = 0.3,
    ulcer_threshold: float = 0.02,
    ulcer_reduction: float = 0.5,
    ulcer_reduction_when_long: Optional[float] = None,
    max_long: float = 1.5,
    max_short: float = -1.0,
    smoothing_alpha: float = 0.3,
    use_multi_timeframe: bool = False,
    fast_weight: float = 0.5,
    medium_weight: float = 0.5,
    rebalance_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Generate trading signals from indicators.
    
    This is the main signal generation function that orchestrates all the steps:
    1. Generate base trend signal
    2. Apply volatility adjustment
    3. Apply volatility momentum adjustment
    4. Apply mean reversion modifier
    5. Apply constraints
    6. Smooth the signal
    7. Shift for execution (avoid lookahead)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all indicators calculated
    [various parameters for each component]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with signal columns added:
        - raw_signal: Signal before constraints and smoothing
        - signal: Final signal after all processing
        - exposure: Shifted signal for actual execution
    """
    
    df = df.copy()
    
    print("\nGenerating trading signals...")
    
    # =========================================================================
    # STEP 1: Generate base trend signal
    # =========================================================================
    if trend_model == "sma_crossover":
        print("  - Generating base trend signal from SMA crossover")
        df['trend_signal'] = generate_sma_crossover_signal(
            df['sma_fast'],
            df['sma_slow'],
            min_threshold=sma_diff_threshold
        )
    elif use_multi_timeframe:
        print(f"  - Generating multi-timeframe signal (fast={fast_weight:.1f}, medium={medium_weight:.1f})")
        df['trend_signal'] = generate_multi_timeframe_signal(
            df['slope_fast_zscore'],
            df['slope_zscore'],
            fast_weight=fast_weight,
            medium_weight=medium_weight,
            min_threshold=min_slope_threshold
        )
    else:
        print("  - Generating base trend signal from FAST slope z-score")
        df['trend_signal'] = generate_trend_signal(
            df['slope_fast_zscore'],
            min_threshold=min_slope_threshold
        )

    # Apply slow-trend regime filter (always available)
    print("  - Applying slow-trend regime filter")
    df['regime_signal'] = apply_trend_regime_filter(
        df['trend_signal'],
        df['slope_slow_zscore'],
        regime_threshold=slow_regime_threshold,
        long_regime_threshold=long_regime_threshold,
        short_regime_threshold=short_regime_threshold,
        weak_regime_scale=weak_regime_scale
    )
    
    # =========================================================================
    # STEP 2: Volatility targeting (position sizing only)
    # =========================================================================
    # Vol targeting scales exposure so strategy vol ≈ target_vol.
    # High-vol regime reduction is handled by the risk scalar below,
    # NOT here, to avoid multiplicative filter stacking.
    print("  - Applying volatility targeting (position sizing)")
    df['vol_adj_signal'] = apply_volatility_adjustment(
        df['regime_signal'],
        df['volatility'],
        df['vol_percentile'],
        target_vol=target_vol,
        target_vol_high_regime=target_vol_high_regime,
        high_vol_percentile=high_vol_percentile,
        high_vol_reduction=1.0,
        high_vol_reduction_long=1.0
    )
    
    # =========================================================================
    # STEP 3: Unified risk scalar
    # =========================================================================
    # Instead of stacking 4 separate multiplicative filters
    # (vol regime * vol momentum * RSI * ulcer = near-zero),
    # we combine them into ONE scalar using min().
    # Only the most pessimistic filter matters on any given day.
    print("  - Computing unified risk scalar (min of vol regime, ulcer)")
    _hvr_long = high_vol_reduction_long if high_vol_reduction_long is not None else high_vol_reduction
    _ulcer_long = ulcer_reduction_when_long if ulcer_reduction_when_long is not None else ulcer_reduction
    df['risk_scalar'] = compute_risk_scalar(
        df['vol_adj_signal'],
        df['vol_percentile'],
        df['ulcer_index'],
        high_vol_percentile=high_vol_percentile,
        high_vol_reduction=high_vol_reduction,
        high_vol_reduction_long=_hvr_long,
        ulcer_threshold=ulcer_threshold,
        ulcer_reduction=ulcer_reduction,
        ulcer_reduction_when_long=_ulcer_long,
    )
    
    # Store raw signal (before constraints and smoothing)
    df['raw_signal'] = df['vol_adj_signal'] * df['risk_scalar']
    
    # =========================================================================
    # STEP 4: Apply exposure constraints
    # =========================================================================
    print(f"  - Applying exposure constraints [{max_short}, {max_long}]")
    df['constrained_signal'] = apply_exposure_constraints(
        df['raw_signal'],
        max_long=max_long,
        max_short=max_short
    )
    
    # =========================================================================
    # STEP 5: Smooth the signal
    # =========================================================================
    print(f"  - Smoothing signal (alpha={smoothing_alpha})")
    df['signal'] = smooth_signal(
        df['constrained_signal'],
        alpha=smoothing_alpha
    )
    
    # Apply constraints again after smoothing (smoothing might push outside bounds)
    df['signal'] = apply_exposure_constraints(
        df['signal'],
        max_long=max_long,
        max_short=max_short
    )
    
    # =========================================================================
    # STEP 5b: Rebalance threshold (reduce turnover)
    # =========================================================================
    if rebalance_threshold > 0:
        print(f"  - Applying rebalance threshold ({rebalance_threshold})")
        df['signal'] = apply_rebalance_threshold(
            df['signal'],
            threshold=rebalance_threshold
        )
    
    # =========================================================================
    # STEP 6: Shift for execution (CRITICAL!)
    # =========================================================================
    print("  - Shifting signal by 1 day to avoid lookahead bias")
    df['exposure'] = shift_signal_for_execution(df['signal'], shift_days=1)
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    valid_signals = df['exposure'].dropna()
    print(f"\nSignal Statistics:")
    print(f"  - Valid signals: {len(valid_signals)} days")
    print(f"  - Mean exposure: {valid_signals.mean():.3f}")
    print(f"  - Median exposure: {valid_signals.median():.3f}")
    print(f"  - Std exposure: {valid_signals.std():.3f}")
    print(f"  - Min exposure: {valid_signals.min():.3f}")
    print(f"  - Max exposure: {valid_signals.max():.3f}")
    
    # Time in market calculation
    active = (abs(valid_signals) > 0.1).sum()
    time_in_market = active / len(valid_signals) * 100
    print(f"  - Time in market (|exposure| > 0.1): {time_in_market:.1f}%")
    
    return df


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Import required modules for testing
    from data import load_gld_data, calculate_price_returns
    from indicators import calculate_all_indicators
    
    # Load and prepare data
    df = load_gld_data("data/processed/gld.csv")
    df = calculate_price_returns(df)
    df = calculate_all_indicators(df)
    
    # Generate signals
    df = generate_signals(df)
    
    # Show sample of signals
    print("\nSample of signals:")
    cols = ['close', 'slope_zscore', 'volatility', 'rsi', 'signal', 'exposure']
    print(df[cols].dropna().tail(20))

