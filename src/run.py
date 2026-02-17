"""
=============================================================================
MAIN EXECUTION MODULE
=============================================================================
This is the main entry point for running the GLD trading strategy.

HOW TO USE:
-----------
From the command line:
    python src/run.py

Or from Python:
    from run import run_strategy
    results = run_strategy()

WHAT THIS DOES:
---------------
1. Loads configuration from configs/default.yaml
2. Loads and preprocesses GLD price data
3. Calculates technical indicators
4. Generates trading signals
5. Runs the backtest simulation
6. Calculates and displays performance metrics
7. Optionally saves results and generates plots

=============================================================================
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_gld_data, calculate_price_returns, filter_date_range, print_data_summary, get_data_summary
from indicators import calculate_all_indicators
from signals import generate_signals
from backtest import run_backtest
from metrics import calculate_all_metrics, print_metrics_report


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    
    # Handle relative paths
    if not os.path.isabs(config_path):
        # Look for config relative to workspace root
        workspace_root = Path(__file__).parent.parent
        config_path = workspace_root / config_path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    
    return config


def run_strategy(
    config_path: str = "configs/default.yaml",
    save_results: bool = True,
    generate_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run the complete trading strategy pipeline.
    
    This function orchestrates the entire workflow:
    1. Load configuration
    2. Load and preprocess data
    3. Calculate indicators
    4. Generate signals
    5. Run backtest
    6. Calculate metrics
    7. (Optional) Save results and generate plots
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    save_results : bool
        Whether to save results to files
    generate_plots : bool
        Whether to generate visualization plots
    verbose : bool
        Whether to print detailed output
        
    Returns
    -------
    dict
        Dictionary containing all results:
        - 'config': Configuration used
        - 'data': Processed DataFrame with all columns
        - 'backtest': Backtest results
        - 'metrics': Performance metrics
    """
    
    print("="*70)
    print("        GLD QUANTITATIVE TRADING STRATEGY")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # STEP 1: Load Configuration
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading Configuration")
    print("="*70)
    
    config = load_config(config_path)
    
    if verbose:
        print("\nKey parameters:")
        print(f"  - Trend lookback: {config['trend']['lookback_days']} days")
        print(f"  - Volatility lookback: {config['volatility']['lookback_days']} days")
        print(f"  - RSI period: {config['mean_reversion']['rsi_period']} days")
        print(f"  - Leverage range: [{config['exposure']['max_short']}, {config['exposure']['max_long']}]")
    
    # =========================================================================
    # STEP 2: Load and Preprocess Data
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Loading and Preprocessing Data")
    print("="*70)
    
    # Get workspace root
    workspace_root = Path(__file__).parent.parent
    data_path = workspace_root / config['data']['file_path']
    
    df = load_gld_data(
        file_path=str(data_path),
        date_column=config['data']['date_column'],
        price_column=config['data']['price_column'],
        volume_column=config['data']['volume_column']
    )
    
    df = calculate_price_returns(df)
    
    if verbose:
        summary = get_data_summary(df)
        print_data_summary(summary)
    
    # =========================================================================
    # STEP 3: Calculate Indicators
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Calculating Technical Indicators")
    print("="*70)
    
    df = calculate_all_indicators(
        df,
        trend_lookback=config['trend']['lookback_days'],
        trend_fast_lookback=config['trend']['fast_lookback_days'],
        trend_slow_lookback=config['trend']['slow_lookback_days'],
        trend_zscore_lookback=config['trend']['zscore_lookback'],
        sma_fast_lookback=config['trend']['sma_fast_lookback_days'],
        sma_slow_lookback=config['trend']['sma_slow_lookback_days'],
        vol_lookback=config['volatility']['lookback_days'],
        vol_percentile_lookback=252,  # 1 year
        vol_trend_lookback=config['volatility']['vol_trend_lookback'],
        rsi_period=config['mean_reversion']['rsi_period'],
        ulcer_lookback=config['ulcer']['lookback_days']
    )
    
    # =========================================================================
    # STEP 4: Generate Signals
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating Trading Signals")
    print("="*70)
    
    df = generate_signals(
        df,
        min_slope_threshold=config['trend']['min_slope_threshold'],
        slow_regime_threshold=config['trend']['slow_regime_threshold'],
        long_regime_threshold=config['trend'].get('long_regime_threshold'),
        short_regime_threshold=config['trend'].get('short_regime_threshold'),
        weak_regime_scale=config['trend'].get('weak_regime_scale', 0.0),
        trend_model=config['trend']['model'],
        sma_diff_threshold=config['trend']['sma_diff_threshold'],
        target_vol=config['volatility']['target_volatility'],
        target_vol_high_regime=config['volatility'].get('target_vol_high_regime'),
        high_vol_percentile=config['volatility']['high_vol_percentile'],
        high_vol_reduction=config['volatility']['high_vol_reduction'],
        high_vol_reduction_long=config['volatility'].get('high_vol_reduction_long'),
        vol_trend_reduction_threshold=0.5,
        oversold_threshold=config['mean_reversion']['oversold_threshold'],
        overbought_threshold=config['mean_reversion']['overbought_threshold'],
        rsi_adjustment=config['mean_reversion']['extreme_adjustment'],
        ulcer_threshold=config['ulcer']['threshold'],
        ulcer_reduction=config['ulcer']['reduction'],
        ulcer_reduction_when_long=config['ulcer'].get('reduction_when_long'),
        max_long=config['exposure']['max_long'],
        max_short=config['exposure']['max_short'],
        smoothing_alpha=config['exposure']['smoothing_alpha']
    )
    
    # =========================================================================
    # STEP 5: Run Backtest
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Running Backtest")
    print("="*70)
    
    dd_scaling = config['backtest'].get('drawdown_scaling', {})
    backtest_results = run_backtest(
        df,
        transaction_cost=config['backtest']['transaction_cost'],
        initial_capital=config['backtest']['initial_capital'],
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
        drawdown_threshold=dd_scaling.get('threshold'),
        drawdown_scale_min=dd_scaling.get('scale_min'),
        max_drawdown_days=dd_scaling.get('max_drawdown_days'),
        time_stop_scale=dd_scaling.get('time_stop_scale', 0.5),
        max_short=config['exposure'].get('max_short', -1.0),
        max_long=config['exposure'].get('max_long', 1.5)
    )
    
    # =========================================================================
    # STEP 6: Calculate Metrics
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: Calculating Performance Metrics")
    print("="*70)
    
    metrics = calculate_all_metrics(
        backtest_results,
        risk_free_rate=0.02,
        min_active_exposure=config['exposure']['min_active_exposure']
    )
    
    print_metrics_report(metrics)
    
    # =========================================================================
    # STEP 7: Generate Plots (Optional)
    # =========================================================================
    if generate_plots:
        print("\n" + "="*70)
        print("STEP 7: Generating Visualizations")
        print("="*70)
        
        plot_results(backtest_results, save=save_results)
    
    # =========================================================================
    # STEP 8: Save Results (Optional)
    # =========================================================================
    if save_results:
        print("\n" + "="*70)
        print("STEP 8: Saving Results")
        print("="*70)
        
        save_backtest_results(backtest_results, metrics, config)
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "="*70)
    print("                    STRATEGY RUN COMPLETE")
    print("="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'config': config,
        'data': backtest_results['data'],
        'backtest': backtest_results,
        'metrics': metrics
    }


def plot_results(backtest_results: Dict, save: bool = False) -> None:
    """
    Generate visualization plots for the backtest results.
    
    Creates 4 subplots:
    1. Equity curve (Strategy vs Buy & Hold)
    2. Drawdown comparison
    3. Exposure over time
    4. Rolling Sharpe ratio
    
    Parameters
    ----------
    backtest_results : dict
        Results from run_backtest()
    save : bool
        Whether to save the plot to file
    """
    
    df = backtest_results['data']
    
    # Filter to valid data
    valid_mask = df['equity'].notna()
    plot_df = df[valid_mask].copy()
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('GLD Trading Strategy Performance', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # PLOT 1: Equity Curve
    # =========================================================================
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['equity'], label='Strategy', linewidth=1.5, color='#2E86AB')
    ax1.plot(plot_df.index, plot_df['bh_equity'], label='Buy & Hold', linewidth=1, color='#A23B72', alpha=0.7)
    ax1.set_title('Equity Curve', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(plot_df.index.min(), plot_df.index.max())
    
    # Add final values annotation
    final_strat = plot_df['equity'].iloc[-1]
    final_bh = plot_df['bh_equity'].iloc[-1]
    ax1.annotate(f'${final_strat:,.0f}', xy=(plot_df.index[-1], final_strat),
                 xytext=(10, 0), textcoords='offset points', fontsize=9, color='#2E86AB')
    ax1.annotate(f'${final_bh:,.0f}', xy=(plot_df.index[-1], final_bh),
                 xytext=(10, 0), textcoords='offset points', fontsize=9, color='#A23B72')
    
    # =========================================================================
    # PLOT 2: Drawdowns
    # =========================================================================
    ax2 = axes[1]
    ax2.fill_between(plot_df.index, plot_df['drawdown'] * 100, 0, 
                     label='Strategy', alpha=0.5, color='#2E86AB')
    ax2.fill_between(plot_df.index, plot_df['bh_drawdown'] * 100, 0,
                     label='Buy & Hold', alpha=0.3, color='#A23B72')
    ax2.set_title('Drawdowns', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(plot_df.index.min(), plot_df.index.max())
    
    # Add max drawdown annotations
    max_dd = plot_df['drawdown'].min() * 100
    max_dd_date = plot_df['drawdown'].idxmin()
    ax2.annotate(f'Max: {max_dd:.1f}%', xy=(max_dd_date, max_dd),
                 xytext=(0, -20), textcoords='offset points', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#2E86AB'), color='#2E86AB')
    
    # =========================================================================
    # PLOT 3: Exposure Over Time
    # =========================================================================
    ax3 = axes[2]
    
    # Color based on long/short
    colors = np.where(plot_df['exposure'] >= 0, '#2E86AB', '#E74C3C')
    ax3.bar(plot_df.index, plot_df['exposure'], color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Max Long')
    ax3.axhline(y=-1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max Short')
    ax3.set_title('Position Exposure Over Time', fontsize=12)
    ax3.set_ylabel('Exposure')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(plot_df.index.min(), plot_df.index.max())
    ax3.set_ylim(-1.5, 2.0)
    
    # =========================================================================
    # PLOT 4: GLD Price with Exposure Overlay
    # =========================================================================
    ax4 = axes[3]
    
    # Plot price
    ax4.plot(plot_df.index, plot_df['close'], label='GLD Price', linewidth=1, color='black')
    ax4.set_ylabel('GLD Price ($)', color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.set_title('GLD Price and Signal', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(plot_df.index.min(), plot_df.index.max())
    
    # Overlay exposure on secondary axis
    ax4_twin = ax4.twinx()
    ax4_twin.fill_between(plot_df.index, plot_df['exposure'], 0, 
                          where=plot_df['exposure'] >= 0, alpha=0.2, color='green', label='Long')
    ax4_twin.fill_between(plot_df.index, plot_df['exposure'], 0,
                          where=plot_df['exposure'] < 0, alpha=0.2, color='red', label='Short')
    ax4_twin.set_ylabel('Exposure', color='gray')
    ax4_twin.tick_params(axis='y', labelcolor='gray')
    ax4_twin.set_ylim(-1.5, 2.0)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        workspace_root = Path(__file__).parent.parent
        reports_dir = workspace_root / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        filename = reports_dir / f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.show()


def save_backtest_results(
    backtest_results: Dict,
    metrics: Dict,
    config: Dict
) -> None:
    """
    Save backtest results to files.
    
    Saves:
    1. Full data as CSV
    2. Metrics as YAML
    3. Summary as text file
    
    Parameters
    ----------
    backtest_results : dict
        Results from run_backtest()
    metrics : dict
        Metrics from calculate_all_metrics()
    config : dict
        Configuration used
    """
    
    workspace_root = Path(__file__).parent.parent
    reports_dir = workspace_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data to CSV
    data_file = reports_dir / f'backtest_data_{timestamp}.csv'
    backtest_results['data'].to_csv(data_file)
    print(f"Data saved to: {data_file}")
    
    # Save metrics to YAML
    metrics_file = reports_dir / f'metrics_{timestamp}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"Metrics saved to: {metrics_file}")


def run_robustness_analysis(
    config_path: str = "configs/default.yaml",
    shock_pct: float = 0.10
) -> Dict:
    """
    Run robustness analysis by shocking key parameters.
    
    Tests the strategy with ±10% changes to key parameters to verify
    it doesn't fall apart with small parameter changes.
    
    Parameters
    ----------
    config_path : str
        Path to base configuration
    shock_pct : float
        Percentage to shock parameters (0.10 = 10%)
        
    Returns
    -------
    dict
        Results for base and shocked parameter runs
    """
    
    print("\n" + "="*70)
    print("        ROBUSTNESS ANALYSIS")
    print("="*70)
    
    # Load base config
    config = load_config(config_path)
    
    # Parameters to shock
    params_to_test = [
        ('trend', 'lookback_days'),
        ('volatility', 'lookback_days'),
        ('mean_reversion', 'rsi_period')
    ]
    
    results = {}
    
    # Run base case
    print("\nRunning BASE case...")
    base_results = run_strategy(config_path, save_results=False, generate_plots=False, verbose=False)
    results['base'] = {
        'calmar': base_results['metrics']['strategy']['calmar_ratio'],
        'cagr': base_results['metrics']['strategy']['cagr'],
        'max_dd': base_results['metrics']['strategy']['max_drawdown'],
        'sharpe': base_results['metrics']['strategy']['sharpe_ratio']
    }
    
    # Test each parameter
    for section, param in params_to_test:
        base_value = config[section][param]
        
        for direction in ['up', 'down']:
            shock_multiplier = 1 + shock_pct if direction == 'up' else 1 - shock_pct
            shocked_value = int(base_value * shock_multiplier)
            
            print(f"\nRunning {section}.{param}: {base_value} → {shocked_value} ({direction} shock)...")
            
            # Modify config
            shocked_config = config.copy()
            shocked_config[section] = config[section].copy()
            shocked_config[section][param] = shocked_value
            
            # Save temp config
            workspace_root = Path(__file__).parent.parent
            temp_config_path = workspace_root / 'configs' / 'temp_shock.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(shocked_config, f)
            
            # Run with shocked config
            try:
                shocked_results = run_strategy(
                    str(temp_config_path), 
                    save_results=False, 
                    generate_plots=False, 
                    verbose=False
                )
                
                key = f"{section}.{param}_{direction}"
                results[key] = {
                    'param_value': shocked_value,
                    'calmar': shocked_results['metrics']['strategy']['calmar_ratio'],
                    'cagr': shocked_results['metrics']['strategy']['cagr'],
                    'max_dd': shocked_results['metrics']['strategy']['max_drawdown'],
                    'sharpe': shocked_results['metrics']['strategy']['sharpe_ratio']
                }
            except Exception as e:
                print(f"  Error: {e}")
                results[f"{section}.{param}_{direction}"] = {'error': str(e)}
            
            # Clean up temp config
            temp_config_path.unlink(missing_ok=True)
    
    # Print summary
    print("\n" + "="*70)
    print("        ROBUSTNESS SUMMARY")
    print("="*70)
    print(f"\n{'Case':<35} {'Calmar':>10} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>10}")
    print("-"*75)
    
    for case, res in results.items():
        if 'error' in res:
            print(f"{case:<35} {'ERROR':>10}")
        else:
            print(f"{case:<35} {res['calmar']:>10.2f} {res['cagr']:>9.1f}% {res['max_dd']:>9.1f}% {res['sharpe']:>10.2f}")
    
    # Check if all cases meet Calmar target
    all_pass = all(
        res.get('calmar', 0) >= 2.0 
        for res in results.values() 
        if 'error' not in res
    )
    
    print("\n" + "-"*75)
    if all_pass:
        print("✅ ROBUSTNESS CHECK PASSED: All cases maintain Calmar ≥ 2.0")
    else:
        print("⚠️  ROBUSTNESS CHECK: Some cases fall below Calmar 2.0 target")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GLD Trading Strategy')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    parser.add_argument('--no-plots', action='store_true',
                        help='Do not generate plots')
    parser.add_argument('--robustness', action='store_true',
                        help='Run robustness analysis')
    
    args = parser.parse_args()
    
    if args.robustness:
        run_robustness_analysis(args.config)
    else:
        results = run_strategy(
            config_path=args.config,
            save_results=not args.no_save,
            generate_plots=not args.no_plots,
            verbose=True
        )

