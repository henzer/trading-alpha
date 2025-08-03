import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class VectorbtEngine:
    """
    High-performance backtesting engine using vectorbt.
    
    Provides institutional-grade backtesting with proper transaction costs,
    slippage modeling, and comprehensive performance analytics.
    """
    
    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.0005,
        initial_cash: float = 100000.0,
        freq: str = "1D"
    ):
        """
        Initialize VectorBT backtesting engine.
        
        Args:
            commission: Commission rate as decimal (0.001 = 0.1%)
            slippage: Slippage rate as decimal (0.0005 = 0.05%)
            initial_cash: Initial portfolio cash
            freq: Data frequency for performance calculations
        """
        self.commission = commission
        self.slippage = slippage
        self.initial_cash = initial_cash
        self.freq = freq
        
        # Setup logging
        self.logger = logging.getLogger("vectorbt_engine")
        
        # Configure vectorbt settings
        vbt.settings.set_theme("dark")
        vbt.settings['plotting']['layout']['width'] = 800
        vbt.settings['plotting']['layout']['height'] = 400
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest with vectorbt.
        
        Args:
            price_data: DataFrame with price data (Close prices)
            weights: DataFrame with portfolio weights over time
            benchmark_returns: Optional benchmark returns for comparison
            leverage: Portfolio leverage multiplier
            
        Returns:
            Dictionary with backtest results and portfolio object
        """
        self.logger.info("Starting vectorbt backtest")
        
        try:
            # Align price data and weights
            price_data, weights = self._align_data(price_data, weights)
            
            if price_data.empty or weights.empty:
                raise ValueError("No overlapping data between prices and weights")
            
            # Calculate returns
            returns = price_data.pct_change().fillna(0)
            
            # Apply leverage to weights
            leveraged_weights = weights * leverage
            
            # Ensure weights sum to target (leverage)
            weight_sums = leveraged_weights.sum(axis=1)
            weight_sums[weight_sums == 0] = 1  # Avoid division by zero
            normalized_weights = leveraged_weights.div(weight_sums, axis=0) * leverage
            
            # Calculate portfolio returns
            portfolio_returns = (normalized_weights.shift(1) * returns).sum(axis=1)
            
            # Apply transaction costs
            portfolio_returns = self._apply_transaction_costs(
                portfolio_returns, normalized_weights
            )
            
            # Create vectorbt portfolio using signals approach
            # Generate buy/sell signals from weights
            signals = normalized_weights.diff().fillna(0)
            entries = signals > 0
            exits = signals < 0
            
            portfolio = vbt.Portfolio.from_signals(
                close=price_data,
                entries=entries,
                exits=exits,
                init_cash=self.initial_cash,
                freq=self.freq,
                fees=self.commission,
                slippage=self.slippage
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio, benchmark_returns
            )
            
            # Calculate additional strategy metrics
            strategy_metrics = self._calculate_strategy_metrics(
                normalized_weights, returns, portfolio_returns
            )
            
            # Combine results
            results = {
                'portfolio': portfolio,
                'portfolio_returns': portfolio_returns,
                'weights': normalized_weights,
                'performance_metrics': performance_metrics,
                'strategy_metrics': strategy_metrics,
                'price_data': price_data,
                'backtest_params': {
                    'commission': self.commission,
                    'slippage': self.slippage,
                    'initial_cash': self.initial_cash,
                    'leverage': leverage,
                    'start_date': price_data.index[0],
                    'end_date': price_data.index[-1]
                }
            }
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
    
    def run_long_short_backtest(
        self,
        price_data: pd.DataFrame,
        long_weights: pd.DataFrame,
        short_weights: pd.DataFrame,
        leverage: float = 1.6
    ) -> Dict[str, Any]:
        """
        Run long/short equity backtest.
        
        Args:
            price_data: DataFrame with price data
            long_weights: DataFrame with long position weights
            short_weights: DataFrame with short position weights (positive values)
            leverage: Total leverage (e.g., 1.6 = 160% long, 60% short)
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting long/short backtest")
        
        # Calculate net weights (long - short)
        net_weights = long_weights - short_weights
        
        # Normalize to target leverage
        long_exposure = long_weights.sum(axis=1)
        short_exposure = short_weights.sum(axis=1)
        gross_exposure = long_exposure + short_exposure
        
        # Scale to target leverage
        scale_factor = leverage / gross_exposure
        scale_factor = scale_factor.fillna(0)
        
        scaled_long_weights = long_weights.multiply(scale_factor, axis=0)
        scaled_short_weights = short_weights.multiply(scale_factor, axis=0)
        
        # Combine into net weights
        combined_weights = scaled_long_weights - scaled_short_weights
        
        # Run backtest
        results = self.run_backtest(price_data, combined_weights, leverage=1.0)
        
        # Add long/short specific metrics
        results['long_short_metrics'] = self._calculate_long_short_metrics(
            scaled_long_weights, scaled_short_weights, price_data
        )
        
        return results
    
    def run_walk_forward_analysis(
        self,
        price_data: pd.DataFrame,
        weight_function: callable,
        train_period_months: int = 12,
        test_period_months: int = 3,
        step_months: int = 1
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            price_data: DataFrame with price data
            weight_function: Function that takes price data and returns weights
            train_period_months: Training period in months
            test_period_months: Testing period in months
            step_months: Step size in months
            
        Returns:
            Dictionary with walk-forward results
        """
        self.logger.info("Starting walk-forward analysis")
        
        results = {
            'periods': [],
            'performance': [],
            'weights_history': []
        }
        
        # Generate walk-forward periods
        start_date = price_data.index[0]
        end_date = price_data.index[-1]
        
        current_date = start_date
        
        while current_date < end_date:
            # Define train and test periods
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_period_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_period_months)
            
            if test_end > end_date:
                break
            
            try:
                # Extract training data
                train_data = price_data.loc[train_start:train_end]
                
                # Generate weights using provided function
                weights = weight_function(train_data)
                
                # Extract test data
                test_data = price_data.loc[test_start:test_end]
                
                # Align weights with test period
                test_weights = weights.reindex(test_data.index, method='ffill')
                
                # Run backtest for this period
                period_results = self.run_backtest(test_data, test_weights)
                
                # Store results
                period_info = {
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'portfolio': period_results['portfolio'],
                    'metrics': period_results['performance_metrics']
                }
                
                results['periods'].append(period_info)
                results['weights_history'].append(test_weights)
                
            except Exception as e:
                self.logger.warning(f"Walk-forward period failed: {e}")
            
            # Move to next period
            current_date += pd.DateOffset(months=step_months)
        
        # Aggregate results
        if results['periods']:
            results['aggregate_metrics'] = self._aggregate_walk_forward_results(
                results['periods']
            )
        
        return results
    
    def _align_data(
        self,
        price_data: pd.DataFrame,
        weights: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align price data and weights by dates and tickers."""
        # Find common dates
        common_dates = price_data.index.intersection(weights.index)
        
        if len(common_dates) == 0:
            self.logger.error("No common dates between price data and weights")
            return pd.DataFrame(), pd.DataFrame()
        
        # Find common tickers
        price_tickers = set(price_data.columns)
        weight_tickers = set(weights.columns)
        common_tickers = price_tickers.intersection(weight_tickers)
        
        if len(common_tickers) == 0:
            self.logger.error("No common tickers between price data and weights")
            return pd.DataFrame(), pd.DataFrame()
        
        # Align data
        aligned_prices = price_data.loc[common_dates, list(common_tickers)]
        aligned_weights = weights.loc[common_dates, list(common_tickers)]
        
        # Fill missing values
        aligned_prices = aligned_prices.fillna(method='ffill').fillna(method='bfill')
        aligned_weights = aligned_weights.fillna(0)
        
        self.logger.info(f"Aligned data: {len(common_dates)} dates, {len(common_tickers)} tickers")
        
        return aligned_prices, aligned_weights
    
    def _apply_transaction_costs(
        self,
        portfolio_returns: pd.Series,
        weights: pd.DataFrame
    ) -> pd.Series:
        """Apply transaction costs to portfolio returns."""
        # Calculate weight changes (turnover)
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1)
        
        # Apply commission costs
        commission_costs = turnover * self.commission
        
        # Apply slippage costs
        slippage_costs = turnover * self.slippage
        
        # Total transaction costs
        total_costs = commission_costs + slippage_costs
        
        # Subtract costs from returns
        adjusted_returns = portfolio_returns - total_costs
        
        return adjusted_returns
    
    def _calculate_performance_metrics(
        self,
        portfolio: vbt.Portfolio,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            # Get returns - handle different VectorBT API versions
            if hasattr(portfolio, 'returns'):
                returns = portfolio.returns()
            else:
                # Calculate returns manually from portfolio value
                returns = portfolio.value.pct_change().fillna(0)
            
            # Handle Series vs DataFrame
            if isinstance(returns, pd.DataFrame):
                returns = returns.sum(axis=1)  # Sum across columns if DataFrame
            
            portfolio_value = portfolio.value
            if isinstance(portfolio_value, pd.DataFrame):
                portfolio_value = portfolio_value.sum(axis=1)  # Sum across columns
            
            # Calculate basic metrics
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(portfolio_value)
            
            metrics = {
                # Return metrics
                'total_return': total_return,
                'annual_return': annual_return,
                'cumulative_return': total_return,
                
                # Risk metrics
                'annual_volatility': volatility,
                'sharpe_ratio': annual_return / volatility if volatility != 0 else 0,
                'sortino_ratio': annual_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0,
                'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                
                # Drawdown metrics (simplified)
                'max_drawdown': max_drawdown,
                'avg_drawdown': 0,  # Simplified for now
                'max_drawdown_duration': 0,  # Simplified for now
                
                # Trade statistics
                'win_rate': (returns > 0).mean(),
                'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf,
                'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
                'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
                
                # Risk-adjusted metrics
                'var_95': returns.quantile(0.05),
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
            # Return default metrics
            metrics = {
                'total_return': 0, 'annual_return': 0, 'cumulative_return': 0,
                'annual_volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
                'max_drawdown': 0, 'avg_drawdown': 0, 'max_drawdown_duration': 0,
                'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
                'var_95': 0, 'cvar_95': 0, 'skewness': 0, 'kurtosis': 0
            }
        
        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            
            # Calculate beta
            covariance = np.cov(returns.dropna(), aligned_benchmark)[0, 1]
            benchmark_var = np.var(aligned_benchmark)
            beta = covariance / benchmark_var if benchmark_var != 0 else 0
            
            # Calculate alpha
            alpha = metrics['annual_return'] - beta * aligned_benchmark.vbt.returns.annualized()
            
            # Information ratio
            active_returns = returns - aligned_benchmark
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
            
            metrics.update({
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            })
        
        return metrics
    
    def _calculate_strategy_metrics(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        portfolio_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate strategy-specific metrics."""
        # Turnover analysis
        weight_changes = weights.diff().abs()
        daily_turnover = weight_changes.sum(axis=1)
        
        # Concentration analysis
        concentration = (weights ** 2).sum(axis=1)  # Herfindahl index
        
        # Long/short analysis if applicable
        long_weights = weights.where(weights > 0, 0)
        short_weights = weights.where(weights < 0, 0).abs()
        
        gross_exposure = weights.abs().sum(axis=1)
        net_exposure = weights.sum(axis=1)
        
        metrics = {
            'avg_daily_turnover': daily_turnover.mean(),
            'avg_monthly_turnover': daily_turnover.mean() * 21,  # Approximate
            'avg_concentration': concentration.mean(),
            'avg_gross_exposure': gross_exposure.mean(),
            'avg_net_exposure': net_exposure.mean(),
            'avg_long_exposure': long_weights.sum(axis=1).mean(),
            'avg_short_exposure': short_weights.sum(axis=1).mean(),
            'avg_num_positions': (weights != 0).sum(axis=1).mean(),
            'max_weight': weights.abs().max().max(),
            'weight_volatility': weights.std().mean()
        }
        
        return metrics
    
    def _calculate_long_short_metrics(
        self,
        long_weights: pd.DataFrame,
        short_weights: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate long/short specific metrics."""
        returns = price_data.pct_change().fillna(0)
        
        # Calculate long and short returns separately
        long_returns = (long_weights.shift(1) * returns).sum(axis=1)
        short_returns = -(short_weights.shift(1) * returns).sum(axis=1)  # Short positions benefit from price declines
        
        metrics = {
            'long_annual_return': long_returns.mean() * 252,
            'short_annual_return': short_returns.mean() * 252,
            'long_volatility': long_returns.std() * np.sqrt(252),
            'short_volatility': short_returns.std() * np.sqrt(252),
            'long_sharpe': (long_returns.mean() * 252) / (long_returns.std() * np.sqrt(252)) if long_returns.std() != 0 else 0,
            'short_sharpe': (short_returns.mean() * 252) / (short_returns.std() * np.sqrt(252)) if short_returns.std() != 0 else 0,
            'long_short_correlation': long_returns.corr(short_returns),
            'avg_long_exposure': long_weights.sum(axis=1).mean(),
            'avg_short_exposure': short_weights.sum(axis=1).mean()
        }
        
        return metrics
    
    def _aggregate_walk_forward_results(
        self,
        periods: List[Dict]
    ) -> Dict[str, float]:
        """Aggregate walk-forward analysis results."""
        # Extract returns from all periods
        all_returns = []
        for period in periods:
            portfolio_returns = period['portfolio'].returns
            all_returns.append(portfolio_returns)
        
        # Combine all returns
        combined_returns = pd.concat(all_returns)
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'total_periods': len(periods),
            'avg_annual_return': combined_returns.mean() * 252,
            'avg_volatility': combined_returns.std() * np.sqrt(252),
            'avg_sharpe_ratio': (combined_returns.mean() * 252) / (combined_returns.std() * np.sqrt(252)) if combined_returns.std() != 0 else 0,
            'win_rate_periods': sum(1 for p in periods if p['metrics']['annual_return'] > 0) / len(periods),
            'consistency_score': 1 - (combined_returns.std() / abs(combined_returns.mean())) if combined_returns.mean() != 0 else 0
        }
        
        return aggregate_metrics
    
    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate maximum drawdown from portfolio value series."""
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        return drawdown.min()
    
    def generate_performance_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive performance report."""
        metrics = results['performance_metrics']
        strategy_metrics = results['strategy_metrics']
        
        report = f"""
TRADING ALPHA - PERFORMANCE REPORT
{'='*50}

RETURN METRICS:
- Total Return: {metrics['total_return']:.2%}
- Annual Return: {metrics['annual_return']:.2%}
- Cumulative Return: {metrics['cumulative_return']:.2%}

RISK METRICS:
- Annual Volatility: {metrics['annual_volatility']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Sortino Ratio: {metrics['sortino_ratio']:.2f}
- Calmar Ratio: {metrics['calmar_ratio']:.2f}

DRAWDOWN METRICS:
- Max Drawdown: {metrics['max_drawdown']:.2%}
- Avg Drawdown: {metrics['avg_drawdown']:.2%}
- Max DD Duration: {metrics['max_drawdown_duration']} days

STRATEGY METRICS:
- Avg Daily Turnover: {strategy_metrics['avg_daily_turnover']:.2%}
- Avg Concentration: {strategy_metrics['avg_concentration']:.3f}
- Avg Gross Exposure: {strategy_metrics['avg_gross_exposure']:.2%}
- Avg Net Exposure: {strategy_metrics['avg_net_exposure']:.2%}
- Avg Positions: {strategy_metrics['avg_num_positions']:.1f}

BACKTEST PARAMETERS:
- Commission: {results['backtest_params']['commission']:.3%}
- Slippage: {results['backtest_params']['slippage']:.3%}
- Initial Cash: ${results['backtest_params']['initial_cash']:,.0f}
- Period: {results['backtest_params']['start_date'].strftime('%Y-%m-%d')} to {results['backtest_params']['end_date'].strftime('%Y-%m-%d')}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {save_path}")
        
        return report