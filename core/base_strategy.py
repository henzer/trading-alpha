from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies in the framework.
    
    This class defines the interface that all strategies must implement
    and provides common functionality for strategy execution.
    """
    
    def __init__(
        self,
        name: str,
        universe: List[str],
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
        rebalance_freq: str = "monthly",
        transaction_cost: float = 0.001,
        leverage: float = 1.0,
        **kwargs
    ):
        """
        Initialize base strategy parameters.
        
        Args:
            name: Strategy name identifier
            universe: List of ticker symbols to trade
            start_date: Strategy start date (YYYY-MM-DD)
            end_date: Strategy end date (YYYY-MM-DD), None for latest
            initial_capital: Starting capital in dollars
            rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
            transaction_cost: Transaction cost as decimal (0.001 = 0.1%)
            leverage: Strategy leverage multiplier
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.universe = universe
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else datetime.now()
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        
        # Store additional parameters
        self.params = kwargs
        
        # Initialize state variables
        self._data = None
        self._signals = None
        self._weights = None
        self._positions = None
        self._returns = None
        
        # Setup logging
        self.logger = logging.getLogger(f"strategy.{self.name}")
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data.
        
        Args:
            data: DataFrame with OHLCV data for all universe tickers
            
        Returns:
            DataFrame with signals (-1, 0, 1) for each ticker and date
        """
        pass
    
    @abstractmethod
    def calculate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to portfolio weights.
        
        Args:
            signals: DataFrame with trading signals
            
        Returns:
            DataFrame with portfolio weights for each ticker and date
        """
        pass
    
    def validate_universe(self) -> bool:
        """
        Validate that the trading universe is properly formatted.
        
        Returns:
            True if universe is valid, False otherwise
        """
        if not isinstance(self.universe, list):
            self.logger.error("Universe must be a list of ticker symbols")
            return False
            
        if len(self.universe) == 0:
            self.logger.error("Universe cannot be empty")
            return False
            
        # Check for valid ticker format (basic validation)
        for ticker in self.universe:
            if not isinstance(ticker, str) or len(ticker) == 0:
                self.logger.error(f"Invalid ticker: {ticker}")
                return False
                
        return True
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        validations = [
            (self.initial_capital > 0, "Initial capital must be positive"),
            (0 <= self.transaction_cost <= 1, "Transaction cost must be between 0 and 1"),
            (self.leverage > 0, "Leverage must be positive"),
            (self.start_date < self.end_date, "Start date must be before end date"),
            (self.rebalance_freq in ['daily', 'weekly', 'monthly'], 
             "Rebalance frequency must be 'daily', 'weekly', or 'monthly'")
        ]
        
        for condition, message in validations:
            if not condition:
                self.logger.error(message)
                return False
                
        return True
    
    def get_rebalance_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """
        Generate rebalance dates based on frequency.
        
        Args:
            start_date: Strategy start date
            end_date: Strategy end date
            
        Returns:
            List of rebalance dates
        """
        freq_map = {
            'daily': 'D',
            'weekly': 'W-FRI',  # Weekly on Fridays
            'monthly': 'ME'     # Monthly end
        }
        
        freq = freq_map.get(self.rebalance_freq, 'ME')
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        return dates.tolist()
    
    def apply_transaction_costs(self, weights: pd.DataFrame, prev_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply transaction costs to weight changes.
        
        Args:
            weights: Current period weights
            prev_weights: Previous period weights
            
        Returns:
            Adjusted weights after transaction costs
        """
        if prev_weights is None:
            return weights
            
        # Calculate turnover (absolute weight changes)
        turnover = np.abs(weights - prev_weights.reindex_like(weights).fillna(0))
        
        # Apply transaction costs
        cost_adjustment = turnover * self.transaction_cost
        
        # Reduce weights proportionally to account for costs
        adjusted_weights = weights * (1 - cost_adjustment.sum(axis=1, keepdims=True))
        
        return adjusted_weights
    
    def apply_leverage(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply leverage to portfolio weights.
        
        Args:
            weights: Base portfolio weights
            
        Returns:
            Leveraged weights
        """
        return weights * self.leverage
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete backtest for the strategy.
        
        Args:
            data: Market data for universe
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest for {self.name}")
        
        # Validate inputs
        if not self.validate_universe() or not self.validate_parameters():
            raise ValueError("Strategy validation failed")
        
        # Store data
        self._data = data
        
        # Generate signals
        self.logger.info("Generating trading signals...")
        self._signals = self.generate_signals(data)
        
        # Calculate weights
        self.logger.info("Calculating portfolio weights...")
        self._weights = self.calculate_weights(self._signals)
        
        # Apply leverage
        self._weights = self.apply_leverage(self._weights)
        
        # Calculate returns (will be implemented by vectorbt engine)
        self.logger.info("Backtest completed successfully")
        
        return {
            'signals': self._signals,
            'weights': self._weights,
            'data': self._data,
            'strategy_name': self.name,
            'parameters': self.get_parameters()
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get all strategy parameters as dictionary.
        
        Returns:
            Dictionary of strategy parameters
        """
        base_params = {
            'name': self.name,
            'universe': self.universe,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'rebalance_freq': self.rebalance_freq,
            'transaction_cost': self.transaction_cost,
            'leverage': self.leverage
        }
        
        # Add strategy-specific parameters
        base_params.update(self.params)
        
        return base_params
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update strategy parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.params[key] = value
        
        self.logger.info(f"Updated parameters: {kwargs}")
    
    def get_signal_statistics(self) -> Dict[str, float]:
        """
        Calculate signal statistics for analysis.
        
        Returns:
            Dictionary with signal statistics
        """
        if self._signals is None:
            return {}
        
        stats = {}
        for ticker in self._signals.columns:
            signals = self._signals[ticker].dropna()
            stats[ticker] = {
                'total_signals': len(signals[signals != 0]),
                'long_signals': len(signals[signals > 0]),
                'short_signals': len(signals[signals < 0]),
                'signal_frequency': len(signals[signals != 0]) / len(signals) if len(signals) > 0 else 0
            }
        
        return stats
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', universe={len(self.universe)} tickers)"