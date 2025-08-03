import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pickle
import os
import hashlib
import logging
from pathlib import Path

from .yahoo_provider import YahooProvider

class DataLoader:
    """
    High-level data loading interface with intelligent caching and data management.
    
    Handles data loading, caching, validation, and preprocessing for trading strategies.
    Provides a clean interface between data providers and strategy implementations.
    """
    
    def __init__(
        self,
        provider: Optional[YahooProvider] = None,
        cache_dir: str = "data/cache",
        cache_expiry_hours: int = 24,
        auto_cache: bool = True
    ):
        """
        Initialize DataLoader.
        
        Args:
            provider: YahooProvider instance (creates default if None)
            cache_dir: Directory for caching data files
            cache_expiry_hours: Hours before cached data expires
            auto_cache: Automatically cache downloaded data
        """
        self.provider = provider or YahooProvider()
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_hours = cache_expiry_hours
        self.auto_cache = auto_cache
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("data_loader")
        
    def load_universe_data(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        fields: List[str] = ["Open", "High", "Low", "Close", "Volume"],
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load data for a universe of tickers with caching.
        
        Args:
            universe: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            fields: List of data fields to include
            interval: Data interval
            force_refresh: Force refresh from provider, ignore cache
            
        Returns:
            DataFrame with MultiIndex columns (field, ticker)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(universe, start_date, end_date, fields, interval)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache first
        if not force_refresh and self._is_cache_valid(cache_file):
            self.logger.info("Loading data from cache")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, fetching fresh data")
        
        # Fetch fresh data
        self.logger.info(f"Fetching fresh data for {len(universe)} tickers")
        data = self.provider.get_panel_data(universe, start_date, end_date, fields, interval)
        
        if data.empty:
            self.logger.error("No data fetched")
            return pd.DataFrame()
        
        # Clean and validate data
        data = self._clean_universe_data(data, universe, fields)
        
        # Cache the data if auto_cache is enabled
        if self.auto_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.info(f"Data cached to {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def load_returns_data(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        return_type: str = "simple",
        period: int = 1
    ) -> pd.DataFrame:
        """
        Load return data for a universe of tickers.
        
        Args:
            universe: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            return_type: 'simple' or 'log' returns
            period: Return period in days
            
        Returns:
            DataFrame with returns for each ticker
        """
        # Load price data
        price_data = self.load_universe_data(
            universe, start_date, end_date, 
            fields=["Close"], force_refresh=False
        )
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Extract close prices
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data["Close"]
        else:
            close_prices = price_data
        
        # Calculate returns
        if return_type == "log":
            returns = np.log(close_prices / close_prices.shift(period))
        else:  # simple returns
            returns = close_prices.pct_change(periods=period)
        
        # Clean returns data
        returns = returns.dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        return returns
    
    def load_fundamental_data(
        self,
        universe: List[str],
        metrics: List[str] = ["pe_ratio", "pb_ratio", "roe", "debt_to_equity"]
    ) -> pd.DataFrame:
        """
        Load fundamental data for a universe of tickers.
        
        Args:
            universe: List of ticker symbols
            metrics: List of fundamental metrics to fetch
            
        Returns:
            DataFrame with fundamental data
        """
        self.logger.info(f"Fetching fundamental data for {len(universe)} tickers")
        
        fundamental_data = []
        
        for ticker in universe:
            try:
                data = self.provider.get_fundamental_data(ticker)
                
                # Extract requested metrics
                ticker_data = {"ticker": ticker}
                for metric in metrics:
                    ticker_data[metric] = data.get(metric)
                
                fundamental_data.append(ticker_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        
        if not fundamental_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(fundamental_data)
        df.set_index("ticker", inplace=True)
        
        return df
    
    def get_trading_calendar(
        self,
        start_date: str,
        end_date: str,
        frequency: str = "D"
    ) -> pd.DatetimeIndex:
        """
        Get trading calendar for the specified period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Calendar frequency ('D', 'W', 'M')
            
        Returns:
            DatetimeIndex with trading dates
        """
        # For now, use simple business day calendar
        # In production, you'd use a proper trading calendar
        
        freq_map = {
            "D": "B",      # Business days
            "W": "W-FRI",  # Weekly (Fridays)
            "M": "BM"      # Business month end
        }
        
        freq_code = freq_map.get(frequency, "B")
        calendar = pd.date_range(start=start_date, end=end_date, freq=freq_code)
        
        return calendar
    
    def align_data_to_calendar(
        self,
        data: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Align data to a specific trading calendar.
        
        Args:
            data: DataFrame with datetime index
            calendar: Target calendar dates
            method: Fill method ('ffill', 'bfill', 'interpolate')
            
        Returns:
            DataFrame aligned to calendar
        """
        # Reindex to calendar
        aligned_data = data.reindex(calendar)
        
        # Apply fill method
        if method == "ffill":
            aligned_data = aligned_data.fillna(method='ffill')
        elif method == "bfill":
            aligned_data = aligned_data.fillna(method='bfill')
        elif method == "interpolate":
            aligned_data = aligned_data.interpolate(method='linear')
        
        return aligned_data
    
    def validate_data_quality(
        self,
        data: pd.DataFrame,
        universe: List[str],
        min_history_days: int = 252,
        max_missing_pct: float = 0.05
    ) -> Dict[str, any]:
        """
        Validate data quality for trading strategy use.
        
        Args:
            data: DataFrame to validate
            universe: Expected universe of tickers
            min_history_days: Minimum required history in days
            max_missing_pct: Maximum allowed missing data percentage
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "failed_tickers": []
        }
        
        if data.empty:
            validation_results["is_valid"] = False
            validation_results["issues"].append("No data available")
            return validation_results
        
        # Check data length
        data_length = len(data)
        if data_length < min_history_days:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Insufficient history: {data_length} < {min_history_days} days")
        
        # Check for missing tickers
        if isinstance(data.columns, pd.MultiIndex):
            available_tickers = set(data.columns.get_level_values(1).unique())
        else:
            available_tickers = set(data.columns)
        
        expected_tickers = set(universe)
        missing_tickers = expected_tickers - available_tickers
        
        if missing_tickers:
            validation_results["issues"].append(f"Missing tickers: {list(missing_tickers)}")
            validation_results["failed_tickers"] = list(missing_tickers)
        
        # Check missing data percentage
        if isinstance(data.columns, pd.MultiIndex):
            # For multi-index, check each ticker separately
            close_data = data.get("Close", pd.DataFrame())
            if not close_data.empty:
                for ticker in close_data.columns:
                    missing_pct = close_data[ticker].isnull().sum() / len(close_data)
                    if missing_pct > max_missing_pct:
                        validation_results["issues"].append(f"Excessive missing data for {ticker}: {missing_pct:.1%}")
        else:
            # For regular DataFrame
            for ticker in data.columns:
                missing_pct = data[ticker].isnull().sum() / len(data)
                if missing_pct > max_missing_pct:
                    validation_results["issues"].append(f"Excessive missing data for {ticker}: {missing_pct:.1%}")
        
        # Calculate statistics
        validation_results["statistics"] = {
            "total_rows": len(data),
            "total_tickers": len(available_tickers),
            "date_range": {
                "start": data.index.min().strftime('%Y-%m-%d'),
                "end": data.index.max().strftime('%Y-%m-%d')
            },
            "missing_tickers": len(missing_tickers),
            "success_rate": len(available_tickers) / len(expected_tickers) if expected_tickers else 0
        }
        
        # Update overall validity
        if validation_results["issues"]:
            validation_results["is_valid"] = False
        
        return validation_results
    
    def _generate_cache_key(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        fields: List[str],
        interval: str
    ) -> str:
        """Generate unique cache key for data request."""
        # Create a string representation of the request
        request_str = f"{sorted(universe)}_{start_date}_{end_date}_{sorted(fields)}_{interval}"
        
        # Generate hash
        cache_key = hashlib.md5(request_str.encode()).hexdigest()
        
        return cache_key
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_file.exists():
            return False
        
        # Check expiry
        file_age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        
        return file_age_hours < self.cache_expiry_hours
    
    def _clean_universe_data(
        self,
        data: pd.DataFrame,
        universe: List[str],
        fields: List[str]
    ) -> pd.DataFrame:
        """Clean and standardize universe data."""
        if data.empty:
            return data
        
        # Ensure MultiIndex columns if multiple fields
        if len(fields) > 1 and not isinstance(data.columns, pd.MultiIndex):
            # Reconstruct MultiIndex if needed
            pass
        
        # Sort by date
        data = data.sort_index()
        
        # Remove weekends and holidays (basic filtering)
        data = data[data.index.dayofweek < 5]  # Monday=0, Friday=4
        
        # Forward fill missing values within reasonable limits
        data = data.fillna(method='ffill', limit=5)
        
        return data
    
    def clear_cache(self) -> None:
        """Clear all cached data files."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete {cache_file}: {e}")
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached data."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "cached_files": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
            "expiry_hours": self.cache_expiry_hours
        }