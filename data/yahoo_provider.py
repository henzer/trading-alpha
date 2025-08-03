import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

class YahooProvider:
    """
    Yahoo Finance data provider with caching, error handling, and parallel downloads.
    
    Provides institutional-grade data fetching capabilities with proper error handling,
    data validation, and performance optimization.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Yahoo Finance provider.
        
        Args:
            max_workers: Maximum number of parallel download threads
            retry_attempts: Number of retry attempts for failed downloads
            retry_delay: Delay between retry attempts in seconds
            cache_dir: Directory for caching data (None = no caching)
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir
        
        # Setup logging
        self.logger = logging.getLogger("yahoo_provider")
        
        # Cache for storing downloaded data
        self._cache = {}
        
    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        # Check cache first
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit for {ticker}")
            return self._cache[cache_key].copy()
        
        for attempt in range(self.retry_attempts):
            try:
                # Create ticker object
                ticker_obj = yf.Ticker(ticker)
                
                # Download data
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,  # Adjust for splits and dividends
                    prepost=False      # No pre/post market data
                )
                
                if data.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    return None
                
                # Validate data quality
                if not self._validate_data(data, ticker):
                    self.logger.warning(f"Data validation failed for {ticker}")
                    return None
                
                # Clean and standardize data
                data = self._clean_data(data, ticker)
                
                # Cache the result
                self._cache[cache_key] = data.copy()
                
                self.logger.debug(f"Successfully fetched {ticker}: {len(data)} rows")
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"Failed to fetch {ticker} after {self.retry_attempts} attempts")
        
        return None
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            
        Returns:
            Dictionary mapping tickers to their DataFrames
        """
        self.logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        results = {}
        failed_tickers = []
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(
                    self.fetch_single_ticker, 
                    ticker, 
                    start_date, 
                    end_date, 
                    interval
                ): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[ticker] = data
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    self.logger.error(f"Exception fetching {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        success_rate = len(results) / len(tickers) * 100
        self.logger.info(f"Download completed: {len(results)}/{len(tickers)} successful ({success_rate:.1f}%)")
        
        if failed_tickers:
            self.logger.warning(f"Failed tickers: {failed_tickers}")
        
        return results
    
    def get_panel_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fields: List[str] = ["Close"],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get panel data (multiple tickers, multiple fields) in a structured format.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: List of fields to include ('Open', 'High', 'Low', 'Close', 'Volume')
            interval: Data interval
            
        Returns:
            DataFrame with MultiIndex columns (ticker, field)
        """
        # Fetch data for all tickers
        ticker_data = self.fetch_multiple_tickers(tickers, start_date, end_date, interval)
        
        if not ticker_data:
            self.logger.error("No data fetched for any tickers")
            return pd.DataFrame()
        
        # Align all data to common date range
        all_dates = set()
        for data in ticker_data.values():
            all_dates.update(data.index)
        
        common_dates = sorted(all_dates)
        
        # Build panel DataFrame
        panel_data = {}
        
        for field in fields:
            field_data = {}
            for ticker, data in ticker_data.items():
                if field in data.columns:
                    # Reindex to common dates
                    aligned_data = data[field].reindex(common_dates)
                    field_data[ticker] = aligned_data
            
            if field_data:
                panel_data[field] = pd.DataFrame(field_data)
        
        if not panel_data:
            return pd.DataFrame()
        
        # Create MultiIndex DataFrame
        result = pd.concat(panel_data, axis=1)
        
        # Sort by date
        result = result.sort_index()
        
        self.logger.info(f"Panel data shape: {result.shape}")
        return result
    
    def get_fundamental_data(self, ticker: str) -> Dict[str, any]:
        """
        Get fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            
            info = ticker_obj.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Failed to fetch fundamentals for {ticker}: {str(e)}")
            return {}
    
    def _validate_data(self, data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
            ticker: Ticker symbol for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            return False
        
        # Check for required columns (be flexible with additional columns)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = list(data.columns)
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            self.logger.warning(f"{ticker}: Missing columns {missing_cols}")
            return False
        
        # Check for obvious data errors
        ohlc_data = data[['Open', 'High', 'Low', 'Close']]
        
        # High should be >= Open, Low, Close
        if not ((ohlc_data['High'] >= ohlc_data[['Open', 'Low', 'Close']].max(axis=1)).all()):
            self.logger.warning(f"{ticker}: Invalid OHLC data - High not highest")
            return False
        
        # Low should be <= Open, High, Close  
        if not ((ohlc_data['Low'] <= ohlc_data[['Open', 'High', 'Close']].min(axis=1)).all()):
            self.logger.warning(f"{ticker}: Invalid OHLC data - Low not lowest")
            return False
        
        # Check for excessive missing data
        missing_pct = data.isnull().sum().max() / len(data)
        if missing_pct > 0.1:  # More than 10% missing
            self.logger.warning(f"{ticker}: Excessive missing data ({missing_pct:.1%})")
            return False
        
        return True
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Clean and standardize data.
        
        Args:
            data: Raw DataFrame
            ticker: Ticker symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        # Keep only required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols].copy()
        
        # Convert timezone-aware index to naive (remove timezone)
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_convert(None)
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values (conservative approach)
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN rows
        data = data.dropna()
        
        # Ensure positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col].abs()
        
        # Ensure positive volume
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].abs()
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def get_sp500_tickers(self) -> List[str]:
        """
        Get current S&P 500 ticker list.
        
        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            # Fetch S&P 500 constituents
            sp500 = yf.Ticker("^GSPC")
            # Note: yfinance doesn't directly provide constituents
            # This is a placeholder - in production, you'd use a dedicated data source
            
            # Hardcoded list of major S&P 500 stocks for demo
            major_sp500 = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'BAC', 'ABBV',
                'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'MRK', 'ABT',
                'VZ', 'ADBE', 'NFLX', 'CRM', 'NKE', 'T', 'DHR', 'XOM', 'CVX',
                'CMCSA', 'LLY', 'ORCL', 'ACN', 'TXN', 'PM', 'MDT', 'HON', 'QCOM'
            ]
            
            return major_sp500
            
        except Exception as e:
            self.logger.error(f"Failed to fetch S&P 500 tickers: {str(e)}")
            return []
    
    def get_russell2000_sample(self) -> List[str]:
        """
        Get sample Russell 2000 tickers for small-cap strategies.
        
        Returns:
            List of Russell 2000 ticker symbols (sample)
        """
        # Sample of Russell 2000 stocks
        russell_sample = [
            'CVLT', 'OLED', 'BGNE', 'SAIA', 'UPST', 'FORM', 'GTLS', 'OMCL',
            'ENSG', 'KRTX', 'FIVE', 'AMED', 'CRVL', 'BILL', 'SAGE', 'TMDX',
            'PCTY', 'MGNI', 'CSOD', 'IRTC', 'HALO', 'BLFS', 'ALRM', 'COLL',
            'CCCS', 'BCPC', 'MASI', 'VICR', 'PRGS', 'EXPO', 'CALX', 'CAKE',
            'POWI', 'FRPT', 'JAMF', 'ALKS', 'MATX', 'CROX', 'SHOO', 'HELE'
        ]
        
        return russell_sample
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_items': len(self._cache),
            'total_memory_mb': sum(df.memory_usage(deep=True).sum() for df in self._cache.values()) / 1024 / 1024
        }