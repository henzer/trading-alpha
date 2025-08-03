import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class FeatureEngine:
    """
    Comprehensive feature engineering for trading strategies.
    
    Provides technical indicators, fundamental features, and cross-sectional rankings
    optimized for institutional multi-factor strategies.
    """
    
    def __init__(self, fillna_method: str = "ffill"):
        """
        Initialize FeatureEngine.
        
        Args:
            fillna_method: Method to handle missing values ('ffill', 'bfill', 'drop')
        """
        self.fillna_method = fillna_method
        self.logger = logging.getLogger("feature_engine")
        
    def calculate_momentum_features(
        self,
        prices: pd.DataFrame,
        periods: List[int] = [1, 5, 22, 63, 126, 252]
    ) -> pd.DataFrame:
        """
        Calculate momentum features for different time periods.
        
        Args:
            prices: DataFrame with price data (tickers as columns)
            periods: List of lookback periods in days
            
        Returns:
            DataFrame with momentum features
        """
        momentum_features = pd.DataFrame(index=prices.index)
        
        for period in periods:
            if period < len(prices):
                # Simple momentum (price change)
                momentum = prices.pct_change(periods=period)
                momentum_features[f'momentum_{period}d'] = momentum.mean(axis=1)
                
                # Risk-adjusted momentum (momentum / volatility)
                vol = prices.pct_change().rolling(window=period).std()
                risk_adj_momentum = momentum / vol
                momentum_features[f'risk_adj_momentum_{period}d'] = risk_adj_momentum.mean(axis=1)
        
        return self._handle_missing_values(momentum_features)
    
    def calculate_mean_reversion_features(
        self,
        prices: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 22
    ) -> pd.DataFrame:
        """
        Calculate mean reversion features.
        
        Args:
            prices: DataFrame with price data
            short_window: Short-term moving average window
            long_window: Long-term moving average window
            
        Returns:
            DataFrame with mean reversion features
        """
        features = pd.DataFrame(index=prices.index)
        
        # Price relative to moving averages
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()
        
        features['price_vs_ma_short'] = ((prices - ma_short) / ma_short).mean(axis=1)
        features['price_vs_ma_long'] = ((prices - ma_long) / ma_long).mean(axis=1)
        
        # Bollinger Bands deviation
        rolling_mean = prices.rolling(window=20).mean()
        rolling_std = prices.rolling(window=20).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        features['bollinger_position'] = bb_position.mean(axis=1)
        
        # RSI (Relative Strength Index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.mean(axis=1)
        
        return self._handle_missing_values(features)
    
    def calculate_volatility_features(
        self,
        returns: pd.DataFrame,
        windows: List[int] = [5, 22, 63]
    ) -> pd.DataFrame:
        """
        Calculate volatility-based features.
        
        Args:
            returns: DataFrame with return data
            windows: List of rolling windows for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        vol_features = pd.DataFrame(index=returns.index)
        
        for window in windows:
            if window < len(returns):
                # Simple volatility (standard deviation)
                vol = returns.rolling(window=window).std()
                vol_features[f'volatility_{window}d'] = vol.mean(axis=1)
                
                # Downside deviation
                downside_returns = returns.where(returns < 0, 0)
                downside_vol = downside_returns.rolling(window=window).std()
                vol_features[f'downside_vol_{window}d'] = downside_vol.mean(axis=1)
                
                # Volatility percentile (current vol vs historical)
                vol_rank = vol.rolling(window=252).rank(pct=True)
                vol_features[f'vol_percentile_{window}d'] = vol_rank.mean(axis=1)
        
        return self._handle_missing_values(vol_features)
    
    def calculate_value_features(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate value-based features.
        
        Args:
            prices: DataFrame with price data
            fundamentals: DataFrame with fundamental data (optional)
            
        Returns:
            DataFrame with value features
        """
        value_features = pd.DataFrame(index=prices.index)
        
        # Price-based value indicators
        # Moving average reversion
        ma_252 = prices.rolling(window=252).mean()
        value_features['price_vs_ma_252'] = ((prices - ma_252) / ma_252).mean(axis=1)
        
        # 52-week high/low ratio
        high_52w = prices.rolling(window=252).max()
        low_52w = prices.rolling(window=252).min()
        
        current_vs_high = (prices / high_52w).mean(axis=1)
        current_vs_low = (prices / low_52w).mean(axis=1)
        
        value_features['vs_52w_high'] = current_vs_high
        value_features['vs_52w_low'] = current_vs_low
        
        # If fundamentals are provided, add fundamental value features
        if fundamentals is not None:
            # This would be expanded with actual fundamental ratios
            self.logger.info("Fundamental data integration not fully implemented")
        
        return self._handle_missing_values(value_features)
    
    def calculate_quality_features(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate quality-based features.
        
        Args:
            prices: DataFrame with price data
            volumes: DataFrame with volume data (optional)
            
        Returns:
            DataFrame with quality features
        """
        quality_features = pd.DataFrame(index=prices.index)
        
        # Price stability (inverse of volatility)
        returns = prices.pct_change()
        vol_22d = returns.rolling(window=22).std()
        quality_features['price_stability'] = (1 / vol_22d).mean(axis=1)
        
        # Trend consistency (% of positive days in rolling window)
        positive_days = (returns > 0).rolling(window=22).mean()
        quality_features['trend_consistency'] = positive_days.mean(axis=1)
        
        # Drawdown recovery (how quickly stocks recover from drawdowns)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(window=252, min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        
        # Time to recovery (simplified)
        recovery_speed = (-drawdown).rolling(window=63).mean()
        quality_features['recovery_speed'] = recovery_speed.mean(axis=1)
        
        # Volume quality (if volume data available)
        if volumes is not None:
            # Price-volume correlation
            price_vol_corr = returns.rolling(window=22).corr(volumes.pct_change())
            quality_features['price_volume_corr'] = price_vol_corr.mean(axis=1)
        
        return self._handle_missing_values(quality_features)
    
    def calculate_growth_features(
        self,
        prices: pd.DataFrame,
        lookback_periods: List[int] = [22, 63, 126]
    ) -> pd.DataFrame:
        """
        Calculate growth-based features.
        
        Args:
            prices: DataFrame with price data
            lookback_periods: Periods for growth calculation
            
        Returns:
            DataFrame with growth features
        """
        growth_features = pd.DataFrame(index=prices.index)
        
        for period in lookback_periods:
            if period < len(prices):
                # Price growth rate
                growth_rate = prices.pct_change(periods=period)
                growth_features[f'growth_rate_{period}d'] = growth_rate.mean(axis=1)
                
                # Growth acceleration (change in growth rate)
                growth_accel = growth_rate.diff(periods=period)
                growth_features[f'growth_accel_{period}d'] = growth_accel.mean(axis=1)
                
                # Growth consistency (std of rolling growth rates)
                growth_consistency = growth_rate.rolling(window=period).std()
                growth_features[f'growth_consistency_{period}d'] = growth_consistency.mean(axis=1)
        
        return self._handle_missing_values(growth_features)
    
    def calculate_cross_sectional_features(
        self,
        data: pd.DataFrame,
        ranking_method: str = "rank"
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional rankings and scores.
        
        Args:
            data: DataFrame with features (tickers as columns)
            ranking_method: 'rank', 'zscore', or 'quantile'
            
        Returns:
            DataFrame with cross-sectional features
        """
        if ranking_method == "rank":
            # Rank-based (0 to 1 scale)
            ranked = data.rank(axis=1, pct=True)
        elif ranking_method == "zscore":
            # Z-score standardization
            ranked = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        elif ranking_method == "quantile":
            # Quantile-based ranking
            ranked = data.rank(axis=1, method='average').div(data.count(axis=1), axis=0)
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        return self._handle_missing_values(ranked)
    
    def create_composite_factor(
        self,
        feature_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create composite factor from multiple feature sets.
        
        Args:
            feature_dict: Dictionary mapping factor names to DataFrames
            weights: Optional weights for each factor (equal weights if None)
            
        Returns:
            DataFrame with composite factor scores
        """
        if weights is None:
            weights = {name: 1.0/len(feature_dict) for name in feature_dict.keys()}
        
        # Standardize each factor first
        standardized_factors = {}
        
        for name, features in feature_dict.items():
            # Calculate composite score for this factor (mean of all columns)
            factor_score = features.mean(axis=1)
            
            # Standardize to z-score
            standardized_score = (factor_score - factor_score.mean()) / factor_score.std()
            standardized_factors[name] = standardized_score
        
        # Combine factors with weights
        composite_score = sum(
            standardized_factors[name] * weights[name] 
            for name in standardized_factors.keys()
        )
        
        return pd.DataFrame({'composite_factor': composite_score})
    
    def calculate_factor_loadings(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        factors: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Calculate factor loadings using multi-factor model.
        
        Args:
            returns: Stock returns DataFrame
            market_returns: Market benchmark returns
            factors: Additional factor returns (SMB, HML, etc.)
            
        Returns:
            DataFrame with factor loadings for each stock
        """
        factor_loadings = []
        
        # Prepare factor matrix
        factor_matrix = pd.DataFrame({'market': market_returns})
        
        if factors:
            for factor_name, factor_returns in factors.items():
                factor_matrix[factor_name] = factor_returns
        
        # Align dates
        common_dates = returns.index.intersection(factor_matrix.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factor_matrix.loc[common_dates]
        
        # Calculate loadings for each stock
        for ticker in returns_aligned.columns:
            try:
                stock_returns = returns_aligned[ticker].dropna()
                factor_subset = factors_aligned.loc[stock_returns.index]
                
                # Run regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(factor_subset, stock_returns)
                
                loading_dict = {'ticker': ticker}
                for i, factor_name in enumerate(factor_subset.columns):
                    loading_dict[f'{factor_name}_loading'] = model.coef_[i]
                
                loading_dict['alpha'] = model.intercept_
                loading_dict['r_squared'] = model.score(factor_subset, stock_returns)
                
                factor_loadings.append(loading_dict)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate loadings for {ticker}: {e}")
        
        return pd.DataFrame(factor_loadings).set_index('ticker')
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to specified method."""
        if self.fillna_method == "ffill":
            return df.fillna(method='ffill')
        elif self.fillna_method == "bfill":
            return df.fillna(method='bfill')
        elif self.fillna_method == "drop":
            return df.dropna()
        else:
            return df
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics for features.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary with feature statistics
        """
        summary = {
            'feature_count': len(features.columns),
            'observation_count': len(features),
            'missing_data_pct': features.isnull().sum().sum() / (len(features) * len(features.columns)),
            'date_range': {
                'start': features.index.min(),
                'end': features.index.max()
            },
            'feature_correlations': features.corr().abs().mean().mean()
        }
        
        return summary