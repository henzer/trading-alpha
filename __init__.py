"""
Trading Alpha - Institutional Trading Framework

A comprehensive framework for building, backtesting, and optimizing 
institutional-level trading strategies targeting 20-30% annual returns.

Uses vectorbt for high-performance backtesting and riskfolio-lib for 
portfolio optimization.

Author: Claude & Henzer
"""

__version__ = "1.0.0"
__author__ = "Claude & Henzer"

from .core.base_strategy import BaseStrategy
from .data.yahoo_provider import YahooProvider
from .backtesting.vectorbt_engine import VectorbtEngine
from .optimization.riskfolio_optimizer import RiskfolioOptimizer

__all__ = [
    "BaseStrategy",
    "YahooProvider", 
    "VectorbtEngine",
    "RiskfolioOptimizer"
]