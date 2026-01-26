# Data Ingestion Layer
# CAC40_Prediction Multimodal ML Pipeline

"""
This module provides the ETL (Extract) framework for the stock prediction pipeline.

Modules:
    - market_data: OHLCV price data ingestion (yfinance, Polygon)
    - news_feeds: Financial news and sentiment (APIs + scrapers)
    - corporate_filings: SEC EDGAR, EU filings, fundamentals
    - macro_indicators: FRED, ECB macroeconomic data
"""

from pathlib import Path

# Module root
INGESTION_ROOT = Path(__file__).parent

# Sub-module paths
MARKET_DATA_DIR = INGESTION_ROOT / "market_data"
NEWS_FEEDS_DIR = INGESTION_ROOT / "news_feeds"
CORPORATE_FILINGS_DIR = INGESTION_ROOT / "corporate_filings"
MACRO_INDICATORS_DIR = INGESTION_ROOT / "macro_indicators"

__all__ = [
    "INGESTION_ROOT",
    "MARKET_DATA_DIR",
    "NEWS_FEEDS_DIR",
    "CORPORATE_FILINGS_DIR",
    "MACRO_INDICATORS_DIR",
]
