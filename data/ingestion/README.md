# Data Ingestion Layer — Architecture Overview

> **CAC40_Prediction Multimodal ML Pipeline**  
> **Module:** `data/ingestion/`  
> **Version:** 1.0.0  
> **Last Updated:** January 2026

---

## 📁 Directory Structure

```
data/ingestion/
│
├── __init__.py                    # Module initialization
├── README.md                      # This file
│
├── market_data/                   # OHLCV Price Data
│   ├── SPECS.md                   # Technical specifications
│   ├── fetchers/                  # Data fetcher implementations
│   │   ├── yfinance_fetcher.py
│   │   ├── polygon_fetcher.py
│   │   └── base_fetcher.py
│   ├── processors/                # Data validation & transformation
│   │   ├── validator.py
│   │   └── weekly_aggregator.py
│   ├── storage/                   # Parquet writers
│   │   └── storage_manager.py
│   └── raw/                       # Raw data storage (git-ignored)
│       └── daily/
│
├── news_feeds/                    # Financial News & Sentiment
│   ├── SPECS.md                   # Technical specifications
│   ├── fetchers/                  # API fetchers
│   │   ├── fmp_fetcher.py
│   │   ├── newsapi_fetcher.py
│   │   └── rss_aggregator.py
│   ├── scrapers/                  # Web scrapers
│   │   ├── yahoo_finance.py
│   │   ├── investing_com.py
│   │   └── boursorama.py
│   ├── processors/                # Text processing
│   │   ├── text_extractor.py
│   │   ├── ticker_extractor.py
│   │   └── timestamp_normalizer.py
│   └── raw/                       # Raw data storage
│
├── corporate_filings/             # Regulatory Filings
│   ├── SPECS.md                   # Technical specifications
│   ├── fetchers/                  # Filing fetchers
│   │   ├── sec_edgar_fetcher.py
│   │   ├── esef_fetcher.py
│   │   └── amf_fetcher.py
│   ├── parsers/                   # XBRL/HTML parsers
│   │   ├── xbrl_parser.py
│   │   └── sec_facts_parser.py
│   ├── scrapers/                  # Fallback scrapers
│   │   └── yahoo_financials_scraper.py
│   └── raw/                       # Raw filings storage
│
└── macro_indicators/              # Macroeconomic Data
    ├── SPECS.md                   # Technical specifications
    ├── fetchers/                  # Data fetchers
    │   ├── fred_fetcher.py
    │   └── ecb_fetcher.py
    ├── processors/                # Feature engineering
    │   ├── frequency_aligner.py
    │   ├── feature_engineer.py
    │   └── yield_curve.py
    └── raw/                       # Raw data storage
```

---

## 🎯 Module Objectives

| Module | Primary Goal | Key Sources |
|--------|--------------|-------------|
| **Market Data** | Adjusted OHLCV for price prediction | Yahoo Finance, Polygon.io |
| **News Feeds** | Sentiment signals & event detection | FMP, NewsAPI, Yahoo RSS |
| **Corporate Filings** | Fundamental data (revenue, EPS) | SEC EDGAR, Yahoo Financials |
| **Macro Indicators** | Economic regime features | FRED, ECB |

---

## 🔧 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ingestion.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Market Data
POLYGON_API_KEY=your_polygon_key

# News
FMP_API_KEY=your_fmp_key
NEWSAPI_KEY=your_newsapi_key

# Macro
FRED_API_KEY=your_fred_key

# SEC (email for User-Agent)
SEC_USER_EMAIL=your@email.com
```

### 3. Run Individual Modules

```python
# Market Data
from data.ingestion.market_data.fetchers.yfinance_fetcher import YFinanceFetcher

fetcher = YFinanceFetcher()
df = fetcher.fetch(tickers=["AAPL", "MC.PA"], start="2020-01-01")

# Macro Indicators
from data.ingestion.macro_indicators.fetchers.fred_fetcher import FREDFetcher

fred = FREDFetcher()
macro_data = fred.fetch_multiple(series_ids=["DGS10", "CPIAUCSL"])
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│   │ Market Data  │   │  News Feeds  │   │   Corporate Filings      │   │
│   │   (OHLCV)    │   │  (Articles)  │   │   (10-K, 10-Q, XBRL)     │   │
│   └──────┬───────┘   └──────┬───────┘   └────────────┬─────────────┘   │
│          │                  │                        │                  │
│          ▼                  ▼                        ▼                  │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    RAW DATA STORAGE                          │     │
│   │                (JSONL / Parquet / HTML)                      │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    PROCESSING LAYER                          │     │
│   │  • Validation  • Normalization  • Feature Engineering        │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                  PROCESSED DATA STORAGE                      │     │
│   │              (Partitioned Parquet Files)                     │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                   ML FEATURE STORE                           │     │
│   │           (Ready for Model Training)                         │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│   ┌──────────────┐                                                     │
│   │Macro Indicators│ ──────────────────────────────────────────────────│
│   │(FRED, ECB)     │                                                   │
│   └────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📅 Scheduling Strategy

| Data Type | Update Frequency | Trigger Time (UTC) |
|-----------|------------------|-------------------|
| OHLCV (Daily) | Daily | 22:00 (after US close) |
| OHLCV (Weekly) | Weekly (Friday) | 22:00 |
| News APIs | Every 4 hours | 00:00, 04:00, 08:00, ... |
| News RSS | Every 30 minutes | Continuous |
| SEC Filings | Daily | 08:00 |
| FRED Data | Daily | 08:00 |
| ECB Data | Daily | 10:00 |

---

## 🗄️ Storage Strategy

### Format: **Parquet** (Primary)

```python
# Partitioning strategy
df.to_parquet(
    "data/processed/",
    partition_cols=["ticker", "year"],
    engine="pyarrow",
    compression="snappy"
)
```

### Advantages over CSV:
- 5-10x compression
- Columnar storage (fast queries)
- Schema enforcement
- Native pandas integration

### Backup: **JSONL** (Raw responses)

```python
# For API responses
with open("raw/2026-01-26.jsonl", "a") as f:
    f.write(json.dumps(response) + "\n")
```

---

## 🔒 Rate Limit Summary

| Source | Limit | Strategy |
|--------|-------|----------|
| Yahoo Finance | ~2000/hour (unofficial) | 0.5s delay, batch requests |
| Polygon.io | Plan-based | Respect headers |
| FMP API | 250-750/day (free) | Prioritize, cache |
| NewsAPI | 100/day (free) | Reserve for key queries |
| SEC EDGAR | 10/second | 0.2s delay, include email |
| FRED | 120/minute | 0.5s delay |
| ECB | No limit | 1s delay (courtesy) |

---

## 🧪 Testing

Run module tests:

```bash
# Test all ingestion modules
pytest data/ingestion/tests/ -v

# Test specific module
pytest data/ingestion/market_data/tests/ -v
```

---

## 📚 Detailed Specifications

Each module contains a `SPECS.md` file with:
- Objective & scope
- Library recommendations
- Data source details (URLs, rate limits)
- Schema definitions
- Implementation plan
- Sample code

**See:**
- [Market Data SPECS](market_data/SPECS.md)
- [News Feeds SPECS](news_feeds/SPECS.md)
- [Corporate Filings SPECS](corporate_filings/SPECS.md)
- [Macro Indicators SPECS](macro_indicators/SPECS.md)

---

## 🚀 Next Steps

1. **Phase 1:** Implement Market Data fetchers (Priority: yfinance)
2. **Phase 2:** Implement FRED/ECB macro fetchers
3. **Phase 3:** Build news aggregation pipeline
4. **Phase 4:** Implement SEC EDGAR integration
5. **Phase 5:** Create unified orchestrator

---

*Authored for CAC40_Prediction ML Pipeline — Data Ingestion Layer*
