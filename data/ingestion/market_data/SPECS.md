# Market Data Ingestion Module — Technical Specifications

> **Module:** `data/ingestion/market_data/`  
> **Data Type:** OHLCV (Open, High, Low, Close, Volume)  
> **Resolution:** Daily & Weekly  
> **Last Updated:** January 2026

---

## 1. Objective

Fetch historical and incremental **adjusted OHLCV price data** for equities traded on major exchanges (NYSE, NASDAQ, Euronext Paris for CAC40, etc.). The data must:

- Handle **corporate actions** (stock splits, dividends) via adjusted close prices.
- Support **multiple resolutions**: Daily (primary), Weekly (aggregated).
- Cover a **minimum 10-year lookback** for training robust ML models.
- Enable **incremental updates** (append new data without full re-download).

---

## 2. Recommended Python Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `yfinance` | Primary OHLCV source (Yahoo Finance wrapper) | `pip install yfinance` |
| `pandas-datareader` | Alternative/fallback data source | `pip install pandas-datareader` |
| `alpha_vantage` | Premium data with adjustments | `pip install alpha_vantage` |
| `polygon-api-client` | Professional-grade real-time & historical | `pip install polygon-api-client` |
| `pyarrow` | Parquet read/write for efficient storage | `pip install pyarrow` |
| `requests` / `httpx` | Custom API calls | `pip install httpx` |
| `tenacity` | Retry logic with exponential backoff | `pip install tenacity` |
| `schedule` | Scheduling periodic fetches | `pip install schedule` |

---

## 3. Data Sources

### 3.1 Free Tier

| Source | API/Method | Rate Limits | Notes |
|--------|------------|-------------|-------|
| **Yahoo Finance** (`yfinance`) | Unofficial API wrapper | ~2,000 requests/hour (unofficial) | Adjusted data included; prone to occasional blocks |
| **Alpha Vantage (Free)** | REST API | 25 requests/day | Reliable adjusted data; very limited for bulk |
| **EODHD (Free Tier)** | REST API | 20 API calls/day | Good European coverage |
| **Stooq** | CSV downloads | No API limits | Manual download; good for Polish/EU markets |

### 3.2 Premium Tier (Recommended for Production)

| Source | API/Method | Pricing (Approx.) | Notes |
|--------|------------|-------------------|-------|
| **Polygon.io** | REST + WebSocket | $29–$199/month | Excellent US coverage, splits/dividends included |
| **Alpha Vantage Premium** | REST API | $49.99/month | 75 requests/minute; reliable adjustments |
| **Tiingo** | REST API | $10–$30/month | Good value; includes IEX data |
| **EOD Historical Data** | REST API | €19.99/month | Strong EU/international coverage |

---

## 4. Data Schema

```python
# Expected DataFrame Schema
columns = {
    "date": "datetime64[ns]",      # Trading date (UTC normalized)
    "ticker": "string",             # Stock symbol (e.g., "AAPL", "MC.PA")
    "open": "float64",              # Adjusted open price
    "high": "float64",              # Adjusted high price
    "low": "float64",               # Adjusted low price
    "close": "float64",             # Adjusted close price
    "volume": "int64",              # Trading volume
    "adj_close": "float64",         # Explicitly stored adjusted close
    "dividends": "float64",         # Dividend amount (if any)
    "stock_splits": "float64",      # Split ratio (if any)
    "source": "string",             # Data provenance tag
    "fetched_at": "datetime64[ns]"  # Ingestion timestamp (UTC)
}
```

---

## 5. Strategy

### 5.1 Handling Adjustments (Splits & Dividends)

```text
┌─────────────────────────────────────────────────────────────────┐
│                    ADJUSTMENT METHODOLOGY                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. ALWAYS use adjusted prices for ML training                  │
│ 2. Store BOTH raw and adjusted in separate columns             │
│ 3. Re-fetch full history if major split detected               │
│ 4. Validate: adj_close[t] / adj_close[t-1] == return           │
└─────────────────────────────────────────────────────────────────┘
```

- **yfinance**: Use `auto_adjust=True` or access `dividends` / `splits` actions.
- **Polygon/Alpha Vantage**: Request adjusted endpoints explicitly.

### 5.2 Rate Limit Handling

```python
# Retry strategy with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60)
)
def fetch_ticker_data(ticker: str, start: str, end: str):
    # Implementation here
    pass
```

**Guidelines:**
- **Batch requests**: Fetch multiple tickers per call where API allows.
- **Stagger requests**: Add `time.sleep(0.5)` between sequential calls.
- **Cache failures**: Log failed tickers and retry in next cycle.

### 5.3 Pagination

Most OHLCV APIs return full history in one call. For APIs with pagination:

```python
def paginated_fetch(ticker: str, page_size: int = 5000):
    all_data = []
    cursor = None
    while True:
        response = api.get_history(ticker, limit=page_size, cursor=cursor)
        all_data.extend(response["results"])
        cursor = response.get("next_cursor")
        if not cursor:
            break
    return pd.DataFrame(all_data)
```

### 5.4 Data Storage Strategy

```text
Storage Format: PARQUET (recommended)

Advantages over CSV:
├── 5-10x compression ratio
├── Columnar storage (faster column selection)
├── Schema enforcement
├── Supports partitioning by date/ticker
└── Native pandas integration

Directory Structure:
data/
└── ingestion/
    └── market_data/
        └── raw/
            ├── daily/
            │   ├── ticker=AAPL/
            │   │   └── data.parquet
            │   ├── ticker=MC.PA/
            │   │   └── data.parquet
            │   └── ...
            └── weekly/
                └── ... (aggregated from daily)
```

**Partitioning Strategy:**
```python
# Save with partitioning
df.to_parquet(
    "data/ingestion/market_data/raw/daily/",
    partition_cols=["ticker"],
    engine="pyarrow",
    compression="snappy"
)
```

---

## 6. Implementation Plan

### Phase 1: Core Fetcher Development

- [ ] **Step 1.1:** Create `base_fetcher.py` with abstract `DataFetcher` class
  ```python
  class DataFetcher(ABC):
      @abstractmethod
      def fetch(self, tickers: List[str], start: date, end: date) -> pd.DataFrame: ...
      @abstractmethod
      def validate(self, df: pd.DataFrame) -> bool: ...
  ```

- [ ] **Step 1.2:** Implement `yfinance_fetcher.py` as primary source
  - Use `yf.download()` with `group_by="ticker"` for batch downloads
  - Handle `auto_adjust=True` for adjusted prices
  - Parse and store `actions` (dividends/splits) separately

- [ ] **Step 1.3:** Implement `polygon_fetcher.py` as premium alternative
  - Use `/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}`
  - Set `adjusted=true` in query params

### Phase 2: Data Quality & Validation

- [ ] **Step 2.1:** Build `validator.py` module
  - Check for missing dates (exclude weekends/holidays)
  - Detect outliers: `|return| > 0.25` (25% daily move = investigate)
  - Verify monotonic dates, no duplicates
  - Validate OHLC relationship: `Low ≤ Open, Close ≤ High`

- [ ] **Step 2.2:** Create `adjustments_checker.py`
  - Detect suspected splits: `close[t] / close[t-1] ≈ 0.5, 0.33, 2.0, 3.0`
  - Cross-reference with known split calendar

### Phase 3: Storage & Incremental Updates

- [ ] **Step 3.1:** Implement `storage_manager.py`
  - Write to Parquet with partitioning
  - Read existing data, detect last date, fetch only new rows
  - Merge logic with deduplication

- [ ] **Step 3.2:** Build `weekly_aggregator.py`
  - Resample daily → weekly using `resample("W-FRI")`
  - OHLC aggregation: `first`, `max`, `min`, `last`
  - Volume: `sum`

### Phase 4: Orchestration

- [ ] **Step 4.1:** Create `orchestrator.py`
  - Load ticker universe from config
  - Parallel fetching with `concurrent.futures.ThreadPoolExecutor`
  - Progress tracking with `tqdm`

- [ ] **Step 4.2:** Implement scheduling
  ```python
  # Run daily at 22:00 UTC (after US market close)
  schedule.every().day.at("22:00").do(run_daily_ingestion)
  ```

---

## 7. Sample Code Skeleton

```python
# market_data/fetchers/yfinance_fetcher.py

import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class YFinanceFetcher:
    """Primary OHLCV data fetcher using Yahoo Finance."""
    
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def fetch(
        self,
        tickers: List[str],
        start: date,
        end: date,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch adjusted OHLCV data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start: Start date
            end: End date  
            interval: "1d" for daily, "1wk" for weekly
            
        Returns:
            DataFrame with standardized schema
        """
        logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")
        
        # Batch download
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            actions=True,
            group_by="ticker",
            threads=True,
            progress=False
        )
        
        # Transform to standardized schema
        records = []
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = raw
                else:
                    ticker_data = raw[ticker]
                
                for idx, row in ticker_data.iterrows():
                    records.append({
                        "date": idx,
                        "ticker": ticker,
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": int(row["Volume"]),
                        "dividends": row.get("Dividends", 0.0),
                        "stock_splits": row.get("Stock Splits", 0.0),
                    })
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
        
        df = pd.DataFrame(records)
        df["source"] = "yfinance"
        df["fetched_at"] = pd.Timestamp.utcnow()
        
        return df
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic data quality checks."""
        # OHLC sanity
        invalid_ohlc = df[(df["low"] > df["open"]) | (df["low"] > df["close"]) |
                         (df["high"] < df["open"]) | (df["high"] < df["close"])]
        if len(invalid_ohlc) > 0:
            logger.warning(f"Found {len(invalid_ohlc)} rows with invalid OHLC")
        
        # Volume sanity
        df = df[df["volume"] >= 0]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["date", "ticker"])
        
        return df


if __name__ == "__main__":
    # Quick test
    fetcher = YFinanceFetcher()
    df = fetcher.fetch(
        tickers=["AAPL", "MSFT", "MC.PA"],
        start=date(2020, 1, 1),
        end=date.today()
    )
    print(df.head())
    print(f"Shape: {df.shape}")
```

---

## 8. Configuration Example

```yaml
# config/market_data.yaml

sources:
  primary: yfinance
  fallback: polygon

universe:
  us:
    - AAPL
    - MSFT
    - GOOGL
    # ... S&P 500 tickers
  cac40:
    - MC.PA      # LVMH
    - OR.PA      # L'Oreal
    - SAN.PA     # Sanofi
    # ... CAC 40 tickers

settings:
  lookback_years: 10
  resolution:
    - daily
    - weekly
  storage:
    format: parquet
    compression: snappy
    partition_by: ticker
  
rate_limits:
  yfinance:
    requests_per_minute: 30
    batch_size: 50
  polygon:
    requests_per_minute: 60
    
scheduling:
  daily_update: "22:00"  # UTC
  timezone: UTC
```

---

## 9. Dependencies (requirements.txt addition)

```txt
# Market Data Module
yfinance>=0.2.36
pandas-datareader>=0.10.0
polygon-api-client>=1.12.0
alpha_vantage>=2.3.1
pyarrow>=14.0.0
tenacity>=8.2.0
schedule>=1.2.0
httpx>=0.26.0
tqdm>=4.66.0
pyyaml>=6.0.1
```

---

## 10. Testing Checklist

- [ ] Unit test: Fetcher returns correct schema
- [ ] Unit test: Adjustments applied correctly (compare to known split)
- [ ] Integration test: Full pipeline run for 5 tickers
- [ ] Validation test: Detect intentionally corrupt data
- [ ] Performance test: Fetch 500 tickers in < 5 minutes
- [ ] Incremental test: Append 1 day, verify no duplicates

---

*Authored for CAC40_Prediction ML Pipeline — Data Ingestion Layer*
