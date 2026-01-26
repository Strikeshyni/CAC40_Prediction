# Macro Indicators Ingestion Module — Technical Specifications

> **Module:** `data/ingestion/macro_indicators/`  
> **Data Type:** Macroeconomic Time Series (Interest Rates, Inflation, GDP)  
> **Resolution:** Daily, Monthly, Quarterly, Annual  
> **Last Updated:** January 2026

---

## 1. Objective

Build a **macroeconomic data ingestion pipeline** to capture global economic indicators that influence equity markets. The system must:

- **US Focus:** Ingest from FRED (Federal Reserve Economic Data).
- **EU Focus:** Ingest from ECB Statistical Data Warehouse.
- **Coverage:** Interest rates, inflation (CPI), GDP, unemployment, yield curves.
- **Multi-frequency:** Handle daily (rates), monthly (CPI), quarterly (GDP) data.
- **Historical Depth:** 20+ years for robust macro-regime detection.

---

## 2. Recommended Python Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `pandas-datareader` | FRED data access | `pip install pandas-datareader` |
| `fredapi` | Official FRED API wrapper | `pip install fredapi` |
| `sdmx1` | ECB/Eurostat SDMX data access | `pip install sdmx1` |
| `ecbdata` | ECB data helper | `pip install ecbdata` |
| `wbgapi` | World Bank data | `pip install wbgapi` |
| `requests` / `httpx` | Custom API calls | `pip install httpx` |
| `pyarrow` | Parquet storage | `pip install pyarrow` |
| `schedule` | Scheduling updates | `pip install schedule` |

---

## 3. Data Sources

### 3.1 FRED (Federal Reserve Bank of St. Louis)

**Base URL:** `https://api.stlouisfed.org/fred/`  
**API Key:** Required (free at https://fred.stlouisfed.org/docs/api/api_key.html)

| Series ID | Indicator | Frequency | Description |
|-----------|-----------|-----------|-------------|
| `FEDFUNDS` | Fed Funds Rate | Daily | Federal Funds Effective Rate |
| `DFF` | Fed Funds Daily | Daily | Daily effective rate |
| `DGS10` | 10Y Treasury Yield | Daily | 10-Year Treasury Constant Maturity |
| `DGS2` | 2Y Treasury Yield | Daily | 2-Year Treasury Constant Maturity |
| `T10Y2Y` | 10Y-2Y Spread | Daily | Yield curve spread (recession indicator) |
| `CPIAUCSL` | CPI (All Urban) | Monthly | Consumer Price Index |
| `CPILFESL` | Core CPI | Monthly | CPI excluding Food & Energy |
| `PCEPI` | PCE Inflation | Monthly | Fed's preferred inflation measure |
| `GDP` | Real GDP | Quarterly | Gross Domestic Product |
| `GDPC1` | Real GDP (Chained) | Quarterly | Real GDP in chained dollars |
| `UNRATE` | Unemployment Rate | Monthly | Civilian unemployment rate |
| `PAYEMS` | Nonfarm Payrolls | Monthly | Total nonfarm employees |
| `UMCSENT` | Consumer Sentiment | Monthly | University of Michigan survey |
| `VIXCLS` | VIX | Daily | CBOE Volatility Index |
| `BAMLH0A0HYM2` | High Yield Spread | Daily | ICE BofA US High Yield Index OAS |

### 3.2 ECB Statistical Data Warehouse

**Base URL:** `https://data-api.ecb.europa.eu/`  
**API Key:** Not required (public access)  
**Format:** SDMX (Statistical Data and Metadata eXchange)

| Flow ID | Indicator | Frequency | Description |
|---------|-----------|-----------|-------------|
| `MNA` | Main Aggregates | Quarterly | GDP components |
| `ICP` | HICP Inflation | Monthly | Harmonized Index of Consumer Prices |
| `FM` | Financial Markets | Daily | Interest rates, yields |
| `BSI` | Bank Statistics | Monthly | Balance sheet items |
| `EXR` | Exchange Rates | Daily | EUR/USD, EUR/GBP, etc. |

**Key ECB Series:**
| Series Key | Description |
|------------|-------------|
| `FM.B.U2.EUR.4F.KR.MRR_FR.LEV` | ECB Main Refinancing Rate |
| `FM.B.U2.EUR.4F.KR.DFR.LEV` | ECB Deposit Facility Rate |
| `ICP.M.U2.N.000000.4.ANR` | Euro Area HICP Inflation YoY |
| `EXR.D.USD.EUR.SP00.A` | EUR/USD Exchange Rate |

### 3.3 Additional Sources (Optional)

| Source | URL | Data Available | Access |
|--------|-----|----------------|--------|
| **World Bank** | `https://data.worldbank.org/` | GDP, Development indicators | Free API |
| **OECD** | `https://data.oecd.org/` | Leading indicators, CLI | Free API |
| **Eurostat** | `https://ec.europa.eu/eurostat/` | EU statistics | Free SDMX |
| **BIS** | `https://www.bis.org/statistics/` | Credit, Property prices | CSV downloads |
| **IMF** | `https://data.imf.org/` | Global financial data | Free API |

---

## 4. Data Schema

```python
# Macro Indicator Schema
macro_schema = {
    "date": "datetime64[ns]",          # Observation date
    "series_id": "string",             # Source series ID (e.g., "DGS10")
    "indicator_name": "string",        # Human-readable name
    "value": "float64",                # Numeric value
    "unit": "string",                  # "percent", "index", "billions_usd", etc.
    "frequency": "string",             # "daily", "monthly", "quarterly", "annual"
    "region": "string",                # "US", "EU", "UK", "Global"
    "category": "string",              # "interest_rate", "inflation", "gdp", "employment"
    
    # Metadata
    "source": "string",                # "fred", "ecb", "world_bank"
    "vintage": "datetime64[ns]",       # Data release date (for revisions)
    "fetched_at": "datetime64[ns, UTC]",
    
    # ML Features (pre-computed)
    "value_pct_change": "float64",     # Period-over-period change
    "value_yoy": "float64",            # Year-over-year change
    "value_zscore": "float64",         # Z-score (rolling window)
}
```

### Indicator Categories for ML

```python
INDICATOR_CATEGORIES = {
    "interest_rates": [
        "FEDFUNDS", "DFF", "DGS2", "DGS10", "T10Y2Y", "T10Y3M",
        "ECB_MRR", "ECB_DFR"
    ],
    "inflation": [
        "CPIAUCSL", "CPILFESL", "PCEPI", "HICP_EA"
    ],
    "growth": [
        "GDP", "GDPC1", "INDPRO", "EA_GDP"
    ],
    "employment": [
        "UNRATE", "PAYEMS", "ICSA"  # Initial jobless claims
    ],
    "sentiment": [
        "UMCSENT", "VIXCLS"
    ],
    "credit": [
        "BAMLH0A0HYM2", "TEDRATE"  # TED spread
    ],
    "forex": [
        "DEXUSEU"  # USD/EUR
    ]
}
```

---

## 5. Strategy

### 5.1 Data Frequency Alignment

```text
┌─────────────────────────────────────────────────────────────────────┐
│                 FREQUENCY ALIGNMENT STRATEGY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Problem: Macro data has mixed frequencies (daily, monthly, qtr)   │
│  Solution: Align all data to target frequency for ML               │
│                                                                     │
│  DAILY TARGET (for daily price prediction):                        │
│  ├── Daily data: Use as-is                                         │
│  ├── Monthly data: Forward-fill until next release                 │
│  └── Quarterly data: Forward-fill until next release               │
│                                                                     │
│  WEEKLY/MONTHLY TARGET (for swing trading):                        │
│  ├── Daily data: Use last value of period                          │
│  ├── Monthly data: Use as-is or interpolate                        │
│  └── Quarterly data: Forward-fill or interpolate                   │
│                                                                     │
│  IMPORTANT: Track actual release dates for forward-looking bias    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```python
def align_to_daily(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Align macro data to daily frequency.
    
    Args:
        df: DataFrame with DateTimeIndex
        method: "ffill" (forward-fill) or "interpolate"
    
    Returns:
        Daily-frequency DataFrame
    """
    # Create daily date range
    daily_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D"
    )
    
    # Reindex and fill
    df_daily = df.reindex(daily_index)
    
    if method == "ffill":
        df_daily = df_daily.ffill()
    elif method == "interpolate":
        df_daily = df_daily.interpolate(method="linear")
    
    return df_daily
```

### 5.2 Handling Data Revisions

Macro data (especially GDP) gets revised over time. For ML training, we need to:

```python
@dataclass
class VintageRecord:
    """Track data revisions by vintage date."""
    series_id: str
    observation_date: date      # The date the data point refers to
    vintage_date: date          # When this value was released/revised
    value: float
    is_preliminary: bool
    is_revised: bool


# Strategy: Store multiple vintages for backtesting
# Use "as-of" queries to avoid lookahead bias
def get_series_as_of(
    series_id: str,
    observation_date: date,
    as_of_date: date
) -> float:
    """
    Get the value that would have been known at as_of_date.
    
    This prevents lookahead bias in backtesting.
    """
    vintages = load_vintages(series_id, observation_date)
    valid = [v for v in vintages if v.vintage_date <= as_of_date]
    return max(valid, key=lambda v: v.vintage_date).value if valid else None
```

### 5.3 Rate Limits

| Source | Rate Limit | Strategy |
|--------|------------|----------|
| **FRED** | 120 requests/minute | Batch requests, 0.5s delay |
| **ECB** | No published limit | Conservative 1 req/sec |
| **World Bank** | Generous | No special handling needed |

### 5.4 Data Storage Strategy

```text
Storage Format: PARQUET with time-series partitioning

Directory Structure:
data/
└── ingestion/
    └── macro_indicators/
        ├── raw/
        │   ├── fred/
        │   │   ├── series=DGS10/
        │   │   │   └── data.parquet
        │   │   ├── series=CPIAUCSL/
        │   │   │   └── data.parquet
        │   │   └── ...
        │   └── ecb/
        │       ├── series=ICP_HICP/
        │       │   └── data.parquet
        │       └── ...
        │
        ├── processed/
        │   ├── daily_aligned/
        │   │   └── macro_daily.parquet       # All indicators, daily freq
        │   ├── weekly_aligned/
        │   │   └── macro_weekly.parquet
        │   └── features/
        │       └── macro_features.parquet    # Pre-computed features
        │
        └── metadata/
            ├── series_catalog.json           # All tracked series
            └── fetch_state.json              # Last fetch timestamps
```

---

## 6. Implementation Plan

### Phase 1: FRED Integration (US Priority)

- [ ] **Step 1.1:** Create `fetchers/fred_fetcher.py`
  ```python
  class FREDFetcher:
      def __init__(self, api_key: str): ...
      async def fetch_series(self, series_id: str, start: date, end: date) -> pd.DataFrame: ...
      async def fetch_multiple(self, series_ids: List[str]) -> Dict[str, pd.DataFrame]: ...
      async def search_series(self, query: str) -> List[SeriesMetadata]: ...
  ```

- [ ] **Step 1.2:** Implement core indicators fetch
  - Interest rates: `FEDFUNDS`, `DGS2`, `DGS10`, `T10Y2Y`
  - Inflation: `CPIAUCSL`, `CPILFESL`, `PCEPI`
  - Growth: `GDP`, `GDPC1`, `INDPRO`
  - Employment: `UNRATE`, `PAYEMS`

- [ ] **Step 1.3:** Create `fetchers/fred_vintage_fetcher.py`
  - FRED ALFRED API for real-time vintage data
  - Store revision history for backtesting

### Phase 2: ECB Integration (EU Priority)

- [ ] **Step 2.1:** Implement `fetchers/ecb_fetcher.py`
  ```python
  class ECBFetcher:
      """Fetch ECB data using SDMX protocol."""
      
      def __init__(self): ...
      async def fetch_series(self, dataflow: str, key: str) -> pd.DataFrame: ...
      def parse_sdmx_response(self, response: bytes) -> pd.DataFrame: ...
  ```

- [ ] **Step 2.2:** Implement key ECB indicators
  - Interest rates: Main Refinancing Rate, Deposit Rate
  - Inflation: HICP (Harmonized Index)
  - Exchange rates: EUR/USD, EUR/GBP

- [ ] **Step 2.3:** Handle SDMX format parsing
  - Use `sdmx1` library
  - Map ECB series keys to human-readable names

### Phase 3: Data Processing

- [ ] **Step 3.1:** Create `processors/frequency_aligner.py`
  - Align all series to daily frequency
  - Create weekly/monthly aggregates
  - Handle missing data appropriately

- [ ] **Step 3.2:** Create `processors/feature_engineer.py`
  - Calculate returns / changes
  - Compute year-over-year comparisons
  - Generate z-scores and percentiles
  - Create regime indicators (expansion/recession)

- [ ] **Step 3.3:** Create `processors/yield_curve.py`
  - Fetch full yield curve (3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)
  - Calculate curve slope, curvature
  - Detect inversions

### Phase 4: Storage & Orchestration

- [ ] **Step 4.1:** Implement `storage/macro_writer.py`
  - Write raw series to partitioned Parquet
  - Write processed/aligned data
  - Track fetch state for incremental updates

- [ ] **Step 4.2:** Create `orchestrator.py`
  - Coordinate FRED and ECB fetches
  - Run frequency alignment
  - Generate ML-ready feature sets

- [ ] **Step 4.3:** Implement scheduling
  - Daily: Interest rates, VIX
  - Weekly: Summarized view
  - Monthly: Inflation, employment
  - Quarterly: GDP

---

## 7. Sample Code: FRED Fetcher

```python
# macro_indicators/fetchers/fred_fetcher.py

import pandas as pd
from fredapi import Fred
from typing import List, Dict, Optional
from datetime import date, timedelta
from dataclasses import dataclass
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class FREDSeriesMetadata:
    series_id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: date


class FREDFetcher:
    """
    Fetcher for Federal Reserve Economic Data (FRED).
    
    API Docs: https://fred.stlouisfed.org/docs/api/fred/
    Rate Limit: 120 requests per minute
    """
    
    # Core indicator series
    CORE_SERIES = {
        # Interest Rates
        "FEDFUNDS": "Federal Funds Effective Rate",
        "DFF": "Fed Funds Rate (Daily)",
        "DGS2": "2-Year Treasury Yield",
        "DGS10": "10-Year Treasury Yield",
        "DGS30": "30-Year Treasury Yield",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "T10Y3M": "10Y-3M Treasury Spread",
        
        # Inflation
        "CPIAUCSL": "CPI All Urban Consumers",
        "CPILFESL": "Core CPI (ex Food & Energy)",
        "PCEPI": "PCE Price Index",
        "PCEPILFE": "Core PCE Price Index",
        
        # Growth
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP (Chained 2017 Dollars)",
        "INDPRO": "Industrial Production Index",
        
        # Employment
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls (Thousands)",
        "ICSA": "Initial Jobless Claims",
        
        # Sentiment & Volatility
        "UMCSENT": "Consumer Sentiment",
        "VIXCLS": "CBOE Volatility Index",
        
        # Credit Spreads
        "BAMLH0A0HYM2": "High Yield Spread (OAS)",
        "TEDRATE": "TED Spread",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher.
        
        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY required")
        
        self.fred = Fred(api_key=self.api_key)
        self.request_delay = 0.5  # Conservative rate limiting
    
    def fetch_series(
        self,
        series_id: str,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series.
        
        Args:
            series_id: FRED series ID (e.g., "DGS10")
            start: Start date (default: 20 years ago)
            end: End date (default: today)
        
        Returns:
            DataFrame with date index and value column
        """
        if start is None:
            start = date.today() - timedelta(days=365 * 20)
        if end is None:
            end = date.today()
        
        logger.info(f"Fetching FRED series: {series_id}")
        
        try:
            series = self.fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end
            )
            
            df = pd.DataFrame({
                "date": series.index,
                "value": series.values,
                "series_id": series_id,
                "indicator_name": self.CORE_SERIES.get(series_id, series_id),
            })
            
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["value"])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            raise
    
    def fetch_multiple(
        self,
        series_ids: Optional[List[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple FRED series.
        
        Args:
            series_ids: List of series IDs (default: CORE_SERIES)
            start: Start date
            end: End date
        
        Returns:
            Dictionary mapping series_id to DataFrame
        """
        if series_ids is None:
            series_ids = list(self.CORE_SERIES.keys())
        
        results = {}
        failed = []
        
        for series_id in series_ids:
            try:
                df = self.fetch_series(series_id, start, end)
                results[series_id] = df
                import time
                time.sleep(self.request_delay)
            except Exception as e:
                logger.warning(f"Failed {series_id}: {e}")
                failed.append(series_id)
        
        if failed:
            logger.warning(f"Failed to fetch {len(failed)} series: {failed}")
        
        return results
    
    def fetch_yield_curve(
        self,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch the US Treasury yield curve.
        
        Returns DataFrame with maturity as columns.
        """
        maturities = {
            "DGS1MO": "1M",
            "DGS3MO": "3M",
            "DGS6MO": "6M",
            "DGS1": "1Y",
            "DGS2": "2Y",
            "DGS5": "5Y",
            "DGS7": "7Y",
            "DGS10": "10Y",
            "DGS20": "20Y",
            "DGS30": "30Y",
        }
        
        if as_of_date is None:
            as_of_date = date.today()
        
        start = as_of_date - timedelta(days=365 * 10)
        
        curve_data = {}
        for series_id, maturity in maturities.items():
            try:
                df = self.fetch_series(series_id, start, as_of_date)
                curve_data[maturity] = df.set_index("date")["value"]
                import time
                time.sleep(self.request_delay)
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        
        curve_df = pd.DataFrame(curve_data)
        curve_df.index = pd.to_datetime(curve_df.index)
        
        return curve_df
    
    def get_series_info(self, series_id: str) -> FREDSeriesMetadata:
        """Get metadata for a FRED series."""
        info = self.fred.get_series_info(series_id)
        
        return FREDSeriesMetadata(
            series_id=series_id,
            title=info.get("title", ""),
            frequency=info.get("frequency_short", ""),
            units=info.get("units_short", ""),
            seasonal_adjustment=info.get("seasonal_adjustment_short", ""),
            last_updated=pd.to_datetime(info.get("last_updated")).date()
        )
    
    def combine_series(
        self,
        series_dict: Dict[str, pd.DataFrame],
        align_freq: str = "D"
    ) -> pd.DataFrame:
        """
        Combine multiple series into a single DataFrame.
        
        Args:
            series_dict: Dictionary of series DataFrames
            align_freq: Target frequency ("D", "W", "M")
        
        Returns:
            Wide-format DataFrame with series as columns
        """
        combined = None
        
        for series_id, df in series_dict.items():
            df_indexed = df.set_index("date")[["value"]].rename(
                columns={"value": series_id}
            )
            
            if combined is None:
                combined = df_indexed
            else:
                combined = combined.join(df_indexed, how="outer")
        
        # Resample to target frequency
        if align_freq == "D":
            combined = combined.resample("D").last().ffill()
        elif align_freq == "W":
            combined = combined.resample("W-FRI").last().ffill()
        elif align_freq == "M":
            combined = combined.resample("M").last().ffill()
        
        return combined


# Usage example
def main():
    fetcher = FREDFetcher()
    
    # Fetch core indicators
    series_data = fetcher.fetch_multiple(
        series_ids=["DGS10", "DGS2", "T10Y2Y", "CPIAUCSL", "UNRATE"],
        start=date(2010, 1, 1)
    )
    
    # Combine into single DataFrame
    combined = fetcher.combine_series(series_data, align_freq="D")
    print(combined.tail(10))
    
    # Fetch yield curve
    curve = fetcher.fetch_yield_curve()
    print("\nYield Curve (latest):")
    print(curve.tail(1).T)


if __name__ == "__main__":
    main()
```

---

## 8. Sample Code: ECB Fetcher

```python
# macro_indicators/fetchers/ecb_fetcher.py

import sdmx
import pandas as pd
from typing import Optional, List, Dict
from datetime import date
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ECBSeriesKey:
    """ECB SDMX series key specification."""
    dataflow: str           # e.g., "FM" for Financial Markets
    key: str                # e.g., "B.U2.EUR.4F.KR.MRR_FR.LEV"
    description: str
    frequency: str


class ECBFetcher:
    """
    Fetcher for ECB Statistical Data Warehouse using SDMX.
    
    API Docs: https://data.ecb.europa.eu/help/api/overview
    """
    
    # Core ECB series
    CORE_SERIES = {
        "ECB_MRR": ECBSeriesKey(
            dataflow="FM",
            key="B.U2.EUR.4F.KR.MRR_FR.LEV",
            description="ECB Main Refinancing Rate",
            frequency="B"  # Business daily
        ),
        "ECB_DFR": ECBSeriesKey(
            dataflow="FM",
            key="B.U2.EUR.4F.KR.DFR.LEV",
            description="ECB Deposit Facility Rate",
            frequency="B"
        ),
        "HICP_EA": ECBSeriesKey(
            dataflow="ICP",
            key="M.U2.N.000000.4.ANR",
            description="Euro Area HICP YoY",
            frequency="M"
        ),
        "HICP_CORE_EA": ECBSeriesKey(
            dataflow="ICP",
            key="M.U2.N.XEF000.4.ANR",
            description="Euro Area Core HICP YoY",
            frequency="M"
        ),
        "EUR_USD": ECBSeriesKey(
            dataflow="EXR",
            key="D.USD.EUR.SP00.A",
            description="EUR/USD Exchange Rate",
            frequency="D"
        ),
        "EUR_GBP": ECBSeriesKey(
            dataflow="EXR",
            key="D.GBP.EUR.SP00.A",
            description="EUR/GBP Exchange Rate",
            frequency="D"
        ),
    }
    
    def __init__(self):
        """Initialize ECB SDMX client."""
        self.ecb = sdmx.Client("ECB")
    
    def fetch_series(
        self,
        series_key: ECBSeriesKey,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch a single ECB series.
        
        Args:
            series_key: ECBSeriesKey specification
            start: Start date
            end: End date
        
        Returns:
            DataFrame with date index and value column
        """
        logger.info(f"Fetching ECB series: {series_key.description}")
        
        # Build date parameters
        params = {}
        if start:
            params["startPeriod"] = start.strftime("%Y-%m-%d")
        if end:
            params["endPeriod"] = end.strftime("%Y-%m-%d")
        
        try:
            # Fetch data using SDMX
            data_msg = self.ecb.data(
                series_key.dataflow,
                key=series_key.key,
                params=params
            )
            
            # Convert to pandas
            df = sdmx.to_pandas(data_msg)
            
            if isinstance(df, pd.Series):
                df = df.reset_index()
                df.columns = ["date", "value"]
            else:
                # Handle multi-dimensional response
                df = df.reset_index()
                # Take first value column
                value_cols = [c for c in df.columns if c not in ["TIME_PERIOD"]]
                df = df[["TIME_PERIOD", value_cols[0]]].rename(
                    columns={"TIME_PERIOD": "date", value_cols[0]: "value"}
                )
            
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch ECB series: {e}")
            raise
    
    def fetch_multiple(
        self,
        series_ids: Optional[List[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple ECB series.
        
        Args:
            series_ids: List of series IDs from CORE_SERIES
            start: Start date
            end: End date
        
        Returns:
            Dictionary mapping series_id to DataFrame
        """
        if series_ids is None:
            series_ids = list(self.CORE_SERIES.keys())
        
        results = {}
        
        for series_id in series_ids:
            if series_id not in self.CORE_SERIES:
                logger.warning(f"Unknown series: {series_id}")
                continue
            
            series_key = self.CORE_SERIES[series_id]
            
            try:
                df = self.fetch_series(series_key, start, end)
                df["series_id"] = series_id
                df["indicator_name"] = series_key.description
                results[series_id] = df
                
                import time
                time.sleep(1.0)  # Conservative rate limiting
                
            except Exception as e:
                logger.warning(f"Failed {series_id}: {e}")
        
        return results
    
    def fetch_interest_rates(
        self,
        start: Optional[date] = None
    ) -> pd.DataFrame:
        """Convenience method to fetch ECB interest rates."""
        series = self.fetch_multiple(
            series_ids=["ECB_MRR", "ECB_DFR"],
            start=start
        )
        
        combined = pd.concat(series.values(), ignore_index=True)
        return combined.pivot(
            index="date",
            columns="series_id",
            values="value"
        )
    
    def fetch_inflation(
        self,
        start: Optional[date] = None
    ) -> pd.DataFrame:
        """Convenience method to fetch Euro Area inflation."""
        series = self.fetch_multiple(
            series_ids=["HICP_EA", "HICP_CORE_EA"],
            start=start
        )
        
        combined = pd.concat(series.values(), ignore_index=True)
        return combined.pivot(
            index="date",
            columns="series_id",
            values="value"
        )


# Usage example
def main():
    fetcher = ECBFetcher()
    
    # Fetch ECB interest rates
    rates = fetcher.fetch_interest_rates(start=date(2010, 1, 1))
    print("ECB Interest Rates:")
    print(rates.tail(10))
    
    # Fetch Euro Area inflation
    inflation = fetcher.fetch_inflation(start=date(2010, 1, 1))
    print("\nEuro Area Inflation:")
    print(inflation.tail(10))
    
    # Fetch exchange rates
    fx = fetcher.fetch_multiple(
        series_ids=["EUR_USD", "EUR_GBP"],
        start=date(2020, 1, 1)
    )
    print("\nEUR/USD (latest):")
    print(fx["EUR_USD"].tail(5))


if __name__ == "__main__":
    main()
```

---

## 9. Sample Code: Feature Engineering

```python
# macro_indicators/processors/feature_engineer.py

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class MacroFeatureEngineer:
    """
    Generate ML-ready features from macro indicators.
    """
    
    def __init__(self, lookback_windows: List[int] = [5, 21, 63, 252]):
        """
        Initialize feature engineer.
        
        Args:
            lookback_windows: Rolling window sizes (in days)
                5 = 1 week, 21 = 1 month, 63 = 1 quarter, 252 = 1 year
        """
        self.lookback_windows = lookback_windows
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for all columns in DataFrame.
        
        Args:
            df: Wide-format DataFrame (date index, indicators as columns)
        
        Returns:
            DataFrame with original and derived features
        """
        features = df.copy()
        
        for col in df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Period-over-period changes
            features[f"{col}_change"] = df[col].diff()
            features[f"{col}_pct_change"] = df[col].pct_change()
            
            # Year-over-year change (for monthly/quarterly data)
            features[f"{col}_yoy"] = df[col].pct_change(periods=252)
            
            # Rolling statistics
            for window in self.lookback_windows:
                # Moving average
                features[f"{col}_ma{window}"] = df[col].rolling(window).mean()
                
                # Distance from moving average
                features[f"{col}_ma{window}_dist"] = (
                    df[col] - features[f"{col}_ma{window}"]
                ) / features[f"{col}_ma{window}"]
                
                # Rolling volatility
                features[f"{col}_vol{window}"] = (
                    df[col].pct_change().rolling(window).std() * np.sqrt(252)
                )
                
                # Z-score
                roll_mean = df[col].rolling(window).mean()
                roll_std = df[col].rolling(window).std()
                features[f"{col}_zscore{window}"] = (df[col] - roll_mean) / roll_std
        
        return features
    
    def compute_yield_curve_features(
        self,
        curve_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute yield curve features.
        
        Args:
            curve_df: DataFrame with maturity columns (1M, 3M, 1Y, 2Y, 10Y, 30Y)
        
        Returns:
            DataFrame with curve features
        """
        features = pd.DataFrame(index=curve_df.index)
        
        # Spreads
        if "10Y" in curve_df.columns and "2Y" in curve_df.columns:
            features["spread_10y_2y"] = curve_df["10Y"] - curve_df["2Y"]
        
        if "10Y" in curve_df.columns and "3M" in curve_df.columns:
            features["spread_10y_3m"] = curve_df["10Y"] - curve_df["3M"]
        
        if "30Y" in curve_df.columns and "2Y" in curve_df.columns:
            features["spread_30y_2y"] = curve_df["30Y"] - curve_df["2Y"]
        
        # Curve slope (linear regression slope)
        maturities = {
            "1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1,
            "2Y": 2, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30
        }
        
        available_mats = [m for m in maturities if m in curve_df.columns]
        x_vals = np.array([maturities[m] for m in available_mats])
        
        def compute_slope(row):
            y_vals = row[available_mats].values.astype(float)
            if np.isnan(y_vals).any():
                return np.nan
            # Simple linear regression slope
            slope = np.polyfit(x_vals, y_vals, 1)[0]
            return slope
        
        features["curve_slope"] = curve_df.apply(compute_slope, axis=1)
        
        # Curvature (2Y + 10Y) / 2 - 5Y
        if all(m in curve_df.columns for m in ["2Y", "5Y", "10Y"]):
            features["curve_curvature"] = (
                (curve_df["2Y"] + curve_df["10Y"]) / 2 - curve_df["5Y"]
            )
        
        # Inversion indicator
        features["curve_inverted"] = (features["spread_10y_2y"] < 0).astype(int)
        
        return features
    
    def compute_regime_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute economic regime indicators.
        
        Args:
            df: DataFrame with macro indicators
        
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)
        
        # Inflation regime (based on CPI if available)
        cpi_cols = [c for c in df.columns if "CPI" in c.upper()]
        if cpi_cols:
            cpi = df[cpi_cols[0]]
            cpi_yoy = cpi.pct_change(12) * 100  # Assuming monthly
            features["inflation_regime"] = pd.cut(
                cpi_yoy,
                bins=[-np.inf, 0, 2, 4, np.inf],
                labels=["deflation", "low", "moderate", "high"]
            )
        
        # Rate regime (based on fed funds if available)
        rate_cols = [c for c in df.columns if "FED" in c.upper() or "DFF" in c.upper()]
        if rate_cols:
            rate = df[rate_cols[0]]
            rate_change_6m = rate.diff(126)  # ~6 months
            features["rate_regime"] = np.where(
                rate_change_6m > 0.25, "hiking",
                np.where(rate_change_6m < -0.25, "cutting", "stable")
            )
        
        return features


# Usage example
def main():
    from fetchers.fred_fetcher import FREDFetcher
    from datetime import date
    
    # Fetch data
    fetcher = FREDFetcher()
    series = fetcher.fetch_multiple(
        series_ids=["DGS10", "DGS2", "CPIAUCSL", "FEDFUNDS"],
        start=date(2010, 1, 1)
    )
    combined = fetcher.combine_series(series, align_freq="D")
    
    # Generate features
    engineer = MacroFeatureEngineer()
    features = engineer.compute_features(combined)
    
    print("Generated Features:")
    print(features.columns.tolist())
    print(features.tail(5))


if __name__ == "__main__":
    main()
```

---

## 10. Configuration Example

```yaml
# config/macro_indicators.yaml

sources:
  fred:
    enabled: true
    api_key_env: FRED_API_KEY
    rate_limit_rpm: 120
    core_series:
      interest_rates:
        - FEDFUNDS
        - DFF
        - DGS2
        - DGS10
        - DGS30
        - T10Y2Y
      inflation:
        - CPIAUCSL
        - CPILFESL
        - PCEPI
      growth:
        - GDP
        - GDPC1
      employment:
        - UNRATE
        - PAYEMS
      sentiment:
        - UMCSENT
        - VIXCLS
  
  ecb:
    enabled: true
    rate_limit_rpm: 60
    core_series:
      - ECB_MRR
      - ECB_DFR
      - HICP_EA
      - EUR_USD

processing:
  align_frequency: daily
  fill_method: ffill
  lookback_windows:
    - 5     # 1 week
    - 21    # 1 month
    - 63    # 1 quarter
    - 252   # 1 year
  compute_features:
    - changes
    - moving_averages
    - z_scores
    - regime_indicators

storage:
  format: parquet
  compression: snappy
  partition_by: year

scheduling:
  daily_update: "08:00"     # UTC (after ECB updates)
  monthly_update: "15:00"   # Day of CPI release
```

---

## 11. Dependencies (requirements.txt addition)

```txt
# Macro Indicators Module
fredapi>=0.5.1
pandas-datareader>=0.10.0
sdmx1>=2.12.0
ecbdata>=0.0.5
wbgapi>=1.0.12
pyarrow>=14.0.0
schedule>=1.2.0
httpx>=0.26.0
```

---

## 12. Testing Checklist

- [ ] Unit test: FRED fetcher returns valid schema for all core series
- [ ] Unit test: ECB fetcher handles SDMX parsing correctly
- [ ] Unit test: Frequency alignment produces no NaN at series start
- [ ] Unit test: Feature engineering produces expected output shape
- [ ] Integration test: Full pipeline for 5 key indicators
- [ ] Validation test: Compare FRED vs OECD for overlapping series
- [ ] Backtest test: "As-of" queries prevent lookahead bias
- [ ] Performance test: Fetch 20 series in < 60 seconds

---

## 13. Economic Regime Detection (Advanced)

```python
# Bonus: Simple recession probability indicator

def compute_recession_probability(
    df: pd.DataFrame,
    spread_col: str = "T10Y2Y",
    window: int = 252
) -> pd.Series:
    """
    Compute recession probability based on yield curve inversion.
    
    Historical rule: Inversion (10Y-2Y < 0) precedes recession by 12-18 months.
    """
    if spread_col not in df.columns:
        raise ValueError(f"Spread column {spread_col} not found")
    
    spread = df[spread_col]
    
    # Count days inverted in rolling window
    inverted = (spread < 0).astype(float)
    inversion_ratio = inverted.rolling(window).mean()
    
    # Map to probability (calibrated heuristic)
    # 0% inverted → 10% base recession probability
    # 50%+ inverted → 70%+ recession probability
    prob = 0.10 + 0.60 * inversion_ratio.clip(0, 1)
    
    return prob
```

---

*Authored for CAC40_Prediction ML Pipeline — Data Ingestion Layer*
