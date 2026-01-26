# Corporate Filings Ingestion Module — Technical Specifications

> **Module:** `data/ingestion/corporate_filings/`  
> **Data Type:** Regulatory Filings (10-K, 10-Q, Annual Reports, EU ESEF)  
> **Resolution:** Quarterly / Annual  
> **Last Updated:** January 2026

---

## 1. Objective

Build a **multi-jurisdictional corporate filings ingestion pipeline** to extract fundamental data from regulatory documents. The system must:

- **US Market:** Fetch 10-K (annual) and 10-Q (quarterly) filings from SEC EDGAR.
- **European Market:** Handle EU ESEF-format filings and national registry variations.
- **Data Extraction:** Parse XBRL/iXBRL for structured financial data (revenue, EPS, debt ratios).
- **Fallback Strategy:** Scrape Yahoo Finance "Financials" tab when direct registry access is fragmented.
- **Historical Coverage:** Minimum 5-year lookback for training data.

---

## 2. Recommended Python Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `sec-edgar-downloader` | Download SEC EDGAR filings | `pip install sec-edgar-downloader` |
| `sec-api` | Premium SEC EDGAR access | `pip install sec-api` |
| `edgartools` | Modern SEC EDGAR parser | `pip install edgartools` |
| `python-xbrl` | XBRL parsing | `pip install python-xbrl` |
| `arelle` | Full XBRL/iXBRL processor | See special install |
| `beautifulsoup4` | HTML/XML parsing | `pip install beautifulsoup4` |
| `lxml` | Fast XML parser | `pip install lxml` |
| `requests` / `httpx` | HTTP client | `pip install httpx` |
| `pdfplumber` | PDF text extraction | `pip install pdfplumber` |
| `playwright` | JS-rendered page scraping | `pip install playwright` |
| `pandas` | Data manipulation | `pip install pandas` |

---

## 3. Data Sources

### 3.1 United States — SEC EDGAR

| Resource | URL | Description |
|----------|-----|-------------|
| **SEC EDGAR Full-Text Search** | `https://efts.sec.gov/LATEST/search-index` | Search filings by company, type |
| **SEC EDGAR REST API** | `https://data.sec.gov/submissions/CIK{cik}.json` | Company submissions metadata |
| **SEC EDGAR Archives** | `https://www.sec.gov/cgi-bin/browse-edgar` | Direct filing access |
| **SEC Financial Statements** | `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` | Pre-parsed XBRL facts |

**Filing Types:**
| Form | Frequency | Content |
|------|-----------|---------|
| **10-K** | Annual | Full financial statements, MD&A, risk factors |
| **10-Q** | Quarterly | Quarterly financials, updates |
| **8-K** | Event-driven | Material events, earnings releases |
| **DEF 14A** | Annual | Proxy statements, executive compensation |

### 3.2 Europe — Fragmented Landscape

| Country | Registry | URL | Format |
|---------|----------|-----|--------|
| **EU (ESEF)** | ESMA ESEF Portal | `https://filings.xbrl.org/` | iXBRL (inline XBRL) |
| **France (AMF)** | AMF BDIF | `https://bdif.amf-france.org/` | PDF + iXBRL |
| **Germany (BaFin)** | Unternehmensregister | `https://www.unternehmensregister.de/` | PDF + XBRL |
| **UK (FCA)** | Companies House | `https://find-and-update.company-information.service.gov.uk/` | iXBRL |
| **Netherlands (AFM)** | Kamer van Koophandel | `https://www.kvk.nl/` | PDF + XBRL |

### 3.3 Fallback — Yahoo Finance Financials Scraping

| URL Pattern | Data Available |
|-------------|----------------|
| `https://finance.yahoo.com/quote/{TICKER}/financials` | Income Statement |
| `https://finance.yahoo.com/quote/{TICKER}/balance-sheet` | Balance Sheet |
| `https://finance.yahoo.com/quote/{TICKER}/cash-flow` | Cash Flow Statement |

---

## 4. Data Schema

### 4.1 Filing Metadata Schema

```python
filing_metadata_schema = {
    "filing_id": "string",             # Unique identifier (e.g., SEC accession number)
    "company_name": "string",          # Legal entity name
    "ticker": "string",                # Trading symbol
    "cik": "string",                   # SEC Central Index Key (US)
    "lei": "string",                   # Legal Entity Identifier (Global)
    
    # Filing details
    "form_type": "string",             # 10-K, 10-Q, Annual Report, etc.
    "filing_date": "datetime64[ns]",   # Date filed with regulator
    "period_end_date": "datetime64[ns]",  # Fiscal period end
    "fiscal_year": "int64",            # e.g., 2025
    "fiscal_quarter": "int64",         # 1, 2, 3, 4 (null for annual)
    
    # Source
    "source": "string",                # "sec_edgar", "amf", "yahoo_scrape"
    "source_url": "string",            # Direct link to filing
    "document_urls": "list[string]",   # Links to all documents in filing
    
    # Processing
    "fetched_at": "datetime64[ns, UTC]",
    "parsed_at": "datetime64[ns, UTC]",
    "status": "string",                # "raw", "parsed", "validated"
}
```

### 4.2 Financial Data Schema (Extracted)

```python
financial_data_schema = {
    "filing_id": "string",             # Link to metadata
    "ticker": "string",
    "period_end_date": "datetime64[ns]",
    "period_type": "string",           # "annual", "quarterly"
    
    # Income Statement
    "revenue": "float64",              # Total Revenue
    "cost_of_revenue": "float64",
    "gross_profit": "float64",
    "operating_income": "float64",
    "net_income": "float64",
    "eps_basic": "float64",
    "eps_diluted": "float64",
    
    # Balance Sheet
    "total_assets": "float64",
    "total_liabilities": "float64",
    "total_equity": "float64",
    "cash_and_equivalents": "float64",
    "total_debt": "float64",
    
    # Cash Flow
    "operating_cash_flow": "float64",
    "investing_cash_flow": "float64",
    "financing_cash_flow": "float64",
    "free_cash_flow": "float64",
    "capex": "float64",
    
    # Derived Ratios
    "debt_to_equity": "float64",
    "current_ratio": "float64",
    "profit_margin": "float64",
    "roe": "float64",                  # Return on Equity
    
    # Metadata
    "currency": "string",              # USD, EUR, GBP
    "unit_scale": "string",            # "units", "thousands", "millions"
    "source": "string",
}
```

---

## 5. Strategy

### 5.1 US Filings — SEC EDGAR Approach

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    SEC EDGAR INGESTION FLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Resolve Ticker → CIK                                           │
│     └── GET https://www.sec.gov/cgi-bin/browse-edgar?              │
│              action=getcompany&company={ticker}&type=&              │
│              dateb=&owner=include&count=40&output=atom              │
│                                                                     │
│  2. Fetch Company Submissions                                       │
│     └── GET https://data.sec.gov/submissions/CIK{cik}.json         │
│                                                                     │
│  3. Filter Relevant Filings (10-K, 10-Q)                           │
│     └── Parse filings array, filter by form type                   │
│                                                                     │
│  4. Download Filing Documents                                       │
│     └── GET https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/   │
│                                                                     │
│  5. Parse XBRL (if available) or HTML                              │
│     └── Use python-xbrl or BeautifulSoup                           │
│                                                                     │
│  6. Extract Financial Facts                                         │
│     └── GET https://data.sec.gov/api/xbrl/companyfacts/            │
│              CIK{cik}.json (pre-parsed)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Rate Limits:**
- SEC EDGAR: 10 requests per second (be conservative: 5/sec)
- Must include User-Agent with contact email

```python
HEADERS = {
    "User-Agent": "CAC40Prediction research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
```

### 5.2 EU Filings — Hybrid Strategy

Given EU fragmentation, we propose a **tiered approach**:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    EU FILINGS STRATEGY                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: ESEF Central Repository (Preferred)                       │
│  ├── Source: filings.xbrl.org                                      │
│  ├── Format: iXBRL (machine-readable)                              │
│  └── Coverage: Growing, but not complete                           │
│                                                                     │
│  TIER 2: National Registries (CAC40 Focus)                         │
│  ├── France (AMF BDIF): Manual PDF + some iXBRL                    │
│  ├── Priority companies: LVMH, L'Oreal, TotalEnergies              │
│  └── Requires custom scraper per registry                          │
│                                                                     │
│  TIER 3: Yahoo Finance Scraping (Fallback)                         │
│  ├── Consistent format across jurisdictions                        │
│  ├── Data: Annual & Quarterly financials                           │
│  └── Limitation: Less detail, potential delays                     │
│                                                                     │
│  TIER 4: Premium Data Providers (Production)                       │
│  ├── Refinitiv, Bloomberg, FactSet                                 │
│  └── Unified API, global coverage                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 XBRL Parsing Strategy

```python
# Key XBRL concepts to extract (US GAAP taxonomy)
XBRL_CONCEPTS = {
    # Income Statement
    "us-gaap:Revenues": "revenue",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "us-gaap:CostOfRevenue": "cost_of_revenue",
    "us-gaap:GrossProfit": "gross_profit",
    "us-gaap:OperatingIncomeLoss": "operating_income",
    "us-gaap:NetIncomeLoss": "net_income",
    "us-gaap:EarningsPerShareBasic": "eps_basic",
    "us-gaap:EarningsPerShareDiluted": "eps_diluted",
    
    # Balance Sheet
    "us-gaap:Assets": "total_assets",
    "us-gaap:Liabilities": "total_liabilities",
    "us-gaap:StockholdersEquity": "total_equity",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "us-gaap:LongTermDebt": "total_debt",
    
    # Cash Flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": "capex",
}

# IFRS equivalents for EU companies
IFRS_CONCEPTS = {
    "ifrs-full:Revenue": "revenue",
    "ifrs-full:GrossProfit": "gross_profit",
    "ifrs-full:ProfitLoss": "net_income",
    "ifrs-full:Assets": "total_assets",
    "ifrs-full:Equity": "total_equity",
    # ... etc
}
```

### 5.4 Rate Limit Handling

```python
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import httpx

@dataclass
class SECRateLimiter:
    """SEC EDGAR rate limiter: max 10 requests/second (use 5 for safety)."""
    max_rps: float = 5.0
    last_request: datetime = field(default_factory=datetime.now)
    
    async def wait(self):
        """Wait if needed to respect rate limit."""
        elapsed = (datetime.now() - self.last_request).total_seconds()
        wait_time = max(0, (1 / self.max_rps) - elapsed)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self.last_request = datetime.now()


class SECClient:
    """HTTP client with SEC-compliant rate limiting."""
    
    def __init__(self, email: str):
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": f"CAC40Prediction {email}",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0
        )
        self.rate_limiter = SECRateLimiter()
    
    async def get(self, url: str) -> httpx.Response:
        await self.rate_limiter.wait()
        return await self.client.get(url)
```

### 5.5 Data Storage Strategy

```text
Storage Format: PARQUET (structured data) + Raw Files (documents)

Directory Structure:
data/
└── ingestion/
    └── corporate_filings/
        ├── raw/
        │   ├── sec_edgar/
        │   │   └── CIK0000320193/              # Apple Inc.
        │   │       ├── 10-K_2024-09-28/
        │   │       │   ├── filing.html
        │   │       │   ├── filing_xbrl.xml
        │   │       │   └── metadata.json
        │   │       └── 10-Q_2024-06-29/
        │   │           └── ...
        │   ├── amf/
        │   │   └── MC.PA/                      # LVMH
        │   │       └── annual_2024.pdf
        │   └── yahoo_scrape/
        │       └── 2026-01-26/
        │           └── financials.jsonl
        │
        ├── processed/
        │   ├── metadata/
        │   │   └── filings_metadata.parquet    # All filing metadata
        │   └── financials/
        │       ├── income_statement.parquet
        │       ├── balance_sheet.parquet
        │       └── cash_flow.parquet
        │
        └── mappings/
            ├── ticker_to_cik.json              # US ticker → CIK lookup
            └── ticker_to_lei.json              # Global LEI lookup
```

---

## 6. Implementation Plan

### Phase 1: SEC EDGAR Integration (US Priority)

- [ ] **Step 1.1:** Create `mappings/ticker_cik_resolver.py`
  - Build ticker → CIK lookup table
  - Cache in JSON for fast resolution
  - Handle edge cases (tickers with dots, preferred shares)

- [ ] **Step 1.2:** Implement `fetchers/sec_edgar_fetcher.py`
  ```python
  class SECEdgarFetcher:
      async def get_company_filings(self, cik: str) -> List[FilingMetadata]: ...
      async def download_filing(self, accession_number: str) -> FilingBundle: ...
      async def get_company_facts(self, cik: str) -> Dict[str, Any]: ...
  ```

- [ ] **Step 1.3:** Implement `parsers/xbrl_parser.py`
  - Parse XBRL documents using `python-xbrl`
  - Map US-GAAP concepts to standardized schema
  - Handle fiscal year variations (non-calendar year-ends)

- [ ] **Step 1.4:** Implement `parsers/sec_facts_parser.py`
  - Use SEC pre-parsed facts API (simpler than raw XBRL)
  - Extract time series of financial metrics

### Phase 2: EU ESEF Integration

- [ ] **Step 2.1:** Implement `fetchers/esef_fetcher.py`
  - Query `filings.xbrl.org` API
  - Download iXBRL reports
  - Filter by LEI or company name

- [ ] **Step 2.2:** Implement `parsers/ixbrl_parser.py`
  - Use `arelle` for inline XBRL parsing
  - Map IFRS concepts to standardized schema

- [ ] **Step 2.3:** Create `fetchers/amf_fetcher.py` (France-specific)
  - Scrape AMF BDIF for CAC40 companies
  - Handle PDF downloads
  - Extract structured data where available

### Phase 3: Yahoo Finance Fallback

- [ ] **Step 3.1:** Implement `scrapers/yahoo_financials_scraper.py`
  - Playwright-based scraper for financial tables
  - Parse income statement, balance sheet, cash flow
  - Handle quarterly vs. annual data

- [ ] **Step 3.2:** Build `normalizer.py`
  - Standardize Yahoo format to our schema
  - Convert currencies if needed
  - Validate against SEC data where overlap exists

### Phase 4: Data Processing & Storage

- [ ] **Step 4.1:** Implement `processors/financial_calculator.py`
  - Calculate derived ratios (debt/equity, ROE, margins)
  - Handle missing data gracefully

- [ ] **Step 4.2:** Implement `storage/filings_writer.py`
  - Write metadata to Parquet
  - Write financial statements to Parquet
  - Store raw documents in organized directory structure

- [ ] **Step 4.3:** Create `orchestrator.py`
  - Coordinate all fetchers and parsers
  - Incremental updates (only fetch new filings)
  - Error handling and retry logic

---

## 7. Sample Code: SEC EDGAR Fetcher

```python
# corporate_filings/fetchers/sec_edgar_fetcher.py

import httpx
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class FilingMetadata:
    accession_number: str
    form_type: str
    filing_date: datetime
    period_end_date: datetime
    primary_document: str
    cik: str
    company_name: str


class SECEdgarFetcher:
    """
    Fetcher for SEC EDGAR filings.
    
    API Docs: https://www.sec.gov/edgar/sec-api-documentation
    """
    
    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    
    def __init__(self, user_email: str):
        """
        Initialize with contact email (required by SEC).
        
        Args:
            user_email: Contact email for SEC User-Agent requirement
        """
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": f"CAC40Prediction {user_email}",
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0
        )
        self.request_interval = 0.2  # 5 requests/second max
    
    async def _rate_limited_get(self, url: str) -> httpx.Response:
        """Make rate-limited GET request."""
        await asyncio.sleep(self.request_interval)
        response = await self.client.get(url)
        response.raise_for_status()
        return response
    
    async def get_company_submissions(self, cik: str) -> Dict[str, Any]:
        """
        Get all submissions for a company.
        
        Args:
            cik: SEC Central Index Key (10 digits, zero-padded)
        
        Returns:
            Full submissions JSON including filings history
        """
        # Ensure CIK is 10 digits with leading zeros
        cik_padded = cik.zfill(10)
        url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
        
        logger.info(f"Fetching submissions for CIK {cik_padded}")
        response = await self._rate_limited_get(url)
        return response.json()
    
    async def get_filings(
        self,
        cik: str,
        form_types: List[str] = ["10-K", "10-Q"],
        limit: int = 20
    ) -> List[FilingMetadata]:
        """
        Get filtered list of filings for a company.
        
        Args:
            cik: SEC Central Index Key
            form_types: List of form types to filter
            limit: Maximum number of filings to return
        
        Returns:
            List of FilingMetadata objects
        """
        submissions = await self.get_company_submissions(cik)
        
        company_name = submissions.get("name", "")
        recent = submissions.get("filings", {}).get("recent", {})
        
        filings = []
        form_list = recent.get("form", [])
        
        for i, form in enumerate(form_list):
            if form in form_types and len(filings) < limit:
                try:
                    filing = FilingMetadata(
                        accession_number=recent["accessionNumber"][i].replace("-", ""),
                        form_type=form,
                        filing_date=datetime.strptime(recent["filingDate"][i], "%Y-%m-%d"),
                        period_end_date=datetime.strptime(recent["reportDate"][i], "%Y-%m-%d"),
                        primary_document=recent["primaryDocument"][i],
                        cik=cik,
                        company_name=company_name
                    )
                    filings.append(filing)
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"Failed to parse filing {i}: {e}")
        
        return filings
    
    async def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """
        Get pre-parsed XBRL facts for a company.
        
        This is the easiest way to get historical financial data
        without parsing XBRL files yourself.
        
        Args:
            cik: SEC Central Index Key
        
        Returns:
            Dictionary of all XBRL facts with time series
        """
        cik_padded = cik.zfill(10)
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        
        logger.info(f"Fetching company facts for CIK {cik_padded}")
        response = await self._rate_limited_get(url)
        return response.json()
    
    async def download_filing_document(
        self,
        cik: str,
        accession_number: str,
        document_name: str
    ) -> str:
        """
        Download a specific document from a filing.
        
        Args:
            cik: SEC Central Index Key
            accession_number: Filing accession number (no dashes)
            document_name: Name of document file
        
        Returns:
            Document content as string
        """
        cik_int = str(int(cik))  # Remove leading zeros for URL
        url = f"{self.ARCHIVES_URL}/{cik_int}/{accession_number}/{document_name}"
        
        response = await self._rate_limited_get(url)
        return response.text
    
    async def close(self):
        await self.client.aclose()


# Ticker to CIK resolver
class TickerToCIKResolver:
    """Resolve stock tickers to SEC CIK numbers."""
    
    MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
    
    def __init__(self):
        self.mapping: Dict[str, str] = {}
    
    async def load(self, client: httpx.AsyncClient):
        """Load the ticker to CIK mapping from SEC."""
        response = await client.get(self.MAPPING_URL)
        response.raise_for_status()
        
        data = response.json()
        for entry in data.values():
            ticker = entry.get("ticker", "")
            cik = str(entry.get("cik_str", ""))
            self.mapping[ticker.upper()] = cik
        
        logger.info(f"Loaded {len(self.mapping)} ticker→CIK mappings")
    
    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker symbol."""
        return self.mapping.get(ticker.upper())


# Usage example
async def main():
    fetcher = SECEdgarFetcher(user_email="research@example.com")
    resolver = TickerToCIKResolver()
    
    try:
        # Load ticker mapping
        await resolver.load(fetcher.client)
        
        # Get Apple's CIK
        cik = resolver.get_cik("AAPL")
        print(f"AAPL CIK: {cik}")
        
        # Get recent 10-K and 10-Q filings
        filings = await fetcher.get_filings(cik, form_types=["10-K", "10-Q"], limit=5)
        for f in filings:
            print(f"{f.form_type} | Filed: {f.filing_date.date()} | Period: {f.period_end_date.date()}")
        
        # Get pre-parsed financial facts
        facts = await fetcher.get_company_facts(cik)
        
        # Extract revenue history
        revenue_facts = facts.get("facts", {}).get("us-gaap", {}).get("Revenues", {})
        if revenue_facts:
            units = revenue_facts.get("units", {}).get("USD", [])
            print("\nRevenue History:")
            for item in units[-5:]:  # Last 5 entries
                print(f"  {item.get('fy')}-Q{item.get('fp', 'FY')}: ${item.get('val'):,.0f}")
        
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Sample Code: Yahoo Finance Financials Scraper

```python
# corporate_filings/scrapers/yahoo_financials_scraper.py

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinancialStatement:
    ticker: str
    statement_type: str  # income_statement, balance_sheet, cash_flow
    period_type: str     # annual, quarterly
    data: pd.DataFrame
    fetched_at: datetime


class YahooFinancialsScraper:
    """
    Scrape financial statements from Yahoo Finance.
    
    Use as fallback when direct regulatory filings are unavailable.
    """
    
    URLS = {
        "income_statement": "https://finance.yahoo.com/quote/{ticker}/financials",
        "balance_sheet": "https://finance.yahoo.com/quote/{ticker}/balance-sheet",
        "cash_flow": "https://finance.yahoo.com/quote/{ticker}/cash-flow",
    }
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    async def scrape_all_statements(
        self,
        ticker: str,
        period_type: str = "annual"  # "annual" or "quarterly"
    ) -> List[FinancialStatement]:
        """
        Scrape all three financial statements for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL", "MC.PA")
            period_type: "annual" or "quarterly"
        
        Returns:
            List of FinancialStatement objects
        """
        statements = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            
            try:
                for statement_type, url_template in self.URLS.items():
                    url = url_template.format(ticker=ticker)
                    logger.info(f"Scraping {statement_type} for {ticker}")
                    
                    page = await context.new_page()
                    await page.goto(url, wait_until="networkidle")
                    
                    # Handle cookie consent
                    try:
                        await page.click('button[name="agree"]', timeout=3000)
                        await asyncio.sleep(1)
                    except:
                        pass
                    
                    # Switch to quarterly if needed
                    if period_type == "quarterly":
                        try:
                            await page.click('button:has-text("Quarterly")', timeout=5000)
                            await asyncio.sleep(2)
                        except:
                            logger.warning(f"Could not switch to quarterly for {ticker}")
                    
                    # Get table HTML
                    html = await page.content()
                    df = self._parse_financial_table(html, statement_type)
                    
                    if df is not None:
                        statements.append(FinancialStatement(
                            ticker=ticker,
                            statement_type=statement_type,
                            period_type=period_type,
                            data=df,
                            fetched_at=datetime.now(ZoneInfo("UTC"))
                        ))
                    
                    await page.close()
                    await asyncio.sleep(1)  # Rate limiting
                    
            finally:
                await browser.close()
        
        return statements
    
    def _parse_financial_table(
        self,
        html: str,
        statement_type: str
    ) -> Optional[pd.DataFrame]:
        """Parse financial table from page HTML."""
        soup = BeautifulSoup(html, "lxml")
        
        # Find the main financial table
        table = soup.select_one('div[data-testid="fin-table"]') or \
                soup.select_one('div.tableContainer')
        
        if not table:
            logger.warning(f"Could not find {statement_type} table")
            return None
        
        rows = table.select('div[data-testid="fin-row"]') or table.select('div.row')
        
        if not rows:
            return None
        
        # Parse header (dates)
        header_row = rows[0]
        headers = ["Item"]
        for cell in header_row.select('div[data-testid="fin-col"]')[1:]:
            text = cell.get_text(strip=True)
            if text:
                headers.append(text)
        
        # Parse data rows
        data = []
        for row in rows[1:]:
            cells = row.select('div[data-testid="fin-col"]')
            if not cells:
                continue
            
            row_data = []
            for cell in cells:
                text = cell.get_text(strip=True)
                # Clean numeric values
                text = text.replace(",", "").replace("(", "-").replace(")", "")
                row_data.append(text)
            
            if row_data and row_data[0]:
                data.append(row_data)
        
        if not data:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure column count matches
        if len(df.columns) <= len(headers):
            df.columns = headers[:len(df.columns)]
        else:
            df = df.iloc[:, :len(headers)]
            df.columns = headers
        
        df = df.set_index("Item")
        
        # Convert to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def statements_to_dict(
        self,
        statements: List[FinancialStatement]
    ) -> Dict[str, pd.DataFrame]:
        """Convert list of statements to dictionary keyed by type."""
        return {s.statement_type: s.data for s in statements}


# Usage example
async def main():
    scraper = YahooFinancialsScraper(headless=True)
    
    statements = await scraper.scrape_all_statements("AAPL", period_type="annual")
    
    for statement in statements:
        print(f"\n=== {statement.statement_type.upper()} ===")
        print(statement.data.head(10))


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Configuration Example

```yaml
# config/corporate_filings.yaml

sources:
  us:
    sec_edgar:
      enabled: true
      user_email: "research@example.com"
      rate_limit_rps: 5
      form_types:
        - 10-K
        - 10-Q
        - 8-K
      lookback_years: 5
  
  eu:
    esef:
      enabled: true
      filings_xbrl_org: true
    amf:
      enabled: true
      target_companies:
        - MC.PA    # LVMH
        - OR.PA    # L'Oreal
        - TTE.PA   # TotalEnergies
        - SAN.PA   # Sanofi
  
  fallback:
    yahoo_finance:
      enabled: true
      priority: 3  # Use only when direct sources fail
      headless: true

processing:
  xbrl_taxonomy: us-gaap  # us-gaap or ifrs-full
  currency_normalize: USD
  calculate_ratios: true

storage:
  raw_documents: true     # Keep original filings
  processed_format: parquet
  partition_by:
    - ticker
    - fiscal_year

scheduling:
  full_refresh: "0 6 * * 0"      # Weekly on Sunday 6 AM
  incremental: "0 8 * * 1-5"     # Daily on weekdays 8 AM
```

---

## 10. Dependencies (requirements.txt addition)

```txt
# Corporate Filings Module
sec-edgar-downloader>=5.0.2
edgartools>=1.0.0
python-xbrl>=1.1.1
arelle-release>=2.20.0
pdfplumber>=0.10.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
playwright>=1.41.0
httpx>=0.26.0
```

**Special Install: Arelle (for advanced XBRL/iXBRL parsing)**
```bash
pip install arelle-release
# or from source for latest features
pip install git+https://github.com/Arelle/Arelle.git
```

---

## 11. Testing Checklist

- [ ] Unit test: SEC fetcher returns valid metadata
- [ ] Unit test: CIK resolver handles edge cases (BRK.A vs BRK-A)
- [ ] Unit test: XBRL parser extracts all target concepts
- [ ] Integration test: Full pipeline for AAPL (10-K + 10-Q)
- [ ] Integration test: Full pipeline for MC.PA (European)
- [ ] Yahoo scraper test: Gracefully handles page structure changes
- [ ] Validation test: Compare SEC data vs Yahoo for consistency
- [ ] Rate limit test: Verify SEC 10 req/sec limit respected

---

*Authored for CAC40_Prediction ML Pipeline — Data Ingestion Layer*
