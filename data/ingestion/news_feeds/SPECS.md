# News Feeds Ingestion Module — Technical Specifications

> **Module:** `data/ingestion/news_feeds/`  
> **Data Type:** Financial News Articles, Headlines, Sentiment Signals  
> **Resolution:** Real-time to Daily aggregation  
> **Last Updated:** January 2026

---

## 1. Objective

Build a **multi-source news aggregation pipeline** to fetch financial news for sentiment analysis and event detection. The system must:

- Aggregate news from **APIs** (structured) and **web scrapers** (unstructured).
- **Normalize timestamps to UTC** for consistent ML feature alignment.
- Extract and store: headlines, summaries, full text (when available), tickers mentioned, publication source.
- Support **historical backfill** and **real-time streaming** modes.
- Enable downstream **NLP/LLM sentiment scoring** integration.

---

## 2. Recommended Python Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `feedparser` | RSS/Atom feed parsing | `pip install feedparser` |
| `beautifulsoup4` | HTML parsing for scrapers | `pip install beautifulsoup4` |
| `lxml` | Fast HTML/XML parser | `pip install lxml` |
| `scrapy` | Full-featured scraping framework | `pip install scrapy` |
| `playwright` | JavaScript-rendered page scraping | `pip install playwright` |
| `newspaper3k` | Article extraction & NLP | `pip install newspaper3k` |
| `httpx` | Async HTTP client | `pip install httpx` |
| `aiohttp` | Async HTTP for high-throughput | `pip install aiohttp` |
| `dateparser` | Robust date/time parsing | `pip install dateparser` |
| `pytz` / `zoneinfo` | Timezone handling | Built-in (Python 3.9+) |
| `langdetect` | Language detection | `pip install langdetect` |
| `tiktoken` | Token counting for LLM prep | `pip install tiktoken` |

---

## 3. Data Sources

### 3.1 API-Based Sources (Structured — Recommended)

| Source | Type | Rate Limits | Coverage | Pricing |
|--------|------|-------------|----------|---------|
| **Financial Modeling Prep (FMP)** | REST API | 250-750 calls/day (free) | US & Global | Free → $19/mo |
| **Alpha Vantage News** | REST API | 25 calls/day (free) | US Focus | Free → $49.99/mo |
| **Polygon.io News** | REST API | Based on plan | US equities | $29-$199/mo |
| **Benzinga** | REST API | Plan-based | US, premium quality | Custom pricing |
| **NewsAPI.org** | REST API | 100 calls/day (free) | General news | Free → $449/mo |
| **GNews API** | REST API | 100 calls/day (free) | Google News aggregator | Free → €64/mo |

### 3.2 RSS Feeds (Free, Reliable)

| Source | Feed URL | Notes |
|--------|----------|-------|
| **Yahoo Finance** | `https://finance.yahoo.com/rss/` | Company-specific feeds available |
| **Reuters Business** | `https://www.reutersagency.com/feed/` | May require parsing main site |
| **CNBC** | `https://www.cnbc.com/id/100003114/device/rss/rss.html` | US market focus |
| **Investing.com** | `https://www.investing.com/rss/` | Multiple category feeds |
| **SeekingAlpha** | `https://seekingalpha.com/feed.xml` | Analysis-heavy content |
| **MarketWatch** | `https://www.marketwatch.com/rss/` | Multiple category feeds |

### 3.3 Scraping Targets (Fallback/Supplementary)

| Source | URL Pattern | Difficulty | Notes |
|--------|-------------|------------|-------|
| **Yahoo Finance News** | `https://finance.yahoo.com/quote/{TICKER}/news` | Medium | JS-rendered, use Playwright |
| **Investing.com** | `https://www.investing.com/equities/{slug}-news` | Medium | Rate limit aggressive |
| **Google Finance** | `https://www.google.com/finance/quote/{TICKER}:{EXCHANGE}` | Hard | Heavy bot protection |
| **Boursorama (FR)** | `https://www.boursorama.com/cours/{TICKER}/actualites` | Easy | Good for CAC40 |
| **Les Echos (FR)** | `https://www.lesechos.fr/` | Medium | French business news |

---

## 4. Data Schema

```python
# Expected DataFrame/Document Schema
schema = {
    "article_id": "string",           # Unique hash (SHA256 of URL)
    "published_at": "datetime64[ns, UTC]",  # Publication time (UTC normalized)
    "fetched_at": "datetime64[ns, UTC]",    # Ingestion timestamp
    "source": "string",               # e.g., "fmp", "yahoo_rss", "investing_scraper"
    "source_url": "string",           # Original article URL
    
    # Content
    "headline": "string",             # Article title
    "summary": "string",              # First paragraph or API summary
    "full_text": "string",            # Full article body (nullable)
    "language": "string",             # ISO 639-1 code (en, fr, de)
    
    # Metadata
    "tickers_mentioned": "list[string]",  # Extracted ticker symbols
    "categories": "list[string]",         # e.g., ["earnings", "merger", "macro"]
    "author": "string",                   # Author name (nullable)
    
    # For LLM processing
    "token_count": "int64",           # Estimated tokens (tiktoken)
    "sentiment_raw": "float64",       # Pre-computed if API provides
}
```

---

## 5. Strategy

### 5.1 Timestamp Normalization (Critical)

```python
from datetime import datetime
from zoneinfo import ZoneInfo
import dateparser

def normalize_to_utc(timestamp_str: str, source_tz: str = None) -> datetime:
    """
    Convert any timestamp string to UTC datetime.
    
    Args:
        timestamp_str: Raw timestamp from source
        source_tz: Known timezone of source (e.g., "America/New_York")
    
    Returns:
        datetime in UTC
    """
    # Parse with dateparser (handles many formats)
    parsed = dateparser.parse(
        timestamp_str,
        settings={
            "TIMEZONE": source_tz or "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TO_TIMEZONE": "UTC"
        }
    )
    
    if parsed is None:
        raise ValueError(f"Could not parse timestamp: {timestamp_str}")
    
    # Ensure UTC
    return parsed.astimezone(ZoneInfo("UTC"))


# Source timezone mappings
SOURCE_TIMEZONES = {
    "yahoo_finance": "America/New_York",
    "reuters": "UTC",
    "cnbc": "America/New_York",
    "investing_com": "UTC",
    "boursorama": "Europe/Paris",
    "les_echos": "Europe/Paris",
    "fmp_api": "UTC",
}
```

### 5.2 Rate Limit Handling

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    requests_per_minute: int
    tokens: float = None
    last_refill: datetime = None
    
    def __post_init__(self):
        self.tokens = float(self.requests_per_minute)
        self.last_refill = datetime.now()
    
    async def acquire(self):
        """Wait until a request token is available."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds() / 60
        self.tokens = min(
            self.requests_per_minute,
            self.tokens + elapsed * self.requests_per_minute
        )
        self.last_refill = now
        
        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.requests_per_minute * 60
            await asyncio.sleep(wait_time)
            self.tokens = 1
        
        self.tokens -= 1


# Rate limits by source
RATE_LIMITS = {
    "fmp_api": RateLimiter(requests_per_minute=5),  # Conservative for free tier
    "newsapi": RateLimiter(requests_per_minute=2),
    "yahoo_scraper": RateLimiter(requests_per_minute=10),
    "investing_scraper": RateLimiter(requests_per_minute=5),
}
```

### 5.3 Pagination Strategies

#### API Pagination (Cursor-based)
```python
async def fetch_all_pages(
    client: httpx.AsyncClient,
    base_url: str,
    params: dict,
    page_key: str = "page",
    limit_key: str = "limit",
    limit: int = 100
) -> list:
    """Generic cursor/offset pagination handler."""
    all_results = []
    page = 1
    
    while True:
        params[page_key] = page
        params[limit_key] = limit
        
        response = await client.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", data.get("articles", []))
        if not results:
            break
        
        all_results.extend(results)
        page += 1
        
        # Rate limit respect
        await asyncio.sleep(0.5)
    
    return all_results
```

#### RSS Feed Pagination (Time-based)
```python
def fetch_rss_incremental(feed_url: str, last_fetch: datetime) -> list:
    """Fetch only new items since last fetch."""
    import feedparser
    
    feed = feedparser.parse(feed_url)
    new_items = []
    
    for entry in feed.entries:
        published = dateparser.parse(entry.get("published", ""))
        if published and published > last_fetch:
            new_items.append(entry)
    
    return new_items
```

### 5.4 Data Storage Strategy

```text
Storage: PARQUET (structured) + JSONL (raw backup)

Directory Structure:
data/
└── ingestion/
    └── news_feeds/
        ├── raw/                          # Original responses
        │   ├── fmp/
        │   │   └── 2026-01-26.jsonl
        │   ├── yahoo_rss/
        │   │   └── 2026-01-26.jsonl
        │   └── scraped/
        │       └── 2026-01-26.jsonl
        │
        ├── processed/                    # Cleaned, normalized
        │   ├── by_date/
        │   │   └── date=2026-01-26/
        │   │       └── news.parquet
        │   └── by_ticker/
        │       └── ticker=AAPL/
        │           └── news.parquet
        │
        └── metadata/
            └── fetch_state.json          # Last fetch timestamps
```

**Why Dual Storage:**
- **JSONL**: Preserves raw API responses for debugging/reprocessing
- **Parquet**: Optimized for ML feature extraction queries

---

## 6. Implementation Plan

### Phase 1: API Fetchers (Priority)

- [ ] **Step 1.1:** Create `base_news_fetcher.py` abstract class
  ```python
  class NewsSource(ABC):
      @abstractmethod
      async def fetch_news(self, tickers: List[str], start: datetime, end: datetime) -> List[Article]: ...
      @abstractmethod
      def get_rate_limiter(self) -> RateLimiter: ...
  ```

- [ ] **Step 1.2:** Implement `fmp_fetcher.py` (Financial Modeling Prep)
  - Endpoint: `https://financialmodelingprep.com/api/v3/stock_news`
  - Params: `tickers`, `limit`, `page`
  - Free tier: 250 calls/day — prioritize for production

- [ ] **Step 1.3:** Implement `newsapi_fetcher.py`
  - Endpoint: `https://newsapi.org/v2/everything`
  - Good for general market news (not ticker-specific)

- [ ] **Step 1.4:** Implement `rss_aggregator.py`
  - Parse multiple RSS feeds concurrently
  - Deduplicate by URL hash

### Phase 2: Web Scrapers (Supplementary)

- [ ] **Step 2.1:** Create `scrapers/base_scraper.py`
  - Playwright-based for JS-rendered pages
  - BeautifulSoup for static HTML

- [ ] **Step 2.2:** Implement `scrapers/yahoo_finance.py`
  ```python
  class YahooFinanceNewsScraper:
      """Scrape Yahoo Finance news for specific tickers."""
      
      async def scrape(self, ticker: str, max_articles: int = 20):
          url = f"https://finance.yahoo.com/quote/{ticker}/news"
          # Use Playwright for JS rendering
          async with async_playwright() as p:
              browser = await p.chromium.launch()
              page = await browser.new_page()
              await page.goto(url)
              await page.wait_for_selector("h3")  # Wait for headlines
              # Extract articles...
  ```

- [ ] **Step 2.3:** Implement `scrapers/investing_com.py`
  - Handle aggressive rate limiting
  - Rotate User-Agents

- [ ] **Step 2.4:** Implement `scrapers/boursorama.py` (French CAC40 coverage)

### Phase 3: Content Processing

- [ ] **Step 3.1:** Build `processors/text_extractor.py`
  - Use `newspaper3k` for article body extraction
  - Fallback to BeautifulSoup patterns

- [ ] **Step 3.2:** Build `processors/ticker_extractor.py`
  - Regex for ticker patterns: `$AAPL`, `(NASDAQ: AAPL)`
  - NER for company name → ticker mapping
  - Maintain company name → ticker lookup table

- [ ] **Step 3.3:** Build `processors/timestamp_normalizer.py`
  - Central UTC normalization logic
  - Handle timezone-naive strings

- [ ] **Step 3.4:** Build `processors/deduplicator.py`
  - Hash-based (URL hash) primary dedup
  - Fuzzy matching for syndicated content (same article, different sources)

### Phase 4: Storage & Orchestration

- [ ] **Step 4.1:** Implement `storage/news_writer.py`
  - Write to JSONL (raw) and Parquet (processed)
  - Partition by date and/or ticker

- [ ] **Step 4.2:** Implement `storage/state_manager.py`
  - Track last fetch timestamp per source
  - Enable incremental fetching

- [ ] **Step 4.3:** Create `orchestrator.py`
  - Coordinate all sources
  - Parallel execution with asyncio
  - Error handling and retry logic

---

## 7. Sample Code: FMP News Fetcher

```python
# news_feeds/fetchers/fmp_fetcher.py

import httpx
import asyncio
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import hashlib
from zoneinfo import ZoneInfo
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    article_id: str
    published_at: datetime
    fetched_at: datetime
    source: str
    source_url: str
    headline: str
    summary: str
    full_text: Optional[str]
    tickers_mentioned: List[str]
    language: str = "en"


class FMPNewsFetcher:
    """
    Financial Modeling Prep News API Fetcher.
    
    API Docs: https://site.financialmodelingprep.com/developer/docs#stock-news
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3/stock_news"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY required")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch_news(
        self,
        tickers: Optional[List[str]] = None,
        limit: int = 100,
        page: int = 0
    ) -> List[NewsArticle]:
        """
        Fetch stock news from FMP API.
        
        Args:
            tickers: List of stock symbols (comma-separated in API)
            limit: Number of articles per page (max 100)
            page: Page number for pagination
        
        Returns:
            List of NewsArticle objects
        """
        params = {
            "apikey": self.api_key,
            "limit": min(limit, 100),
            "page": page
        }
        
        if tickers:
            params["tickers"] = ",".join(tickers)
        
        logger.info(f"Fetching FMP news: tickers={tickers}, page={page}")
        
        response = await self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        articles = []
        for item in response.json():
            try:
                article = self._parse_article(item)
                articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
        
        return articles
    
    def _parse_article(self, raw: dict) -> NewsArticle:
        """Parse raw API response into NewsArticle."""
        url = raw.get("url", "")
        article_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        # FMP returns ISO format timestamps in UTC
        published_str = raw.get("publishedDate", "")
        published_at = datetime.fromisoformat(
            published_str.replace("Z", "+00:00")
        ).astimezone(ZoneInfo("UTC"))
        
        return NewsArticle(
            article_id=article_id,
            published_at=published_at,
            fetched_at=datetime.now(ZoneInfo("UTC")),
            source="fmp",
            source_url=url,
            headline=raw.get("title", ""),
            summary=raw.get("text", "")[:500],  # Truncate summary
            full_text=raw.get("text"),
            tickers_mentioned=[raw.get("symbol", "")] if raw.get("symbol") else [],
        )
    
    async def fetch_all_for_tickers(
        self,
        tickers: List[str],
        max_pages: int = 5,
        delay_seconds: float = 1.0
    ) -> List[NewsArticle]:
        """Fetch all available news for given tickers with pagination."""
        all_articles = []
        
        for page in range(max_pages):
            articles = await self.fetch_news(tickers=tickers, page=page)
            if not articles:
                break
            all_articles.extend(articles)
            logger.info(f"Page {page}: fetched {len(articles)} articles")
            await asyncio.sleep(delay_seconds)  # Rate limit
        
        return all_articles
    
    async def close(self):
        await self.client.aclose()


# Usage example
async def main():
    fetcher = FMPNewsFetcher()
    try:
        articles = await fetcher.fetch_all_for_tickers(
            tickers=["AAPL", "MSFT", "MC.PA"],
            max_pages=3
        )
        for article in articles[:5]:
            print(f"[{article.published_at}] {article.headline}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Sample Code: Yahoo Finance Scraper

```python
# news_feeds/scrapers/yahoo_finance.py

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import asyncio
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ScrapedArticle:
    article_id: str
    source_url: str
    headline: str
    summary: str
    published_at: datetime
    fetched_at: datetime
    ticker: str


class YahooFinanceNewsScraper:
    """
    Playwright-based scraper for Yahoo Finance news pages.
    
    Note: Use responsibly. Respect robots.txt and rate limits.
    """
    
    BASE_URL = "https://finance.yahoo.com/quote/{ticker}/news"
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    async def scrape_ticker_news(
        self,
        ticker: str,
        max_articles: int = 20,
        scroll_count: int = 3
    ) -> List[ScrapedArticle]:
        """
        Scrape news articles for a specific ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            max_articles: Maximum articles to return
            scroll_count: Number of scroll actions (for lazy loading)
        
        Returns:
            List of ScrapedArticle objects
        """
        url = self.BASE_URL.format(ticker=ticker)
        articles = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            try:
                logger.info(f"Navigating to {url}")
                await page.goto(url, wait_until="networkidle")
                
                # Handle cookie consent if present
                try:
                    await page.click('button[name="agree"]', timeout=3000)
                except:
                    pass
                
                # Scroll to load more content
                for _ in range(scroll_count):
                    await page.keyboard.press("End")
                    await asyncio.sleep(1)
                
                # Get page content
                html = await page.content()
                soup = BeautifulSoup(html, "lxml")
                
                # Extract article links (adjust selectors as needed)
                news_items = soup.select("li.stream-item, div[data-testid='news-item']")[:max_articles]
                
                for item in news_items:
                    try:
                        article = self._parse_news_item(item, ticker)
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse item: {e}")
                
            finally:
                await browser.close()
        
        logger.info(f"Scraped {len(articles)} articles for {ticker}")
        return articles
    
    def _parse_news_item(self, item, ticker: str) -> ScrapedArticle:
        """Parse a single news item from the page."""
        # Find headline link
        link_elem = item.select_one("a[href*='/news/']") or item.select_one("h3 a")
        if not link_elem:
            return None
        
        headline = link_elem.get_text(strip=True)
        href = link_elem.get("href", "")
        
        # Build full URL
        if href.startswith("/"):
            source_url = f"https://finance.yahoo.com{href}"
        else:
            source_url = href
        
        # Extract summary if available
        summary_elem = item.select_one("p")
        summary = summary_elem.get_text(strip=True) if summary_elem else ""
        
        # Generate unique ID
        article_id = hashlib.sha256(source_url.encode()).hexdigest()[:16]
        
        return ScrapedArticle(
            article_id=article_id,
            source_url=source_url,
            headline=headline,
            summary=summary,
            published_at=datetime.now(ZoneInfo("UTC")),  # Would need deeper parsing
            fetched_at=datetime.now(ZoneInfo("UTC")),
            ticker=ticker
        )


# Usage
async def main():
    scraper = YahooFinanceNewsScraper(headless=True)
    articles = await scraper.scrape_ticker_news("AAPL", max_articles=10)
    
    for article in articles:
        print(f"- {article.headline}")
        print(f"  URL: {article.source_url}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Configuration Example

```yaml
# config/news_feeds.yaml

sources:
  apis:
    fmp:
      enabled: true
      priority: 1
      api_key_env: FMP_API_KEY
      rate_limit_rpm: 5
      
    newsapi:
      enabled: true
      priority: 2
      api_key_env: NEWSAPI_KEY
      rate_limit_rpm: 2
  
  rss:
    - name: yahoo_finance
      url: "https://finance.yahoo.com/rss/headline"
      enabled: true
    - name: cnbc_markets
      url: "https://www.cnbc.com/id/100003114/device/rss/rss.html"
      enabled: true
    - name: reuters_business
      url: "https://www.reutersagency.com/feed/?best-regions=north-america&post_type=best"
      enabled: true
  
  scrapers:
    yahoo_finance:
      enabled: true
      rate_limit_rpm: 10
      headless: true
    investing_com:
      enabled: false  # Aggressive blocking
      rate_limit_rpm: 3
    boursorama:
      enabled: true
      rate_limit_rpm: 10

processing:
  timestamp_normalization: UTC
  deduplication:
    method: url_hash
    fuzzy_threshold: 0.85
  languages:
    - en
    - fr

storage:
  raw_format: jsonl
  processed_format: parquet
  partition_by:
    - date
  retention_days: 365

scheduling:
  api_fetch_interval: "0 */4 * * *"  # Every 4 hours
  rss_fetch_interval: "*/30 * * * *"  # Every 30 minutes
  scrape_interval: "0 8,12,18 * * *"  # 3 times daily
```

---

## 10. Dependencies (requirements.txt addition)

```txt
# News Feeds Module
feedparser>=6.0.10
beautifulsoup4>=4.12.0
lxml>=5.1.0
scrapy>=2.11.0
playwright>=1.41.0
newspaper3k>=0.2.8
httpx>=0.26.0
aiohttp>=3.9.0
dateparser>=1.2.0
langdetect>=1.0.9
tiktoken>=0.5.0
tenacity>=8.2.0
```

**Post-install for Playwright:**
```bash
playwright install chromium
```

---

## 11. Testing Checklist

- [ ] Unit test: FMP fetcher returns valid schema
- [ ] Unit test: Timestamp normalization handles all common formats
- [ ] Unit test: Deduplication correctly identifies duplicates
- [ ] Integration test: End-to-end pipeline for 1 ticker
- [ ] Scraper test: Yahoo Finance scraper handles page changes gracefully
- [ ] Rate limit test: Verify backoff behavior
- [ ] Timezone test: All outputs are UTC regardless of source

---

## 12. Anti-Bot Considerations

```text
╔══════════════════════════════════════════════════════════════════╗
║                    ETHICAL SCRAPING GUIDELINES                   ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. Always check robots.txt before scraping                       ║
║ 2. Implement reasonable delays (1-5 seconds between requests)    ║
║ 3. Use realistic User-Agent strings                              ║
║ 4. Rotate IP addresses for high-volume scraping (proxies)        ║
║ 5. Respect rate limit headers (Retry-After, X-RateLimit-*)       ║
║ 6. Cache responses to minimize redundant requests                ║
║ 7. Prefer official APIs when available                           ║
║ 8. Stop immediately if you receive a legal notice                ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*Authored for CAC40_Prediction ML Pipeline — Data Ingestion Layer*
