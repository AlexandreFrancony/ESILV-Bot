"""
ESILV Smart Assistant - Web Scraper
====================================

Web scraper for ESILV website.
- Handles errors gracefully
- Retry logic for failed requests
- Rate limiting respect
- Logging & progress tracking
- Saves to data/esilv_documents.txt

Date: 2025-12-16
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(f"logs/scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.stream = open(1, 'w', encoding='utf-8', buffering=1)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


logger = setup_logger("ESILVScraper")


# ============================================================================
# CONFIGURATION
# ============================================================================

# ESILV URLs to scrape (from config)
ESILV_URLS = [
    "https://www.esilv.fr/formations/",
    "https://www.esilv.fr/formations/cycle-ingenieur/",
    "https://www.esilv.fr/formations/cycle-ingenieur/majeures/",
    "https://www.esilv.fr/admissions/",
    "https://www.esilv.fr/en/programs/",
]

# Output file
OUTPUT_FILE = "data/esilv_documents.txt"

# HTTP Request configuration
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5
RATE_LIMIT_DELAY = 1  # seconds between requests

# User-Agent to avoid blocking
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# HTML elements to scrape
CONTENT_SELECTORS = {
    "main_content": ["main", ".content", ".page-content", "article", ".entry-content"],
    "headings": ["h1", "h2", "h3"],
    "paragraphs": ["p"],
    "lists": ["ul", "ol"],
    "divs": ["div.text", "div.description"],
}

# Elements to exclude (noise)
EXCLUDE_SELECTORS = [
    "nav",
    ".navigation",
    ".sidebar",
    ".comments",
    ".related-posts",
    "footer",
    "script",
    "style",
    "[role='complementary']",
    ".widget",
]


# ============================================================================
# SCRAPER CLASS
# ============================================================================

class ESILVScraper:
    """Production-ready scraper for ESILV website"""
    
    def __init__(self, output_file: str = OUTPUT_FILE):
        self.output_file = output_file
        self.session = self._create_session()
        self.scraped_urls = set()
        self.failed_urls = []
        self.total_documents = 0
        self.logger = logger
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry strategy.
        
        Handles:
        - Connection errors
        - Timeouts
        - HTTP 429 (Too Many Requests)
        - HTTP 500+ (Server errors)
        """
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers
        session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        
        return session
    
    def scrape(self, urls: List[str] = None) -> bool:
        """
        Main scraping method.
        
        Args:
            urls: List of URLs to scrape. If None, uses ESILV_URLS
        
        Returns:
            True if successful, False if all failed
        """
        if urls is None:
            urls = ESILV_URLS
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 Starting ESILV Web Scraping")
        self.logger.info(f"📍 URLs to scrape: {len(urls)}")
        self.logger.info(f"💾 Output file: {self.output_file}")
        self.logger.info("=" * 80)
        
        # Clear output file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"ESILV Smart Assistant - Scraped Documents\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
        
        # Scrape each URL
        for i, url in enumerate(tqdm(urls, desc="Scraping URLs"), 1):
            self.logger.info(f"\n[{i}/{len(urls)}] Scraping: {url}")
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            
            try:
                self._scrape_url(url)
            except Exception as e:
                self.logger.error(f"❌ Failed to scrape {url}: {e}")
                self.failed_urls.append((url, str(e)))
        
        # Log summary
        self._log_summary(urls)
        
        return len(self.failed_urls) < len(urls)
    
    def _scrape_url(self, url: str) -> None:
        """
        Scrape a single URL and extract content.
        
        Args:
            url: URL to scrape
        
        Raises:
            Various exceptions from requests/BeautifulSoup
        """
        # Fetch page
        self.logger.debug(f"Fetching: {url}")
        response = self._fetch_page(url)
        
        if response is None:
            raise RuntimeError(f"Failed to fetch {url}")
        
        self.logger.debug(f"Status code: {response.status_code}")
        
        # Parse HTML
        soup = BeautifulSoup(response.content, "lxml")
        
        # Clean HTML (remove noise)
        self._clean_html(soup)
        
        # Extract text
        text = self._extract_text(soup)
        
        if not text.strip():
            self.logger.warning(f"⚠️  No content extracted from {url}")
            return
        
        # Save to file
        self._save_to_file(url, text)
        
        self.scraped_urls.add(url)
        self.total_documents += 1
        
        self.logger.info(f"✅ Successfully scraped: {url}")
        self.logger.debug(f"   Content length: {len(text)} chars")
    
    def _fetch_page(self, url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
        """
        Fetch a single page with timeout and error handling.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
        
        Returns:
            Response object or None if failed
        """
        try:
            self.logger.debug(f"Making request to: {url}")
            response = self.session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()  # Raise exception for bad status
            return response
            
        except requests.exceptions.Timeout:
            self.logger.error(f"⏱️  Timeout fetching {url}")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"🔌 Connection error fetching {url}")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"❌ HTTP error {e.response.status_code} for {url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Request error: {e}")
            return None
    
    def _clean_html(self, soup: BeautifulSoup) -> None:
        """
        Remove noise elements from HTML (navigation, sidebar, etc).
        
        Modifies soup in-place.
        """
        for selector in EXCLUDE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract main content text from HTML.
        
        Tries multiple selectors to find main content.
        """
        text_parts = []
        
        # Try main content selectors
        for selector in CONTENT_SELECTORS["main_content"]:
            elements = soup.select(selector)
            if elements:
                self.logger.debug(f"Found content with selector: {selector}")
                for elem in elements:
                    text_parts.append(elem.get_text(separator="\n", strip=True))
                break
        
        # Fallback: extract all text
        if not text_parts:
            self.logger.debug("Using fallback: extracting all text")
            text_parts.append(soup.body.get_text(separator="\n", strip=True) if soup.body else "")
        
        # Join and clean text
        text = "\n\n".join(text_parts)
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        - Remove extra whitespace
        - Remove special characters
        - Normalize line breaks
        """
        # Remove multiple blank lines
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Join back
        text = "\n".join(lines)
        
        # Remove multiple spaces
        text = " ".join(text.split())
        
        return text
    
    def _save_to_file(self, url: str, text: str) -> None:
        """
        Append content to output file.
        
        Args:
            url: Source URL
            text: Extracted text
        """
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"SOURCE: {url}\n")
            f.write(f"SCRAPED: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            f.write(text)
            f.write("\n\n")
    
    def _log_summary(self, urls: List[str]) -> None:
        """Log scraping summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 SCRAPING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Successfully scraped: {len(self.scraped_urls)}/{len(urls)} URLs")
        self.logger.info(f"❌ Failed URLs: {len(self.failed_urls)}")
        self.logger.info(f"📄 Total documents: {self.total_documents}")
        
        if self.failed_urls:
            self.logger.info("\n⚠️  Failed URLs:")
            for url, error in self.failed_urls:
                self.logger.info(f"   - {url}: {error}")
        
        # File size
        if os.path.exists(self.output_file):
            size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
            self.logger.info(f"\n💾 Output file size: {size_mb:.2f} MB")
            self.logger.info(f"📁 Output file: {os.path.abspath(self.output_file)}")
        
        self.logger.info("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    # Create scraper
    scraper = ESILVScraper(output_file=OUTPUT_FILE)
    
    # Run scraping
    success = scraper.scrape()
    
    if success:
        logger.info("✅ Scraping completed successfully!")
        return 0
    else:
        logger.warning("⚠️  Scraping completed with some failures")
        return 1


if __name__ == "__main__":
    exit(main())
