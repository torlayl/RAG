# Web Content Crawler
# This script crawls websites using Selenium WebDriver to extract text content systematically.
# Features:
# - Configurable crawl depth and maximum page limits
# - Internal link detection to stay within the same domain
# - Exclusion of non-textual content and specific URL patterns
# - Headless browser operation using Chrome WebDriver
# - Structured JSON output with crawl metadata
# Dependencies:
# - selenium: For browser automation
# - beautifulsoup4: For HTML parsing
# - webdriver_manager: For ChromeDriver management
# Usage:
#     python 1_Crawling_raw_json.py https://example.com --depth 2 --max-pages 20 --timeout 3
# Arguments:
#     url:        Starting URL to crawl
#     --depth:    How many links deep to crawl (default: 2)
#     --max-pages: Maximum number of pages to crawl (default: 20)
#     --timeout:  Seconds to wait between page loads (default: 20)
# Results are saved to the 'raw_results_json' directory as timestamped JSON files.


import argparse
import json
import time
import urllib.parse
from datetime import datetime
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import re


def extract_page_text_content(driver, url):
    """Extract detailed text content from the current page using BeautifulSoup"""
    try:
        title = driver.title
        
        # Get the current page source and parse with BeautifulSoup
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title separately
        if not title and soup.title:
            title = soup.title.string.strip()
            
        # Create a single text content string
        text_parts = []
        
        # Add title to the text content if available
        if title:
            text_parts.append(f"Title: {title}\n")
        
        # Extract headings (h1-h6)
        for heading_level in range(1, 7):
            for heading in soup.find_all(f'h{heading_level}'):
                heading_text = heading.get_text(strip=True)
                if heading_text:
                    text_parts.append(heading_text)
        
        # Extract paragraphs
        for paragraph in soup.find_all('p'):
            paragraph_text = paragraph.get_text(strip=True)
            if paragraph_text:
                text_parts.append(paragraph_text)
        
        # Extract lists (ul, ol)
        for list_elem in soup.find_all(['ul', 'ol']):
            list_items = []
            for li in list_elem.find_all('li'):
                item_text = li.get_text(strip=True)
                if item_text:
                    list_items.append(f"• {item_text}")
            
            if list_items:
                text_parts.extend(list_items)
        
        # Extract table text
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                row_texts = []
                for cell in row.find_all(['td', 'th']):
                    cell_text = cell.get_text(strip=True)
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    text_parts.append(" | ".join(row_texts))
        
        # Get text from other common content containers
        # for container in soup.select('div, section, article, main, aside'):
        #     # Only process direct text nodes of this container (not children already processed)
        #     if container.contents:
        #         for content in container.contents:
        #             if content.string and content.string.strip():
        #                 text = content.string.strip()
        #                 if text and text not in text_parts:
        #                     text_parts.append(text)
        
        # Combine all text parts with newlines for separation
        combined_text = "\n\n".join(text_parts)
        
        # Extract links for crawling purposes
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                # Convert relative URLs to absolute
                full_url = urllib.parse.urljoin(url, href)
                links.append(full_url)
                
        return {
            "url": url,
            "title": title,
            "text_content": combined_text,
            "links": links,  # Keep links for crawling
            "crawl_time": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return {
            "url": url,
            "error": str(e),
            "crawl_time": datetime.now().isoformat()
        }


def is_internal_link(base_url, link):
    """Check if a link is internal to the base URL"""
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link).netloc
    return link_domain == base_domain or link_domain == ''


def crawl_website(url, depth=2, max_pages=20, timeout=20):
    """
    Crawl a website using Selenium WebDriver, focusing on text content
    
    Args:
        url (str): Starting URL to crawl
        depth (int): How many links deep to crawl
        max_pages (int): Maximum number of pages to crawl
        timeout (int): Seconds to wait between page loads
    
    Returns:
        dict: Crawled data with text content
    """
    print(f"Starting text content crawl of {url} with depth {depth}, max pages {max_pages}")
    
    # Set up Selenium WebDriver with Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=1")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    driver.set_page_load_timeout(30)  # Set page load timeout to 30 seconds
    
    # Move regex compilation outside the loop for better performance
    excluded_extensions = re.compile(
        r".*\.(png|jpe?g|gif|bmp|svg|webp|pdf|zip|tar|gz|tar\\.gz|rar|7z"
        r"|docx?|xlsx?|pptx?|exe|msi|sh|bin|iso|dmg|apk|jar"
        r"|mp3|mp4|avi|mov|ogg|wav"
        r"|ttf|woff2?|eot"
        r"|ics|csv|dat)(\?.*)?$", re.IGNORECASE
    )
    
    # Pattern to exclude URLs containing specific terms
    excluded_terms = re.compile(r"search|query|assay", re.IGNORECASE)
    
    # Initialize variables
    queue = [(url, 0)]  # (url, depth)
    visited = set()
    results = []
    pages_crawled = 0
    
    try:
        # Breadth-first search crawling
        while queue and pages_crawled < max_pages:
            current_url, current_depth = queue.pop(0)
            
            if current_url in visited:
                continue
                
            visited.add(current_url)
            
            try:
                print(f"Crawling text from: {current_url} (depth {current_depth})")
                driver.get(current_url)
                time.sleep(timeout)  # Wait for page to load and JavaScript to execute
                
                # Extract text content from current page
                page_data = extract_page_text_content(driver, current_url)
                results.append(page_data)
                pages_crawled += 1
                
                # Add links to queue if not at max depth (to continue crawling)
                if current_depth < depth:
                    for link in page_data["links"]:
                        if (link and isinstance(link, str) and link.startswith("http") 
                            and is_internal_link(url, link) 
                            and link not in visited
                            and not excluded_extensions.match(link)
                            and not excluded_terms.search(link)):  # Skip URLs with excluded terms
                            queue.append((link, current_depth + 1))
                                
            except TimeoutException:
                print(f"Timeout on {current_url}")
                results.append({
                    "url": current_url,
                    "error": "Timeout",
                    "crawl_time": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                results.append({
                    "url": current_url,
                    "error": str(e),
                    "crawl_time": datetime.now().isoformat()
                })
    
    finally:
        driver.quit()
    
    return {
        "start_url": url,
        "depth": depth,
        "max_pages": max_pages,
        "pages_crawled": pages_crawled,
        "crawl_date": datetime.now().isoformat(),
        "results": results
    }


def save_results(data, url):
    """Save crawl results to a JSON file"""
    # Ensure results directory exists
    results_dir = "raw_results_json"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create filename based on domain and date
    domain = urlparse(url).netloc.replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crawl_{domain}_{timestamp}.json"
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return filename


def main():
    parser = argparse.ArgumentParser(description="Crawl website text content using Selenium WebDriver")
    parser.add_argument("url", help="URL to start crawling from")
    parser.add_argument("--depth", type=int, default=2, help="How many links deep to crawl (default: 1)")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to crawl (default: 10)")
    parser.add_argument("--timeout", type=int, default=20, help="Seconds to wait between page loads (default: 2)")
    
    args = parser.parse_args()
    
    # Crawl the website
    crawl_data = crawl_website(
        args.url, 
        depth=args.depth,
        max_pages=args.max_pages,
        timeout=args.timeout
    )
    
    # Save the results
        
    filename = save_results(crawl_data, args.url)
    print(f"Text crawl completed. Results saved to {filename}")


if __name__ == "__main__":
    main()