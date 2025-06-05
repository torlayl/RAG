# -----------------------------------------------------------------------------
# This script performs part of RAG (Retrieval-Augmented Generation) pipeline:
# - Crawls HTML pages from a specified base URL
# - Extracts readable content and chunks it into paragraphs
# - Sends each page to a local LLM via Ollama to generate:
#     • a summary of the page
#     • a list of keywords
# - Saves the enriched data to a .jsonl file
# - Does not perform vector indexing, only enrichment
# -----------------------------------------------------------------------------

import os
import re
import time
import json
import uuid
import torch
import logging
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ Using device: {device}")

DEFAULT_CONFIG = {
    #"model": "qwen3:8b",
    "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", # recent improvment of Qwen3 model by DeepSeek
    "max_pages": 10,
    "max_depth": 1, # Maximum depth to crawl from the base URL
    "chunk_size": 1024,
    "min_content_length": 100, # Minimum content length to consider a page valid
    "delay_seconds": 10.0, # Delay between requests to avoid overloading servers
    "timeout_seconds": 40 # Timeout for requests
}

def generate_filename_from_url(url: str) -> str:
    """Generate a safe filename based on the URL"""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path.strip('/')
    
    # Remove common prefixes and clean domain
    domain = re.sub(r'^(www\.|doc\.|docs\.)', '', domain)
    
    # Combine domain and path, replace unsafe characters
    if path:
        filename_base = f"{domain}_{path}"
    else:
        filename_base = domain
    
    # Replace unsafe characters with underscores
    filename_base = re.sub(r'[^\w\-_.]', '_', filename_base)
    # Remove multiple consecutive underscores
    filename_base = re.sub(r'_+', '_', filename_base)
    # Remove leading/trailing underscores
    filename_base = filename_base.strip('_')
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"enriched_{filename_base}_{timestamp}.jsonl"

def run_pipeline(base_url: str, config: dict = None, **kwargs):
    """Run the RAG pipeline with configurable parameters"""
    cfg = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    # Check Ollama connection before starting
    if not check_ollama_connection():
        logger.error("❌ Ollama is not running or accessible at http://localhost:11434")
        return
    
    # Generate filename if not provided
    jsonl_output_path = kwargs.get('jsonl_output_path') or generate_filename_from_url(base_url)
    
    logger.info("🚀 Starting RAG pipeline")
    logger.info(f"📁 Output file: {jsonl_output_path}")
    start_time = time.time()  # Add timing

    def ollama_generate(prompt, model=cfg["model"]):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': model, 'prompt': prompt, 'stream': False},
                timeout=60  # Increase timeout for complex processing
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def extract_summary_and_keywords(text):
        prompt = f"""
Here is a web article:

{text[:2000]}  # Increase from 1500 to get more context

Please return:
1. A short summary (3–5 sentences) 
2. A list of 5 to 10 keywords

Expected JSON format:
{{
  "summary": "...",
  "keywords": ["word1", "word2", ...]
}}
"""
        try:
            raw = ollama_generate(prompt)
            # More robust JSON extraction
            start = raw.find('{')
            end = raw.rfind('}') + 1
            if start == -1 or end == 0:
                logger.warning(f"No JSON found in LLM response: {raw[:100]}...")
                return {"summary": "", "keywords": []}
            json_part = raw[start:end]
            return json.loads(json_part)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"summary": "", "keywords": []}

    def split_into_paragraphs(text, max_len=1024):
        """Split text into chunks respecting sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ''
        
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            if len(current) + len(sentence) + 1 <= max_len:
                current += ' ' + sentence if current else sentence
            else:
                # Save current chunk if it exists
                if current:
                    chunks.append(current.strip())
                
                # Handle oversized sentences
                if len(sentence) > max_len:
                    # Split long sentences by words
                    words = sentence.split()
                    temp_chunk = ''
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_len:
                            temp_chunk += ' ' + word if temp_chunk else word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    current = temp_chunk
                else:
                    current = sentence
        
        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk.strip()]

    # Move regex compilation outside the loop for better performance
    excluded_extensions = re.compile(
        r".*\.(png|jpe?g|gif|bmp|svg|webp|pdf|zip|tar|gz|tar\\.gz|rar|7z"
        r"|docx?|xlsx?|pptx?|exe|msi|sh|bin|iso|dmg|apk|jar"
        r"|mp3|mp4|avi|mov|ogg|wav"
        r"|ttf|woff2?|eot"
        r"|ics|csv|dat)(\?.*)?$", re.IGNORECASE
    )
    
    visited, to_visit = set(), set([base_url])
    page_counter = 0
    chunk_counter = 0
    depth_counter = {base_url: 0}

    with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
        while to_visit and page_counter < cfg["max_pages"]:
            url = to_visit.pop()
            if url in visited:
                continue

            try:
                response = requests.get(url, timeout=cfg["timeout_seconds"])  # Add timeout
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith("text/html"):
                    continue

                visited.add(url)

                if response.status_code == 200:
                    page_counter += 1
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    # Skip pages with minimal content
                    if len(text.strip()) < cfg["min_content_length"]:
                        logger.info(f"⏭️ Skipping {url} - minimal content ({len(text)} chars)")
                        continue
                    
                    title = ''
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                    web_path = urlparse(url).path
                    enrichments = extract_summary_and_keywords(text)

                    chunks = split_into_paragraphs(text, cfg["chunk_size"])
                    for idx, chunk in enumerate(chunks):
                        chunk_counter += 1
                        doc = {
                            "id": str(uuid.uuid4()),
                            "url": url,
                            "web_path": web_path,
                            "title": title,
                            "text": chunk,
                            "summary": enrichments.get("summary", ""),
                            "keywords": enrichments.get("keywords", []),
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "chunk_id": idx
                        }
                        f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')

                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])
                        normalized_url = normalize_url(full_url)
                        if (
                            normalized_url.startswith(base_url)
                            and normalized_url not in visited
                            and not excluded_extensions.match(normalized_url)
                        ):
                            # Add safety check for depth_counter
                            current_depth = depth_counter.get(url, 0)
                            if current_depth + 1 <= cfg["max_depth"]:
                                to_visit.add(normalized_url)
                                depth_counter[normalized_url] = current_depth + 1

                # Add periodic logging with ETA estimation and memory usage
                if page_counter % 10 == 0:
                    elapsed_time = time.time() - start_time
                    pages_per_sec = page_counter / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (cfg["max_pages"] - page_counter) / pages_per_sec if pages_per_sec > 0 else 0
                    memory_mb = len(visited) * 0.1 + len(to_visit) * 0.1  # Rough estimate
                    logger.info(f"📊 Progress: {page_counter}/{cfg['max_pages']} pages ({pages_per_sec:.1f}/s), {chunk_counter} chunks, {len(to_visit)} queued, ~{memory_mb:.1f}MB, ETA: {eta_seconds/60:.1f}min")

                time.sleep(cfg["delay_seconds"])

            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Timeout for {url} ({cfg['timeout_seconds']}s limit)")
            except requests.exceptions.RequestException as e:
                logger.error(f"🌐 Network error for {url}: {type(e).__name__}: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error with {url}: {type(e).__name__}: {e}")

    # Final summary with timing
    total_time = time.time() - start_time
    logger.info(f"🌐 Finished crawling {page_counter} pages | {chunk_counter} chunks in {total_time/60:.1f} minutes")
    logger.info(f"📄 JSONL file saved: {jsonl_output_path}")
    logger.info("✅ Data enrichment complete. No vector indexing performed.")

# Add connection check for Ollama
def check_ollama_connection():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def normalize_url(url):
    """Remove query parameters and fragments for deduplication"""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://cansar.ai/"
    run_pipeline(base_url=url)
