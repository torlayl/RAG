# Web Crawling and Content Processing Pipeline

A suite of Python scripts for crawling websites, processing content, and enriching data using local LLM

## Features

- **Intelligent Web Crawling**: Extract text content from websites with configurable depth and page limits
- **Content Deduplication**: Remove duplicate content based on content hashing
- **AI-Powered Enrichment**: Generate summaries and keywords using local LLM models
- **Structured Output**: Clean JSON format for downstream processing
- **Robust Error Handling**: Comprehensive logging and retry mechanisms

## 📋 Scripts Overview

### 1. `1_Crawling_raw_json.py` - Web Content Crawler
Crawls websites using Selenium WebDriver to extract text content systematically.

**Key Features:**
- Configurable crawl depth and maximum page limits
- Internal link detection to stay within the same domain
- Exclusion of non-textual content (images, documents, media files)
- Headless browser operation using Chrome WebDriver
- Structured JSON output with crawl metadata

### 2. `2_Process_json.py` - JSON Data Processor
Processes JSON files containing web crawl data by removing duplicates and cleaning the data structure.

**Key Features:**
- Content deduplication based on MD5 hash
- Keeps only essential information (title, text_content, url, crawl_time)
- Adds content hash for each entry
- Optimized data structure for downstream processing

### 3. `3_Enriching_Json.py` - AI Content Enrichment
Enriches crawled website data by generating summaries and keywords using local LLM models.

**Key Features:**
- Integration with Ollama for local LLM inference
- Generates concise summaries (3-5 sentences)
- Extracts 8 relevant keywords per page
- Comprehensive error handling and retry logic
- Progress tracking and detailed logging

## Installation

### Prerequisites
- Python 3.8 or higher
- Chrome browser installed
- Ollama installed and running (for enrichment step)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup Ollama (for enrichment)
1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull a compatible model:
   ```bash
   ollama pull deepseek-r1:latest (or more stable qwen3:8b)
   ```
3. Start Ollama service:
   ```bash
   ollama serve
   ```


### Common Issues

1. **ChromeDriver Issues**
   - The script automatically manages ChromeDriver via `webdriver_manager`
   - Ensure Chrome browser is installed and up to date

2. **Ollama Connection Failed**
   - Verify Ollama is running: `ollama list`
   - Check if the service is accessible: `curl http://localhost:11434/api/tags`
   - Restart Ollama service if needed

3. **Memory Issues**
   - For large websites, reduce `--max-pages` parameter
   - Monitor system resources during crawling

4. **Network Timeouts**
   - Increase `--timeout` parameter for slow websites
   - Check internet connection stability


## License

See LICENSE file for details.

## Dependencies

Key dependencies:
- `selenium`: Web browser automation
- `beautifulsoup4`: HTML parsing
- `webdriver-manager`: ChromeDriver management

For a complete list, see `requirements.txt`.
