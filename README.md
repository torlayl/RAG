# Web Crawling and Content Processing Pipeline

A comprehensive suite of Python scripts for crawling websites, processing content, and enriching data with AI-powered analysis using local LLM models via Ollama.

## 🚀 Features

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

## 🛠️ Installation

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
   ollama pull deepseek-r1:latest
   ```
3. Start Ollama service:
   ```bash
   ollama serve
   ```

## 📖 Usage

### Step 1: Crawl Website Content
```bash
python 1_Crawling_raw_json.py https://example.com --depth 2 --max-pages 50 --timeout 3
```

**Arguments:**
- `url`: Starting URL to crawl (required)
- `--depth`: How many links deep to crawl (default: 2)
- `--max-pages`: Maximum number of pages to crawl (default: 20)
- `--timeout`: Seconds to wait between page loads (default: 20)

**Output:** Raw crawl data saved to `raw_results_json/crawl_<domain>_<timestamp>.json`

### Step 2: Process and Clean Data
```bash
python 2_Process_json.py raw_results_json/crawl_example_com_20250605_120000.json
```

**Arguments:**
- `input_file`: Path to the input JSON file from step 1 (required)
- `output_file`: Optional output filename (auto-generated if not specified)

**Output:** Processed data saved to `raw_results_json/<filename>_processed.json`

### Step 3: Enrich with AI Analysis
```bash
python 3_Enriching_Json.py raw_results_json/crawl_example_com_20250605_120000_processed.json --model deepseek-r1:latest
```

**Arguments:**
- `input_json`: Path to the processed JSON file from step 2 (required)
- `--output, -o`: Path for the enriched output JSON file (optional)
- `--model`: Ollama model to use for generation (default: deepseek-r1:latest)
- `--delay`: Delay between LLM requests in seconds (default: 5.0)
- `--timeout`: Timeout for LLM requests in seconds (default: 120)
- `--retries`: Maximum number of retries for failed requests (default: 3)
- `--verbose, -v`: Enable verbose debug logging

**Output:** Enriched data saved to `enriched_results_json/enriched_<filename>_<timestamp>.json`

## 📁 Directory Structure

After running the pipeline, you'll have the following structure:

```
├── raw_results_json/           # Raw and processed crawl data
│   ├── crawl_<domain>_<timestamp>.json
│   └── crawl_<domain>_<timestamp>_processed.json
├── enriched_results_json/      # AI-enriched data
│   └── enriched_<filename>_<timestamp>.json
└── logs/                       # Processing logs
    └── enrichment_log_<timestamp>.log
```

## 🔧 Configuration

### Crawler Settings
The crawler excludes certain file types and URL patterns by default:
- **Excluded file types**: Images (png, jpg, gif), documents (pdf, docx), archives (zip, tar), media files (mp3, mp4), etc.
- **Excluded URL patterns**: URLs containing "search", "query", "assay" (case-insensitive)

### Enrichment Settings
Default configuration in `3_Enriching_Json.py`:
- **Model**: `deepseek-r1:latest`
- **Timeout**: 120 seconds per request
- **Delay**: 5 seconds between requests
- **Max retries**: 3 attempts per page
- **Retry delay**: 10 seconds between retries

## 📊 Output Format

### Raw Crawl Data
```json
{
  "start_url": "https://example.com",
  "depth": 2,
  "max_pages": 20,
  "pages_crawled": 15,
  "crawl_date": "2025-06-05T14:30:00",
  "results": [
    {
      "url": "https://example.com/page1",
      "title": "Page Title",
      "text_content": "Page content...",
      "links": ["https://example.com/page2"],
      "crawl_time": "2025-06-05T14:30:15"
    }
  ]
}
```

### Processed Data
```json
{
  "results": [
    {
      "title": "Page Title",
      "text_content": "Page content...",
      "crawl_time": "2025-06-05T14:30:15",
      "url": "https://example.com/page1",
      "content_hash": "abc123def456..."
    }
  ]
}
```

### Enriched Data
```json
{
  "results": [
    {
      "title": "Page Title",
      "text_content": "Page content...",
      "crawl_time": "2025-06-05T14:30:15",
      "url": "https://example.com/page1",
      "content_hash": "abc123def456...",
      "summary": "AI-generated summary of the page content...",
      "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"]
    }
  ]
}
```

## 🚨 Troubleshooting

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

### Performance Tips

- **Crawling**: Use appropriate timeout values to balance speed vs. completeness
- **Processing**: Large JSON files are processed in memory; ensure sufficient RAM
- **Enrichment**: Adjust delay between requests to avoid overwhelming the LLM

## 📝 Logging

All scripts provide comprehensive logging:
- **Console output**: Real-time progress and status updates
- **Log files**: Detailed logs saved to `logs/` directory
- **Error tracking**: Full error details with retry attempts

## 🤝 Contributing

This pipeline is part of the DAISY RAG system. For improvements or bug reports, please follow the project's contribution guidelines.

## 📄 License

See LICENSE file for details.

## 🔗 Dependencies

Key dependencies:
- `selenium`: Web browser automation
- `beautifulsoup4`: HTML parsing
- `requests`: HTTP requests for Ollama API
- `webdriver-manager`: ChromeDriver management

For a complete list, see `requirements.txt`.
