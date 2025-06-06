# enrich_crawled_json.py
# -----------------------------------------------------------------------------
# This script enriches crawled website data in JSON format by:
# - Reading a crawled JSON file containing webpage content
# - Sending each page's content to a local LLM via Ollama to generate:
#     • a summary of the page
#     • a list of 8 keywords
# - Adding these enrichments to each page object in the JSON
# - Saving the enriched data back to a new JSON file
# -----------------------------------------------------------------------------

import os
import json
import time
import logging
import requests
import argparse
from datetime import datetime

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"enrichment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Configuration ===
DEFAULT_CONFIG = {
    "model": "qwen3:8b",  # Default Ollama model to use
    #"model": "deepseek-r1:latest",  # Alternative model, trouble with timeout, qwen3:8b is more stable and more quick
    "timeout_seconds": 300,  # Timeout for LLM requests (increased from 60)
    "delay_seconds": 5.0,   # Delay between LLM requests to avoid overloading
    "max_retries": 3,       # Maximum number of retries for failed requests
    "retry_delay": 10.0,    # Delay between retries in seconds
}

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama connection verified successfully")
            return True
        else:
            logger.error(f"❌ Ollama returned status code {response.status_code}: {response.text}")
            return False
    except requests.ConnectionError as e:
        logger.error(f"❌ Ollama connection failed: Could not connect to server at localhost:11434 - {e}")
        return False
    except requests.Timeout as e:
        logger.error(f"❌ Ollama connection failed: Request timed out after 5 seconds - {e}")
        return False
    except requests.RequestException as e:
        logger.error(f"❌ Ollama connection failed: {type(e).__name__} - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error checking Ollama connection: {type(e).__name__} - {e}")
        return False

def verify_model_availability(model_name):
    """Verify that the specified model is available in Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            # Check exact match first
            if model_name in available_models:
                logger.info(f"✅ Model '{model_name}' is available")
                return True
            
            # Check if model name without tag matches any available model
            model_base_name = model_name.split(':')[0]
            matching_models = [m for m in available_models if m.startswith(model_base_name + ':')]
            
            if matching_models:
                logger.info(f"✅ Model '{model_name}' found as '{matching_models[0]}'")
                logger.info(f"💡 Using available version: {matching_models[0]}")
                return True
            else:
                logger.error(f"❌ Model '{model_name}' is not available")
                logger.info(f"Available models: {', '.join(available_models)}")
                logger.error(f"💡 To install the model, run: ollama pull {model_name}")
                return False
        else:
            logger.warning(f"Could not verify model availability: HTTP {response.status_code}")
            return True  # Assume it's available if we can't check
    except Exception as e:
        logger.warning(f"Could not verify model availability: {e}")
        return True  # Assume it's available if we can't check

def generate_enrichments(text, model, config=None):
    """Generate summary and keywords using Ollama with improved error handling and retry logic"""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # Trim to avoid sending too much text
    text_sample = text  # Using full text for summary generation
    prompt = f"""
Please analyze this webpage content:

{text_sample}

Return only a JSON object with:
1. A concise summary (3-5 sentences)
2. Exactly 8 keywords (important terms from the content)

Expected format:
{{
  "summary": "...",
  "keywords": ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8"]
}}
"""
    
    last_error = None
    for attempt in range(cfg['max_retries']):
        try:
            logger.debug(f"Attempting enrichment generation (attempt {attempt + 1}/{cfg['max_retries']})")
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': model, 'prompt': prompt, 'stream': False},
                timeout=cfg['timeout_seconds']
            )
            response.raise_for_status()
            
            result = response.json()['response']
            
            # Extract the JSON part from the response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start >= 0 and json_end > 0:
                json_str = result[json_start:json_end]
                try:
                    result_json = json.loads(json_str)
                    
                    # Validate the structure
                    if 'summary' not in result_json or 'keywords' not in result_json:
                        logger.warning(f"Invalid JSON structure in LLM response: missing required fields")
                        logger.debug(f"Response content: {json_str[:200]}...")
                    elif not isinstance(result_json['keywords'], list):
                        logger.warning(f"Keywords field is not a list: {type(result_json['keywords'])}")
                        logger.debug(f"Response content: {json_str[:200]}...")
                    else:
                        logger.debug(f"Successfully parsed enrichment response")
                        return result_json
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"JSON string that failed to parse: {json_str[:300]}...")
                    last_error = f"JSON parsing error: {e}"
            else:
                logger.warning(f"No valid JSON structure found in LLM response")
                logger.debug(f"Response content: {result[:300]}...")
                last_error = "No JSON structure found in response"
                
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout after {cfg['timeout_seconds']} seconds: {e}"
            logger.error(error_msg)
            last_error = error_msg
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to Ollama server: {e}"
            logger.error(error_msg)
            last_error = error_msg
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from Ollama: {e} - Status: {response.status_code}"
            logger.error(error_msg)
            if hasattr(response, 'text'):
                logger.debug(f"Response content: {response.text[:300]}...")
            last_error = error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {type(e).__name__}: {e}"
            logger.error(error_msg)
            last_error = error_msg
            
        except KeyError as e:
            error_msg = f"Missing 'response' key in Ollama API response: {e}"
            logger.error(error_msg)
            last_error = error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error generating enrichments: {type(e).__name__}: {e}"
            logger.error(error_msg)
            last_error = error_msg
        
        # Wait before retry (except for the last attempt)
        if attempt < cfg['max_retries'] - 1:
            logger.info(f"Retrying in {cfg['retry_delay']} seconds... (attempt {attempt + 1}/{cfg['max_retries']})")
            time.sleep(cfg['retry_delay'])
    
    # All retries failed, return fallback response
    logger.error(f"All {cfg['max_retries']} attempts failed. Last error: {last_error}")
    return {
        "summary": f"Summary generation failed after {cfg['max_retries']} attempts. Last error: {last_error}",
        "keywords": ["error", "failed", "retry", "timeout"]
    }

def enrich_json_data(input_json_path, output_json_path=None, config=None):
    """Process the crawled JSON file and add enrichments"""
    cfg = {**DEFAULT_CONFIG, **(config or {})}    # Check Ollama connection
    if not check_ollama_connection():
        logger.error("❌ Ollama is not running or accessible at http://localhost:11434")
        logger.error("💡 Please ensure Ollama is installed and running:")
        logger.error("   • Start Ollama: 'ollama serve'")
        logger.error("   • Verify model is available: 'ollama list'")
        logger.error(f"   • Pull model if needed: 'ollama pull {cfg['model']}'")
        return False
    
    # Verify model availability
    if not verify_model_availability(cfg['model']):
        logger.error(f"❌ Cannot proceed without model '{cfg['model']}'")
        return False
    
    # Define output directory and create if it doesn't exist
    output_dir = "enriched_results_json"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path if not provided
    if not output_json_path:
        base = os.path.splitext(os.path.basename(input_json_path))[0]
        output_json_path = os.path.join(output_dir, f"enriched_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        # If path was provided but doesn't include the output directory, add it
        if not os.path.dirname(output_json_path):
            output_json_path = os.path.join(output_dir, output_json_path)
    
    logger.info(f"🚀 Starting JSON enrichment process")
    logger.info(f"📄 Input file: {input_json_path}")
    logger.info(f"📁 Output file: {output_json_path}")
      # Read the input JSON file
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Successfully loaded input JSON file")
    except FileNotFoundError:
        logger.error(f"❌ Input JSON file not found: {input_json_path}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format in input file: {e}")
        return False
    except PermissionError:
        logger.error(f"❌ Permission denied reading input file: {input_json_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to read input JSON file: {type(e).__name__}: {e}")
        return False
        
    # Check if the file has the expected structure
    if 'results' not in data:
        logger.error("❌ Input JSON doesn't have 'results' key. Expected format: {'results': [...]}")
        logger.debug(f"Available keys in JSON: {list(data.keys())}")
        return False
        
    if not isinstance(data['results'], list):
        logger.error(f"❌ 'results' should be a list, got {type(data['results'])}")
        return False
        
    # Track progress
    total_pages = len(data['results'])
    logger.info(f"Found {total_pages} pages to process")
    
    start_time = time.time()
      # Process each page
    processed_pages = 0
    failed_pages = 0
    
    for i, page in enumerate(data['results']):
        if 'text_content' not in page:
            logger.warning(f"Page {i+1} missing text_content, skipping - URL: {page.get('url', 'unknown URL')}")
            failed_pages += 1
            continue
        
        page_url = page.get('url', 'unknown URL')
        text_length = len(page['text_content'])
        
        # Generate enrichments
        logger.info(f"Processing page {i+1}/{total_pages}: {page_url} ({text_length} chars)")
        
        try:
            enrichments = generate_enrichments(page['text_content'], cfg['model'], cfg)
            
            # Add enrichments to the page
            page['summary'] = enrichments.get('summary', 'No summary generated')
            page['keywords'] = enrichments.get('keywords', [])
            page['enrichment_status'] = 'success'
            page['enrichment_timestamp'] = datetime.now().isoformat()
            
            processed_pages += 1
            logger.debug(f"Successfully enriched page: {page_url}")
            
        except Exception as e:
            logger.error(f"Failed to enrich page {i+1} ({page_url}): {type(e).__name__}: {e}")
            page['summary'] = f"Enrichment failed: {str(e)}"
            page['keywords'] = ["error", "failed"]
            page['enrichment_status'] = 'failed'
            page['enrichment_timestamp'] = datetime.now().isoformat()
            page['enrichment_error'] = str(e)
            failed_pages += 1
        
        # Log progress periodically
        if (i+1) % 5 == 0 or i+1 == total_pages:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)
            eta = avg_time * (total_pages - (i+1))
            success_rate = (processed_pages / (i+1)) * 100 if i+1 > 0 else 0
            logger.info(f"📊 Progress: {i+1}/{total_pages} pages ({(i+1)/total_pages*100:.1f}%) | "
                       f"Success: {processed_pages}, Failed: {failed_pages} ({success_rate:.1f}% success) | "
                       f"Avg: {avg_time:.1f}s/page | ETA: {eta/60:.1f}min")
            
        time.sleep(cfg["delay_seconds"])
      # Write the enriched data
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Enriched JSON saved to {output_json_path}")
        
        # Log final statistics
        file_size = os.path.getsize(output_json_path) / (1024 * 1024)  # Size in MB
        logger.info(f"📁 Output file size: {file_size:.2f} MB")
        
    except PermissionError as e:
        logger.error(f"❌ Permission denied writing to {output_json_path}: {e}")
        return False
    except OSError as e:
        logger.error(f"❌ OS error writing to {output_json_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to write output JSON: {type(e).__name__}: {e}")
        return False
    
    total_time = time.time() - start_time
    success_rate = (processed_pages / total_pages) * 100 if total_pages > 0 else 0
    
    logger.info(f"🎉 Finished processing {total_pages} pages in {total_time/60:.1f} minutes")
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   • Successfully enriched: {processed_pages}/{total_pages} pages ({success_rate:.1f}%)")
    logger.info(f"   • Failed enrichments: {failed_pages}/{total_pages} pages")
    logger.info(f"   • Average time per page: {total_time/total_pages:.1f} seconds")
    
    if failed_pages > 0:
        logger.warning(f"⚠️  {failed_pages} pages failed enrichment. Check logs for details.")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enrich crawled JSON with summaries and keywords')
    parser.add_argument('input_json', help='Path to the crawled JSON file')
    parser.add_argument('--output', '-o', help='Path for the enriched output JSON file')
    parser.add_argument('--model', help='Ollama model to use for generation')
    parser.add_argument('--delay', type=float, help='Delay between LLM requests in seconds')
    parser.add_argument('--timeout', type=int, help='Timeout for LLM requests in seconds')
    parser.add_argument('--retries', type=int, help='Maximum number of retries for failed requests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Verbose debug logging enabled")
    
    config = {}
    if args.model:
        config['model'] = args.model
        logger.info(f"🤖 Using model: {args.model}")
    if args.delay:
        config['delay_seconds'] = args.delay
        logger.info(f"⏱️  Delay between requests: {args.delay}s")
    if args.timeout:
        config['timeout_seconds'] = args.timeout
        logger.info(f"⏰ Request timeout: {args.timeout}s")
    if args.retries:
        config['max_retries'] = args.retries
        logger.info(f"🔄 Max retries: {args.retries}")
        
    success = enrich_json_data(args.input_json, args.output, config)
    
    if success:
        logger.info("✅ Enrichment process completed successfully")
    else:
        logger.error("❌ Enrichment process failed")
        exit(1)