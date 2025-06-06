#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# This script processes a JSON file from the enriched_results_json directory,
# re-chunks the text content of each result, and creates a new JSON file.
# It adds summaries to the beginning of EACH chunk (when available) and removes
# "original_id", "chunk_id", "chunk_count", "processing_timestamp", "summary", and "url" 
# fields from each result. It uses the same chunking approach as the
# generate_chunks_jsonl.py script.
# -----------------------------------------------------------------------------

import os
import re
import json
import uuid
import logging
import sys
import traceback
from datetime import datetime
from urllib.parse import urlparse

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"rechunk_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

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

def generate_output_filename(input_filename):
    """Generate output filename based on the input filename, keeping only the URL part"""
    # Extract just the domain part from the filename (e.g., cansar_ai from enriched_crawl_cansar_ai_20250605_...)
    filename_base = os.path.basename(input_filename).split('.')[0]
    # Extract domain from pattern like 'enriched_crawl_domain_name_timestamp'
    match = re.search(r'enriched_crawl_([^_]+)', filename_base)
    domain = match.group(1) if match else filename_base
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{domain}_chunked_{timestamp}.json"

def process_json_file(input_file, chunk_size=1024):
    """Process the JSON file, rechunk the text content, and add summaries to each chunk"""
    logger.info(f"Processing file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return None
    
    results = data.get('results', [])
    if not results:
        logger.warning("No results found in the JSON file")
        return None
    
    new_results = []
    total_original_results = len(results)
    total_new_chunks = 0
    
    for idx, result in enumerate(results):
        if not result.get('text_content') or not isinstance(result.get('text_content'), str):
            logger.warning(f"Skipping result {idx} - no valid text_content")
            continue
        
        # Get original data
        title = result.get('title', '')
        url = result.get('url', '')
        web_path = urlparse(url).path if url else ''
        summary = result.get('summary', '')
        keywords = result.get('keywords', [])
        
        # Get the original text_content
        original_text = result['text_content']
        
        # Create the summary prefix if it exists
        summary_prefix = f"{summary}\n\n" if summary else ""
        
        # Split original text content into chunks first
        content_chunks = split_into_paragraphs(original_text, chunk_size)
        
        # Now add summary to each chunk
        chunks = []
        for content_chunk in content_chunks:
            if not content_chunk.startswith("Summary:"):  # Avoid duplicate summaries
                enhanced_chunk = f"{summary_prefix}Content: {content_chunk}"
                chunks.append(enhanced_chunk)
            else:
                # The chunk already has a summary format
                chunks.append(content_chunk)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Create a new result with the chunk - without the specified fields and url
            new_result = {
                "id": str(uuid.uuid4()),
                "web_path": web_path,
                "title": title,
                "text_content": chunk,
                "keywords": keywords
            }
            new_results.append(new_result)
        
        total_new_chunks += len(chunks)
        if (idx + 1) % 10 == 0 or idx + 1 == total_original_results:
            logger.info(f"Processed {idx + 1}/{total_original_results} results, created {total_new_chunks} chunks")
    
    return {"results": new_results}

def main():
    """Main function to process the JSON file and create a new one with rechunked data"""
    import argparse
    import sys
    
    try:
        parser = argparse.ArgumentParser(description="Process and rechunk JSON data")
        parser.add_argument("input_file", help="Path to the input JSON file")
        parser.add_argument("--output-file", help="Path to the output JSON file")
        parser.add_argument("--chunk-size", type=int, default=1024, 
                            help="Maximum size of each text chunk (default: 1024)")
        args = parser.parse_args()
        
        print(f"Processing input file: {args.input_file}")
        
        if not os.path.isfile(args.input_file):
            print(f"ERROR: Input file not found: {args.input_file}")
            logger.error(f"Input file not found: {args.input_file}")
            return
        
        print(f"File exists, starting processing...")
        
        # Process the JSON file
        new_data = process_json_file(args.input_file, args.chunk_size)
        if not new_data:
            print("ERROR: Failed to process JSON file - no data returned")
            return
        
        print(f"Processing complete. Got {len(new_data['results'])} new chunks.")
        
        # Generate output filename if not provided
        output_file = args.output_file or os.path.join(
            os.path.dirname(args.input_file),
            generate_output_filename(args.input_file)
        )
        
        print(f"Saving output to: {output_file}")
        
        # Save the new JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully created new JSON file: {output_file}")
            print(f"Total chunks created: {len(new_data['results'])}")
            logger.info(f"Successfully created new JSON file: {output_file}")
            logger.info(f"Total chunks created: {len(new_data['results'])}")
        except Exception as e:
            print(f"ERROR saving JSON file: {e}")
            logger.error(f"Error saving JSON file: {e}")
            
    except Exception as e:
        print(f"Unexpected ERROR: {e}")
        print(f"Type: {type(e)}")
        print(f"Args: {e.args}")
        import traceback
        traceback.print_exc()
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
