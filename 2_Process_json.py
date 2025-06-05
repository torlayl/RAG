# Web Crawl Data Processor
# This script processes JSON files containing web crawl data by removing duplicate 
# entries based on content hash, keeping only essential information (title, text_content, 
# url, crawl_time), and adding a content hash for each entry.
# Usage:
#     python 2_Process_json.py <input_file> [output_file]
# Arguments:
#     input_file: Path to the input JSON file containing web crawl data
#     output_file: Optional path for the output file. If not specified, 
#                  creates '<input_filename>_processed.json'
# Output:
#     Processed JSON file saved in the 'raw_results_json' directory



import json
import hashlib
import os
import sys

def process_json_file(input_path):
    """
    Process a JSON file containing web crawl data:
    1. Generate hash ID based on text_content
    2. Remove duplicate entries based on content hash
    3. Keep only title, text_content, crawl_time, url, and content_hash
    """
    # Read the JSON file
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None    # Dictionary to track unique content based on hash
    unique_entries = {}
    hash_to_url_map = {}
    
    # Process each entry in the results list
    for entry in data.get('results', []):
        url = entry.get('url')
        
        # Skip entries without a URL
        if not url:
            continue
            
        # Generate hash from text_content
        text = entry.get('text_content', '')
        hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Only keep the first instance of each unique content
        if hash_value not in hash_to_url_map:
            hash_to_url_map[hash_value] = url
            
            # Create a simplified entry with only the fields we want
            unique_entries[url] = {
                'title': entry.get('title', ''),
                'text_content': text,
                'crawl_time': entry.get('crawl_time', ''),
                'url': url,  # Also store the URL in the entry
                'content_hash': hash_value
            }
    
    # Create a new structure for the processed data
    processed_data = {
        'results': list(unique_entries.values())
    }
    
    return processed_data

def save_json(data, output_path):
    """Save processed data to a JSON file"""
    try:
        # Create raw_results_json directory if it doesn't exist
        output_dir = "raw_results_json"
        os.makedirs(output_dir, exist_ok=True)
        
        # Update the output path to be in the raw_results_json directory
        filename = os.path.basename(output_path)
        new_output_path = os.path.join(output_dir, filename)
        
        with open(new_output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Successfully saved processed data to {new_output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python process_crawl_data.py <input_file> [output_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_processed.json"
    
    # Process the file
    processed_data = process_json_file(input_file)
    
    if processed_data:
        save_json(processed_data, output_file)
    else:
        print("Processing failed. No output file created.")
        sys.exit(1)

if __name__ == "__main__":
    main()