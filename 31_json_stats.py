#     Analyze text content lengths from a JSON file and print statistics.
#     This function reads a JSON file that contains a 'results' array with items that have
#     'text_content' fields. It calculates and prints the maximum and average lengths
#     of those text contents, along with the number of entries analyzed.
#     Example:
#         python 31_json_stats.py path/to/results.json

import json
import sys


def analyze_text_content(file_path):
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract lengths of text_content fields
    text_lengths = []
    for result in data.get("results", []):
        if "text_content" in result and result["text_content"]:
            text_lengths.append(len(result["text_content"]))
    
    # Calculate statistics
    if text_lengths:
        max_length = max(text_lengths)
        avg_length = sum(text_lengths) / len(text_lengths)
        
        print(f"Maximum text content length: {max_length:,} characters")
        print(f"Average text content length: {avg_length:,.2f} characters")
        print(f"Number of entries analyzed: {len(text_lengths)}")
    else:
        print("No text content found in the JSON file.")


# Get file path from command-line argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 31_json_stats.py <json_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"Analyzing file: {file_path}")
    analyze_text_content(file_path)