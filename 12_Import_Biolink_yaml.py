
# Script to extract and chunk the Biolink Model YAML file into logical segments for RAG (Retrieval-Augmented Generation) applications.
# This script parses the biolink-model.yaml file and divides it into meaningful chunks, 
# such as metadata, prefixes, types, slots, classes, and enums. Each chunk includes relevant
# metadata and keywords to facilitate information retrieval in RAG systems.
# Usage:
#     python 12_Import_Biolink_yaml.py path/to/biolink-model.yaml [-o output_file.txt]
#     yaml_file: Path to the biolink-model.yaml file
#     -o, --output: Optional output file path (default: biolink_chunks.txt)

import yaml
from typing import Dict, List, Any
import re
import argparse  # Add this import for command-line argument parsing

def extract_biolink_chunks(yaml_file_path: str) -> List[Dict[str, Any]]:
    """
    Extract logical chunks from biolink-model.yaml for RAG purposes.
    
    Args:
        yaml_file_path: Path to the biolink-model.yaml file
        
    Returns:
        List of dictionaries containing chunked content with metadata
    """
    
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    chunks = []
    
    # 1. Extract model metadata
    metadata_chunk = {
        'chunk_type': 'metadata',
        'title': 'Biolink Model Metadata',
        'content': f"""
Biolink Model Information:
- ID: {data.get('id', 'N/A')}
- Name: {data.get('name', 'N/A')}
- Description: {data.get('description', 'N/A')}
- Version: {data.get('version', 'N/A')}
- License: {data.get('license', 'N/A')}
- Default Prefix: {data.get('default_prefix', 'N/A')}
        """.strip(),
        'metadata': {
            'section': 'model_info',
            'keywords': ['biolink', 'model', 'metadata', 'version']
        }
    }
    chunks.append(metadata_chunk)
    
    # 2. Extract prefixes section
    if 'prefixes' in data:
        prefix_groups = {}
        prefixes = data['prefixes']
        
        # Group prefixes by domain/category
        for prefix, uri in prefixes.items():
            category = categorize_prefix(prefix, uri)
            if category not in prefix_groups:
                prefix_groups[category] = []
            prefix_groups[category].append((prefix, uri))
        
        for category, prefix_list in prefix_groups.items():
            content = f"Biolink Model Prefixes - {category}:\n\n"
            for prefix, uri in prefix_list:
                content += f"- {prefix}: {uri}\n"
            
            chunks.append({
                'chunk_type': 'prefixes',
                'title': f'Prefixes - {category}',
                'content': content.strip(),
                'metadata': {
                    'section': 'prefixes',
                    'category': category,
                    'keywords': ['prefixes', 'namespaces', category.lower()]
                }
            })
    
    # 3. Extract types section
    if 'types' in data:
        for type_name, type_info in data['types'].items():
            content = f"Type: {type_name}\n\n"
            if isinstance(type_info, dict):
                if 'description' in type_info:
                    content += f"Description: {type_info['description']}\n\n"
                if 'typeof' in type_info:
                    content += f"Type of: {type_info['typeof']}\n"
                if 'uri' in type_info:
                    content += f"URI: {type_info['uri']}\n"
                if 'exact_mappings' in type_info:
                    if isinstance(type_info['exact_mappings'], (list, tuple)):
                        content += f"Exact mappings: {', '.join(type_info['exact_mappings'])}\n"
                    else:
                        content += f"Exact mappings: {type_info['exact_mappings']}\n"
            
            chunks.append({
                'chunk_type': 'type',
                'title': f'Type: {type_name}',
                'content': content.strip(),
                'metadata': {
                    'section': 'types',
                    'type_name': type_name,
                    'keywords': ['types', 'data_types', type_name.replace(' ', '_')]
                }
            })
    
    # 4. Extract slots section
    if 'slots' in data:
        slots = data['slots']
        slot_groups = group_slots_by_category(slots)
        
        for category, slot_list in slot_groups.items():
            for slot_name in slot_list:
                slot_info = slots[slot_name]
                chunks.append(create_slot_chunk(slot_name, slot_info, category))
    
    # 5. Extract classes section
    if 'classes' in data:
        classes = data['classes']
        class_groups = group_classes_by_hierarchy(classes)
        
        for group_name, class_list in class_groups.items():
            for class_name in class_list:
                class_info = classes[class_name]
                chunks.append(create_class_chunk(class_name, class_info, group_name))
    
    # 6. Extract enums section
    if 'enums' in data:
        for enum_name, enum_info in data['enums'].items():
            content = f"Enum: {enum_name}\n\n"
            if isinstance(enum_info, dict):
                if 'description' in enum_info:
                    content += f"Description: {enum_info['description']}\n\n"
                if 'permissible_values' in enum_info:
                    content += "Permissible values:\n"
                    for value in enum_info['permissible_values']:
                        content += f"- {value}\n"
            
            chunks.append({
                'chunk_type': 'enum',
                'title': f'Enum: {enum_name}',
                'content': content.strip(),
                'metadata': {
                    'section': 'enums',
                    'enum_name': enum_name,
                    'keywords': ['enums', 'enumeration', enum_name.replace('Enum', '').lower()]
                }
            })
    
    return chunks

def categorize_prefix(prefix: str, uri: str) -> str:
    """Categorize prefixes based on their domain."""
    prefix_lower = prefix.lower()
    uri_lower = uri.lower()
    
    if any(term in prefix_lower for term in ['ncbi', 'ensembl', 'uniprot', 'gene']):
        return 'Genomics'
    elif any(term in prefix_lower for term in ['chebi', 'pubchem', 'drug', 'chem']):
        return 'Chemistry'
    elif any(term in prefix_lower for term in ['mondo', 'hp', 'doid', 'disease']):
        return 'Disease'
    elif any(term in prefix_lower for term in ['go', 'so', 'uberon', 'anatomy']):
        return 'Ontology'
    elif any(term in prefix_lower for term in ['pmid', 'doi', 'isbn']):
        return 'Publications'
    elif any(term in prefix_lower for term in ['wikidata', 'schema', 'rdf', 'owl']):
        return 'Semantic_Web'
    else:
        return 'Other'

def group_slots_by_category(slots: Dict) -> Dict[str, List[str]]:
    """Group slots by their functional category."""
    categories = {
        'Node_Properties': [],
        'Association_Properties': [],
        'Qualifiers': [],
        'Relationships': [],
        'Attributes': []
    }
    
    for slot_name, slot_info in slots.items():
        if isinstance(slot_info, dict):
            if slot_info.get('is_a') == 'association slot':
                categories['Association_Properties'].append(slot_name)
            elif slot_info.get('is_a') == 'qualifier':
                categories['Qualifiers'].append(slot_name)
            elif slot_info.get('is_a') == 'related to':
                categories['Relationships'].append(slot_name)
            elif slot_info.get('is_a') == 'node property':
                categories['Node_Properties'].append(slot_name)
            else:
                categories['Attributes'].append(slot_name)
    
    return {k: v for k, v in categories.items() if v}  # Remove empty categories

def group_classes_by_hierarchy(classes: Dict) -> Dict[str, List[str]]:
    """Group classes by their hierarchical category."""
    categories = {
        'Core_Entities': [],
        'Biological_Entities': [],
        'Chemical_Entities': [],
        'Clinical_Entities': [],
        'Information_Entities': [],
        'Associations': [],
        'Other': []
    }
    
    for class_name, class_info in classes.items():
        if isinstance(class_info, dict):
            if 'association' in class_name.lower():
                categories['Associations'].append(class_name)
            elif any(term in class_name.lower() for term in ['gene', 'protein', 'biological', 'organism']):
                categories['Biological_Entities'].append(class_name)
            elif any(term in class_name.lower() for term in ['chemical', 'drug', 'molecule']):
                categories['Chemical_Entities'].append(class_name)
            elif any(term in class_name.lower() for term in ['clinical', 'disease', 'phenotype']):
                categories['Clinical_Entities'].append(class_name)
            elif any(term in class_name.lower() for term in ['information', 'publication', 'dataset']):
                categories['Information_Entities'].append(class_name)
            elif class_name in ['entity', 'named thing', 'ontology class']:
                categories['Core_Entities'].append(class_name)
            else:
                categories['Other'].append(class_name)
    
    return {k: v for k, v in categories.items() if v}

def create_slot_chunk(slot_name: str, slot_info: Dict, category: str) -> Dict[str, Any]:
    """Create a chunk for a slot definition."""
    content = f"Slot: {slot_name}\n\n"
    
    if 'description' in slot_info:
        content += f"Description: {slot_info['description']}\n\n"
    
    if 'is_a' in slot_info:
        content += f"Is a: {slot_info['is_a']}\n"
    
    if 'domain' in slot_info:
        content += f"Domain: {slot_info['domain']}\n"
    
    if 'range' in slot_info:
        content += f"Range: {slot_info['range']}\n"
    
    if 'exact_mappings' in slot_info:
        if isinstance(slot_info['exact_mappings'], (list, tuple)):
            content += f"Exact mappings: {', '.join(slot_info['exact_mappings'])}\n"
        else:
            content += f"Exact mappings: {slot_info['exact_mappings']}\n"
    
    if 'aliases' in slot_info:
        if isinstance(slot_info['aliases'], (list, tuple)):
            content += f"Aliases: {', '.join(slot_info['aliases'])}\n"
        else:
            content += f"Aliases: {slot_info['aliases']}\n"
    
    return {
        'chunk_type': 'slot',
        'title': f'Slot: {slot_name}',
        'content': content.strip(),
        'metadata': {
            'section': 'slots',
            'slot_name': slot_name,
            'category': category,
            'keywords': ['slots', 'properties', category.lower(), slot_name.replace(' ', '_')]
        }
    }

def create_class_chunk(class_name: str, class_info: Dict, group: str) -> Dict[str, Any]:
    """Create a chunk for a class definition."""
    content = f"Class: {class_name}\n\n"
    
    if 'description' in class_info:
        content += f"Description: {class_info['description']}\n\n"
    
    if 'is_a' in class_info:
        content += f"Is a: {class_info['is_a']}\n"
    
    if 'mixins' in class_info:
        if isinstance(class_info['mixins'], (list, tuple)):
            content += f"Mixins: {', '.join(class_info['mixins'])}\n"
        else:
            content += f"Mixins: {class_info['mixins']}\n"
    
    if 'exact_mappings' in class_info:
        if isinstance(class_info['exact_mappings'], (list, tuple)):
            content += f"Exact mappings: {', '.join(class_info['exact_mappings'])}\n"
        else:
            content += f"Exact mappings: {class_info['exact_mappings']}\n"
    
    if 'id_prefixes' in class_info:
        if isinstance(class_info['id_prefixes'], (list, tuple)):
            content += f"ID prefixes: {', '.join(class_info['id_prefixes'])}\n"
        else:
            content += f"ID prefixes: {class_info['id_prefixes']}\n"
    
    if 'aliases' in class_info:
        if isinstance(class_info['aliases'], (list, tuple)):
            content += f"Aliases: {', '.join(class_info['aliases'])}\n"
        else:
            content += f"Aliases: {class_info['aliases']}\n"
    
    return {
        'chunk_type': 'class',
        'title': f'Class: {class_name}',
        'content': content.strip(),
        'metadata': {
            'section': 'classes',
            'class_name': class_name,
            'group': group,
            'keywords': ['classes', 'entities', group.lower(), class_name.replace(' ', '_')]
        }
    }

def save_chunks_to_file(chunks: List[Dict], output_file: str):
    """Save chunks to a file for inspection."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== CHUNK {i+1} ===\n")
            f.write(f"Type: {chunk['chunk_type']}\n")
            f.write(f"Title: {chunk['title']}\n")
            f.write(f"Keywords: {', '.join(chunk['metadata']['keywords'])}\n")
            f.write(f"Content:\n{chunk['content']}\n\n")

# Main execution
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Extract logical chunks from Biolink model YAML file for RAG systems.')
    parser.add_argument('yaml_file', help='Path to the biolink-model.yaml file')
    parser.add_argument('-o', '--output', default='biolink_chunks.txt',
                        help='Output file path (default: biolink_chunks.txt)')
    args = parser.parse_args()
    
    try:
        chunks = extract_biolink_chunks(args.yaml_file)
        
        print(f"Extracted {len(chunks)} chunks from the biolink model")
        
        # Print summary by chunk type
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        print("\nChunk distribution:")
        for chunk_type, count in chunk_types.items():
            print(f"  {chunk_type}: {count}")
        
        # Save chunks to a text file for inspection
        output_path = args.output
        save_chunks_to_file(chunks, output_path)
        print(f"\nChunks saved to {output_path}")
        
        # Example: Show first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Type: {chunk['chunk_type']}")
            print(f"Title: {chunk['title']}")
            print(f"Content (first 200 chars): {chunk['content'][:200]}...")
    
    except FileNotFoundError:
        print(f"File {args.yaml_file} not found. Please check the path.")
    except Exception as e:
        print(f"Error processing file: {e}")