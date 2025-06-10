#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# File        : 5_text_content_embedding_and_storage.py
# Description : Indexes text_content from JSON into website-specific ChromaDB with 
#               GPU-accelerated embeddings. Uses website name and date for collection naming.
# -----------------------------------------------------------------------------

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from tqdm import tqdm

# === Constants and Configuration ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_INDEX_BATCH_SIZE = 100
CHROMA_BASE_DIR = "./ChromaDB"
MODEL_CACHE_DIR = "./models"  # Directory to store downloaded models

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"text_content_indexing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Custom Fast GPU Embedder ===
class GPUEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        # Ensure model cache directory exists
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_CACHE_DIR, model_name.split('/')[-1])
        
        try:
            if os.path.exists(local_model_path):
                # Load model from local cache
                logger.info(f"📂 Loading model from local cache: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
                logger.info(f"✅ Local embedding model loaded on: {self.device}")
            else:
                # Download and save the model
                logger.info(f"⬇️ Downloading model '{model_name}' to {local_model_path}...")
                self.model = SentenceTransformer(model_name, device=self.device)
                logger.info(f"✅ Embedding model '{model_name}' loaded on: {self.device}")
                
                # Save the model to local directory
                logger.info(f"💾 Saving model to {local_model_path}...")
                self.model.save(local_model_path)
                logger.info(f"✅ Model saved successfully to {local_model_path}")
            
            # Add GPU memory logging
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"📊 GPU Memory Available: {gpu_memory:.1f} GB")
                
        except Exception as e:
            logger.error(f"❌ Could not load model '{model_name}': {e}")
            raise

    def __call__(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("⚠️ Empty text list provided")
            return np.array([])
        
        # GPU memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before processing
            memory_before = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"📊 GPU Memory Before: {memory_before:.2f} GB")
        
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False, 
                batch_size=self.batch_size
            )
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings with model: {e}")
            raise
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"📊 GPU Memory After: {memory_after:.2f} GB")
        
        if np.all(embeddings == 0):
            logger.warning("⚠️ All generated embeddings are zeros.")
        
        return embeddings

    def __del__(self):
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# === Process JSON chunks ===
def process_json_chunks(json_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process JSON file with results array in chunks to avoid memory issues.
    
    Args:
        json_path: Path to the JSON file
        chunk_size: Number of documents per chunk
        
    Yields:
        List of document dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = data.get('results', [])
        total_chunks = len(results)
        logger.info(f"📊 Total entries in JSON: {total_chunks}")
        
        for i in range(0, total_chunks, chunk_size):
            chunk = results[i:i+chunk_size]
            # Filter out empty or incomplete entries
            valid_chunk = [entry for entry in chunk if entry.get('id') and entry.get('text_content')]
            if valid_chunk:
                yield valid_chunk
                
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"❌ Error processing JSON file: {e}")

# === Index JSON content to vector store ===
def index_text_content(
    json_input_path: str,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> bool:
    logger.info("🧠 Starting text content vector indexing...")
    
    # Input validation
    if not os.path.exists(json_input_path):
        logger.error(f"❌ Input file does not exist: {json_input_path}")
        return False
        
    if not os.path.isfile(json_input_path):
        logger.error(f"❌ Path is not a file: {json_input_path}")
        return False
    
    index_start = time.time()

    # Generate collection name and directory from JSON filename with website name and date
    base_name = os.path.splitext(os.path.basename(json_input_path))[0]
    # Extract website name from the filename (assuming format like "website_chunked_date")
    website_name = base_name.split('_')[0] if '_' in base_name else base_name
    current_date = datetime.now().strftime('%Y%m%d')
    
    if collection_name is None:
        collection_name = f"{website_name}_{current_date}_collection"
        
    # Set custom persist directory with website name and date if not provided
    if persist_directory is None:
        persist_directory = os.path.join(CHROMA_BASE_DIR, f"{website_name}_{current_date}")
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    logger.info(f"📂 Using collection name: {collection_name}")
    logger.info(f"📁 Using persist directory: {persist_directory}")

    # Ensure the persist_directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize embedder and database
    try:
        logger.info(f"⚙️ Initializing embedding model on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        embedder = GPUEmbedder()
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedder: {e}")
        return False
    
    logger.info(f"📦 Opening ChromaDB collection at: {persist_directory}")
    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None  # We supply embeddings manually
    )

    total_processed = 0
    
    try:
        # Process file in chunks
        for chunk_num, chunk_docs in enumerate(process_json_chunks(json_input_path, chunk_size)):
            documents = []
            metadatas = []
            ids = []
            
            # Extract text_content from each document
            for entry in chunk_docs:
                if not entry.get('text_content'):
                    continue
                    
                documents.append(entry['text_content'])
                
                # Include valuable metadata
                metadatas.append({
                    "title": entry.get('title', ''),
                    "web_path": entry.get('web_path', ''),
                    "id": entry.get('id', '')
                })
                
                ids.append(entry['id'])

            chunk_total = len(documents)
            if chunk_total == 0:
                logger.warning(f"⚠️ Chunk {chunk_num+1} has no valid documents, skipping")
                continue
                
            total_processed += chunk_total
            logger.info(f"📦 Processing chunk {chunk_num+1} with {chunk_total} documents (total: {total_processed})")
            
            # Generate embeddings for the text_content
            logger.info("⚙️ Encoding text_content...")
            embeddings = embedder(documents)
            
            # Convert embeddings to list if they're already numpy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Index in batches
            num_batches = (chunk_total + batch_size - 1) // batch_size
            for i in tqdm(range(0, chunk_total, batch_size), desc=f"Indexing chunk (batch size: {batch_size})"):
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embs = embeddings[i:i + batch_size]
                
                try:
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids,
                        embeddings=batch_embs
                    )
                    batch_num = i // batch_size + 1
                    logger.info(f"✅ Indexed batch {batch_num}/{num_batches} — {len(batch_docs)} documents")
                except Exception as e:
                    logger.error(f"❌ Failed to index batch {batch_num}: {e}")
                    raise

        # Log performance metrics
        index_end = time.time()
        elapsed_time = index_end - index_start
        logger.info(f"✅ Vector store built in {elapsed_time:.2f} seconds")
        logger.info(f"📚 Total documents indexed: {total_processed}")
        
        if total_processed > 0:
            logger.info(f"⚡ Processing rate: {total_processed / elapsed_time:.1f} docs/sec")
        
        # Calculate directory size
        try:
            dir_size = sum(os.path.getsize(os.path.join(persist_directory, f)) 
                         for f in os.listdir(persist_directory) 
                         if os.path.isfile(os.path.join(persist_directory, f)))
            logger.info(f"💾 Vector store persisted under: {persist_directory} ({dir_size / 1024**2:.1f} MB)")
        except Exception:
            logger.info(f"💾 Vector store persisted under: {persist_directory}")

    except Exception as e:
        logger.error(f"❌ Failed to index vector store: {e}")
        return False

    logger.info("🎉 Text content vector indexing completed successfully")
    return True

# === Entry Point ===
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Index JSON text_content into ChromaDB vector store")
    parser.add_argument("json_path", nargs="?", default="/home/ltor/Nextcloud/LIG/DAISY/RAG/enriched_results_json/biolink_chunked_20250606_153430.json", 
                       help="Path to the JSON file to index")
    parser.add_argument("--collection-name", type=str, 
                       help="Name for the ChromaDB collection")
    parser.add_argument("--persist-dir", type=str, default=CHROMA_BASE_DIR,
                       help=f"Base directory to store ChromaDB (default: {CHROMA_BASE_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INDEX_BATCH_SIZE,
                       help=f"Batch size for indexing (default: {DEFAULT_INDEX_BATCH_SIZE})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f"Chunk size for processing large files (default: {DEFAULT_CHUNK_SIZE})")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Extract website name from filename for custom directory structure
    if args.persist_dir == CHROMA_BASE_DIR and not args.collection_name:
        input_base_name = os.path.splitext(os.path.basename(args.json_path))[0]
        website_name = input_base_name.split('_')[0] if '_' in input_base_name else input_base_name
        current_date = datetime.now().strftime('%Y%m%d')
        custom_dir = os.path.join(CHROMA_BASE_DIR, f"{website_name}_{current_date}")
    else:
        custom_dir = args.persist_dir
        
    success = index_text_content(
        json_input_path=args.json_path,
        collection_name=args.collection_name,
        persist_directory=custom_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"🎉 Indexing process completed successfully in {total_time:.2f} seconds")
    else:
        logger.error(f"❌ Indexing process failed after {total_time:.2f} seconds")
        sys.exit(1)
