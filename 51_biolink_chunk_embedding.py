#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# File        : biolink_chunk_embedding.py
# Description : Indexes biolink_chunks.txt text content into ChromaDB with 
#               GPU-accelerated embeddings.
# -----------------------------------------------------------------------------

import os
import re
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
DEFAULT_INDEX_BATCH_SIZE = 100
CHROMA_BASE_DIR = "./ChromaDB"
MODEL_CACHE_DIR = "./models"  # Directory to store downloaded models
BIOLINK_CHUNK_FILE = "/home/ltor/Nextcloud/LIG/DAISY/RAG/Biolink_model/biolink_chunks.txt"

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"biolink_indexing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# === Process Biolink Chunks ===
def parse_biolink_chunks(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Parse the biolink_chunks.txt file and extract structured chunk data.
    
    Args:
        file_path: Path to the biolink_chunks.txt file
        
    Yields:
        Dictionary with chunk data (id, type, title, keywords, content)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the file into chunks using the chunk delimiter pattern
        chunks = content.split("=== CHUNK ")
        # Remove the first empty element (before first delimiter)
        if chunks and not chunks[0].strip():
            chunks.pop(0)
        
        total_chunks = len(chunks)
        logger.info(f"📊 Total chunks found in file: {total_chunks}")
        
        # Process each chunk
        for i, chunk_text in enumerate(chunks, 1):
            if not chunk_text.strip():
                continue
            
            # Create chunk content with the delimiter restored (except for first instance)
            full_content = chunk_text
            
            # Extract chunk number from first line
            chunk_num = chunk_text.split("===", 1)[0].strip()
            
            # Create a unique ID for the chunk
            chunk_id = f"biolink-chunk-{chunk_num}"
            
            # Try to extract minimal metadata without changing content structure
            lines = chunk_text.split("\n", 4)
            chunk_type = lines[1].replace("Type:", "").strip() if len(lines) > 1 else ""
            title = lines[2].replace("Title:", "").strip() if len(lines) > 2 else ""
            keywords = lines[3].replace("Keywords:", "").strip().split(',') if len(lines) > 3 else []
            keywords = [kw.strip() for kw in keywords]
            
            # Yield the chunk data
            yield {
            "id": chunk_id,
            "type": chunk_type,
            "title": title,
            "keywords": keywords,
            "content": full_content.strip()
            }
            
    except Exception as e:
        logger.error(f"❌ Error processing Biolink chunks file: {e}")
        raise

# === Index Biolink chunks to vector store ===
def index_biolink_chunks(
    input_file_path: str = BIOLINK_CHUNK_FILE,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE
) -> bool:
    logger.info("🧠 Starting Biolink chunks vector indexing...")
    
    # Input validation
    if not os.path.exists(input_file_path):
        logger.error(f"❌ Input file does not exist: {input_file_path}")
        return False
        
    if not os.path.isfile(input_file_path):
        logger.error(f"❌ Path is not a file: {input_file_path}")
        return False
    
    index_start = time.time()

    # Generate collection name and directory
    current_date = datetime.now().strftime('%Y%m%d')
    
    if collection_name is None:
        collection_name = f"biolink_model_{current_date}_collection"
        
    # Set custom persist directory if not provided
    if persist_directory is None:
        persist_directory = os.path.join(CHROMA_BASE_DIR, f"biolink_model_{current_date}")
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    logger.info(f"📂 Using collection name: {collection_name}")
    logger.info(f"📁 Using persist directory: {persist_directory}")

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
        # Prepare batches for processing
        documents = []
        metadatas = []
        ids = []
        
        # Extract chunks from the file
        for chunk in parse_biolink_chunks(input_file_path):
            # Extract content from the chunk
            documents.append(chunk["content"])
            
            # Include valuable metadata
            metadatas.append({
                "title": chunk["title"],
                "type": chunk["type"],
                "keywords": ", ".join(chunk["keywords"]),
                "id": chunk["id"]
            })
            
            ids.append(chunk["id"])
            
            # Process in batches
            if len(documents) >= batch_size:
                batch_total = len(documents)
                total_processed += batch_total
                logger.info(f"📦 Processing batch with {batch_total} documents (total: {total_processed})")
                
                # Generate embeddings for the content
                logger.info("⚙️ Encoding chunk content...")
                embeddings = embedder(documents)
                
                # Convert embeddings to list if they're already numpy arrays
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                
                # Add to collection
                try:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                    logger.info(f"✅ Indexed batch — {batch_total} documents")
                except Exception as e:
                    logger.error(f"❌ Failed to index batch: {e}")
                    raise
                
                # Reset for next batch
                documents = []
                metadatas = []
                ids = []
        
        # Process any remaining documents
        if documents:
            batch_total = len(documents)
            total_processed += batch_total
            logger.info(f"📦 Processing final batch with {batch_total} documents (total: {total_processed})")
            
            # Generate embeddings
            logger.info("⚙️ Encoding chunk content...")
            embeddings = embedder(documents)
            
            # Convert embeddings to list if they're already numpy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Add to collection
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                logger.info(f"✅ Indexed final batch — {batch_total} documents")
            except Exception as e:
                logger.error(f"❌ Failed to index final batch: {e}")
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

        # Save chunk metadata to a JSON file for reference
        try:
            all_chunks = list(parse_biolink_chunks(input_file_path))
            metadata_file = os.path.join(persist_directory, "chunk_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2)
            logger.info(f"📄 Chunk metadata saved to: {metadata_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save chunk metadata: {e}")

    except Exception as e:
        logger.error(f"❌ Failed to index vector store: {e}")
        return False

    logger.info("🎉 Biolink model vector indexing completed successfully")
    return True

# === Entry Point ===
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Index biolink_chunks.txt into ChromaDB vector store")
    parser.add_argument("--input-file", type=str, default=BIOLINK_CHUNK_FILE, 
                       help="Path to the biolink_chunks.txt file to index")
    parser.add_argument("--collection-name", type=str, 
                       help="Name for the ChromaDB collection")
    parser.add_argument("--persist-dir", type=str, default=None,
                       help=f"Base directory to store ChromaDB (default: {CHROMA_BASE_DIR}/biolink_model_<date>)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INDEX_BATCH_SIZE,
                       help=f"Batch size for indexing (default: {DEFAULT_INDEX_BATCH_SIZE})")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    success = index_biolink_chunks(
        input_file_path=args.input_file,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        batch_size=args.batch_size
    )
    total_time = time.time() - start_time
    
    if success:
        logger.info(f"🎉 Indexing process completed successfully in {total_time:.2f} seconds")
    else:
        logger.error(f"❌ Indexing process failed after {total_time:.2f} seconds")
        sys.exit(1)
