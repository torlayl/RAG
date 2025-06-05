# -----------------------------------------------------------------------------
# File        : vector_indexing.py
# Description : Indexes enriched JSONL into ChromaDB with GPU-accelerated embeddings
# -----------------------------------------------------------------------------

# Move all imports to the top
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Generator, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from tqdm import tqdm

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"indexing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Embedding Configuration ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_WEIGHT = 0.8
KEYWORDS_WEIGHT = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_INDEX_BATCH_SIZE = 1000

# Add validation for weights
if SUMMARY_WEIGHT + KEYWORDS_WEIGHT != 1.0:
    logger.warning(f"⚠️ Weights don't sum to 1.0: {SUMMARY_WEIGHT + KEYWORDS_WEIGHT}")

# === Custom Fast GPU Embedder ===
class GPUEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"✅ Embedding model '{model_name}' loaded on: {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model '{model_name}': {e}")
            raise
        
        # Add GPU memory logging
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"📊 GPU Memory Available: {gpu_memory:.1f} GB")

    def __call__(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("⚠️ Empty text list provided")
            return np.array([])
            
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
            logger.error(f"❌ Failed to generate embeddings: {e}")
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

# === Generate Weighted Embeddings in Batch ===
def generate_weighted_embeddings(
    embedder: GPUEmbedder, 
    summaries: List[str], 
    keywords_list: List[str], 
    log_samples: int = 5
) -> np.ndarray:
    logger.info("⚙️ Encoding summaries in batch...")
    summary_embs = embedder(summaries)
    logger.info("⚙️ Encoding keywords in batch...")
    keyword_embs = embedder(keywords_list)

    # Convert to numpy arrays and use numpy operations for better performance
    summary_embs = np.array(summary_embs)
    keyword_embs = np.array(keyword_embs)
    weighted_embeddings = SUMMARY_WEIGHT * summary_embs + KEYWORDS_WEIGHT * keyword_embs
    
    # Log sample norms
    for idx in range(min(log_samples, len(weighted_embeddings))):
        norm = np.linalg.norm(weighted_embeddings[idx])
        logger.info(f"🔍 Norm of embedding for sample {idx + 1}: {norm:.4f}")
    
    return weighted_embeddings

# === Indexer Function ===
def index_vector_store(
    jsonl_input_path: str,
    chroma_collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> bool:
    logger.info("🧠 Starting vector indexing...")
    
    # Add comprehensive input validation
    if not os.path.exists(jsonl_input_path):
        logger.error(f"❌ Input file does not exist: {jsonl_input_path}")
        return False
        
    if not os.path.isfile(jsonl_input_path):
        logger.error(f"❌ Path is not a file: {jsonl_input_path}")
        return False
        
    if batch_size <= 0:
        logger.error("❌ Batch size must be positive")
        return False
        
    if chunk_size <= 0:
        logger.error("❌ Chunk size must be positive")
        return False
    
    index_start = time.time()

    # Generate collection name and directory from JSONL filename if not provided
    if chroma_collection_name is None or persist_directory is None:
        base_name = os.path.splitext(os.path.basename(jsonl_input_path))[0]
        if chroma_collection_name is None:
            chroma_collection_name = f"{base_name}_collection"
        if persist_directory is None:
            persist_directory = f"./chroma_db_{base_name}"
    
    logger.info(f"📂 Using collection name: {chroma_collection_name}")
    logger.info(f"📁 Using persist directory: {persist_directory}")

    embedder = GPUEmbedder()
    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=None  # We supply embeddings manually
    )

    total_processed = 0
    
    try:
        # Process file in chunks to avoid memory issues
        for chunk_docs in process_in_chunks(jsonl_input_path, chunk_size):
            documents, metadatas, ids = [], [], []
            summaries, keywords_list = [], []
            
            # Process current chunk
            for entry in chunk_docs:
                summary = entry.get("summary", "")
                keywords = ", ".join(entry["keywords"]) if isinstance(entry["keywords"], list) else entry.get("keywords", "")

                summaries.append(summary)
                keywords_list.append(keywords)
                documents.append(entry['text'])
                metadatas.append({
                    "title": entry['title'],
                    "url": entry['url'],
                    "keywords": keywords,
                    "summary": summary,
                    "web_path": entry['web_path']                })
                ids.append(entry['id'])

            chunk_total = len(documents)
            total_processed += chunk_total
            logger.info(f"📦 Processing chunk with {chunk_total} documents (total: {total_processed})")
            
            # Generate embeddings for current chunk
            embeddings = generate_weighted_embeddings(embedder, summaries, keywords_list)
            
            # Index current chunk in batches
            num_batches = (chunk_total + batch_size - 1) // batch_size
            for i in tqdm(range(0, chunk_total, batch_size), desc=f"Indexing chunk (batch size: {batch_size})"):
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embs = embeddings[i:i + batch_size].tolist()
                
                try:
                    collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids, embeddings=batch_embs)
                    batch_num = i // batch_size + 1
                    logger.info(f"✅ Indexed batch {batch_num}/{num_batches} — {len(batch_docs)} documents")
                except Exception as e:
                    logger.error(f"❌ Failed to index batch {batch_num}: {e}")
                    raise

        index_end = time.time()
        logger.info(f"✅ Vector store built in {index_end - index_start:.2f} seconds")
        logger.info(f"📚 Total documents indexed: {total_processed}")
        # Comprehensive metrics tracking
        logger.info(f"⚡ Processing rate: {total_processed / (index_end - index_start):.1f} docs/sec")
        
        # Calculate directory size
        try:
            dir_size = sum(os.path.getsize(os.path.join(persist_directory, f)) 
                          for f in os.listdir(persist_directory) 
                          if os.path.isfile(os.path.join(persist_directory, f)))
            logger.info(f"💾 Vector store persisted under: {persist_directory} ({dir_size / 1024**2:.1f} MB)")
        except Exception:
            logger.info(f"💾 Vector store persisted under: {persist_directory}")

    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {jsonl_input_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to index vector store: {e}")
        return False

    logger.info("💾 Persisted vector store to disk.")
    return True  # on success

# Consider processing in chunks instead of loading everything at once
def process_in_chunks(jsonl_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process large JSONL files in chunks to avoid memory issues.
    
    Args:
        jsonl_path: Path to the JSONL file
        chunk_size: Number of documents per chunk
        
    Yields:
        List of document dictionaries
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        chunk_docs = []
        line_number = 0
        
        for line in f:
            line_number += 1
            try:
                entry = json.loads(line)
                
                # Validate required fields
                required_fields = ['text', 'title', 'url', 'id', 'web_path']
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    logger.warning(f"⚠️ Missing fields at line {line_number}: {missing_fields}")
                    continue
                
                chunk_docs.append(entry)
                
                # Process chunk when it reaches the specified size
                if len(chunk_docs) >= chunk_size:
                    yield chunk_docs
                    chunk_docs = []
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON at line {line_number}: {e}")
                continue
        
        # Process remaining documents
        if chunk_docs:
            yield chunk_docs

# === Entry Point ===
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Index JSONL files into ChromaDB vector store")
    parser.add_argument("jsonl_path", nargs="?", default="enriched_pages.jsonl", 
                       help="Path to the JSONL file to index")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INDEX_BATCH_SIZE,
                       help=f"Batch size for indexing (default: {DEFAULT_INDEX_BATCH_SIZE})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f"Chunk size for processing large files (default: {DEFAULT_CHUNK_SIZE})")
    
    args = parser.parse_args()
    
    start_time = time.time()
    success = index_vector_store(
        jsonl_input_path=args.jsonl_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    total = time.time() - start_time
    
    if success:
        logger.info(f"🎉 Indexing process completed successfully in {total:.2f} seconds")
    else:
        logger.error(f"❌ Indexing process failed after {total:.2f} seconds")
        sys.exit(1)
