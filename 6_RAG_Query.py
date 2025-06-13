#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# File        : 6_rag_query.py
# Description : RAG query system that answers questions using ChromaDB collections
# -----------------------------------------------------------------------------

import os
import csv
import logging
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Optional, Union
import tempfile
import shutil

# === Constants and Configuration ===
CHROMA_BASE_DIR = "./ChromaDB"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = "./models"
OLLAMA_MODEL = "qwen3:8b"
MAX_RETRIEVAL_RESULTS = 5
CSV_OUTPUT_DIR = "./rag_results"

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"rag_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure model cache directory exists
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_CACHE_DIR, model_name.split('/')[-1])
        
        try:
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local cache: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
            else:
                logger.info(f"Downloading model '{model_name}'...")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.model.save(local_model_path)
                
            logger.info(f"Embedding model loaded on: {self.device}")
        except Exception as e:
            logger.error(f"Could not load model '{model_name}': {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

def find_website_collection(website: str) -> Optional[str]:
    """Find the most recent ChromaDB collection for a specific website"""
    collections = []
    
    try:
        for item in os.listdir(CHROMA_BASE_DIR):
            full_path = os.path.join(CHROMA_BASE_DIR, item)
            if os.path.isdir(full_path) and item.startswith(f"{website}_"):
                collections.append(full_path)
        
        if not collections:
            logger.warning(f"No ChromaDB collection found for website: {website}")
            return None
        
        # Return the most recent collection (assumed to be last in sorted order)
        return sorted(collections)[-1]
    except Exception as e:
        logger.error(f"Error finding collection for {website}: {e}")
        return None

def find_biolink_collection() -> Optional[str]:
    """Find the biolink_merged collection"""
    try:
        for item in os.listdir(CHROMA_BASE_DIR):
            full_path = os.path.join(CHROMA_BASE_DIR, item)
            if os.path.isdir(full_path) and "biolink" in item:
                return full_path
        return None
    except Exception as e:
        logger.error(f"Error finding biolink collection: {e}")
        return None

def merge_collections(website_path: str, biolink_path: str) -> str:
    """Create a temporary merged collection"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="merged_chromadb_")
        logger.info(f"Created temporary directory for merged collection: {temp_dir}")
        
        # Setup clients
        website_client = PersistentClient(path=website_path)
        biolink_client = PersistentClient(path=biolink_path)
        merged_client = PersistentClient(path=temp_dir)
        
        # Get collections
        website_collections = website_client.list_collections()
        biolink_collections = biolink_client.list_collections()
        
        if not website_collections or not biolink_collections:
            logger.error("One or both collections are empty")
            return ""
            
        website_collection = website_client.get_collection(website_collections[0].name)
        biolink_collection = biolink_client.get_collection(biolink_collections[0].name)
        
        # Create new merged collection
        merged_collection = merged_client.create_collection(name="merged_collection")
        
        # Add documents from both collections
        for source_name, collection in [("website", website_collection), ("biolink", biolink_collection)]:
            batch_size = 100
            count = collection.count()
            
            for i in range(0, count, batch_size):
                items = collection.get(limit=batch_size, offset=i)
                
                # Add source identifier to metadata
                for metadata in items["metadatas"]:
                    metadata["source"] = source_name
                
                merged_collection.add(
                    documents=items["documents"],
                    metadatas=items["metadatas"],
                    ids=items["ids"],
                    embeddings=items["embeddings"]
                )
                
            logger.info(f"Added {count} items from {source_name}")
        
        return temp_dir
        
    except Exception as e:
        logger.error(f"Failed to merge collections: {e}")
        return ""

def query_and_answer(question: str, chroma_path: str, embedder: Embedder) -> str:
    """Query the ChromaDB collection and generate an answer"""
    try:
        # Connect to ChromaDB
        client = PersistentClient(path=chroma_path)
        collections = client.list_collections()
        
        if not collections:
            return f"Error: No collections found in {chroma_path}"
        
        collection = client.get_collection(name=collections[0].name)
        logger.info(f"Querying collection: {collections[0].name}")
        
        # Embed the question
        question_embedding = embedder.embed_text(question)
        
        # Query for similar documents
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=MAX_RETRIEVAL_RESULTS,
            include=["documents", "metadatas"]
        )
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # Combine documents into context
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            title = meta.get("title", "No title")
            source = meta.get("source", "website")
            context_parts.append(f"[Document {i+1} - Title: {title} - Source: {source}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with Ollama
        prompt = f"""
You are a helpful assistant answering questions based on retrieved documents.
Answer the following question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Your answer should be concise, no more than 5 sentences.
"""
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )
        
        return response.json()['response'].strip()
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: {str(e)}"

def run_rag_queries(websites: List[str], questions_file: str) -> None:
    """Process all questions for each website"""
    start_time = time.time()
    
    # Initialize embedder
    embedder = Embedder()
    
    # Create output directory
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    
    # Read questions dataframe
    try:
        question_base_df = pd.read_csv(questions_file)
        logger.info(f"Loaded {len(question_base_df)} questions from {questions_file}")
    except Exception as e:
        logger.error(f"Failed to load questions file: {e}")
        return
    
    # Process each website
    for website in websites:
        logger.info(f"Processing website: {website}")
        
        # Find website collection
        website_collection = find_website_collection(website)
        if not website_collection:
            logger.error(f"No collection found for website: {website}")
            continue
        
        # Find biolink collection
        biolink_collection = find_biolink_collection()
        if not biolink_collection:
            logger.warning("No biolink collection found")
        
        # Process questions and store results
        results = []
        
        for index, row in question_base_df.iterrows():
            question = row[0]
            search_type = row[1]
            logger.info(f"Processing question {index+1}: {question[:50]}...")
            
            temp_merged_dir = None
            
            try:
                # Determine which collection to query
                if search_type == "website":
                    collection_to_use = website_collection
                elif search_type == "website+biolink" and biolink_collection:
                    # Create temporary merged collection
                    temp_merged_dir = merge_collections(website_collection, biolink_collection)
                    if temp_merged_dir:
                        collection_to_use = temp_merged_dir
                    else:
                        logger.error("Failed to create merged collection")
                        continue
                else:
                    logger.error(f"Invalid search type or missing biolink collection: {search_type}")
                    continue
                
                # Generate answer
                answer = query_and_answer(question, collection_to_use, embedder)
                results.append([question, search_type, answer])
                logger.info(f"Answer generated for question {index+1}")
            
            finally:
                # Clean up temporary directory if created
                if temp_merged_dir and os.path.exists(temp_merged_dir):
                    shutil.rmtree(temp_merged_dir)
        
        # Save results to CSV
        output_file = os.path.join(CSV_OUTPUT_DIR, f"{website}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Search Type', 'Answer'])
            writer.writerows(results)
            
        logger.info(f"Results saved to {output_file}")
    
    logger.info(f"All processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG query system for multiple websites")
    parser.add_argument("--questions", "-q", required=True, help="Path to questions CSV file")
    parser.add_argument("--websites", "-w", required=True, nargs="+", help="List of websites to process")
    
    args = parser.parse_args()
    
    run_rag_queries(args.websites, args.questions)