# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# Organization: Centrale Méditerranée
# File        : search_engine_webApp.py
# Description : RAG (Retrieval-Augmented Generation) Web Application
#               Supports RAG-only, LLM-only, and hybrid query modes.
#               Features:
#               - Multi-database ChromaDB support with merging capabilities
#               - Multiple LLM models via Ollama API
#               - Bilingual interface (English/French)
#               - PDF export functionality
#               - Real-time performance monitoring
#               - Comprehensive error handling and logging
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.6 (Enhanced with validation, rate limiting, and improved UX)
# Dependencies: See requirements.txt
# Usage       : python search_engine_webApp.py [--db DATABASE_PATH] [--port PORT]
# -----------------------------------------------------------------------------

"""
RAG Web Application

A sophisticated web-based Retrieval-Augmented Generation system that combines
document search capabilities with Large Language Models to provide intelligent
question-answering functionality.

Key Features:
- Multiple ChromaDB database support
- Hybrid, RAG-only, and LLM-only query modes
- Bilingual interface (English/French)
- PDF export of conversations
- Real-time performance monitoring
- Comprehensive error handling

Example usage:
    python search_engine_webApp.py --db ./my_database --port 8080
    python search_engine_webApp.py --merge ./db1 ./db2 --debug
"""

import json
import numpy as np
import argparse
import os
import glob
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from chromadb import PersistentClient
from xhtml2pdf import pisa
from markdown2 import markdown
import tempfile
import base64
import requests
import time
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import shutil
import logging
import threading
from collections import defaultdict
import psutil

# --- Configuration ---
class Config:
    def __init__(self):
        self.TOP_K = int(os.getenv("TOP_K", "30"))
        self.THRESHOLD_GOOD = float(os.getenv("THRESHOLD_GOOD", "0.70"))
        self.MAX_CHARS = int(os.getenv("MAX_CHARS", "8000"))
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize config
config = Config()

def validate_config():
    """Validate configuration parameters"""
    issues = []
    if config.TOP_K <= 0:
        issues.append(f"TOP_K must be positive, got {config.TOP_K}")
    if config.THRESHOLD_GOOD < 0 or config.THRESHOLD_GOOD > 1:
        issues.append(f"THRESHOLD_GOOD must be between 0 and 1, got {config.THRESHOLD_GOOD}")
    if config.MAX_CHARS <= 0:
        issues.append(f"MAX_CHARS must be positive, got {config.MAX_CHARS}")
    
    if issues:
        print("Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise ValueError("Invalid configuration parameters")
    return True

TOP_K = config.TOP_K
THRESHOLD_GOOD = config.THRESHOLD_GOOD
DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_LANGUAGE = "EN"
DEFAULT_QUERY_MODE = "rag_only"
MAX_CHARS = config.MAX_CHARS  # Truncate LLM context if too long
TOP_K_RELEVANT = 20  # Number of most relevant documents to inject in the prompt
OLLAMA_URL = config.OLLAMA_URL

# Validate configuration early
validate_config()

# --- Logging Setup ---
os.makedirs('logs', exist_ok=True)  # Ensure logs directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("[SUCCESS] Configuration validation passed")

# --- Global variables for ChromaDB ---
client = None
collection = None
available_databases = []
current_db_info = ""

# --- Database Management Functions ---
def discover_chroma_databases(base_path="."):
    """Discover all available ChromaDB databases in the current directory"""
    databases = []
    
    # Look for directories starting with "chroma_db"
    pattern = os.path.join(base_path, "chroma_db*")
    db_paths = glob.glob(pattern)
    
    for db_path in db_paths:
        if os.path.isdir(db_path):
            try:
                # Try to connect to verify it's a valid ChromaDB
                test_client = PersistentClient(path=db_path)
                collections = test_client.list_collections()
                if collections:
                    db_name = os.path.basename(db_path)
                    collection_names = [col.name for col in collections]
                    databases.append({
                        "name": db_name,
                        "path": db_path,
                        "collections": collection_names
                    })
            except Exception as e:
                print(f"⚠️ Skipping invalid database {db_path}: {e}")
    
    return databases

def load_single_database(db_path, collection_name="web_chunks"):
    """Load a single ChromaDB database with improved error handling"""
    global client, collection, current_db_info
    
    try:
        logger.info(f"Loading database: {db_path}")
        
        # Check if database path exists
        if not os.path.exists(db_path):
            error_msg = f"Database path does not exist: {db_path}"
            logger.error(error_msg)
            return False
            
        client = PersistentClient(path=db_path)
        collections = client.list_collections()
        
        if not collections:
            logger.warning(f"No collections found in database: {db_path}")
            return False
        
        # Try to find the requested collection or use the first available
        target_collection = None
        available_names = [col.name for col in collections]
        
        for col in collections:
            if col.name == collection_name:
                target_collection = col
                break
        
        if not target_collection:
            target_collection = collections[0]
            original_name = collection_name
            collection_name = target_collection.name
            logger.info(f"Requested collection '{original_name}' not found. Available: {available_names}. Using '{collection_name}'")
        
        collection = client.get_collection(name=collection_name)
        doc_count = collection.count()
        current_db_info = f"Loaded: {os.path.basename(db_path)} (collection: {collection_name}, {doc_count:,} documents)"
        logger.info(f"Successfully loaded database: {db_path} with {doc_count:,} documents")
        return True
            
    except Exception as e:
        error_msg = f"Error loading database {db_path}: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False

def merge_databases(db_paths, collection_name="web_chunks"):
    """Merge multiple ChromaDB databases into a single collection with improved resource management"""
    global client, collection, current_db_info
    
    temp_db_path = "./temp_merged_db"
    
    try:
        # Clean up any existing temporary database
        cleanup_temp_databases()
        
        logger.info(f"Merging {len(db_paths)} databases")
        
        # Create a new client for the merged collection
        client = PersistentClient(path=temp_db_path)
        
        # Create a new collection for merged data
        try:
            client.delete_collection(name="merged_collection")
        except:
            pass  # Collection doesn't exist, which is fine
        
        merged_collection = client.create_collection(name="merged_collection")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        id_counter = 0
        
        loaded_dbs = []
        
        for db_path in db_paths:
            try:
                temp_client = PersistentClient(path=db_path)
                collections = temp_client.list_collections()
                
                # Find the target collection
                target_collection = None
                for col in collections:
                    if col.name == collection_name:
                        target_collection = col
                        break
                
                if not target_collection and collections:
                    target_collection = collections[0]
                
                if target_collection:
                    temp_collection = temp_client.get_collection(name=target_collection.name)
                    
                    # Get all data from this collection
                    results = temp_collection.get()
                    
                    if results["documents"]:
                        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
                            # Add source database info to metadata
                            metadata = metadata or {}
                            metadata["source_db"] = os.path.basename(db_path)
                            
                            all_documents.append(doc)
                            all_metadatas.append(metadata)
                            all_ids.append(f"merged_{id_counter}")
                            id_counter += 1
                    
                    loaded_dbs.append(os.path.basename(db_path))
                    print(f"✅ Merged data from: {db_path}")
                
            except Exception as e:
                print(f"⚠️ Error merging database {db_path}: {e}")
        
        if all_documents:
            # Add all documents to the merged collection in batches
            batch_size = 1000
            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i+batch_size]
                batch_metas = all_metadatas[i:i+batch_size]
                batch_ids = all_ids[i:i+batch_size]
                
                merged_collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            collection = merged_collection
            current_db_info = f"Merged databases: {', '.join(loaded_dbs)} ({len(all_documents)} documents)"
            print(f"✅ Successfully merged {len(loaded_dbs)} databases with {len(all_documents)} total documents")
            return True
        else:
            print("❌ No documents found in any of the databases")
            return False
            
    except Exception as e:
        print(f"❌ Error merging databases: {e}")
        return False

def get_database_options():
    """Get available database options for the UI"""
    global available_databases
    
    options = []
    
    # Single database options
    for db in available_databases:
        options.append({
            "label": f"{db['name']} ({len(db['collections'])} collections)",
            "value": db['path']
        })
    
    # Merged database options (all combinations of 2+ databases)
    if len(available_databases) >= 2:
        options.append({"label": "--- Merged Options ---", "value": "", "disabled": True})
        
        # Add option to merge all databases
        all_paths = [db['path'] for db in available_databases]
        all_names = [db['name'] for db in available_databases]
        options.append({
            "label": f"Merge All ({len(available_databases)} databases)",
            "value": "|".join(all_paths)
        })
    
    return options

# --- Embedding Utility (cached + in-memory) ---
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=500)
def cached_embed(text):
    return embed_model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

def embed_texts(texts):
    return [cached_embed(text) for text in texts]

# --- Validation Functions ---
def validate_ollama_connection():
    """Validate that Ollama service is available"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

# --- Rate Limiting ---
# Rate limiting storage
last_request_time = defaultdict(float)
request_lock = threading.Lock()

def check_rate_limit(identifier="default", min_interval=2.0):
    """Simple rate limiting - minimum interval between requests"""
    with request_lock:
        current_time = time.time()
        last_time = last_request_time[identifier]
        
        if current_time - last_time < min_interval:
            wait_time = min_interval - (current_time - last_time)
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
        
        last_request_time[identifier] = time.time()

# --- LLM Call ---
def call_ollama_llm(prompt, model, temperature=0.1):
    """Call Ollama LLM with improved error handling and logging"""
    try:
        # Apply rate limiting
        check_rate_limit("ollama_api", min_interval=1.0)
        
        # Validate connection first
        if not validate_ollama_connection():
            logger.error("Ollama service is not available")
            return "❌ Ollama service is not available. Please ensure Ollama is running."
        
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature}
        }
        
        logger.info(f"Calling Ollama with model: {model}, temperature: {temperature}")
        
        # Check rate limit before the request
        check_rate_limit("ollama_llm_call", min_interval=2.0)  # 2 seconds interval
        
        response = requests.post(f"{OLLAMA_URL}/api/generate", 
                                json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            answer_parts = []
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode("utf-8"))
                        if "response" in json_line:
                            answer_parts.append(json_line["response"])
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in line: {line}")
            
            result = ''.join(answer_parts)
            logger.info(f"Successfully received response from Ollama ({len(result)} characters)")
            return result
        else:
            error_msg = f"LLM API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "❌ Request timed out. The model might be taking too long to respond."
        logger.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Connection error. Please check if Ollama is running."
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"LLM API exception: {str(e)}"
        logger.error(error_msg)
        return error_msg

# --- Traductions pour l'interface ---
def get_translations(lang):
    if lang == "FR":
        return {
            "ask_placeholder": "Posez votre question...",
            "model_label": "Modèle",
            "language_label": "Langue",
            "mode_label": "Mode « Assistant »",
            "mode_tooltip": "Choisissez comment l'assistant doit répondre :\n- Hybride : RAG avec complément LLM\n- RAG seul : uniquement depuis les documents\n- LLM seul : uniquement les connaissances du LLM",
            "submit": "Soumettre",
            "clear": "Effacer",
            "show_sources": "Afficher les sources",
            "you": "🧑 Vous",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources utilisées :",
            "download_pdf": "📄 Télécharger PDF",
            "no_question": "❗ Veuillez entrer une question.",
            "no_relevant_docs": "⚠️ Aucun document pertinent trouvé."
        }
    else:
        return {
            "ask_placeholder": "Ask your question...",
            "model_label": "Model",
            "language_label": "Language",
            "mode_label": "Mode",
            "mode_tooltip": "Choose how the assistant should answer:\n- Hybrid: RAG with LLM enhancement\n- RAG Only: answer only from documents\n- LLM Only: use only the LLM's internal knowledge",
            "submit": "Submit",
            "clear": "Clear Output",
            "show_sources": "Show sources",
            "you": "🧑 You",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources used:",
            "download_pdf": "📄 Download PDF",
            "no_question": "❗ Please enter a question.",
            "no_relevant_docs": "⚠️ No relevant documents found."
        }

# --- Adaptation dynamique du prompt ---
def process_query(user_question, llm_model, lang, mode=DEFAULT_QUERY_MODE):
    start_time = time.time()
    
    # Sanitize input
    user_question = sanitize_query(user_question)
    
    temperature = 0.1 if mode == "rag_only" else 0.4 if mode == "hybrid" else 0.7

    translations = get_translations(lang)

    if lang == "FR":
        intro = "Vous êtes un assistant expert qui aide les utilisateurs à comprendre de la documentation technique."
        instruction_rag = "Fournissez une réponse détaillée, structurée et pratique uniquement à partir de la documentation fournie."
        instruction_hybrid = "Ajoutez des compléments issus du LLM si pertinent, en les identifiant clairement."
    else:
        intro = "You are an expert assistant helping users understand technical documentation."
        instruction_rag = "Provide a detailed, structured, and practical answer using only the provided documentation."
        instruction_hybrid = "If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM."

    if mode == "llm_only":
        prompt = f"""
        {intro}

        Question: {user_question}

        Answer strictly using the LLM's internal knowledge.
        """
        duration = time.time() - start_time
        return call_ollama_llm(prompt, llm_model, temperature=temperature), [], duration

    query_emb = embed_texts([user_question])[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    relevant_data = [
        (doc, meta, score)
        for doc, meta, score in zip(docs, metas, scores)
        if score >= THRESHOLD_GOOD
    ]

    if not relevant_data:
        duration = time.time() - start_time
        return translations["no_relevant_docs"], [], duration    # Keep only the N most relevant documents
    relevant_data = sorted(relevant_data, key=lambda x: x[2], reverse=True)[:TOP_K_RELEVANT]

    page_map = {}
    for doc, meta, score in relevant_data:
        url = meta.get("url", "")
        if url not in page_map:
            page_map[url] = {"text": doc, "score": round(score, 4), "metadata": meta}
        else:
            page_map[url]["text"] += "\n" + doc

    page_contexts = [
        {"url": url, "text": data["text"], "score": data["score"], "metadata": data["metadata"]}
        for url, data in page_map.items()
    ]

    all_text = "\n\n".join(p["text"] for p in page_contexts)[:MAX_CHARS]
    all_urls = [p["url"] for p in page_contexts]

    prompt = f"""
{intro}

Sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation:
{all_text}

Question: {user_question}

{instruction_rag}
"""
    if mode == "hybrid":
        prompt += f"\n{instruction_hybrid}"

    duration = time.time() - start_time
    return call_ollama_llm(prompt, llm_model, temperature=temperature), page_contexts, duration


# --- PDF with markdown rendering ---
def generate_pdf(content, lang):
    """Generate a PDF from markdown content with proper error handling and cleanup"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            html_template = f"""
            <html>
            <head>
                <meta charset='UTF-8'>
                <style>
                    body {{ font-family: Helvetica, sans-serif; line-height: 1.4; font-size: 12pt; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    code {{ background-color: #f4f4f4; padding: 2px 4px; }}
                </style>
            </head>
            <body>{markdown(content)}</body>
            </html>
            """
            pisa_status = pisa.CreatePDF(html_template, dest=tmp_file)
            
            if pisa_status.err:
                logger.warning(f"PDF generation had errors: {pisa_status.err}")
            
            tmp_file.seek(0)
            pdf_data = tmp_file.read()
            tmp_file_path = tmp_file.name
        
        # Clean up the temporary file
        try:
            os.unlink(tmp_file_path)
        except OSError as e:
            logger.warning(f"Failed to clean up temporary PDF file: {e}")
        
        return html.A(
            get_translations(lang)["download_pdf"],
            href=f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode('utf-8')}",
            download="rag_answer.pdf",
            target="_blank",
            className="btn btn-outline-info mt-3"
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return html.Div(f"❌ Error generating PDF: {str(e)}", className="text-danger")

# Add memory monitoring
def get_memory_usage():
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return "psutil not available"

# --- Health Check Functions ---
def perform_startup_health_checks():
    """Perform health checks during startup"""
    logger.info("Performing startup health checks...")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False
      # Check Ollama connection
    if validate_ollama_connection():
        logger.info("[SUCCESS] Ollama service is available")
        print("[SUCCESS] Ollama service is available")
    else:
        logger.warning("⚠️ Ollama service is not available")
        print("⚠️ Ollama service is not available. Some features may not work.")
      # Check embedding model
    try:
        test_embedding = cached_embed("test")
        logger.info("[SUCCESS] Embedding model loaded successfully")
        print("[SUCCESS] Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading embedding model: {e}")
        print(f"❌ Error loading embedding model: {e}")
        return False
    
    # Check memory usage
    memory_usage = get_memory_usage()
    if isinstance(memory_usage, (int, float)):
        logger.info(f"📊 Current memory usage: {memory_usage:.1f} MB")
        print(f"📊 Current memory usage: {memory_usage:.1f} MB")
    
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = {
        'dash': 'Dash',
        'dash_bootstrap_components': 'Dash Bootstrap Components',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'Sentence Transformers',
        'xhtml2pdf': 'xhtml2pdf',
        'markdown2': 'markdown2',
        'requests': 'requests',
        'numpy': 'NumPy'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"[SUCCESS] {name} is available")
        except ImportError:
            missing_packages.append(name)
            logger.error(f"❌ {name} is not installed")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    logger.info("[SUCCESS] All required dependencies are available")
    return True

# --- Dash App Setup ---
# Perform health checks
if not perform_startup_health_checks():
    logger.error("❌ Startup health checks failed. Exiting.")
    print("❌ Startup health checks failed. Please fix the issues above before running the application.")
    exit(1)

# Initialize available databases
available_databases = discover_chroma_databases()
print(f"🔍 Found {len(available_databases)} ChromaDB databases:")
for db in available_databases:
    print(f"  - {db['name']}: {db['collections']}")

# Load default database if available
if available_databases:
    default_db = available_databases[0]["path"]
    if load_single_database(default_db):
        print("[SUCCESS] Default database loaded successfully")
    else:
        print("⚠️ Failed to load default database")
else:
    logger.warning("No ChromaDB databases found")
    print("⚠️ No ChromaDB databases found. Please ensure you have valid ChromaDB databases in the directory.")

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "RAG Assistant"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🤖 RAG Assistant", className="text-primary"), width=8),
        dbc.Col(html.Img(src="/assets/logo_centrale.svg", height="60px"), width=4, style={"textAlign": "right"})
    ], align="center"),

    html.Hr(),    # Database selection row
    dbc.Row([
        dbc.Col([
            html.Label("Database Selection", className="text-info fw-bold"),
            dbc.Select(
                id="database-selector",
                options=get_database_options() if available_databases else [{"label": "No databases found", "value": "", "disabled": True}],
                value=available_databases[0]["path"] if available_databases else "",
                className="mb-2"
            ),
            html.Div(id="database-status", className="text-muted", style={"fontSize": "0.85em"},
                    children=current_db_info if current_db_info else "No database loaded")
        ], width=12)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Textarea(id="question-input", placeholder="Ask your question...", style={"height": "120px"}, className="mb-2"), width=8),
        dbc.Col([
            html.Label("Model", className="text-info fw-bold"),
            dbc.Select(
                id="llm-selector",
                options=[
                    {"label": "Qwen 3 (8b)", "value": "qwen3:8b"},
                    {"label": "Gemma 3 (4b)", "value": "gemma3:4b"},
                    {"label": "Mistral 7B", "value": "mistral:7b"}
                ],
                value=DEFAULT_LLM_MODEL,
                className="mb-3"
            ),
            html.Label("Language", className="text-info fw-bold"),
            dbc.Select(
                id="lang-selector",
                options=[
                    {"label": "English", "value": "EN"},
                    {"label": "Français", "value": "FR"}
                ],
                value=DEFAULT_LANGUAGE,
                className="mb-3"
            ),
            html.Label([
                "Mode ",
                html.Span(
                    "ⓘ",
                    id="mode-tooltip",
                    style={"textDecoration": "underline dotted", "cursor": "pointer"},
                    title="Choose how the assistant should answer:\n- Hybrid: RAG with LLM enhancement\n- RAG Only: answer only from documents\n- LLM Only: use only the LLM's internal knowledge"
                )
            ], className="text-info fw-bold"),

            dbc.Select(
                id="mode-selector",
                options=[
                    {"label": "Hybrid (RAG + LLM)", "value": "hybrid"},
                    {"label": "RAG Only", "value": "rag_only"},
                    {"label": "LLM Only", "value": "llm_only"}
                ],
                value=DEFAULT_QUERY_MODE,
                className="mb-3"
            )
        ], width=4)
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit-button", color="success"), width="auto"),
        dbc.Col(dbc.Button("🧹 Clear Output", id="clear-button", color="warning", className="ms-2"), width="auto"),
        dbc.Col(dbc.Checkbox(id="show-sources-toggle", value=True, className="ms-3"), width="auto"),
        dbc.Col(html.Label("Show sources", className="mt-2"), width="auto")
    ], className="my-3", align="center"),

    dcc.Loading(
        id="loading-output",
        type="circle",
        color="#00ff99",
        children=[
            dbc.Card([html.Div(id="chat-history", children=[], style={"margin": "10px"})], color="dark", inverse=True),
            html.Div(id="pdf-download", className="mt-3 text-end")
        ]
    ),
    dcc.Store(id="clear-question", data="")
], fluid=True, className="p-4", style={"backgroundColor": "#1e1e1e"})

# Database selection callback
@app.callback(
    Output("database-status", "children"),
    Input("database-selector", "value"),
    prevent_initial_call=True
)
def update_database(selected_db):
    global current_db_info
    
    if not selected_db:
        return "No database selected"
    
    try:
        if "|" in selected_db:
            # Multiple databases to merge
            db_paths = selected_db.split("|")
            if merge_databases(db_paths):
                return f"✅ {current_db_info}"
            else:
                return "❌ Failed to merge databases"
        else:
            # Single database
            if load_single_database(selected_db):
                return f"✅ {current_db_info}"
            else:
                return "❌ Failed to load database"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.callback(
    Output("chat-history", "children"),
    Output("pdf-download", "children"),
    Output("question-input", "value"),
    Input("submit-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("question-input", "value"),
    State("show-sources-toggle", "value"),
    State("llm-selector", "value"),
    State("lang-selector", "value"),
    State("mode-selector", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True
)
def update_chat(submit_clicks, clear_clicks, question, show_sources, llm_model, lang, mode, history):
    triggered_id = ctx.triggered_id
    if triggered_id == "clear-button":
        logger.info("Chat history cleared")
        return [], "", ""

    if not question:
        return history + [html.Div("❗ Please enter a question.")], "", question

    # Sanitize input
    original_question = question
    question = sanitize_query(question)
    if len(question) != len(original_question):
        logger.warning(f"Question was truncated from {len(original_question)} to {len(question)} characters")

    # Check if a database is loaded
    if not collection:
        error_msg = "❌ No database loaded. Please select a database first."
        logger.warning("User attempted to query without database loaded")
        return history + [html.Div(error_msg)], "", question

    try:
        logger.info(f"Processing query: '{question[:50]}...' with model: {llm_model}, mode: {mode}")
        answer, source_data, latency = process_query(question, llm_model, lang, mode)
        
        # Check if the answer indicates an error
        if answer.startswith("❌"):
            logger.error(f"Query processing failed: {answer}")
            error_exchange = html.Div([
                html.H5("🧑 You:", className="text-warning"),
                html.Div(question, style={"whiteSpace": "pre-wrap", "marginBottom": "10px"}),
                html.H5("❌ Error:", className="text-danger"),
                html.Div(answer, style={
                    "backgroundColor": "#4a2a2a",
                    "padding": "10px",
                    "borderRadius": "10px",
                    "marginBottom": "10px",
                    "border": "1px solid #ff6b6b"
                })            ], style={"marginBottom": "30px"})
            return [error_exchange] + history, "", ""
        
        formatted_answer = dcc.Markdown(answer)
        latency_info = html.Div(f"⏱️ Answered in {latency:.2f} seconds", 
                               className="text-muted", 
                               style={"fontSize": "0.8em", "marginTop": "5px"})

        sorted_sources = sorted(source_data, key=lambda x: x["score"], reverse=True)
        if show_sources and sorted_sources:
            source_items = []
            for item in sorted_sources:
                source_db = item.get("metadata", {}).get("source_db", "")
                source_text = f"- {item['url']} (score: {item['score']})"
                if source_db:
                    source_text += f" [from: {source_db}]"
                source_items.append(source_text)
            source_block = dcc.Markdown("\n".join(source_items))
        else:
            source_block = ""

        pdf_sources = "\n".join([f"- {item['url']} (score: {item['score']})" for item in sorted_sources])
        pdf_content = f"You: {question}\n\nAnswer:\n{answer}\n\nSources:\n{pdf_sources}\n\nDatabase: {current_db_info}"

        download_link = generate_pdf(pdf_content, lang)

        new_exchange = html.Div([
            html.H5("🧑 You:", className="text-warning"),
            html.Div(question, style={"whiteSpace": "pre-wrap", "marginBottom": "10px"}),
            html.H5("🤖 Assistant:", className="text-success"),
            html.Div([formatted_answer, latency_info], style={
                "backgroundColor": "#2a2a2a",
                "padding": "10px",
                "borderRadius": "10px",
                "marginBottom": "10px"
            }),
            html.Div([html.Strong("Sources used:"), source_block], style={
                "marginTop": "10px",
                "color": "#ccc",
                "fontSize": "0.85em",
                "backgroundColor": "#1e1e1e",
                "padding": "8px",
                "borderRadius": "6px"
            })
        ], style={"marginBottom": "30px"})

        logger.info(f"Query processed successfully in {latency:.2f} seconds")
        return [new_exchange] + history, download_link, ""
        
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        logger.error(f"Unexpected error in update_chat: {e}")
        error_exchange = html.Div([
            html.H5("🧑 You:", className="text-warning"),
            html.Div(question, style={"whiteSpace": "pre-wrap", "marginBottom": "10px"}),
            html.H5("❌ Error:", className="text-danger"),
            html.Div(error_msg, style={
                "backgroundColor": "#4a2a2a",
                "padding": "10px",
                "borderRadius": "10px",
                "marginBottom": "10px",
                "border": "1px solid #ff6b6b"
            })
        ], style={"marginBottom": "30px"})
        return [error_exchange] + history, "", ""

def parse_arguments():
    """Parse command line arguments for database selection"""
    parser = argparse.ArgumentParser(description="RAG Assistant with ChromaDB support")
    parser.add_argument("--db", "--database", type=str, 
                       help="Path to ChromaDB database directory")
    parser.add_argument("--merge", nargs='+', 
                       help="Paths to multiple ChromaDB databases to merge")
    parser.add_argument("--collection", type=str, default="web_chunks",
                       help="Collection name to use (default: web_chunks)")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port to run the web application (default: 8050)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")
    return parser.parse_args()

# Connection validation is already defined above

# Add input validation
def sanitize_query(query):
    """Sanitize user input to prevent potential issues"""
    if not query:
        return ""
    
    # Remove any null bytes and control characters
    query = ''.join(char for char in query if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length to prevent extremely long queries
    if len(query) > 2000:  # Increased from 1000 for better usability
        logger.warning(f"Query truncated from {len(query)} to 2000 characters")
        query = query[:2000]
    
    return query.strip()

# Add cleanup for temporary databases
def cleanup_temp_databases():
    temp_path = "./temp_merged_db"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

def cleanup_temp_databases():
    temp_path = "./temp_merged_db"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Override database loading if command line arguments are provided
    if args.merge:
        print(f"🔄 Merging databases from command line: {args.merge}")
        if merge_databases(args.merge, args.collection):
            print("✅ Successfully merged databases from command line")
        else:
            print("❌ Failed to merge databases from command line")
    elif args.db:
        print(f"📁 Loading database from command line: {args.db}")
        if load_single_database(args.db, args.collection):
            print("✅ Successfully loaded database from command line")
        else:
            print("❌ Failed to load database from command line")
    
    app.run(debug=args.debug, port=args.port)
