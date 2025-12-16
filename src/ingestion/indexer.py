"""
ESILV Smart Assistant - Vector Store Indexer
=============================================

Indexer for ESILV chunks.
- Loads chunks from data/chunks.json
- Generates embeddings with Nomic (via Ollama)
- Stores in Chromadb v1.0 (persistent storage)
- Supports semantic search and retrieval

Date: 2025-12-16
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
import ollama


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/indexer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("ESILVIndexer")


# ============================================================================
# CONFIGURATION
# ============================================================================

CHUNKS_FILE = "data/chunks.json"
CHROMA_PATH = "data/chromadb"
COLLECTION_NAME = "esilv_documents"

# Embedding model (must run: ollama pull nomic-embed-text)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"


# ============================================================================
# CUSTOM EMBEDDING FUNCTION FOR OLLAMA
# ============================================================================

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function using Ollama"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            try:
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                logger.error(f"Embedding error for text: {e}")
                raise
        return embeddings


# ============================================================================
# INDEXER CLASS
# ============================================================================

class ESILVIndexer:
    """Vector store indexer using Chromadb v1.0 and Ollama embeddings."""
    
    def __init__(self, chunks_file: str = CHUNKS_FILE, chroma_path: str = CHROMA_PATH):
        self.chunks_file = chunks_file
        self.chroma_path = chroma_path
        self.logger = logger
        
        # Initialize Chromadb persistent client (v1.0 style)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = None
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
    
    def process(self) -> bool:
        """Main indexing pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Vector Store Indexing")
        self.logger.info(f"Chunks file: {self.chunks_file}")
        self.logger.info(f"Chroma path: {self.chroma_path}")
        self.logger.info("=" * 60)
        
        # 1. Verify Ollama and embeddings model
        if not self._verify_ollama():
            return False
        
        # 2. Load chunks
        if not os.path.exists(self.chunks_file):
            self.logger.error(f"Chunks file not found: {self.chunks_file}")
            return False
        
        chunks = self._load_chunks()
        if not chunks:
            self.logger.error("No chunks loaded")
            return False
        
        self.logger.info(f"Loaded {len(chunks)} chunks")
        
        # 3. Create/get Chromadb collection
        self._init_collection()
        
        # 4. Generate embeddings and index
        self._index_chunks(chunks)
        
        # 5. Verify indexing
        self._verify_index()
        
        return True
    
    def _verify_ollama(self) -> bool:
        """Check if Ollama is running and embedding model is available"""
        try:
            # Check if Ollama is running
            self.logger.info("Checking Ollama connection...")
            response = self.ollama_client.list()
            self.logger.info("Ollama is running")
            
            # Parse models list (handle different formats)
            models_list = []
            if isinstance(response, dict) and "models" in response:
                for item in response["models"]:
                    if isinstance(item, dict):
                        # Format: {"name": "model_name", ...}
                        if "name" in item:
                            models_list.append(item["name"])
                    elif isinstance(item, str):
                        # Format: just model name string
                        models_list.append(item)
            
            # Check if embedding model is available (substring match)
            model_available = any(EMBEDDING_MODEL in m for m in models_list)
            
            if not model_available:
                self.logger.warning(f"{EMBEDDING_MODEL} not found. Pulling...")
                try:
                    self.ollama_client.pull(EMBEDDING_MODEL)
                    self.logger.info(f"Pulled {EMBEDDING_MODEL}")
                except Exception as pull_error:
                    self.logger.error(f"Failed to pull model: {pull_error}")
                    return False
            else:
                self.logger.info(f"Embedding model available: {EMBEDDING_MODEL}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            self.logger.error("Make sure Ollama is running: ollama serve")
            return False
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file"""
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks
    
    def _init_collection(self):
        """Initialize Chromadb collection"""
        # Delete existing collection if it exists (fresh start)
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass
        
        # Create custom embedding function
        embedding_fn = OllamaEmbeddingFunction()
        
        # Create new collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger.info(f"Created collection: {COLLECTION_NAME}")
    
    def _index_chunks(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings and add chunks to collection"""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["id"]
            chunk_text = chunk["text"]
            chunk_meta = chunk["metadata"]
            
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append(chunk_meta)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Prepared {i + 1}/{len(chunks)} chunks")
        
        # Add all to collection at once (embeddings generated automatically)
        self.logger.info(f"Adding {len(ids)} chunks to Chromadb...")
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            self.logger.info(f"Successfully indexed {len(ids)} chunks")
        except Exception as e:
            self.logger.error(f"Error indexing chunks: {e}")
            raise
    
    def _verify_index(self):
        """Verify indexing worked by testing a few queries"""
        self.logger.info("Verifying index with test queries...")
        
        test_queries = [
            "formations",
            "admissions",
            "programs"
        ]
        
        for query in test_queries:
            try:
                results = self.search(query, top_k=3)
                self.logger.info(f"Query '{query}': Found {len(results)} results")
                for i, result in enumerate(results[:1]):
                    text_preview = result["text"][:100] + "..."
                    self.logger.info(f"  [{i+1}] {result['metadata']['source']}: {text_preview}")
            except Exception as e:
                self.logger.warning(f"Query '{query}' failed: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        # Query collection (embedding generated automatically)
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {}
        
        return {
            "collection_name": COLLECTION_NAME,
            "total_documents": self.collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM
        }


# ============================================================================
# INTERACTIVE SEARCH (for testing)
# ============================================================================

def interactive_search(indexer: ESILVIndexer):
    """Interactive search mode"""
    print("\n" + "=" * 60)
    print("Interactive Vector Search (type 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        query = input("Search query: ").strip()
        
        if query.lower() == "quit":
            break
        
        if not query:
            continue
        
        results = indexer.search(query, top_k=5)
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results):
            print(f"[{i + 1}] Source: {result['metadata']['source']}")
            print(f"    Type: {result['metadata']['type']}")
            print(f"    Tokens: {result['metadata']['tokens']}")
            text_preview = result["text"][:200].replace("\n", " ")
            print(f"    Text: {text_preview}...")
            if result["distance"]:
                print(f"    Distance: {result['distance']:.4f}")
            print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    indexer = ESILVIndexer()
    success = indexer.process()
    
    if success:
        logger.info("Indexing completed successfully!")
        
        # Show stats
        stats = indexer.get_stats()
        logger.info("\nIndex Statistics:")
        logger.info(f"  Total Documents: {stats['total_documents']}")
        logger.info(f"  Embedding Model: {stats['embedding_model']}")
        logger.info(f"  Embedding Dimension: {stats['embedding_dim']}")
        
        # Optional: Interactive search
        print("\nStart interactive search? (y/n): ", end="")
        if input().lower() == "y":
            interactive_search(indexer)
        
        return 0
    else:
        logger.error("Indexing failed.")
        return 1


if __name__ == "__main__":
    exit(main())