"""
ESILV Smart Assistant - Retriever Agent
=======================================

RAG retriever agent that searches vector store and returns top-K documents.
- Connects to Chromadb vector store
- Performs semantic search on user queries
- Returns ranked results with metadata
- Integrates with LangGraph state machine

Date: 2025-12-16
"""

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
        import os
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/retriever_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("ESILVRetriever")


# ============================================================================
# CONFIGURATION
# ============================================================================

CHROMA_PATH = "data/chromadb"
COLLECTION_NAME = "esilv_documents"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TOP_K = 5


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
                logger.error(f"Embedding error: {e}")
                raise
        return embeddings


# ============================================================================
# RETRIEVER AGENT CLASS
# ============================================================================

class RetrieverAgent:
    """
    Retriever agent for semantic search over indexed documents.
    
    This agent:
    - Loads documents from Chromadb vector store
    - Performs similarity search on queries
    - Returns top-K ranked results
    """
    
    def __init__(self, chroma_path: str = CHROMA_PATH, top_k: int = DEFAULT_TOP_K):
        self.chroma_path = chroma_path
        self.top_k = top_k
        self.logger = logger
        
        # Initialize Chromadb client
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = None
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        
        # Load collection
        self._load_collection()
    
    def _load_collection(self):
        """Load collection from Chromadb"""
        try:
            # Get embedding function
            embedding_fn = OllamaEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            count = self.collection.count()
            self.logger.info(f"Loaded collection with {count} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to load collection: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-K documents for a query.
        
        Args:
            query: User query string
            top_k: Number of results (default: self.top_k)
        
        Returns:
            List of documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if not self.collection:
            self.logger.error("Collection not loaded")
            return []
        
        try:
            self.logger.debug(f"Querying: {query}")
            
            # Query collection (embedding generated automatically)
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    result = {
                        "id": chunk_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i] if results["distances"] else None,
                        "rank": i + 1
                    }
                    formatted_results.append(result)
            
            self.logger.info(f"Retrieved {len(formatted_results)} documents for query")
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}")
            return []
    
    def retrieve_with_context(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieve documents and format for RAG context.
        
        Returns formatted context suitable for LLM prompt.
        """
        results = self.retrieve(query, top_k)
        
        context_text = ""
        sources = []
        
        for result in results:
            context_text += f"\n[{result['rank']}] {result['text']}\n"
            sources.append({
                "id": result["id"],
                "source": result["metadata"]["source"],
                "type": result["metadata"]["type"],
                "score": result["score"],
                "rank": result["rank"]
            })
        
        return {
            "context": context_text,
            "sources": sources,
            "num_results": len(results),
            "query": query
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {}
        
        return {
            "collection_name": COLLECTION_NAME,
            "total_documents": self.collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "default_top_k": self.top_k
        }


# ============================================================================
# LANGRAPH INTEGRATION
# ============================================================================

def create_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for retriever agent.
    
    Expected state keys:
        - "query": User query string
        - "user_message": Original user message
    
    Returns:
        Updated state with:
        - "retrieved_context": Formatted context from documents
        - "retrieved_sources": Source metadata
        - "retrieval_count": Number of documents retrieved
    """
    try:
        retriever = RetrieverAgent()
        
        # Get query from state
        query = state.get("query") or state.get("user_message", "")
        
        if not query:
            logger.warning("No query provided to retriever")
            return state
        
        # Retrieve documents
        context_result = retriever.retrieve_with_context(query)
        
        # Update state
        state["retrieved_context"] = context_result["context"]
        state["retrieved_sources"] = context_result["sources"]
        state["retrieval_count"] = context_result["num_results"]
        
        logger.info(f"Retriever returned {context_result['num_results']} documents")
        
        return state
    
    except Exception as e:
        logger.error(f"Retriever node error: {e}")
        state["retrieval_error"] = str(e)
        return state


# ============================================================================
# TESTING & UTILITIES
# ============================================================================

def test_retriever():
    """Test retriever with sample queries"""
    print("\n" + "=" * 60)
    print("Testing Retriever Agent")
    print("=" * 60 + "\n")
    
    retriever = RetrieverAgent(top_k=3)
    
    # Show stats
    stats = retriever.get_stats()
    print("Retriever Stats:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print()
    
    # Test queries
    test_queries = [
        "What are the programs offered?",
        "How to apply to ESILV?",
        "Tell me about the 5th year",
        "What majors are available?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        results = retriever.retrieve(query, top_k=3)
        
        for result in results:
            print(f"\n[{result['rank']}] Score: {result['score']:.4f}")
            print(f"Source: {result['metadata']['source']}")
            text_preview = result["text"][:150].replace("\n", " ")
            print(f"Text: {text_preview}...")
        
        print("\n")


def interactive_mode():
    """Interactive retriever mode"""
    print("\n" + "=" * 60)
    print("Interactive Retriever (type 'quit' to exit)")
    print("=" * 60 + "\n")
    
    retriever = RetrieverAgent()
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() == "quit":
            break
        
        if not query:
            continue
        
        results = retriever.retrieve(query, top_k=5)
        
        print(f"\nFound {len(results)} results:\n")
        
        for result in results:
            print(f"[{result['rank']}] Score: {result['score']:.4f}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Type: {result['metadata']['type']}")
            text_preview = result["text"][:200].replace("\n", " ")
            print(f"Text: {text_preview}...\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    logger.info("Starting Retriever Agent")
    
    # Test mode
    test_retriever()
    
    # Interactive mode
    print("\nStart interactive mode? (y/n): ", end="")
    if input().lower() == "y":
        interactive_mode()
    
    logger.info("Retriever Agent stopped")


if __name__ == "__main__":
    main()
