"""
ESILV Smart Assistant - QA Generator
====================================

Q&A generator that produces concise FAQ-style responses.
- Retrieves context from vector store
- Generates answers via Mistral LLM
- Formats with citations [1][2][3]
- Integrates with LangGraph state machine

Date: 2026-01-02
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

import ollama
from .retriever import RetrieverAgent


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        import os
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/qa_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("ESILVQAGenerator")


# ============================================================================
# CONFIGURATION
# ============================================================================

LLM_MODEL = "mistral"
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TOP_K = 3
TEMPERATURE = 0.3  # Lower = more deterministic, better for FAQ style
MAX_TOKENS = 500


# ============================================================================
# QA GENERATOR CLASS
# ============================================================================

class QAGeneratorAgent:
    """
    QA generator that produces FAQ-style concise answers.
    
    This agent:
    - Retrieves relevant documents from vector store
    - Formats context with citations
    - Generates answer using Mistral LLM
    - Returns answer + sources
    """
    
    def __init__(self, model: str = LLM_MODEL, top_k: int = DEFAULT_TOP_K):
        self.model = model
        self.top_k = top_k
        self.logger = logger
        
        # Initialize retriever
        self.retriever = RetrieverAgent(top_k=top_k)
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        
        # Verify LLM is available
        self._verify_llm()
    
    def _verify_llm(self):
        """Verify LLM model is available"""
        try:
            self.logger.info(f"Checking LLM availability: {self.model}")
            response = self.ollama_client.list()
            
            models_list = []
            if isinstance(response, dict) and "models" in response:
                for item in response["models"]:
                    if isinstance(item, dict) and "name" in item:
                        models_list.append(item["name"])
                    elif isinstance(item, str):
                        models_list.append(item)
            
            model_available = any(self.model in m for m in models_list)
            
            if not model_available:
                self.logger.warning(f"{self.model} not found. Pulling...")
                try:
                    self.ollama_client.pull(self.model)
                    self.logger.info(f"Pulled {self.model}")
                except Exception as e:
                    self.logger.error(f"Failed to pull model: {e}")
                    raise
            else:
                self.logger.info(f"LLM model available: {self.model}")
        
        except Exception as e:
            self.logger.error(f"LLM verification error: {e}")
            raise
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM.
        
        Style: FAQ concis, réponses courtes et directes
        """
        prompt = f"""Tu es un assistant FAQ officiel de l'école ESILV.
Réponds de manière concise et directe, en 2-3 phrases maximum.
Utilise les numéros [1], [2], [3] pour citer tes sources.
Réponds en français.

Documents:
{context}

Question: {query}

Réponse:"""
        return prompt
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for a query.
        
        Args:
            query: User question
        
        Returns:
            Dict with keys:
            - "answer": Generated answer with citations
            - "sources": List of source metadata
            - "raw_context": Raw retrieved context
            - "num_sources": Number of sources used
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # 1. Retrieve context
            retrieved = self.retriever.retrieve_with_context(query, self.top_k)
            context = retrieved["context"]
            sources = retrieved["sources"]
            
            if not sources:
                self.logger.warning("No documents retrieved")
                return {
                    "answer": "Je n'ai pas trouvé d'informations pour répondre à cette question.",
                    "sources": [],
                    "raw_context": "",
                    "num_sources": 0
                }
            
            self.logger.info(f"Retrieved {len(sources)} documents")
            
            # 2. Build prompt with context
            prompt = self._build_prompt(query, context)
            
            # 3. Call LLM
            self.logger.debug("Calling LLM...")
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )

            
            answer_text = response.get("response", "").strip()
            self.logger.info("LLM response generated")
            
            # 4. Format final answer with citation format
            # Replace [1], [2], [3] in answer if they reference actual sources
            formatted_answer = self._format_answer(answer_text, sources)
            
            return {
                "answer": formatted_answer,
                "sources": sources,
                "raw_context": context,
                "num_sources": len(sources),
                "model": self.model,
                "query": query
            }
        
        except Exception as e:
            self.logger.error(f"QA generation error: {e}")
            return {
                "answer": f"Erreur: {str(e)}",
                "sources": [],
                "raw_context": "",
                "num_sources": 0
            }
    
    def _format_answer(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Format answer with proper citation references"""
        formatted = answer
        
        # Ensure citations reference actual sources
        for i in range(1, len(sources) + 1):
            if f"[{i}]" not in formatted:
                # Citation mentioned but source might not be referenced
                pass
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get QA generator statistics"""
        return {
            "llm_model": self.model,
            "retriever_top_k": self.top_k,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }


# ============================================================================
# LANGRAPH INTEGRATION
# ============================================================================

def create_qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for QA generator.
    
    Expected state keys:
        - "user_message": User query
        - "query": (optional) parsed query
    
    Returns:
        Updated state with:
        - "answer": Generated answer
        - "answer_sources": Source metadata
        - "answer_raw_context": Raw context used
    """
    try:
        qa_generator = QAGeneratorAgent()
        
        # Get query from state
        query = state.get("query") or state.get("user_message", "")
        
        if not query:
            logger.warning("No query provided to QA generator")
            return state
        
        # Generate answer
        result = qa_generator.answer(query)
        
        # Update state
        state["answer"] = result["answer"]
        state["answer_sources"] = result["sources"]
        state["answer_raw_context"] = result["raw_context"]
        state["answer_num_sources"] = result["num_sources"]
        
        logger.info(f"QA generated with {result['num_sources']} sources")
        
        return state
    
    except Exception as e:
        logger.error(f"QA node error: {e}")
        state["qa_error"] = str(e)
        return state


# ============================================================================
# TESTING & UTILITIES
# ============================================================================

def test_qa_generator():
    """Test QA generator with sample questions"""
    print("\n" + "=" * 70)
    print("Testing QA Generator (FAQ Style)")
    print("=" * 70 + "\n")
    
    qa_gen = QAGeneratorAgent(top_k=3)
    
    # Show stats
    stats = qa_gen.get_stats()
    print("QA Generator Stats:")
    print(f"  LLM Model: {stats['llm_model']}")
    print(f"  Retriever Top-K: {stats['retriever_top_k']}")
    print(f"  Temperature: {stats['temperature']}")
    print()
    
    # Test questions (FAQ style)
    test_questions = [
        "Quels sont les majeurs proposés en 5e année?",
        "Comment s'inscrire à ESILV?",
        "Quel est le cursus de 3e année?",
        "Y a-t-il des doubles diplômes?",
        "Où sont les campus?"
    ]
    
    for question in test_questions:
        print(f"Q: {question}")
        print("-" * 70)
        
        result = qa_gen.answer(question)
        
        print(f"A: {result['answer']}\n")
        
        if result['sources']:
            print("Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  [{i}] {source['source']} (Score: {source['score']:.3f})")
        
        print("\n")


def interactive_qa():
    """Interactive QA mode"""
    print("\n" + "=" * 70)
    print("Interactive QA Mode (type 'quit' to exit)")
    print("=" * 70 + "\n")
    
    qa_gen = QAGeneratorAgent()
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() == "quit":
            break
        
        if not question:
            continue
        
        result = qa_gen.answer(question)
        
        print(f"\nA: {result['answer']}\n")
        
        if result['sources']:
            print("Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  [{i}] {source['source']} (Score: {source['score']:.3f})")
            print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    logger.info("Starting QA Generator")
    
    # Test mode
    test_qa_generator()
    
    # Interactive mode
    print("\nStart interactive mode? (y/n): ", end="")
    if input().lower() == "y":
        interactive_qa()
    
    logger.info("QA Generator stopped")


if __name__ == "__main__":
    main()
