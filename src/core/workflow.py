"""
ESILV Smart Assistant - Core Workflow
=====================================

Simple mono-turn RAG workflow.
- Takes a user question
- Runs Retriever node
- Runs QA Generator node
- Returns answer + sources

Date: 2026-01-02
"""

import logging
from typing import Dict, Any
from datetime import datetime

from src.agents.retriever import create_retriever_node
from src.agents.qa_generator import create_qa_node


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        import os
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = setup_logger("ESILVWorkflow")


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_workflow(question: str) -> Dict[str, Any]:
    """
    Run mono-turn RAG workflow.

    Args:
        question: user question in natural language

    Returns:
        dict with:
            - answer: final answer string
            - sources: list of source metadata
            - retrieved_context: raw context from retriever
            - retrieval_count: number of retrieved documents
    """
    # Initial state
    state: Dict[str, Any] = {
        "user_message": question,
        "query": question,
    }

    logger.info(f"Running workflow for question: {question}")

    # 1. Retriever node
    state = create_retriever_node(state)

    # 2. QA node
    state = create_qa_node(state)

    # 3. Extract final data
    answer = state.get("answer", "")
    sources = state.get("answer_sources", [])
    retrieved_context = state.get("retrieved_context", "")
    retrieval_count = state.get("retrieval_count", 0)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_context": retrieved_context,
        "retrieval_count": retrieval_count,
    }


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_loop():
    """Simple CLI to test the workflow."""
    print("\n" + "=" * 70)
    print("ESILV Smart Assistant - RAG Workflow (type 'quit' to exit)")
    print("=" * 70 + "\n")

    while True:
        q = input("Vous: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        result = run_workflow(q)

        print("\nAssistant:", result["answer"], "\n")

        if result["sources"]:
            print("Sources:")
            for i, s in enumerate(result["sources"], 1):
                print(f"  [{i}] {s['source']} (Score: {s['score']:.3f})")
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("Starting ESILV Workflow")
    interactive_loop()
    logger.info("Workflow stopped")


if __name__ == "__main__":
    main()
