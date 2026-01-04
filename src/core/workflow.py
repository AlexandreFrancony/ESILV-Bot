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

from src.agents.router import RouterAgent
from src.agents.retriever import RetrieverAgent
from src.agents.qa_generator import QAGeneratorAgent

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
# WORKFLOW MANAGER
# ============================================================================
class WorkflowManager:
    """
    Gestionnaire de workflow qui maintient les agents en mémoire.
    Utilisé par Streamlit avec cache pour éviter de recharger les agents.
    """
    
    def __init__(self):
        logger.info("Initialisation du WorkflowManager")
        self.router = RouterAgent()
        self.retriever = RetrieverAgent()
        self.qa_generator = QAGeneratorAgent()
        logger.info("WorkflowManager initialisé")
    
    def run(self, question: str, intent_data: dict = None, chat_history: list = None) -> Dict[str, Any]:
        """
        Execute le workflow complet.
        
        Args:
            question: Question de l'utilisateur
            intent_data: Intent pré-classifié (optionnel)
            chat_history: Historique du chat (pour contexte, optionnel)
        
        Returns:
            Dict avec answer, sources, intent, intent_confidence
        """
        logger.info(f"Running workflow for question: {question}")
        
        # Classification de l'intent
        if intent_data:
            intent = intent_data["intent"]
            confidence = intent_data["confidence"]
        else:
            result = self.router.classify(question)
            intent = result["intent"]
            confidence = result["confidence"]
        
        logger.info(f"Intent: {intent} ({confidence:.2f})")
        
        # Routing selon l'intent
        rag_intents = ["program_info", "admission_help", "course_details", "general_info"]
        
        if intent in rag_intents:
            # RAG flow
            logger.info("Using RAG flow")
            retrieved = self.retriever.retrieve_with_context(question)
            prompt = self.qa_generator._build_prompt(question, retrieved["context"])
            response = self.qa_generator.ollama_client.generate(
                model=self.qa_generator.model,
                prompt=prompt,
                stream=False
            )
            answer = response.get("response", "").strip()
            sources = retrieved["sources"]
        
        elif intent == "small_talk":
            # Small talk sans RAG
            logger.info("Small talk - direct response")
            prompt = self.qa_generator._build_prompt(question, "")
            response = self.qa_generator.ollama_client.generate(
                model=self.qa_generator.model,
                prompt=prompt,
                stream=False
            )
            answer = response.get("response", "").strip()
            sources = []
        
        elif intent == "contact_collection":
            logger.info("Contact collection flow")

            if not chat_history:
                chat_history = []
                            
            # Initialiser FormFiller
            from src.agents.form_filler import FormFillerAgent
            form_filler = FormFillerAgent()
            
            # Traiter
            result = form_filler.process(chat_history)
            
            if result["complete"]:
                # Sauvegarder
                from src.utils.save_data import save_contact
                save_contact(result["data"])
                
                answer = result["next_question"]  # Message de confirmation personnalisé
            else:
                # Demander le prochain champ
                answer = result["next_question"]
            
            sources = []
            
            # MODIFICATION: Ajouter flag pour reset
            return {
                "answer": answer,
                "sources": sources,
                "intent": intent,
                "intent_confidence": confidence,
                "form_complete": result["complete"]
            }
        
        else:
            logger.warning("Unknown intent")
            answer = "Je peux vous aider avec des questions sur ESILV."
            sources = []
        
        return {
            "answer": answer,
            "sources": sources,
            "intent": intent,
            "intent_confidence": confidence,
            "form_complete": False
        }
    

_manager_instance = None

def get_manager() -> WorkflowManager:
    """Retourne une instance singleton du WorkflowManager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = WorkflowManager()
    return _manager_instance

def run_workflow(question: str, intent_data: dict = None) -> Dict[str, Any]:
    """
    Fonction wrapper pour run_workflow (compatibilité).
    Utilise un singleton pour éviter de recharger les agents.
    """
    manager = get_manager()
    return manager.run(question, intent_data)


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
