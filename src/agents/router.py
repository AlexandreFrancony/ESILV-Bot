"""
ESILV Smart Assistant - Router Agent
"""
import logging
from typing import Dict, Any
from datetime import datetime
import json
import ollama

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        import os
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/router_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logger("ESILVRouter")

LLM_MODEL = "mistral:7b-instruct-q4_0"
OLLAMA_HOST = "http://localhost:11434"

INTENT_TYPES = {
    "program_info": "Questions sur programmes/formations/majeures",
    "admission_help": "Questions sur admission/candidature",
    "course_details": "Questions sur cours/matières",
    "general_info": "Infos générales ESILV",
    "contact_collection": "Collecte d'informations de contact (nom, email, téléphone, programme d'intérêt)",
    "small_talk": "Salutations (bonjour/merci/bye) MÊME avec 'peux-tu m'aider', 'comment vas-tu', etc.",
    "unknown": "Non classifiable"
}

class RouterAgent:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.logger = logger
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        self._verify_llm()
    
    def _verify_llm(self):
        try:
            response = self.ollama_client.list()
            models_list = []
            if isinstance(response, dict) and "models" in response:
                for item in response["models"]:
                    if isinstance(item, dict) and "name" in item:
                        models_list.append(item["name"])
            if not any(self.model in m for m in models_list):
                self.ollama_client.pull(self.model)
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            raise
    
    def classify(self, query: str) -> Dict[str, Any]:
        try:
            prompt = self._build_prompt(query)
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1},
                stream=False
            )
            response_text = response.get("response", "").strip()
            return self._parse_response(response_text, query)
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return self._fallback_classification(query)
    
    def _build_prompt(self, query: str) -> str:
        intent_desc = "\n".join([
            f"{i+1}. {intent}: {desc}"
            for i, (intent, desc) in enumerate(INTENT_TYPES.items())
        ])
        return f"""Tu es un classificateur d'intentions pour un chatbot ESILV.

Classe ce message dans UNE catégorie:

{intent_desc}

Message: "{query}"

Réponds en JSON (sans markdown):
{{"intent": "program_info", "confidence": 0.95, "entities": {{}}}}

JSON:"""
    
    def _parse_response(self, response: str, query: str) -> Dict[str, Any]:
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            data = json.loads(response)
            intent = data.get("intent", "unknown")
            
            if intent not in INTENT_TYPES:
                return self._fallback_classification(query)
            
            return {
                "intent": intent,
                "confidence": float(data.get("confidence", 0.5)),
                "entities": data.get("entities", {})
            }
        except:
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        keywords = {
            "program_info": ["programme", "formation", "majeure", "cursus"],
            "admission_help": ["admission", "postuler", "candidature", "inscription"],
            "course_details": ["cours", "matière", "module"],
            "general_info": ["esilv", "école", "campus", "où"],
            "contact_collection": ["contact", "brochures", "téléphone", "rappelé", "visite", "documentation", "informations", "intéressé"],
            "small_talk": ["bonjour", "salut", "merci", "bye"]
        }
        
        for intent, words in keywords.items():
            if any(word in query_lower for word in words):
                return {"intent": intent, "confidence": 0.6, "entities": {}}
        
        return {"intent": "unknown", "confidence": 0.3, "entities": {}}

def create_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        router = RouterAgent()
        query = state.get("user_message", "")
        if not query:
            return state
        
        result = router.classify(query)
        state["intent"] = result["intent"]
        state["intent_confidence"] = result["confidence"]
        state["intent_entities"] = result["entities"]
        return state
    except Exception as e:
        logger.error(f"Router error: {e}")
        state["intent"] = "unknown"
        state["intent_confidence"] = 0.0
        return state