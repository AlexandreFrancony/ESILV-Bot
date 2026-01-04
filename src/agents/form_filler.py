"""
FormFillerAgent - Agent IA pour collecter les coordonnées utilisateur.
================================

Version avec prints détaillés pour débugger l'extraction.
"""

import re
import json
from typing import Dict, Any, Optional, Tuple
import ollama


class FormFillerAgent:
    """Agent IA pour collecter les coordonnées utilisateur."""
    
    def __init__(self):
        self.ollama_client = ollama.Client(host="http://localhost:11434")
        self.model = "mistral:7b-instruct-q4_0"
        self.required_fields = ["first_name", "last_name", "email", "phone"]
    
    def detect_request_type(self, first_message: str) -> str:
        """Détecte le type de demande."""
        message_lower = first_message.lower()
        
        if any(w in message_lower for w in ["brochure", "documentation", "plaquette"]):
            return "brochures"
        elif any(w in message_lower for w in ["visite", "visiter", "campus"]):
            return "visit"
        elif any(w in message_lower for w in ["rendez-vous", "rencontrer", "conseiller"]):
            return "meeting"
        elif any(w in message_lower for w in ["candidat", "postuler", "inscription"]):
            return "application"
        else:
            return "callback"
    
    def get_intro_message(self, request_type: str) -> str:
        """Message d'introduction."""
        intros = {
            "brochures": "Avec plaisir ! Je vais vous envoyer les brochures de nos programmes. Pour cela, j'ai besoin de quelques informations.",
            "callback": "Bien sûr, notre équipe va vous recontacter. Laissez-moi vos coordonnées.",
            "visit": "Excellente idée de visiter notre campus ! Je vais organiser cela. Quelques informations d'abord.",
            "meeting": "Un conseiller sera ravi de vous rencontrer. Donnez-moi vos coordonnées pour fixer un rendez-vous.",
            "application": "Parfait pour candidater ! Je transmets vos informations à notre service admission."
        }
        return intros.get(request_type, "Je note vos informations.")
    
    def get_confirmation_message(self, request_type: str, data: dict) -> str:
        """Message de confirmation."""
        name = data["first_name"]
        email = data["email"]
        phone = data.get("phone", "")
        
        confirmations = {
            "brochures": f"Merci {name} ! Les brochures seront envoyées à {email}. Notre équipe pourra vous contacter au {phone} si besoin.",
            "callback": f"Merci {name}. Notre équipe vous contactera à {email} ou au {phone} sous 48h.",
            "visit": f"Merci {name} ! Notre équipe va vous contacter à {email} pour organiser votre visite.",
            "meeting": f"Merci {name}. Un conseiller va vous contacter à {email} pour fixer un rendez-vous.",
            "application": f"Merci {name}. Vous recevrez un email à {email} avec les prochaines étapes."
        }
        return confirmations.get(request_type, f"Merci {name}. Nous vous recontacterons à {email}.")
    
    def _get_expected_field(self, history: list) -> Optional[str]:
        """Devine quel champ on attend."""
        for msg in reversed(history):
            if msg["role"] == "assistant":
                content = msg["content"].lower()
                if "prénom" in content:
                    return "first_name"
                elif "nom de famille" in content or ("nom" in content and "prénom" not in content):
                    return "last_name"
                elif "email" in content or "e-mail" in content:
                    return "email"
                elif "téléphone" in content or "phone" in content:
                    return "phone"
                break
        return None
    
    def extract_info_simple(self, history: list) -> Dict[str, Any]:
        """
        Extraction depuis TOUT l'historique (accumulation).
        """
        data = {}
        
        
        # Parcourir TOUS les messages
        for i, msg in enumerate(history):
            if msg["role"] == "user":
                user_text = msg["content"].strip()
                
                # Quelle question précédait ?
                question = ""
                if i > 0 and history[i-1]["role"] == "assistant":
                    question = history[i-1]["content"].lower()
                                
                # Identifier et extraire
                if "prénom" in question and "first_name" not in data:
                    parts = user_text.split()
                    data["first_name"] = parts[0].capitalize()
                    if len(parts) > 1:
                        data["last_name"] = parts[1].capitalize()
                
                elif ("nom de famille" in question or ("nom" in question and "prénom" not in question)) and "last_name" not in data:
                    data["last_name"] = user_text.capitalize()
                
                elif ("email" in question or "e-mail" in question) and "email" not in data:
                    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_text)
                    if match:
                        data["email"] = match.group(0)
                
                elif ("téléphone" in question or "phone" in question) and "phone" not in data:
                    phone = re.sub(r'[^\d]', '', user_text)
                    if len(phone) >= 10:
                        data["phone"] = phone[:10]
        
        return data
    
    def extract_info_llm(self, history: list, expected_field: str = None) -> Dict[str, Any]:
        """Extraction avec LLM."""
        recent = history[-8:]
        messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
        
        hint = ""
        if expected_field:
            hints = {
                "first_name": "Le dernier message utilisateur contient probablement son PRÉNOM.",
                "last_name": "Le dernier message utilisateur contient probablement son NOM DE FAMILLE.",
                "email": "Le dernier message utilisateur contient probablement son EMAIL.",
                "phone": "Le dernier message utilisateur contient probablement son TÉLÉPHONE."
            }
            hint = f"\n\n{hints.get(expected_field, '')}"
        
        prompt = f"""Extrais les informations de cette conversation.

Conversation:
{messages_text}
{hint}

Réponds UNIQUEMENT en JSON (pas de markdown):
{{
    "first_name": "...",
    "last_name": "...",
    "email": "...",
    "phone": "..."
}}

Mets null pour les champs absents.

JSON:"""
        
        
        try:
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0, "num_predict": 200}
            )
            
            text = response["response"].strip()
            
            # Nettoyer
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            text = text.strip()
            data = json.loads(text)
            
            return data
        except Exception as e:
            print(f"[ERROR] Extraction LLM échouée: {e}")
            return {}
    
    def validate_field(self, field: str, value: str) -> Tuple[bool, Optional[str]]:
        """Valide un champ."""
        if not value or value == "null":
            return False, "Valeur manquante"
        
        if field == "email":
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                return False, "Email invalide"
        
        elif field == "phone":
            clean = re.sub(r'[^\d]', '', value)
            if not re.match(r'^\d{10}$', clean):
                return False, "Téléphone invalide"
        
        return True, None
    
    def get_missing_fields(self, data: Dict) -> list:
        """Retourne les champs manquants."""
        return [f for f in self.required_fields if not data.get(f)]
    
    def ask_for_field(self, field: str) -> str:
        """Question pour un champ."""
        questions = {
            "first_name": "Quel est votre prénom ?",
            "last_name": "Et votre nom de famille ?",
            "email": "Quelle est votre adresse email ?",
            "phone": "Quel est votre numéro de téléphone ?"
        }
        return questions.get(field, f"Pouvez-vous fournir: {field} ?")
    
    def process(self, conversation_history: list) -> Dict[str, Any]:
        """
        Traite le dialogue.
        """
        
        if not conversation_history:
            return {
                "complete": False,
                "data": {},
                "next_question": "Comment puis-je vous aider ?",
                "request_type": "callback"
            }
        
        # Détecter type de demande
        first_message = conversation_history[0]["content"]
        request_type = self.detect_request_type(first_message)
        
        # Premier message
        if len(conversation_history) == 1:
            intro = self.get_intro_message(request_type)
            next_q = self.ask_for_field("first_name")
            return {
                "complete": False,
                "data": {},
                "next_question": f"{intro} {next_q}",
                "request_type": request_type
            }
        
        data = self.extract_info_simple(conversation_history)
        
        # Valider
        validated_data = {}
        validation_error = None
        for field, value in data.items():
            if value and value != "null":
                is_valid, error = self.validate_field(field, value)
                if is_valid:
                    validated_data[field] = value
                elif error:
                    validation_error = (field, error)

        validated_data["request_type"] = request_type

        # Si erreur de validation, redemander le champ
        if validation_error:
            field, error_msg = validation_error
            return {
                "complete": False,
                "data": validated_data,
                "next_question": error_msg,
                "request_type": request_type
            }

        # Vérifier ce qui manque
        missing = self.get_missing_fields(validated_data)
        
        if missing:
            next_field = missing[0]
            return {
                "complete": False,
                "data": validated_data,
                "next_question": self.ask_for_field(next_field),
                "request_type": request_type
            }
        else:
            confirmation = self.get_confirmation_message(request_type, validated_data)
            return {
                "complete": True,
                "data": validated_data,
                "next_question": confirmation,
                "request_type": request_type
            }