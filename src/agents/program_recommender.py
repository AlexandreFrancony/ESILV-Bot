"""
ESILV smart assistant - Programme Recommender Agent
"""

from typing import Dict, Any, List
import ollama


class ProgramRecommenderAgent:
    """Agent conversationnel pour recommander un programme ESILV."""
    
    def __init__(self):
        self.ollama_client = ollama.Client(host="http://localhost:11434")
        self.model = "mistral:7b-instruct-q4_0"
        
        # Liste des programmes disponibles
        self.programs = """
- Intelligence Artificielle & Data Science (IA) (Data & intelligence artificielle / Data Engineering & AI) 
- Cybersécurité (Cyber) (Cloud Computing & cybersécurité / Cybersecurity & Cloud Computing) 
- Développement Logiciel & Cloud (Software / Ingénierie logicielle & IA) 
- Fintech & Finance Digitale (Fintech / Ingénierie financière) 
- Énergie & Environnement (Énergie et ville durable / Éco-innovation) 
- Ingénierie Mécanique & Robotique (Conception mécanique et industrie durable / Industrie & robotique) 
- Creative Technology 
- MedTech & Santé 
- Actuariat 
- Aérospatial & Défense (Aerospace & Defence) 
- Modélisation & Mécanique Numérique (Modelling & Computational Mechanics) 
- Objets connectés & Cybersécurité (IoT & Security)
"""
    
    def process(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Laisse le LLM gérer la conversation de manière naturelle.
        """
        # Premier message
        if len(conversation_history) <= 1:
            return {
                "complete": False,
                "next_question": """Je vais vous aider à trouver le programme ESILV qui vous correspond ! 🎯

    Pour commencer, parlez-moi de vos centres d'intérêt : qu'est-ce qui vous passionne ? Qu'aimez-vous faire ? Dans quoi êtes-vous doué(e) ?""",
                "recommended_program": None,
                "recommendation_query": None,
                "recommendation_complete": False
            }
        
        # Compter les échanges (user messages seulement, sauf le premier)
        user_messages = [msg for msg in conversation_history if msg["role"] == "user"]
        num_exchanges = len(user_messages) - 1  # -1 pour exclure le premier "aide-moi"
        
        # Construire l'historique pour le LLM
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        # Prompt pour le LLM
        prompt = f"""Important Instructions pour le LLM:
    Tu es un conseiller d'orientation pour l'école d'ingénieurs ESILV.

{self.programs}

Conversation actuelle ({num_exchanges} échanges):
{messages_text}

RÈGLES:
1. TOUJOURS en français (pas d'anglais)
2. Si MOINS de 2 échanges: pose UNE question pour comprendre le profil
3. Si 2 échanges ou PLUS: fais une recommandation

FORMATAGE:
- Pas de blocs de code
- Texte naturel

RÉPONSE:
- Question: écris la question en français
- Recommandation: format exact:
RECOMMANDATION: [nom du programme]
JUSTIFICATION: [2-3 phrases EN FRANÇAIS]

Maintenant réponds:"""
        
        # Appeler le LLM
        response = self.ollama_client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.7, "num_predict": 200}
        )
        
        answer = response["response"].strip()
        
        # Vérifier si c'est une recommandation
        if "RECOMMANDATION:" in answer:
            # Extraire le programme recommandé
            lines = answer.split("\n")
            recommended_program = None
            justification = ""
            
            for line in lines:
                if line.startswith("RECOMMANDATION:"):
                    recommended_program = line.replace("RECOMMANDATION:", "").strip()
                elif line.startswith("JUSTIFICATION:"):
                    justification = line.replace("JUSTIFICATION:", "").strip()
            
            if recommended_program:
                # Construire la query pour le RAG
                rag_query = f"Donne-moi tous les détails sur {recommended_program}: durée, cours principaux, compétences développées, débouchés professionnels"
                
                return {
                    "complete": True,
                    "next_question": f""" **Je vous recommande: {recommended_program}**

    {justification}

    Laissez-moi vous donner plus de détails sur ce programme...""",
                    "recommended_program": recommended_program,
                    "recommendation_query": rag_query,
                    "recommendation_complete": True
                }
        
        # Sinon, c'est une question
        return {
            "complete": False,
            "next_question": answer,
            "recommended_program": None,
            "recommendation_query": None,
            "recommendation_complete": False
        }


# Test
if __name__ == "__main__":
    agent = ProgramRecommenderAgent()
    
    # Simulation
    history = [
        {"role": "user", "content": "Aide-moi à choisir"},
        {"role": "assistant", "content": "Parlez-moi de vos centres d'intérêt"},
        {"role": "user", "content": "J'adore coder et créer des trucs intelligents avec l'IA"}
    ]
    
    result = agent.process(history)
    print(result)