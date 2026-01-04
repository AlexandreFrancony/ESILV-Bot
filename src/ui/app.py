import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ---------------------------------------------------------
# Caching des agents pour performance
# ---------------------------------------------------------
@st.cache_resource
def get_workflow_manager():
    from src.core.workflow import WorkflowManager
    return WorkflowManager()

# ---------------------------------------------------------
# Config générale
# ---------------------------------------------------------
st.set_page_config(
    page_title="ESILV Smart Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------
# Initialisation du WorkflowManager
# ---------------------------------------------------------
with st.spinner("🚀 Initialisation des agents IA..."):
    manager = get_workflow_manager()

st.title("🎓 ESILV Smart Assistant")

# Header avec bouton toggle pour le contact
col1, col2 = st.columns([8, 1])
with col1:
    st.caption("Assistant RAG basé sur la brochure ESILV – répond à tes questions sur l'école.")
with col2:
    show_contact = st.toggle("💬 Contact", key="toggle_contact")

# ---------------------------------------------------------
# Initialisation de l'état de session
# ---------------------------------------------------------
if "messages" not in st.session_state:
    # Chaque message : {"role": "user"/"assistant", "content": str}
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ---------------------------------------------------------
# Affichage de l'historique de chat
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# Formulaire de contact dans la sidebar (conditionnel)
# --------------------------------------------------------
if show_contact:
    with st.sidebar:
        st.header("💬 Demande de Contact")
        st.caption("Chat avec notre assistant pour être recontacté")
        
        if "form_chat" not in st.session_state:
            st.session_state.form_chat = []
        
        # Afficher historique formulaire
        for msg in st.session_state.form_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        form_input = st.chat_input("Vos coordonnées", key="form_input")
        
        if form_input:
            st.session_state.form_chat.append({
                "role": "user",
                "content": form_input
            })
            
            manager = get_workflow_manager()
            
            result = manager.run(
                form_input,
                intent_data={"intent": "contact_collection", "confidence": 1.0},
                chat_history=st.session_state.form_chat
            )
            
            st.session_state.form_chat.append({
                "role": "assistant",
                "content": result["answer"]
            })
            
            if result.get("form_complete", False):
                st.success("Contact enregistré avec succès !")
                st.balloons()
                import time
                time.sleep(1.5)
                st.session_state.form_chat = []
            
            st.rerun()


# ---------------------------------------------------------
# Saisie utilisateur
# ---------------------------------------------------------
prompt = st.chat_input("Pose une question sur l'ESILV (admissions, majeures, campus, etc.)")

if prompt:
    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse
    with st.chat_message("assistant"):
        import time
        start_time = time.time()
        # Créer les placeholders pour affichage progressif
        intent_placeholder = st.empty()
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        manager = get_workflow_manager()

        # Phase 1: Classification uniquement (rapide)
        with st.spinner("Classification de l'intent..."):
            intent_result = manager.router.classify(prompt)
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
        
        # Afficher l'intent immédiatement
        intent_placeholder.info(f"Intent détecté: {intent} (confiance: {confidence:.0%})")
        
        # Phase 2: Génération de la réponse
        with st.spinner("Génération de la réponse..."):
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ]
            result = manager.run(prompt, intent_data={"intent": intent, "confidence": confidence}, chat_history=chat_history)
            answer = result["answer"]
            sources = result["sources"]

        elapsed_time = time.time() - start_time


        # Afficher la réponse
        answer_placeholder.markdown(answer or "Aucune réponse générée.")
        
        st.caption(f"Réponse générée en {elapsed_time:.2f} secondes.")

        # Afficher les sources
        if sources:
            with sources_placeholder.expander("Sources utilisées"):
                for i, s in enumerate(sources, 1):
                    source_name = s.get("source", "Document")
                    score = s.get("score", 0.0)
                    st.markdown(f"[{i}] {source_name} (score: {score:.3f})")

    # Sauvegarder dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": intent,
        "intent_confidence": confidence
    })
    st.session_state.last_sources = sources