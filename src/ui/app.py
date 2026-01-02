import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.core.workflow import run_workflow

# ---------------------------------------------------------
# Config générale
# ---------------------------------------------------------
st.set_page_config(
    page_title="ESILV Smart Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 ESILV Smart Assistant")
st.caption("Assistant RAG basé sur la brochure ESILV – répond à tes questions sur l'école.")


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
# Saisie utilisateur
# ---------------------------------------------------------
prompt = st.chat_input("Pose une question sur l'ESILV (admissions, majeures, campus, etc.)")

if prompt:
    # 1) Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Appeler le workflow RAG
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            result = run_workflow(prompt)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

        # Affichage de la réponse principale
        st.markdown(answer or "_Aucune réponse générée._")

        # Affichage des sources en dessous
        if sources:
            st.markdown("**Sources utilisées :**")
            for i, s in enumerate(sources, 1):
                source_name = s.get("source", "Document")
                score = s.get("score", 0.0)
                st.markdown(f"- [{i}] {source_name} (score: {score:.3f})")

    # 3) Sauvegarder le message assistant dans l'historique
    # (on ne remet pas les sources en texte brut dans l'historique pour éviter de polluer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = sources
