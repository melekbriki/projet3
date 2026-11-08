"""Interface Streamlit du chatbot. Lancer : streamlit run app.py"""
import uuid

import streamlit as st

from chatbot import PERSONAS, ask_stream, build_chatbot, reset_history

st.set_page_config(page_title="Chatbot LangChain", page_icon="🤖")
st.title("🤖 Chatbot LangChain")

provider = st.sidebar.selectbox("Fournisseur LLM", ["groq", "gemini"])
persona = st.sidebar.selectbox("Personnalité", list(PERSONAS))

# session_id unique par utilisateur/onglet
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.messages = []

if st.sidebar.button("🗑️ Nouvelle conversation"):
    reset_history(st.session_state.sid)
    st.session_state.messages = []

# le bot est reconstruit si on change de fournisseur ou de personnalité
key = (provider, persona)
if st.session_state.get("bot_key") != key:
    st.session_state.bot = build_chatbot(provider, persona)
    st.session_state.bot_key = key

# affichage de l'historique
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Écris ton message..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                ask_stream(st.session_state.bot, question, st.session_state.sid)
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Erreur : {e}")
