"""Projet 3 : Chatbot avec LangChain.

Chaine : trim(historique) -> prompt(system + historique + question) -> LLM -> texte
La memoire est geree par RunnableWithMessageHistory (un historique par session).
"""
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

MAX_HISTORY_MESSAGES = 10  # on garde seulement les 10 derniers messages (5 echanges)

PERSONAS = {
    "Assistant general": "You are a helpful, concise and friendly assistant.",
    "Prof de programmation": (
        "You are a patient programming teacher. Explain simply, with short code "
        "examples, and ask a check question at the end."
    ),
    "Traducteur de poche": (
        "You are a language buddy. Answer in the language the user writes in, and "
        "correct their mistakes politely."
    ),
}


def get_llm(provider: str = "groq", temperature: float = 0.7):
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=temperature,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    raise ValueError(f"Fournisseur inconnu : {provider}")


# Un historique par session_id (en memoire : perdu quand on ferme le programme)
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def reset_history(session_id: str) -> None:
    _store.pop(session_id, None)


def _trim(messages):
    return messages[-MAX_HISTORY_MESSAGES:]


def build_chatbot(provider: str = "groq", persona: str = "Assistant general"):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PERSONAS[persona]),
            MessagesPlaceholder("history"),  # ici LangChain injecte les anciens messages
            ("human", "{question}"),
        ]
    )
    chain = (
        RunnablePassthrough.assign(history=itemgetter("history") | RunnableLambda(_trim))
        | prompt
        | get_llm(provider)
        | StrOutputParser()
    )
    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def ask(bot, question: str, session_id: str = "default") -> str:
    return bot.invoke(
        {"question": question}, config={"configurable": {"session_id": session_id}}
    )


def ask_stream(bot, question: str, session_id: str = "default"):
    """Generateur : renvoie la reponse morceau par morceau (effet 'machine a ecrire')."""
    yield from bot.stream(
        {"question": question}, config={"configurable": {"session_id": session_id}}
    )


def run_demo(provider: str = "groq"):
    """Test automatique : verifie que le bot se souvient du contexte."""
    bot = build_chatbot(provider)
    sid = "demo"
    scenario = [
        "Hi! My name is Briki and I study software engineering in Tunis.",
        "What is a Python list comprehension? Answer in 2 sentences.",
        "What is my name and what do I study?",  # test memoire
    ]
    for q in scenario:
        print(f"\nUser: {q}")
        print(f"Bot : {ask(bot, q, sid)}")
    reset_history(sid)
    print("\n--- apres reset ---")
    print("User: What is my name?")
    print(f"Bot : {ask(bot, 'What is my name?', sid)}")  # ne doit plus savoir


def chat_cli(provider: str = "groq"):
    bot = build_chatbot(provider)
    print("Chatbot pret. Commandes : /reset (oublier), /exit (quitter)\n")
    while True:
        q = input("Vous : ").strip()
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/reset":
            reset_history("cli")
            print("Memoire effacee.\n")
            continue
        print("Bot  : ", end="", flush=True)
        for chunk in ask_stream(bot, q, "cli"):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    import sys

    provider = os.getenv("LLM_PROVIDER", "groq")
    if "--demo" in sys.argv:
        run_demo(provider)
    else:
        chat_cli(provider)
