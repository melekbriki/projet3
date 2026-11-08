# Projet 3 : Chatbot avec LangChain

## Objectif
Chatbot conversationnel qui répond en langage naturel et **se souvient** du contexte de la conversation.

## Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Fichier `.env` (le même que les projets précédents) :
```
GROQ_API_KEY=ta_cle
GROQ_MODEL=openai/gpt-oss-20b
LLM_PROVIDER=groq
```

## Lancer
```powershell
python chatbot.py --demo   # test automatique de la mémoire
python chatbot.py          # chat dans le terminal (/reset, /exit)
streamlit run app.py       # interface web (streaming + personnalités)
```

## Architecture
`trim(historique)` -> `ChatPromptTemplate(system + MessagesPlaceholder + question)` -> `LLM` -> `StrOutputParser`.
`RunnableWithMessageHistory` charge/sauvegarde l'historique de chaque `session_id` automatiquement.

## Cas de test
1. "My name is Briki..." puis question sans rapport.
2. "What is my name and what do I study?" -> doit répondre correctement (mémoire).
3. `/reset` puis "What is my name?" -> ne doit plus savoir.
4. Changer de personnalité (prof de programmation) -> style de réponse différent.

## Limites et améliorations
- Mémoire en RAM : perdue à la fermeture -> sauvegarde SQLite/Redis (`SQLChatMessageHistory`).
- Fenêtre fixe de 10 messages : le bot oublie les anciens échanges -> résumé automatique de l'ancien contexte.
- Pas de connaissances externes -> RAG (documents) ou outils (recherche web).
- Le LLM peut halluciner -> citer des sources, ajouter des garde-fous.
- Pas d'authentification ni de limite d'usage.
