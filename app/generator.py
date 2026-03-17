import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = None  # Lazy-load


def get_client():
    """Return cached Gemini client instance."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(query: str, docs: list, chat_history: list = None) -> str:
    """
    Generate a context-aware answer using retrieved docs and chat history.

    Args:
        query: The user's question.
        docs: List of LangChain Document objects (retrieved chunks).
        chat_history: List of previous Q/A strings for multi-turn context.

    Returns:
        The model's answer as a string.
    """
    if chat_history is None:
        chat_history = []

    # Build context from retrieved chunks (include source metadata)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Build conversation history string (last 5 turns = 10 lines)
    history_text = "\n".join(chat_history[-10:])

    prompt = f"""You are an expert AI Research Assistant. Your job is to answer questions
accurately and clearly based ONLY on the provided context documents.

Rules:
- Answer only from the context. If the answer is not there, say so honestly.
- Be concise but complete. Do NOT include any chunk references or source citations in your answer.
- Write in clean, natural language as if explaining to a person.
- If the question is a follow-up, use the chat history for context.

---
Chat History (last few turns):
{history_text if history_text else "None"}

---
Context Documents:
{context}

---
User Question: {query}

Answer:"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nPlease check your GOOGLE_API_KEY in the .env file."