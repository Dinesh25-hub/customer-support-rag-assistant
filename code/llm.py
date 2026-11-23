import json

import requests

from rag import retrieve

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
NUM_CHUNKS = 4
MAX_DISTANCE = 0.6  # Tune this using the distances rag.py prints

BASE_PROMPT = """You are a support assistant for Threadline Clothing.
Answer using ONLY the policy excerpts below. If the answer is not in the excerpts,
say you don't know and suggest contacting support. Mention which policy you used.

POLICY EXCERPTS:
{context}"""


def search_query(messages):
    """Combine the last two user messages so follow-up questions keep their context."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_msgs[-2:])


def retrieve_context(messages):
    """Fetch relevant policy excerpts and format them for the system prompt."""
    hits = [
        h for h in retrieve(search_query(messages), k=NUM_CHUNKS)
        if h["distance"] <= MAX_DISTANCE
    ]
    if not hits:
        return "No relevant policy excerpts were found."
    return "\n\n---\n\n".join(
        f"[{h['source']} > {h['section']}]\n{h['text']}" for h in hits
    )


def stream_answer(messages):
    """Yield the model's reply in chunks. `messages` is the chat history, without a system prompt."""
    system = {
        "role": "system",
        "content": BASE_PROMPT.format(context=retrieve_context(messages)),
    }
    payload = {
        "model": MODEL,
        "messages": [system] + messages,
        "stream": True,
        "options": {"num_predict": 300},
    }
    with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "message" in chunk:
                yield chunk["message"]["content"]
            if chunk.get("done"):
                break