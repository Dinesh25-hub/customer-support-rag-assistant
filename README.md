# customer-support-rag-assistant

A customer support chat assistant for Threadline Clothing. It answers questions using the company's policy documents, and it says so when the policies don't cover a question.

## How it works

1. **Ingest.** `ingest.py` reads the policy files in `policies/`, splits each one into sections, and stores the sections in a local ChromaDB database (`chroma_db/`).
2. **Retrieve.** When a user asks a question, `rag.py` finds the policy sections most similar to it.
3. **Answer.** `llm.py` puts those sections into the prompt and sends it to a local language model (`llama3.2` through Ollama). The model writes the reply.
4. **Chat.** `app.py` shows the chat interface in Streamlit and keeps the conversation history.

Retrieval and answering are separate steps. If an answer is wrong, the cause is either the retrieved sections (retrieval) or how the model uses them (prompt and model).

## Project structure

```
support-bot/
├── app.py          # Streamlit chat interface
├── llm.py          # Prompt building, retrieval context, Ollama calls
├── rag.py          # Policy retrieval from ChromaDB
├── ingest.py       # Reads, chunks, and stores the policies
├── policies/       # Policy documents (.md)
│   ├── damaged_items_policy.md
│   ├── exchanges_policy.md
│   ├── returns_policy.md
│   ├── shipping_policy.md
│   └── sizing_guide.md
└── chroma_db/      # Created by ingest.py. Do not edit by hand.
```

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download), running locally
- Two Ollama models:
  - `llama3.2`, for writing answers
  - `nomic-embed-text`, for turning text into vectors

## Setup

Install the Python packages:

```bash
pip install streamlit chromadb requests
```

Download the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Check that Ollama is running and both models are listed:

```bash
ollama list
```

## Usage

Build the policy database. Run this once, and again every time a file in `policies/` changes:

```bash
python ingest.py
```

You should see a chunk count for each file and a total of 32 chunks.

Start the chat:

```bash
streamlit run app.py
```

Open the URL Streamlit prints, usually http://localhost:8501.

## Updating the policies

1. Edit or replace the `.md` files in `policies/`.
2. Run `python ingest.py`. It removes each file's old chunks before adding the new ones.
3. Restart the app if it was already running.

To start from a clean database, delete `chroma_db/` and run `ingest.py` again.

## Testing retrieval

Check what the retriever returns for a question, without the chat model:

```bash
python rag.py "Can I return a sale item?"
```

Each result shows a distance score (lower is closer), the source file, and the section. Use this to debug wrong answers before changing the prompt.

## Important settings

These are in `llm.py` and `ingest.py`:

| Setting | File | Purpose |
|---|---|---|
| `MODEL` | `llm.py` | Chat model used for answers |
| `NUM_CHUNKS` | `llm.py` | Number of sections retrieved per question |
| `MAX_DISTANCE` | `llm.py` | Sections farther than this are dropped |
| `EMBED_MODEL` | `ingest.py` | Embedding model. Changing it requires rebuilding `chroma_db/` |
| `MAX_CHARS` | `ingest.py` | Target size of each chunk |

## Troubleshooting

- **`Policy collection not found`**: run `python ingest.py` first.
- **`Could not reach Ollama`**: start Ollama with `ollama serve`, or open the Ollama app.
- **`404 ... /api/embed`**: the `nomic-embed-text` model is missing or Ollama is out of date. Run `ollama list`, then `ollama pull nomic-embed-text`.
- **Embedding conflict error**: the database was built with a different embedder. Delete `chroma_db/` and run `ingest.py` again.
- **The bot answers from the wrong policy**: check retrieval with `rag.py` first, then adjust `MAX_DISTANCE` or the section headings in the policy files.

## Limitations

- The bot answers only from the five policy files. Questions outside them get "I don't know" or a suggestion to contact support.
- A small local model can still misread or ignore excerpts. Test important answers before relying on them.
- Chat history lives in the browser session and is lost on refresh.
- The sample policies are marked confidential, so keep this repository private.
