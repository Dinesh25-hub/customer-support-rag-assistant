import re
from pathlib import Path

import chromadb
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Paths are built from this file's location, so they work from any folder
BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policies"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION = "policies"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
MAX_CHARS = 1000

# Lines present in every file that carry no policy content
NOISE_RE = re.compile(r"^(Threadline Clothing - Confidential|Page \d+ of \d+)$")


class OllamaEmbedder(EmbeddingFunction[Documents]):
    """Embedding function that calls Ollama, so ingest and queries use the same model."""

    def __init__(self, model=EMBED_MODEL, url=OLLAMA_EMBED_URL):
        self.model = model
        self.url = url

    def __call__(self, input: Documents) -> Embeddings:
        r = requests.post(
            self.url,
            json={"model": self.model, "input": list(input)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    @staticmethod
    def name() -> str:
        # Chroma stores this name with the collection to detect embedding mismatches
        return "ollama_embedder"

    def get_config(self):
        return {"model": self.model, "url": self.url}

    @staticmethod
    def build_from_config(config):
        return OllamaEmbedder(model=config["model"], url=config["url"])


def clean_text(text):
    """Drop the confidentiality header and page markers, keeping everything else."""
    lines = [line for line in text.splitlines() if not NOISE_RE.match(line.strip())]
    return "\n".join(lines).strip()


def split_sections(text):
    """Return (title, [(heading, body), ...]) from cleaned policy text."""
    title, _, rest = text.partition("\n")
    parts = re.split(r"^\d+\. (.+)$", rest, flags=re.MULTILINE)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    sections = [
        (heading.strip(), body.strip())
        for heading, body in zip(parts[1::2], parts[2::2])
    ]
    return title.strip(), sections


def pack_paragraphs(body, max_chars):
    """Group paragraphs into pieces of roughly max_chars without cutting a paragraph."""
    pieces, current = [], ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        pieces.append(current)
    return pieces


def build_chunks(path):
    """Turn one policy file into chunks that carry their policy title and section."""
    title, sections = split_sections(clean_text(path.read_text(encoding="utf-8")))
    chunks = []
    for heading, body in sections:
        for piece in pack_paragraphs(body, MAX_CHARS):
            n = len(chunks)
            chunks.append({
                "id": f"{path.stem}-{n}",
                # Title and section go into the text so the embedding captures them
                "text": f"{title} > {heading}\n\n{piece}",
                "metadata": {"source": path.name, "title": title, "section": heading},
            })
    return chunks


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=OllamaEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )

    for path in sorted(POLICY_DIR.glob("*.md")):
        chunks = build_chunks(path)
        # Remove this file's old chunks first, so edited or deleted sections don't linger
        collection.delete(where={"source": path.name})
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        print(f"{path.name}: {len(chunks)} chunks")

    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    main()