import chromadb

from ingest import COLLECTION, DB_DIR, OllamaEmbedder

_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _client.get_collection(name=COLLECTION, embedding_function=OllamaEmbedder())


def retrieve(question, k=4):
    """Return the k chunks most similar to the question."""
    res = _collection.query(query_texts=[question], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "section": meta["section"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the refund window?"
    for hit in retrieve(question):
        print(f"[{hit['distance']:.3f}] {hit['source']} > {hit['section']}")
        print(hit["text"][:200], "...\n")