"""
ingest.py -- Module 3: chunking, embedding, and storing the Zepto policy
corpus in ChromaDB.

Each of the 8 corpus documents (docs/doc_01.txt ... docs/doc_08.txt) is
short enough to serve as its own single chunk -- a simple per-document
chunking scheme, which the assignment spec explicitly allows given their
length. Each chunk is embedded locally with sentence-transformers'
all-MiniLM-L6-v2 model (no API key, no network call beyond the one-time
model download) and stored in a persistent ChromaDB collection on disk, so
the FastAPI app can query it without re-running ingestion on every startup.

Run:
    python ingest.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    """
    Load each doc_0N.txt file as a single chunk. Returns a list of dicts:
    {"id": "doc_01", "text": "...", "source": "doc_01.txt"}
    """
    chunks = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem  # e.g. "doc_01"
        text = path.read_text(encoding="utf-8").strip()
        chunks.append({"id": doc_id, "text": text, "source": path.name})
    return chunks


def main():
    print(f"Loading documents from {DOCS_DIR} ...")
    chunks = load_documents()
    print(f"Loaded {len(chunks)} document(s):")
    for c in chunks:
        print(f"  {c['id']} ({len(c['text'])} chars)")

    if len(chunks) != 8:
        print(
            f"WARNING: expected exactly 8 documents, found {len(chunks)}. "
            f"Check that docs/doc_01.txt ... doc_08.txt all exist."
        )

    print(f"\nLoading embedding model ''{EMBEDDING_MODEL_NAME}'' (local, no API key)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding all chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    print(f"\nInitializing persistent ChromaDB at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Start fresh each run so ingestion is idempotent/re-runnable
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- fine
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    print(f"Stored {collection.count()} chunk(s) in collection ''{COLLECTION_NAME}''.")

    # Quick sanity check: query with one of the corpus texts itself and
    # confirm the top result is that same document.
    print("\nSanity check query: ''How long do I have to return a damaged item?''")
    query_embedding = model.encode(["How long do I have to return a damaged item?"]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    for doc_id, distance, doc_text in zip(
        results["ids"][0], results["distances"][0], results["documents"][0]
    ):
        print(f"  {doc_id} (distance={distance:.4f}): {doc_text[:80]}...")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
