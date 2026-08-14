"""
Ingestion pipeline for the Zepto support-assistant corpus.

Stage: ingestion -> embedding -> storage
  1. Load the 8 .txt documents from docs/.
  2. Chunk them (each document is short enough that we use a simple
     per-document chunk; one chunk == one document == one policy topic).
  3. Embed each chunk locally with sentence-transformers/all-MiniLM-L6-v2
     (no API key, no network call required at inference time once the
     model has been downloaded once).
  4. Store the embeddings + text + metadata in a persistent ChromaDB
     collection ("zepto_policies") on disk under ./chroma_db.

Run this once before starting the API:
    python ingest.py
"""

import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    """Load each doc_XX.txt file as one chunk, keyed by its filename stem."""
    chunks = []
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    if not paths:
        raise FileNotFoundError(f"No corpus documents found in {DOCS_DIR}")
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_index() -> None:
    chunks = load_documents()
    print(f"Loaded {len(chunks)} corpus documents from {DOCS_DIR}")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (sentence-transformers)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    print(f"Writing to persistent ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh collection each run so re-ingesting is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"doc_id": doc_id} for doc_id in ids],
    )

    count = collection.count()
    print(f"Stored {count} document embeddings in ChromaDB collection '{COLLECTION_NAME}'.")
    assert count == 8, f"Expected 8 corpus docs stored, found {count}"


if __name__ == "__main__":
    build_index()
