"""
Build FAISS index from rag/data.json.

Run from message-api directory:
    python -m rag.embed

Outputs:
    rag/index.faiss        — FAISS index (normalized vectors, inner product)
    rag/chunk_meta.json    — [{chunk_id, source_name, text}, ...] aligned with index rows
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np

from rag.config import CHUNK_META_PATH, DATA_PATH, INDEX_PATH, RAG_DIR
from rag.corpus import normalize_records
from rag.embedding.bi_encoder import SentenceBiEncoder
from rag.vectorstore.faiss_ip import FaissIPVectorStore


def main() -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = normalize_records(payload)
    texts = [r["text"] for r in records]

    encoder = SentenceBiEncoder()
    embeddings = encoder.encode(texts, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    store = FaissIPVectorStore.from_normalized_matrix(embeddings)

    os.makedirs(RAG_DIR, exist_ok=True)
    store.save(INDEX_PATH)
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    dim = embeddings.shape[1]
    print(f"Wrote {len(texts)} vectors, dim={dim}")
    print(f"  {INDEX_PATH}")
    print(f"  {CHUNK_META_PATH}")


if __name__ == "__main__":
    main()
