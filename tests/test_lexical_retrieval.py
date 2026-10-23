import json
from unittest.mock import patch

from rag.pipeline.orchestrator import RetrievalPipeline
from rag.retriever import session


def test_small_service_retrieves_cited_book_passages_without_local_models(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"chunk_id": "rose", "source_name": "book", "text": "玫瑰需要浇水。", "chapter": "玫瑰"},
        {"chunk_id": "fox", "source_name": "book", "text": "狐狸说驯养就是建立联系。", "chapter": "狐狸", "content": "请你驯养我。"},
    ], ensure_ascii=False))
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "lexical")
    monkeypatch.setattr(session, "CHUNK_META_PATH", str(path))
    with patch.object(session, "get_retriever", side_effect=AssertionError("No vector model")), patch("rag.reranker.cross_encoder.CrossEncoderReranker.rerank", side_effect=AssertionError("No cross encoder")):
        rows = RetrievalPipeline().run("狐狸说的驯养是什么意思？")
    assert rows[0]["chunk_id"] == "fox"
    assert rows[0]["source_name"] == "book"
    assert rows[0]["content"] == "请你驯养我。"


def test_unrelated_query_does_not_force_a_citation(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"chunk_id": "fox", "text": "狐狸和麦田"}]))
    monkeypatch.setattr(session, "CHUNK_META_PATH", str(path))
    assert session.lexical_candidates("火锅辣椒") == []
    assert session.lexical_candidates("") == []


def test_lexical_results_are_bounded_and_normalize_source(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"chunk_id": str(i), "text": "狐狸建立联系", "source": "original"} for i in range(5)]))
    monkeypatch.setattr(session, "CHUNK_META_PATH", str(path))
    rows = session.lexical_candidates("狐狸", top_k=2)
    assert len(rows) == 2
    assert all(row["source_name"] == "original" for row in rows)
