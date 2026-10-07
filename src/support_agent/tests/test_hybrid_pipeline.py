"""Deterministic tests for the configured hybrid retrieval pipeline."""

import os

from langchain_core.documents import Document

from src.support_agent.vector_store import (
    DEFAULT_EMBEDDINGS_MODEL,
    KnowledgeBaseVectorStore,
    create_cross_encoder_reranker,
)


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def similarity_search_with_score(self, _query, k, filter=None):
        return self.hits[:k]


class FakeCrossEncoder:
    def __init__(self):
        self.seen_pairs = None

    def predict(self, pairs):
        self.seen_pairs = pairs
        # Prefer the generic document to prove cross-encoder order is applied.
        return [0.9 if "generic" in text else 0.2 for _, text in pairs]


def test_bge_m3_is_the_default_embedding_configuration(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    store = KnowledgeBaseVectorStore(embeddings=object())

    assert store.embeddings_model == DEFAULT_EMBEDDINGS_MODEL == "BAAI/bge-m3"


def test_cross_encoder_factory_reranks_candidates():
    first = Document(page_content="return policy", metadata={"source_id": "return"})
    second = Document(page_content="generic product details", metadata={"source_id": "generic"})
    encoder = FakeCrossEncoder()
    rerank = create_cross_encoder_reranker(encoder=encoder)

    ranked = rerank("refund", [first, second])

    assert [document.metadata["source_id"] for document in ranked] == ["generic", "return"]
    assert encoder.seen_pairs == [
        ("refund", "return policy"),
        ("refund", "generic product details"),
    ]


def test_hybrid_search_calls_injected_reranker_after_rrf():
    generic = Document(
        page_content="generic product details", metadata={"source_id": "generic", "category": "product"}
    )
    returns = Document(
        page_content="return refund policy", metadata={"source_id": "returns", "category": "return"}
    )
    encoder = FakeCrossEncoder()
    store = KnowledgeBaseVectorStore(
        embeddings=object(), reranker=create_cross_encoder_reranker(encoder=encoder)
    )
    store.documents_by_id = {"generic": generic, "returns": returns}
    store.vector_store = FakeVectorStore([(generic, 0.8), (returns, 0.6)])

    results = store.search_with_scores("return refund", k=2)

    assert [document.metadata["source_id"] for document, _ in results] == ["generic", "returns"]
    assert len(encoder.seen_pairs) == 2


def test_bge_model_can_be_overridden_without_loading_it(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MODEL", "test/local-embedding")
    store = KnowledgeBaseVectorStore(embeddings=object())

    assert store.embeddings_model == "test/local-embedding"
