"""Validation des données des ports RAG et conformité du faux Embedder.

Les tests de comportement d'un VectorStore sont dans ``test_vector_store_contract.py``.
"""

import pytest

from app.ports.embeddings import Embedder
from app.ports.vector_store import IndexedChunk, SearchResult
from tests.fixtures.fakes import FakeEmbedder

# L'annotation fait vérifier par mypy que le faux respecte le Protocol.
EMBEDDER: Embedder = FakeEmbedder()


@pytest.mark.parametrize("field", ["client_id", "document_id"])
def test_indexed_chunk_rejects_empty_identifiers(field: str) -> None:
    values = {"client_id": "client-a", "document_id": "doc-1", field: "  "}

    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        IndexedChunk(**values, chunk_id=1, source="f.pdf", page_number=1, text="t", vector=(1.0,))


def test_indexed_chunk_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="vector must not be empty"):
        IndexedChunk("c", "d", chunk_id=1, source="f.pdf", page_number=None, text="t", vector=())


def test_indexed_chunk_rejects_invalid_chunk_id() -> None:
    with pytest.raises(ValueError, match="chunk_id must be >= 1"):
        IndexedChunk(
            "c", "d", chunk_id=0, source="f.pdf", page_number=None, text="t", vector=(1.0,)
        )


def test_search_result_rejects_non_finite_score() -> None:
    with pytest.raises(ValueError, match="score must be a finite number"):
        SearchResult("c", "d", 1, "f.pdf", None, "t", score=float("nan"))


def test_embedder_returns_one_vector_per_text_with_the_announced_dimension() -> None:
    vectors = EMBEDDER.embed_documents(["prix du riz", "stock client"])

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDER.dimension for vector in vectors)
