"""Validation des données des ports RAG et conformité du faux Embedder.

Les tests de comportement d'un VectorStore sont dans ``test_vector_store_contract.py``.
"""

import pytest

from app.ports.embeddings import Embedder
from app.ports.vector_store import IndexedChunk, SearchResult
from tests.fixtures.fakes import FakeEmbedder

# L'annotation fait vérifier par mypy que le faux respecte le Protocol.
EMBEDDER: Embedder = FakeEmbedder()


@pytest.mark.parametrize(
    ("client_id", "document_id", "field"),
    [("  ", "doc-1", "client_id"), ("client-a", "  ", "document_id")],
)
def test_indexed_chunk_rejects_empty_identifiers(
    client_id: str, document_id: str, field: str
) -> None:
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        IndexedChunk(client_id, document_id, 1, "f.pdf", page_number=1, text="t", vector=(1.0,))


def test_indexed_chunk_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="vector must not be empty"):
        IndexedChunk("c", "d", chunk_id=1, source="f.pdf", page_number=None, text="t", vector=())


def test_indexed_chunk_rejects_invalid_chunk_id() -> None:
    with pytest.raises(ValueError, match="chunk_id must be >= 1"):
        IndexedChunk(
            "c", "d", chunk_id=0, source="f.pdf", page_number=None, text="t", vector=(1.0,)
        )


def test_page_end_cannot_be_before_the_first_page() -> None:
    with pytest.raises(ValueError, match="page_end must be >= page_number"):
        IndexedChunk("c", "d", 1, "f.pdf", page_number=5, text="t", vector=(1.0,), page_end=4)
    with pytest.raises(ValueError, match="page_end must be >= page_number"):
        SearchResult("c", "d", 1, "f.pdf", page_number=None, text="t", score=0.5, page_end=4)


@pytest.mark.parametrize(
    ("first", "last", "pages"),
    [(None, None, ()), (4, None, (4,)), (4, 4, (4,)), (4, 6, (4, 5, 6))],
)
def test_search_result_lists_all_covered_pages(
    first: int | None, last: int | None, pages: tuple[int, ...]
) -> None:
    result = SearchResult("c", "d", 1, "f.pdf", first, "t", score=0.5, page_end=last)

    assert result.pages == pages


def test_search_result_rejects_non_finite_score() -> None:
    with pytest.raises(ValueError, match="score must be a finite number"):
        SearchResult("c", "d", 1, "f.pdf", None, "t", score=float("nan"))


def test_embedder_returns_one_vector_per_text_with_the_announced_dimension() -> None:
    vectors = EMBEDDER.embed_documents(["prix du riz", "stock client"])

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDER.dimension for vector in vectors)
