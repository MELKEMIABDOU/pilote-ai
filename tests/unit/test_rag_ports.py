"""Contrats des ports RAG : validation des données et comportement attendu d'un VectorStore.

Les annotations ``Embedder`` / ``VectorStore`` ci-dessous font vérifier par mypy que les faux
respectent bien les Protocols.
"""

import pytest

from app.ports.embeddings import Embedder
from app.ports.vector_store import IndexedChunk, SearchResult, VectorStore
from tests.fixtures.fakes import FakeEmbedder, InMemoryVectorStore

EMBEDDER: Embedder = FakeEmbedder()


def make_chunk(
    text: str, *, client_id: str = "client-a", document_id: str = "doc-1", chunk_id: int = 1
) -> IndexedChunk:
    return IndexedChunk(
        client_id=client_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source="horizon.pdf",
        page_number=3,
        text=text,
        vector=EMBEDDER.embed_query(text),
    )


def make_store(*chunks: IndexedChunk) -> VectorStore:
    store = InMemoryVectorStore()
    store.add(chunks)
    return store


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


def test_search_returns_results_sorted_by_score_with_citation() -> None:
    store = make_store(
        make_chunk("stock client", chunk_id=1),
        make_chunk("prix du riz", chunk_id=2),
    )

    results = store.search(EMBEDDER.embed_query("quel prix pour le riz"), client_id="client-a")

    assert [r.chunk_id for r in results] == [2, 1]
    assert results[0].score > results[1].score
    assert (results[0].source, results[0].page_number) == ("horizon.pdf", 3)


def test_search_never_returns_another_clients_documents() -> None:
    store = make_store(make_chunk("prix du riz", client_id="client-b"))

    assert store.search(EMBEDDER.embed_query("prix du riz"), client_id="client-a") == []


def test_search_respects_top_k_and_rejects_invalid_values() -> None:
    store = make_store(*(make_chunk(f"riz {i}", chunk_id=i) for i in range(1, 4)))
    query = EMBEDDER.embed_query("riz")

    assert len(store.search(query, client_id="client-a", top_k=2)) == 2
    with pytest.raises(ValueError, match="top_k"):
        store.search(query, client_id="client-a", top_k=0)


def test_search_rejects_a_query_of_the_wrong_dimension() -> None:
    store = make_store(make_chunk("riz"))

    with pytest.raises(ValueError, match="dimension"):
        store.search((1.0,), client_id="client-a")


def test_adding_the_same_chunk_twice_does_not_create_duplicates() -> None:
    store = make_store(make_chunk("riz"), make_chunk("riz"))

    assert len(store.search(EMBEDDER.embed_query("riz"), client_id="client-a")) == 1


def test_same_file_is_indexed_separately_for_each_client() -> None:
    store = make_store(make_chunk("riz", client_id="client-a"))

    assert store.has_document("doc-1", client_id="client-a")
    assert not store.has_document("doc-1", client_id="client-b")


def test_delete_document_only_removes_that_clients_passages() -> None:
    store = make_store(
        make_chunk("riz", client_id="client-a", chunk_id=1),
        make_chunk("prix", client_id="client-a", chunk_id=2),
        make_chunk("riz", client_id="client-b", chunk_id=1),
    )

    assert store.delete_document("doc-1", client_id="client-a") == 2
    assert not store.has_document("doc-1", client_id="client-a")
    assert store.has_document("doc-1", client_id="client-b")
