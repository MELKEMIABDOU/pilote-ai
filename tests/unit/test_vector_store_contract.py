"""Contrat du port VectorStore, vérifié sur TOUTES les implémentations.

Chaque test tourne deux fois : sur ``InMemoryVectorStore`` (la référence) et sur
``SqliteVecStore``. Une nouvelle implémentation (pgvector…) s'ajoute dans ``STORES``.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.adapters.vector_store.sqlite_vec import SqliteVecStore
from app.ports.embeddings import Embedder
from app.ports.vector_store import IndexedChunk, VectorStore
from tests.fixtures.fakes import FakeEmbedder, InMemoryVectorStore

EMBEDDER: Embedder = FakeEmbedder()

type StoreFactory = Callable[..., VectorStore]


def make_chunk(
    text: str,
    *,
    client_id: str = "client-a",
    document_id: str = "doc-1",
    chunk_id: int = 1,
    page_end: int | None = None,
) -> IndexedChunk:
    return IndexedChunk(
        client_id=client_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source="horizon.pdf",
        page_number=3,
        text=text,
        vector=EMBEDDER.embed_query(text),
        page_end=page_end,
    )


def _memory(_: Path) -> VectorStore:
    return InMemoryVectorStore()


def _sqlite(tmp_path: Path) -> VectorStore:
    return SqliteVecStore(
        tmp_path / "store.db", dimension=EMBEDDER.dimension, model_name=EMBEDDER.model_name
    )


STORES = {"memory": _memory, "sqlite_vec": _sqlite}


@pytest.fixture(params=sorted(STORES))
def new_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreFactory]:
    created: list[VectorStore] = []

    def factory(*chunks: IndexedChunk) -> VectorStore:
        store = STORES[request.param](tmp_path)
        created.append(store)
        store.add(chunks)
        return store

    yield factory
    for store in created:
        if isinstance(store, SqliteVecStore):
            store.close()


def test_search_returns_results_sorted_by_score_with_citation(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("stock client", chunk_id=1), make_chunk("prix du riz", chunk_id=2))

    results = store.search(EMBEDDER.embed_query("quel prix pour le riz"), client_id="client-a")

    assert [r.chunk_id for r in results] == [2, 1]
    assert results[0].score > results[1].score
    assert (results[0].source, results[0].page_number) == ("horizon.pdf", 3)


def test_page_range_is_kept_for_citations(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("prix du riz", page_end=5))

    [result] = store.search(EMBEDDER.embed_query("riz"), client_id="client-a")

    assert result.pages == (3, 4, 5)


def test_scores_are_cosine_similarities(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("prix du riz"))

    [result] = store.search(EMBEDDER.embed_query("prix du riz"), client_id="client-a")

    assert result.score == pytest.approx(1.0, abs=1e-5)


def test_search_never_returns_another_clients_documents(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("prix du riz", client_id="client-b"))

    assert store.search(EMBEDDER.embed_query("prix du riz"), client_id="client-a") == []


def test_search_respects_top_k_and_rejects_invalid_values(new_store: StoreFactory) -> None:
    store = new_store(*(make_chunk(f"riz {i}", chunk_id=i) for i in range(1, 4)))
    query = EMBEDDER.embed_query("riz")

    assert len(store.search(query, client_id="client-a", top_k=2)) == 2
    with pytest.raises(ValueError, match="top_k"):
        store.search(query, client_id="client-a", top_k=0)


def test_search_rejects_a_query_of_the_wrong_dimension(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("riz"))

    with pytest.raises(ValueError, match="dimension"):
        store.search((1.0,), client_id="client-a")


def test_adding_the_same_chunk_again_replaces_it(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("riz"), make_chunk("riz et prix"))

    results = store.search(EMBEDDER.embed_query("riz"), client_id="client-a")

    assert [r.text for r in results] == ["riz et prix"]


def test_same_file_is_indexed_separately_for_each_client(new_store: StoreFactory) -> None:
    store = new_store(make_chunk("riz", client_id="client-a"))

    assert store.has_document("doc-1", client_id="client-a")
    assert not store.has_document("doc-1", client_id="client-b")


def test_delete_document_only_removes_that_clients_passages(new_store: StoreFactory) -> None:
    store = new_store(
        make_chunk("riz", client_id="client-a", chunk_id=1),
        make_chunk("prix", client_id="client-a", chunk_id=2),
        make_chunk("riz", client_id="client-b", chunk_id=1),
    )

    assert store.delete_document("doc-1", client_id="client-a") == 2
    assert not store.has_document("doc-1", client_id="client-a")
    assert store.search(EMBEDDER.embed_query("riz"), client_id="client-a") == []
    assert store.has_document("doc-1", client_id="client-b")


def test_all_implementations_return_the_same_ranking_and_scores(tmp_path: Path) -> None:
    chunks = [
        make_chunk(text, chunk_id=i)
        for i, text in enumerate(["prix du riz", "stock client", "riz riz stock", "client"], 1)
    ]
    query = EMBEDDER.embed_query("prix riz stock")
    rankings = {}
    for name, build in STORES.items():
        store = build(tmp_path / name)
        store.add(chunks)
        rankings[name] = [
            (r.chunk_id, round(r.score, 5)) for r in store.search(query, client_id="client-a")
        ]
        if isinstance(store, SqliteVecStore):
            store.close()

    assert rankings["sqlite_vec"] == rankings["memory"]
