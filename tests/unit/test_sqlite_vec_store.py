"""Comportements propres à SqliteVecStore.

Le contrat commun à tous les VectorStore est dans ``test_vector_store_contract.py``.
"""

import sqlite3
from pathlib import Path

import pytest

from app.adapters.vector_store.sqlite_vec import SqliteVecStore
from app.domain.errors import VectorStoreError
from app.ports.vector_store import IndexedChunk, VectorStore

MODEL = "fake-model"


def chunk(vector: tuple[float, ...]) -> IndexedChunk:
    return IndexedChunk("client-a", "doc-1", 1, "horizon.pdf", 2, "riz", vector)


def test_respects_the_vector_store_contract(tmp_path: Path) -> None:
    store: VectorStore = SqliteVecStore(tmp_path / "s.db", dimension=2, model_name=MODEL)

    assert not store.has_document("doc-1", client_id="client-a")


def test_data_survives_closing_and_reopening(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    with SqliteVecStore(path, dimension=2, model_name=MODEL) as store:
        store.add([chunk((1.0, 0.0))])

    with SqliteVecStore(path, dimension=2, model_name=MODEL) as reopened:
        [result] = reopened.search((1.0, 0.0), client_id="client-a")

    assert (result.text, result.page_number) == ("riz", 2)


def test_store_created_before_page_end_existed_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with SqliteVecStore(path, dimension=2, model_name=MODEL) as store:
        store.add([chunk((1.0, 0.0))])
    # Recrée l'ancien schéma : la table chunks sans la colonne page_end.
    db = sqlite3.connect(path)
    with db:
        db.execute("ALTER TABLE chunks DROP COLUMN page_end")
    db.close()

    with SqliteVecStore(path, dimension=2, model_name=MODEL) as migrated:
        [old] = migrated.search((1.0, 0.0), client_id="client-a")
        migrated.add([IndexedChunk("client-a", "doc-2", 1, "b.pdf", 4, "t", (1.0, 0.0), 5)])
        pages = {r.document_id: r.pages for r in migrated.search((1.0, 0.0), client_id="client-a")}

    assert old.text == "riz"  # les anciennes données sont conservées
    assert pages == {"doc-1": (2,), "doc-2": (4, 5)}


def test_reopening_with_another_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    SqliteVecStore(path, dimension=2, model_name=MODEL).close()

    with pytest.raises(VectorStoreError, match="re-index"):
        SqliteVecStore(path, dimension=2, model_name="other-model")
    with pytest.raises(VectorStoreError, match="re-index"):
        SqliteVecStore(path, dimension=3, model_name=MODEL)


def test_adding_a_vector_of_the_wrong_dimension_stores_nothing(tmp_path: Path) -> None:
    with SqliteVecStore(tmp_path / "s.db", dimension=2, model_name=MODEL) as store:
        with pytest.raises(ValueError, match="dimension"):
            store.add([chunk((1.0, 0.0)), chunk((1.0, 0.0, 0.0))])

        assert not store.has_document("doc-1", client_id="client-a")


def test_invalid_dimension_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimension must be >= 1"):
        SqliteVecStore(tmp_path / "s.db", dimension=0, model_name=MODEL)


def test_top_k_above_the_sqlite_vec_limit_is_rejected(tmp_path: Path) -> None:
    with (
        SqliteVecStore(tmp_path / "s.db", dimension=2, model_name=MODEL) as store,
        pytest.raises(ValueError, match="top_k"),
    ):
        store.search((1.0, 0.0), client_id="client-a", top_k=5000)
