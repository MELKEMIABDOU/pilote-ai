"""Stockage vectoriel dans un fichier SQLite, avec l'extension sqlite-vec.

Deux tables :
- ``chunks`` : le texte et la provenance de chaque passage (une ligne par passage) ;
- ``chunk_vectors`` : table virtuelle ``vec0`` qui contient les vecteurs. Son ``rowid`` est
  l'``id`` de ``chunks``. ``client_id`` y est une *partition key* : une recherche ne parcourt
  que les vecteurs du client demandé.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

import sqlite_vec

from app.domain.errors import VectorStoreError
from app.ports.embeddings import Vector
from app.ports.vector_store import IndexedChunk, SearchResult

# Limite de sqlite-vec pour une recherche des k plus proches voisins.
MAX_TOP_K = 4096


class SqliteVecStore:
    """Implémente le port ``VectorStore``.

    Une base est liée à UN modèle d'embeddings : le nom du modèle et la dimension sont
    enregistrés à la création, et une ouverture avec un autre modèle est refusée (ses
    vecteurs ne seraient pas comparables).
    """

    def __init__(self, path: str | Path, *, dimension: int, model_name: str) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False : FastAPI appelle depuis plusieurs threads ; le verrou
        # garantit qu'un seul thread utilise la connexion à la fois.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        try:
            if str(path) != ":memory:":
                self._db.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
            self._check_model(model_name)
        except BaseException:
            self._db.close()
            raise

    # --- Port VectorStore -------------------------------------------------------------------

    def add(self, chunks: Sequence[IndexedChunk]) -> None:
        for chunk in chunks:
            self._check_dimension(chunk.vector)
        with self._transaction() as db:
            for chunk in chunks:
                (row_id,) = db.execute(
                    """
                    INSERT INTO chunks
                        (client_id, document_id, chunk_id, source, page_number, page_end, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (client_id, document_id, chunk_id) DO UPDATE SET
                        source = excluded.source,
                        page_number = excluded.page_number,
                        page_end = excluded.page_end,
                        text = excluded.text
                    RETURNING id
                    """,
                    (
                        chunk.client_id,
                        chunk.document_id,
                        chunk.chunk_id,
                        chunk.source,
                        chunk.page_number,
                        chunk.page_end,
                        chunk.text,
                    ),
                ).fetchone()
                # vec0 ne gère pas l'upsert : on supprime l'ancien vecteur avant d'insérer.
                db.execute("DELETE FROM chunk_vectors WHERE rowid = ?", (row_id,))
                db.execute(
                    "INSERT INTO chunk_vectors (rowid, client_id, embedding) VALUES (?, ?, ?)",
                    (row_id, chunk.client_id, sqlite_vec.serialize_float32(list(chunk.vector))),
                )

    def search(self, query: Vector, *, client_id: str, top_k: int = 5) -> list[SearchResult]:
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        self._check_dimension(query)
        with self._lock:
            rows = self._db.execute(
                """
                WITH nearest AS (
                    SELECT rowid, distance FROM chunk_vectors
                    WHERE embedding MATCH ? AND k = ? AND client_id = ?
                )
                SELECT c.client_id, c.document_id, c.chunk_id, c.source, c.page_number, c.text,
                       nearest.distance, c.page_end
                FROM nearest JOIN chunks AS c ON c.id = nearest.rowid
                ORDER BY nearest.distance
                """,
                (sqlite_vec.serialize_float32(list(query)), top_k, client_id),
            ).fetchall()
        return [
            SearchResult(
                client_id=row[0],
                document_id=row[1],
                chunk_id=row[2],
                source=row[3],
                page_number=row[4],
                text=row[5],
                score=1.0 - row[6],  # distance cosinus → similarité cosinus
                page_end=row[7],
            )
            for row in rows
        ]

    def has_document(self, document_id: str, *, client_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM chunks WHERE client_id = ? AND document_id = ? LIMIT 1",
                (client_id, document_id),
            ).fetchone()
        return row is not None

    def delete_document(self, document_id: str, *, client_id: str) -> int:
        with self._transaction() as db:
            row_ids = [
                row_id
                for (row_id,) in db.execute(
                    "SELECT id FROM chunks WHERE client_id = ? AND document_id = ?",
                    (client_id, document_id),
                )
            ]
            db.executemany("DELETE FROM chunk_vectors WHERE rowid = ?", [(i,) for i in row_ids])
            db.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in row_ids])
        return len(row_ids)

    # --- Cycle de vie -----------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> SqliteVecStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- Interne ----------------------------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Tout ou rien : en cas d'erreur, aucune modification n'est gardée."""
        with self._lock, self._db:
            yield self._db

    def _create_schema(self) -> None:
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    page_number INTEGER,
                    page_end INTEGER,
                    text TEXT NOT NULL,
                    UNIQUE (client_id, document_id, chunk_id)
                )
                """
            )
            # Migration : les bases créées avant l'ajout de page_end n'ont pas la colonne.
            # Leurs passages gardent page_end = NULL (une seule page) jusqu'à un --reindex.
            columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
            if "page_end" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN page_end INTEGER")
            # La dimension est un int validé : on peut l'insérer dans le DDL sans risque.
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
                "client_id TEXT PARTITION KEY, "
                f"embedding FLOAT[{int(self._dimension)}] distance_metric=cosine)"
            )

    def _check_model(self, model_name: str) -> None:
        expected = {"model_name": model_name, "dimension": str(self._dimension)}
        with self._transaction() as db:
            stored = dict(db.execute("SELECT key, value FROM store_meta").fetchall())
            if not stored:
                db.executemany("INSERT INTO store_meta VALUES (?, ?)", expected.items())
                return
        if stored != expected:
            raise VectorStoreError(
                f"this store was created for {stored}, not {expected}; "
                "re-index the documents into a new store"
            )

    def _check_dimension(self, vector: Vector) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"vector dimension {len(vector)} does not match the store ({self._dimension})"
            )
