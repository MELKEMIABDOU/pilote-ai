"""Cas d'usage RAG (ingestion + recherche) avec des faux : rapide, sans modèle ni base."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.application.ingest_document import IngestDocument, file_sha256
from app.application.search_documents import MAX_TOP_K, SearchDocuments
from app.domain.errors import ExtractionError
from app.ports.extraction import DocumentExtractor
from tests.fixtures.fakes import FakeEmbedder, FakeExtractor, InMemoryVectorStore

PASSAGES = [(3, "Le prix du riz est de 2,20 €."), (4, "Le stock client est suivi chaque mois.")]


@pytest.fixture
def document(tmp_path: Path) -> Path:
    path = tmp_path / "rapport.pdf"
    path.write_bytes(b"contenu du rapport")
    return path


class Rag:
    """Les deux cas d'usage branchés sur les mêmes faux, comme dans l'application."""

    def __init__(self, passages: Sequence[tuple[int | None, str]] = PASSAGES) -> None:
        self.extractor = FakeExtractor(passages)
        self.store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        extractor: DocumentExtractor = self.extractor  # mypy vérifie le contrat
        self.ingest = IngestDocument(extractor, embedder, self.store)
        self.search = SearchDocuments(embedder, self.store)


def test_document_id_is_the_sha256_of_the_content(tmp_path: Path) -> None:
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    first.write_bytes(b"same")
    second.write_bytes(b"same")

    assert file_sha256(first) == file_sha256(second)
    assert len(file_sha256(first)) == 64


def test_indexed_document_can_be_searched_with_its_citation(document: Path) -> None:
    rag = Rag()

    result = rag.ingest(document, client_id="client-a")
    [best, *_] = rag.search("prix du riz", client_id="client-a")

    assert (result.chunks, result.already_indexed) == (2, False)
    assert result.document_id == file_sha256(document)
    assert (best.text, best.source, best.page_number) == (PASSAGES[0][1], "rapport.pdf", 3)


def test_indexing_the_same_file_twice_does_nothing_the_second_time(document: Path) -> None:
    rag = Rag()
    rag.ingest(document, client_id="client-a")

    second = rag.ingest(document, client_id="client-a")

    assert second.already_indexed
    assert len(rag.extractor.calls) == 1  # pas de nouvelle extraction


def test_the_same_file_is_indexed_again_for_another_client(document: Path) -> None:
    rag = Rag()
    rag.ingest(document, client_id="client-a")

    other = rag.ingest(document, client_id="client-b")

    assert not other.already_indexed
    assert rag.search("prix du riz", client_id="client-b")


def test_reindex_replaces_the_previous_passages(document: Path) -> None:
    rag = Rag()
    rag.ingest(document, client_id="client-a")
    rag.extractor.chunks = ((5, "Nouveau prix du riz."),)

    result = rag.ingest(document, client_id="client-a", reindex=True)

    texts = [r.text for r in rag.search("riz", client_id="client-a", top_k=MAX_TOP_K)]
    assert result.chunks == 1
    assert texts == ["Nouveau prix du riz."]


def test_document_without_text_is_an_error(document: Path) -> None:
    rag = Rag(passages=[])

    with pytest.raises(ExtractionError, match="no text could be extracted"):
        rag.ingest(document, client_id="client-a")
    assert not rag.store.has_document(file_sha256(document), client_id="client-a")


def test_failed_reindex_keeps_the_previous_version(document: Path) -> None:
    rag = Rag()
    rag.ingest(document, client_id="client-a")
    rag.extractor.chunks = ()

    with pytest.raises(ExtractionError):
        rag.ingest(document, client_id="client-a", reindex=True)

    assert rag.search("prix du riz", client_id="client-a")


def test_search_only_sees_the_clients_own_documents(document: Path) -> None:
    rag = Rag()
    rag.ingest(document, client_id="client-a")

    assert rag.search("prix du riz", client_id="client-b") == []


@pytest.mark.parametrize(
    ("query", "client_id", "top_k", "message"),
    [
        ("  ", "client-a", 5, "query must not be empty"),
        ("riz", " ", 5, "client_id must not be empty"),
        ("riz", "client-a", 0, "top_k must be between"),
        ("riz", "client-a", MAX_TOP_K + 1, "top_k must be between"),
    ],
)
def test_invalid_search_requests_are_rejected(
    query: str, client_id: str, top_k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Rag().search(query, client_id=client_id, top_k=top_k)


def test_ingest_requires_a_client(document: Path) -> None:
    with pytest.raises(ValueError, match="client_id must not be empty"):
        Rag().ingest(document, client_id="")
