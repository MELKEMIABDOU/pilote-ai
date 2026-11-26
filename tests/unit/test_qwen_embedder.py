"""QwenEmbedder sans télécharger le modèle : on injecte un faux encodeur de lots."""

from collections.abc import Sequence

import pytest

from app.adapters.embeddings.qwen import QwenEmbedder
from app.domain.errors import EmbeddingError
from app.ports.embeddings import Embedder, Vector


class RecordingEncoder:
    """Faux encodeur : enregistre les lots reçus et renvoie des vecteurs de taille fixe."""

    def __init__(self, dimension: int = 3, vectors_per_batch: int | None = None) -> None:
        self.dimension = dimension
        self.vectors_per_batch = vectors_per_batch
        self.batches: list[list[str]] = []

    def __call__(self, batch: Sequence[str]) -> list[Vector]:
        self.batches.append(list(batch))
        count = len(batch) if self.vectors_per_batch is None else self.vectors_per_batch
        return [(1.0,) * self.dimension for _ in range(count)]


def make_embedder(
    encoder: RecordingEncoder, *, batch_size: int = 8, max_length: int = 1024
) -> QwenEmbedder:
    return QwenEmbedder(
        dimension=3,
        query_instruction="Find passages",
        batch_size=batch_size,
        max_length=max_length,
        encoder=encoder,
    )


def test_respects_the_embedder_contract() -> None:
    embedder: Embedder = make_embedder(RecordingEncoder())

    assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.dimension == 3


def test_documents_are_encoded_as_is() -> None:
    encoder = RecordingEncoder()

    vectors = make_embedder(encoder).embed_documents(["prix du riz", "stock"])

    assert encoder.batches == [["prix du riz", "stock"]]
    assert vectors == [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]


def test_query_is_prefixed_with_the_instruction() -> None:
    encoder = RecordingEncoder()

    make_embedder(encoder).embed_query("quel est le prix du riz ?")

    assert encoder.batches == [["Instruct: Find passages\nQuery:quel est le prix du riz ?"]]


def test_texts_are_sent_in_batches() -> None:
    encoder = RecordingEncoder()

    make_embedder(encoder, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert encoder.batches == [["a", "b"], ["c", "d"], ["e"]]


def test_empty_document_list_does_not_load_the_model() -> None:
    encoder = RecordingEncoder()

    assert make_embedder(encoder).embed_documents([]) == []
    assert encoder.batches == []


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        make_embedder(RecordingEncoder()).embed_query("   ")


def test_wrong_vector_count_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="expected 2 vectors, got 1"):
        make_embedder(RecordingEncoder(vectors_per_batch=1)).embed_documents(["a", "b"])


def test_wrong_vector_dimension_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="dimension 3"):
        make_embedder(RecordingEncoder(dimension=4)).embed_documents(["a"])


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        make_embedder(RecordingEncoder(), batch_size=0)


def test_invalid_max_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_length"):
        make_embedder(RecordingEncoder(), max_length=0)


def test_injected_encoder_requires_an_explicit_dimension() -> None:
    with pytest.raises(ValueError, match="dimension is required"):
        QwenEmbedder(encoder=RecordingEncoder())
