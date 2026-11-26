"""Vrai modèle Qwen3-Embedding : ``pytest -m integration``.

Le premier lancement télécharge le modèle (≈ 1,2 Go) dans le cache Hugging Face.
"""

import math

import pytest

pytest.importorskip("transformers")

from app.adapters.embeddings.qwen import QwenEmbedder  # noqa: E402


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))  # vecteurs déjà normalisés


@pytest.fixture(scope="module")
def embedder() -> QwenEmbedder:
    return QwenEmbedder()


@pytest.mark.integration
def test_vectors_have_the_announced_dimension_and_are_normalized(embedder: QwenEmbedder) -> None:
    [vector] = embedder.embed_documents(["Le riz basmati est vendu 2,50 € le kilo."])

    assert embedder.dimension == 1024
    assert len(vector) == embedder.dimension
    assert math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-3)


@pytest.mark.integration
def test_relevant_passage_scores_higher_than_unrelated_one(embedder: QwenEmbedder) -> None:
    relevant, unrelated = embedder.embed_documents(
        [
            "Le riz basmati est vendu 2,50 € le kilo.",
            "Le magasin ferme à 20 heures le samedi.",
        ]
    )
    query = embedder.embed_query("Quel est le prix du riz ?")

    assert cosine(query, relevant) > cosine(query, unrelated)
