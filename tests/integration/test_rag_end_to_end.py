"""Chaîne complète réelle : Docling + Qwen3 + sqlite-vec (lent : ``pytest -m integration``)."""

from pathlib import Path

import pytest

pytest.importorskip("docling")

from app.config import Settings  # noqa: E402
from app.rag_container import build_rag_services  # noqa: E402

HORIZON = Path(__file__).parents[1] / "fixtures" / "horizon.pdf"


@pytest.mark.integration
def test_index_then_search_the_horizon_report(tmp_path: Path) -> None:
    settings = Settings(vector_store_path=tmp_path / "store.db")

    with build_rag_services(settings) as rag:
        first = rag.ingest(HORIZON, client_id="client-a")
        again = rag.ingest(HORIZON, client_id="client-a")
        results = rag.search("Quel est le prix d'achat du riz ?", client_id="client-a")
        other_client = rag.search("Quel est le prix d'achat du riz ?", client_id="client-b")

    assert first.chunks > 0
    assert again.already_indexed
    assert results[0].source == "horizon.pdf"
    assert {3, 4, 5} & {r.page_number for r in results[:3]}
    assert other_client == []
