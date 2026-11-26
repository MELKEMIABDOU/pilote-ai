"""Test sur un vrai PDF avec le vrai Docling (lent : lancer avec ``pytest -m integration``)."""

from pathlib import Path

import pytest

from tests.fixtures.rules import HORIZON_RULES

pytest.importorskip("docling")

from app.adapters.extraction.docling import DoclingExtractor  # noqa: E402

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.integration
def test_horizon_report_is_extracted_like_the_reference() -> None:
    result = DoclingExtractor(cleaning_rules=HORIZON_RULES).extract(FIXTURES / "horizon.pdf")

    expected = (FIXTURES / "horizon_expected.md").read_text(encoding="utf-8")
    assert result.markdown.strip() == expected.strip()
    assert result.chunks, "aucun passage produit"

    with_toc = [c.chunk_id for c in result.chunks if "sommaire" in c.text.casefold()]
    assert not with_toc, f"passages contenant encore le sommaire : {with_toc}"

    with_header = [c.chunk_id for c in result.chunks if "magasin horizon |" in c.text.casefold()]
    assert not with_header, f"passages contenant encore l'en-tête répété : {with_header}"
