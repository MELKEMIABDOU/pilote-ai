"""Nettoyage des passages, avec des entrées qui ressemblent aux vrais chunks Docling.

Rappel : chez Docling, les titres sont dans ``meta.headings`` et PAS dans le texte du chunk.
"""

from app.adapters.extraction.docling import (
    ChunkCleaningRules,
    _clean_and_repair_chunks,
    _RawChunk,
)
from tests.fixtures.rules import HORIZON_RULES

DEFAULT_RULES = ChunkCleaningRules()
HORIZON_HEADER = "MAGASIN HORIZON | RAPPORT DE GESTION DES PRODUITS"


def raw(
    body: str,
    page: int | None = None,
    headings: tuple[str, ...] = (),
    captions: tuple[str, ...] = (),
) -> _RawChunk:
    return _RawChunk(headings=headings, captions=captions, body=body, page_number=page)


def texts(chunks: list[_RawChunk], rules: ChunkCleaningRules = DEFAULT_RULES) -> list[str]:
    return [chunk.text for chunk in _clean_and_repair_chunks(chunks, rules)]


def test_joined_chunks_cover_the_pages_of_both_parts() -> None:
    left = _RawChunk(headings=("4. Achats",), captions=(), body="Ligne P001,", page_number=8)
    right = _RawChunk(
        headings=("4. Achats",), captions=(), body="Riz, 20 unités", page_number=9, page_end=10
    )

    [chunk] = _clean_and_repair_chunks([left, right], DEFAULT_RULES)

    assert (chunk.page_number, chunk.page_end) == (8, 10)


def test_table_of_contents_chunk_is_dropped_entirely() -> None:
    chunks = [
        # Cas réel du PDF Horizon : la dernière entrée n'a qu'un seul numéro.
        raw("1. 1. Objet du rapport\n16. Annexe - Fiche produit exemple", 2, ("Sommaire",)),
        raw("Ce rapport couvre 2026.", 3, ("1. Objet du rapport",)),
    ]

    result = _clean_and_repair_chunks(chunks, DEFAULT_RULES)

    assert [(c.chunk_id, c.page_number, c.text) for c in result] == [
        (1, 3, "1. Objet du rapport\nCe rapport couvre 2026.")
    ]


def test_numbered_toc_lines_are_removed_from_the_body() -> None:
    assert texts([raw("2. 2. Synthèse générale\nTexte utile.", 3)]) == ["Texte utile."]


def test_running_header_heading_is_removed_only_with_document_rules() -> None:
    # Cas réel : Docling a classé l'en-tête répété comme un titre de section.
    chunks = [raw("Distrib A, Catégorie = Boissons.", 6, (HORIZON_HEADER,))]

    assert texts(chunks, HORIZON_RULES) == ["Distrib A, Catégorie = Boissons."]
    assert texts(chunks) == [f"{HORIZON_HEADER}\nDistrib A, Catégorie = Boissons."]


def test_chunk_whose_only_heading_was_removed_takes_the_previous_section() -> None:
    # Cas réel (question q20 de l'évaluation) : le tableau des fournisseurs, page 6, n'avait
    # que l'en-tête répété comme titre ; une fois retiré, il perdait sa section.
    section = "11. Suivi des responsables et fournisseurs"
    chunks = [
        raw("Karim Benali, Produits suivis = 10.", 5, (section,)),
        raw("CleanPro, Évaluation = À surveiller.", 6, (HORIZON_HEADER,)),
    ]

    assert texts(chunks, HORIZON_RULES) == [
        f"{section}\nKarim Benali, Produits suivis = 10.",
        f"{section}\nCleanPro, Évaluation = À surveiller.",
    ]


def test_chunk_that_never_had_a_heading_does_not_take_one() -> None:
    chunks = [
        raw("Premier passage.", 1, ("1. Objet",)),
        raw("Passage sans titre chez Docling.", 2),
    ]

    assert texts(chunks)[1] == "Passage sans titre chez Docling."


def test_running_header_line_is_removed_from_the_body() -> None:
    chunks = [raw(f"{HORIZON_HEADER}\nTexte utile.", 6)]

    assert texts(chunks, HORIZON_RULES) == ["Texte utile."]


def test_headings_and_captions_are_placed_before_the_body() -> None:
    chunks = [raw("Riz, Qté = 3.", 5, ("3. Référentiel", "Produits secs"), ("Tableau 2",))]

    assert texts(chunks) == ["3. Référentiel\nProduits secs\nTableau 2\nRiz, Qté = 3."]


def test_markdown_table_separators_are_removed() -> None:
    chunks = [raw("| Produit | Qté |\n|---|---|\n| P001 | 3 |", 7, ("5. Ventes",))]

    assert texts(chunks) == ["5. Ventes\n| Produit | Qté |\n| P001 | 3 |"]


def test_line_cut_in_the_middle_is_joined_without_repeating_the_heading() -> None:
    chunks = [
        raw("Ligne P001,", 8, ("4. Achats",)),
        raw("Riz basmati, 20 unités", 9, ("4. Achats",)),
    ]

    result = _clean_and_repair_chunks(chunks, DEFAULT_RULES)

    assert [(c.page_number, c.text) for c in result] == [
        (8, "4. Achats\nLigne P001,\nRiz basmati, 20 unités")
    ]


def test_label_is_joined_with_its_value_on_the_next_chunk() -> None:
    chunks = [
        raw("Prix de vente :", 10, ("8. Tarifs",)),
        raw("12,50 € par unité", 11, ("8. Tarifs",)),
    ]

    assert texts(chunks) == ["8. Tarifs\nPrix de vente :\n12,50 € par unité"]


def test_short_sentence_is_joined_with_the_table_that_follows() -> None:
    chunks = [
        raw("Voici les ventes du mois.", 12, ("5. Ventes",)),
        raw("| Produit | Qté |\n|---|---|\n| P001 | 3 |", 13, ("5. Ventes",)),
    ]

    assert texts(chunks) == [
        "5. Ventes\nVoici les ventes du mois.\n| Produit | Qté |\n| P001 | 3 |"
    ]


def test_independent_paragraphs_are_not_joined() -> None:
    long_paragraph = "Analyse détaillée. " * 20
    chunks = [
        raw(long_paragraph, 14, ("9. Analyse",)),
        raw("Autre paragraphe.", 15, ("9. Analyse",)),
    ]

    assert len(texts(chunks)) == 2


def test_empty_chunks_are_dropped() -> None:
    assert texts([raw("|---|---|", 16, ("Vide",))]) == []
