"""Évalue la qualité de la recherche RAG sur le rapport Horizon.

Chaîne évaluée : extraction Docling → passages → embeddings Qwen3 → recherche cosinus exacte.
Le stockage n'influe pas sur la qualité : une recherche exacte en mémoire donne le résultat
de référence que sqlite-vec / pgvector devront reproduire.

Lancer depuis la racine du projet :
    uv run python -m scripts.eval_retrieval
    uv run python -m scripts.eval_retrieval --top-k 10 --save
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

type Report = dict[str, Any]

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
DEFAULT_QUESTIONS = FIXTURES / "eval" / "horizon_questions.json"
DEFAULT_DOCUMENT = FIXTURES / "horizon.pdf"
RESULTS_DIR = ROOT / "outputs" / "eval"
CLIENT_ID = "eval"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    expected_pages: tuple[int, ...]
    expected_answer: str | None
    type: str

    @property
    def answerable(self) -> bool:
        return bool(self.expected_pages)


@dataclass(frozen=True, slots=True)
class Hit:
    page_number: int | None
    score: float


@dataclass(frozen=True, slots=True)
class QuestionResult:
    id: str
    type: str
    question: str
    expected_pages: tuple[int, ...]
    rank: int | None  # rang (1 = premier) du premier passage sur une page attendue
    top_hits: tuple[Hit, ...]

    @property
    def top_score(self) -> float:
        return self.top_hits[0].score if self.top_hits else 0.0


# --- Calculs (fonctions pures, testées dans tests/unit/test_eval_retrieval.py) -------------


def first_relevant_rank(hits: Sequence[Hit], expected_pages: Sequence[int]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.page_number in expected_pages:
            return rank
    return None


def recall_at(results: Sequence[QuestionResult], k: int) -> float:
    """Part des questions dont la bonne page est dans les ``k`` premiers résultats."""
    if not results:
        return 0.0
    return sum(r.rank is not None and r.rank <= k for r in results) / len(results)


def mean_reciprocal_rank(results: Sequence[QuestionResult]) -> float:
    """1 si la bonne page est toujours première, 0,5 si toujours deuxième, 0 si jamais trouvée."""
    if not results:
        return 0.0
    return sum(1 / r.rank if r.rank else 0.0 for r in results) / len(results)


# --- Chaîne réelle --------------------------------------------------------------------------


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=item["id"],
            question=item["question"],
            expected_pages=tuple(item["expected_pages"]),
            expected_answer=item.get("expected_answer"),
            type=item["type"],
        )
        for item in data["questions"]
    ]


def run(
    questions: list[Question], document: Path, top_k: int, store_kind: str = "memory"
) -> tuple[list[QuestionResult], Report]:
    # Imports ici : ils chargent Docling / PyTorch (lents), inutile pour les tests des calculs.
    from app.adapters.embeddings.qwen import QwenEmbedder
    from app.adapters.extraction.docling import DoclingExtractor
    from app.adapters.vector_store.sqlite_vec import SqliteVecStore
    from app.ports.vector_store import IndexedChunk, VectorStore
    from tests.fixtures.fakes import InMemoryVectorStore
    from tests.fixtures.rules import HORIZON_RULES

    started = time.perf_counter()
    extracted = DoclingExtractor(cleaning_rules=HORIZON_RULES).extract(document)
    extraction_seconds = time.perf_counter() - started

    embedder = QwenEmbedder()
    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk.text for chunk in extracted.chunks])
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_store_"))
    store: VectorStore
    if store_kind == "sqlite":
        store = SqliteVecStore(
            tmp_dir / "eval.db", dimension=embedder.dimension, model_name=embedder.model_name
        )
    else:
        store = InMemoryVectorStore()
    try:
        store.add(
            [
                IndexedChunk(
                    client_id=CLIENT_ID,
                    document_id=document.name,
                    chunk_id=chunk.chunk_id,
                    source=extracted.source,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    vector=vector,
                )
                for chunk, vector in zip(extracted.chunks, vectors, strict=True)
            ]
        )
        results = []
        for question in questions:
            found = store.search(
                embedder.embed_query(question.question), client_id=CLIENT_ID, top_k=top_k
            )
            hits = tuple(Hit(page_number=r.page_number, score=r.score) for r in found)
            results.append(
                QuestionResult(
                    id=question.id,
                    type=question.type,
                    question=question.question,
                    expected_pages=question.expected_pages,
                    rank=first_relevant_rank(hits, question.expected_pages)
                    if question.answerable
                    else None,
                    top_hits=hits,
                )
            )
    finally:
        if isinstance(store, SqliteVecStore):
            store.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    config = {
        "model": embedder.model_name,
        "store": store_kind,
        "chunks": len(extracted.chunks),
        "top_k": top_k,
        "extraction_seconds": round(extraction_seconds, 1),
        "embedding_and_search_seconds": round(time.perf_counter() - started, 1),
    }
    return results, config


def summarize(results: list[QuestionResult]) -> Report:
    answerable = [r for r in results if r.expected_pages]
    unanswerable = [r for r in results if not r.expected_pages]
    by_type = {
        kind: round(recall_at([r for r in answerable if r.type == kind], 5), 3)
        for kind in sorted(Counter(r.type for r in answerable))
    }
    found_scores = [r.top_score for r in answerable if r.rank == 1]
    return {
        "questions": len(answerable),
        "recall@1": round(recall_at(answerable, 1), 3),
        "recall@5": round(recall_at(answerable, 5), 3),
        "mrr": round(mean_reciprocal_rank(answerable), 3),
        "recall@5_by_type": by_type,
        # Pour choisir le seuil « rien trouvé » : les questions sans réponse doivent avoir un
        # meilleur score nettement plus bas que les questions bien trouvées.
        "top_score_when_found_min": round(min(found_scores), 3) if found_scores else None,
        "top_score_when_found_median": round(statistics.median(found_scores), 3)
        if found_scores
        else None,
        "top_score_unanswerable": {r.id: round(r.top_score, 3) for r in unanswerable},
    }


def print_report(results: list[QuestionResult], summary: Report, config: Report) -> None:
    print(
        f"\nModèle : {config['model']} | passages : {config['chunks']} | top_k : {config['top_k']}"
    )
    print(f"\n{'id':<5} {'type':<14} {'rang':>4}  {'score':>5}  pages trouvées (top 3)  attendues")
    for r in results:
        rank = "-" if r.rank is None else str(r.rank)
        pages = ", ".join(str(h.page_number) for h in r.top_hits[:3])
        expected = ", ".join(map(str, r.expected_pages)) or "aucune"
        flag = (
            ""
            if (r.rank == 1 or not r.expected_pages)
            else "  <-- raté"
            if r.rank is None
            else "  <-- pas 1er"
        )
        print(
            f"{r.id:<5} {r.type:<14} {rank:>4}  {r.top_score:>5.3f}  {pages:<22}  {expected}{flag}"
        )

    print("\nRésumé (questions avec réponse) :")
    print(
        f"  recall@1 = {summary['recall@1']:.0%}   "
        f"recall@5 = {summary['recall@5']:.0%}   "
        f"MRR = {summary['mrr']:.2f}"
    )
    print(f"  recall@5 par type : {summary['recall@5_by_type']}")
    print(
        "  meilleur score quand la bonne page est 1re : "
        f"min {summary['top_score_when_found_min']}, "
        f"médiane {summary['top_score_when_found_median']}"
    )
    print(f"  meilleur score des questions sans réponse : {summary['top_score_unanswerable']}")

    misses = [r for r in results if r.expected_pages and r.rank is None]
    if misses:
        print("\nQuestions ratées (à analyser) :")
        for r in misses:
            print(f"  {r.id} « {r.question} » attendu p.{list(r.expected_pages)}")


def save(results: list[QuestionResult], summary: Report, config: Report) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"retrieval_{timestamp}.json"
    payload = {"config": config, "summary": summary, "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--store",
        choices=["memory", "sqlite"],
        default="memory",
        help="memory = recherche exacte de référence ; sqlite = vrai stockage sqlite-vec",
    )
    parser.add_argument(
        "--save", action="store_true", help="enregistre le résultat dans outputs/eval/"
    )
    args = parser.parse_args()

    results, config = run(load_questions(args.questions), args.document, args.top_k, args.store)
    summary = summarize(results)
    print_report(results, summary, config)
    if args.save:
        print(f"\nRésultat enregistré : {save(results, summary, config)}")


if __name__ == "__main__":
    main()
