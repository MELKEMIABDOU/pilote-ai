import json
from pathlib import Path

import pytest

from scripts.eval_retrieval import (
    DEFAULT_QUESTIONS,
    Hit,
    QuestionResult,
    first_relevant_rank,
    load_questions,
    mean_reciprocal_rank,
    recall_at,
)


def result(rank: int | None) -> QuestionResult:
    return QuestionResult("q", "texte", "?", (3,), rank, (Hit(3, 0.9),))


def test_first_relevant_rank_returns_the_rank_of_the_first_expected_page() -> None:
    hits = [Hit(page_number=5, score=0.9), Hit(page_number=None, score=0.8), Hit(3, 0.7)]

    assert first_relevant_rank(hits, [3, 4]) == 3
    assert first_relevant_rank(hits, [7]) is None


def test_recall_counts_questions_found_within_k() -> None:
    results = [result(1), result(3), result(None), result(6)]

    assert recall_at(results, 1) == 0.25
    assert recall_at(results, 5) == 0.5
    assert recall_at([], 5) == 0.0


def test_mrr_averages_reciprocal_ranks() -> None:
    assert mean_reciprocal_rank([result(1), result(2), result(None)]) == pytest.approx(0.5)


def test_question_file_is_valid() -> None:
    questions = load_questions(DEFAULT_QUESTIONS)
    ids = [q.id for q in questions]

    assert len(ids) == len(set(ids)), "identifiants de question en double"
    assert all(q.question.strip() for q in questions)
    for q in questions:
        assert all(1 <= page <= 7 for page in q.expected_pages), q.id
        assert q.answerable == (q.expected_answer is not None), q.id
        assert (q.type == "sans_reponse") == (not q.answerable), q.id


def test_load_questions_reads_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "a", "question": "?", "expected_pages": [], "type": "sans_reponse"}
                ]
            }
        ),
        encoding="utf-8",
    )

    [question] = load_questions(path)

    assert question.expected_answer is None
    assert not question.answerable
