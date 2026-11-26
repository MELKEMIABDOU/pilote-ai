import pytest

from app.cli import format_pages


@pytest.mark.parametrize(
    ("pages", "label"),
    [((), "page ?"), ((4,), "p.4"), ((4, 5), "p.4-5"), ((4, 5, 6), "p.4-6")],
)
def test_format_pages(pages: tuple[int, ...], label: str) -> None:
    assert format_pages(pages) == label
