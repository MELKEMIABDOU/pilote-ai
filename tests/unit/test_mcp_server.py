"""Serveur MCP testé comme un vrai client MCP le voit, mais dans le même processus.

``mcp.Client(server)`` se connecte directement au serveur, sans sous-processus ni réseau ;
la recherche utilise les faux (pas de modèle, pas de base).
"""

from pathlib import Path

import pytest
from mcp import Client

from app.application.ingest_document import IngestDocument
from app.application.search_documents import MAX_TOP_K, SearchDocuments
from app.config import Settings
from app.mcp_server.__main__ import main
from app.mcp_server.server import create_server
from tests.fixtures.fakes import FakeEmbedder, FakeExtractor, InMemoryVectorStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def search(tmp_path: Path) -> SearchDocuments:
    """Un document indexé pour client-a uniquement."""
    store, embedder = InMemoryVectorStore(), FakeEmbedder()
    document = tmp_path / "rapport.pdf"
    document.write_bytes(b"contenu")
    extractor = FakeExtractor([(4, "Le prix du riz est de 1,20 €."), (6, "Stock client.")])
    IngestDocument(extractor, embedder, store)(document, client_id="client-a")
    return SearchDocuments(embedder, store)


async def test_only_the_read_only_search_tool_is_exposed(search: SearchDocuments) -> None:
    async with Client(create_server(search, client_id="client-a")) as client:
        [tool] = (await client.list_tools()).tools

    assert tool.name == "chercher_documents"
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False


async def test_the_agent_cannot_choose_the_client(search: SearchDocuments) -> None:
    async with Client(create_server(search, client_id="client-a")) as client:
        [tool] = (await client.list_tools()).tools

    assert set(tool.input_schema["properties"]) == {"question", "top_k"}


async def test_search_returns_passages_with_their_citation(search: SearchDocuments) -> None:
    async with Client(create_server(search, client_id="client-a")) as client:
        result = await client.call_tool("chercher_documents", {"question": "prix du riz"})

    assert not result.is_error
    assert result.structured_content is not None
    best = result.structured_content["passages"][0]
    assert (best["source"], best["pages"]) == ("rapport.pdf", [4])
    assert "1,20 €" in best["text"]


async def test_server_only_sees_its_own_clients_documents(search: SearchDocuments) -> None:
    async with Client(create_server(search, client_id="client-b")) as client:
        result = await client.call_tool("chercher_documents", {"question": "prix du riz"})

    assert result.structured_content == {"passages": []}


@pytest.mark.parametrize(
    "arguments",
    [
        {"question": ""},
        {"question": "riz", "top_k": 0},
        {"question": "riz", "top_k": MAX_TOP_K + 1},
    ],
)
async def test_invalid_arguments_are_reported_as_tool_errors(
    search: SearchDocuments, arguments: dict[str, object]
) -> None:
    async with Client(create_server(search, client_id="client-a")) as client:
        result = await client.call_tool("chercher_documents", arguments)

    assert result.is_error


async def test_server_requires_a_client(search: SearchDocuments) -> None:
    with pytest.raises(ValueError, match="client_id must not be empty"):
        create_server(search, client_id=" ")


async def test_main_refuses_to_start_without_a_configured_client(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(Settings(mcp_client_id=None)) == 2
    assert "COPILOTE_MCP_CLIENT_ID" in capsys.readouterr().err
