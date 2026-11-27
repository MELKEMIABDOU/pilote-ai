"""L'adaptateur Ollama sans Ollama : ``httpx.MockTransport`` simule le serveur."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.adapters.llm.ollama import OllamaLLM
from app.domain.errors import LLMError
from app.ports.llm import StructuredLLM

SCHEMA = {"type": "object", "properties": {"produit": {"type": "string"}}}


def llm_answering(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaLLM:
    return OllamaLLM("qwen3:8b", transport=httpx.MockTransport(handler))


def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_respects_the_structured_llm_contract() -> None:
    llm: StructuredLLM = llm_answering(lambda request: chat_reply("{}"))

    assert llm.model_name == "qwen3:8b"


def test_sends_the_schema_and_returns_the_json_object() -> None:
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return chat_reply('{"produit": "Café 250 g"}')

    result = llm_answering(handler).generate_json(system="sys", prompt="doc", schema=SCHEMA)

    assert result == {"produit": "Café 250 g"}
    assert sent["format"] == SCHEMA
    assert sent["think"] is False
    assert sent["options"]["temperature"] == 0
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "num_gpu" not in sent["options"]  # par défaut, Ollama choisit


def test_gpu_layers_are_sent_as_num_gpu() -> None:
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return chat_reply("{}")

    llm = OllamaLLM("qwen3:8b", gpu_layers=12, transport=httpx.MockTransport(handler))
    llm.generate_json(system="sys", prompt="doc", schema=SCHEMA)

    assert sent["options"]["num_gpu"] == 12


def test_negative_gpu_layers_are_refused() -> None:
    with pytest.raises(ValueError, match="gpu_layers"):
        OllamaLLM(gpu_layers=-1)


def test_ollama_not_running_gives_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError, match="lancer l'application Ollama"):
        llm_answering(handler).generate_json(system="s", prompt="p", schema=SCHEMA)


def test_missing_model_tells_how_to_download_it() -> None:
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        llm_answering(lambda request: httpx.Response(404)).generate_json(
            system="s", prompt="p", schema=SCHEMA
        )


@pytest.mark.parametrize("content", ["pas du json", "[1, 2]"])
def test_invalid_answers_are_errors(content: str) -> None:
    with pytest.raises(LLMError, match="JSON"):
        llm_answering(lambda request: chat_reply(content)).generate_json(
            system="s", prompt="p", schema=SCHEMA
        )
