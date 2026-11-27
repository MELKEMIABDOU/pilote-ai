"""Configuration lue depuis les variables d'environnement (préfixe ``COPILOTE_``) ou ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOTE_", env_file=".env", extra="ignore")

    app_name: str = "Copilote ADV"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: LogLevel = "INFO"

    # RAG
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    vector_store_path: Path = Path("data/vector_store.db")

    # Serveur MCP : le SEUL client dont l'agent peut lire les documents (obligatoire pour le
    # lancer). Fixé par la configuration, jamais choisi par l'agent.
    mcp_client_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
