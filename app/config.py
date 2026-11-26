"""Configuration lue depuis les variables d'environnement (préfixe ``COPILOTE_``) ou ``.env``."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOTE_", env_file=".env", extra="ignore")

    app_name: str = "Copilote ADV"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: LogLevel = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
