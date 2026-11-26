"""Point d'entrée de l'API : ``uv run uvicorn app.main:app --reload``."""

from fastapi import FastAPI

from app.api.routes import health
from app.config import Settings, get_settings
from app.logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    application = FastAPI(title=settings.app_name)
    application.include_router(health.router)
    return application


app = create_app()
