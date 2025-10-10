"""
backend/api/main.py
FastAPI application entry point.
"""
from __future__ import annotations
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .routes.ingest import router as ingest_router
from .routes.query import router as query_router
from .routes.eval import router as eval_router
from .routes.sessions import router as sessions_router
from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger("api")


class _RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }))
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    settings = get_settings()
    logger.info(f"Starting RAG Research Agent [{settings.app_env}]")
    logger.info(f"Vector DB: {settings.vector_db}")
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set — queries will fail. Add it to .env")
    else:
        logger.info("Gemini API key: found")
    logger.info("Ready. Embeddings load on first request.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Research Agent API",
        description=(
            "Multi-source AI research agent with hybrid search, "
            "re-ranking, citation tracking, and RAGAS evaluation."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(_RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(eval_router)
    app.include_router(sessions_router)

    @app.get("/health")
    async def health():
        from ..retrieval.vector_store import get_vector_store
        try:
            vs = get_vector_store()
            num_sources = len(vs.get_source_ids())
        except Exception:
            num_sources = 0
        return {
            "status": "ok",
            "vector_db": settings.vector_db,
            "num_sources": num_sources,
            "version": "1.0.0",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=s.app_host,
        port=s.app_port,
        reload=s.app_env == "development",
        log_level=s.log_level.lower(),
    )
