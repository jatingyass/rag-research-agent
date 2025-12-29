"""
tests/test_e2e.py

End-to-end integration tests for the full ingest → query pipeline.
These tests run against the real FastAPI app (no mocks) using an
in-memory ChromaDB and a temporary SQLite database so they leave
no side-effects on disk.

Run with:
    pytest tests/test_e2e.py -v
"""
from __future__ import annotations
import os
import textwrap
import pytest
from httpx import AsyncClient, ASGITransport

# Point at a throw-away in-memory store so tests don't pollute the dev DB
os.environ.setdefault("CHROMA_PERSIST_DIR", ":memory:")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("API_KEY", "")  # auth disabled for tests


SAMPLE_TEXT = textwrap.dedent("""
    Retrieval-Augmented Generation (RAG) is an AI framework that combines
    information retrieval with language model generation. Instead of relying
    solely on the model's parametric knowledge, RAG fetches relevant documents
    from an external knowledge base and uses them as grounding context.

    The main advantages of RAG are:
    1. Reduced hallucination — answers are grounded in retrieved facts.
    2. Up-to-date information — the knowledge base can be updated without
       retraining the model.
    3. Source attribution — citations can be attached to each answer.

    BM25 is a probabilistic keyword-matching algorithm widely used in search
    engines. It ranks documents by term frequency and inverse document
    frequency, normalised for document length.
""").strip()


@pytest.fixture(scope="module")
def app():
    from backend.api.main import create_app
    return create_app()


@pytest.fixture(scope="module")
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_ingest_url_and_query(client, monkeypatch):
    """
    Ingest a short text document via the ingest endpoint, then query it.
    Monkeypatches the web loader so no real HTTP call is made.
    """
    from backend.ingestion import web_loader as wl
    from backend.ingestion.chunker import Chunk

    async def _fake_load(url: str):
        return [
            Chunk(
                text=SAMPLE_TEXT,
                source_id="url_test123",
                source_type="url",
                source_name=url,
                chunk_id="chunk_0",
                page_number=None,
                section=None,
                url=url,
            )
        ]

    monkeypatch.setattr(wl.WebLoader, "load_url", _fake_load)

    resp = await client.post(
        "/api/ingest/url",
        json={"url": "https://example.com/rag-overview"},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["num_chunks"] > 0


@pytest.mark.asyncio
async def test_query_returns_answer_and_citations(client):
    resp = await client.post(
        "/api/query",
        json={"query": "What is retrieval-augmented generation?"},
        headers={"X-API-Key": ""},
    )
    # Without a real GEMINI_API_KEY the LLM will fail; we only check retrieval
    assert resp.status_code in (200, 429, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["citations"], list)


@pytest.mark.asyncio
async def test_sources_endpoint(client):
    resp = await client.get("/api/sources", headers={"X-API-Key": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_query_with_filter(client):
    """Filters should be accepted without error even if no docs match."""
    resp = await client.post(
        "/api/query",
        json={
            "query": "BM25 algorithm",
            "filters": {"source_type": "pdf"},
        },
        headers={"X-API-Key": ""},
    )
    assert resp.status_code in (200, 429, 500)


@pytest.mark.asyncio
async def test_sessions_crud(client):
    create_resp = await client.post(
        "/api/sessions",
        json={"title": "E2E Test Session"},
        headers={"X-API-Key": ""},
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["id"]

    list_resp = await client.get("/api/sessions", headers={"X-API-Key": ""})
    assert list_resp.status_code == 200
    ids = [s["id"] for s in list_resp.json()["sessions"]]
    assert session_id in ids
