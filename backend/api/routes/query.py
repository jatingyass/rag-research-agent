"""
backend/api/routes/query.py
Query endpoints — standard JSON and streaming SSE.
All retrieval and sources are scoped to the requesting user.
"""
from __future__ import annotations
import json
import traceback
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...core.user import get_user_id


def _gemini_friendly(e: Exception) -> tuple[int, str]:
    msg = str(e)
    if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
        return 429, (
            "Rate limit reached (free tier: 5 requests/min for Gemini 2.5 Flash). "
            "Please wait ~60 seconds and try again, or upgrade your Google AI plan."
        )
    if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower():
        return 503, (
            "Gemini API is experiencing high demand right now. "
            "Please try again in a few seconds."
        )
    return 500, msg


from ...agents.research_agent import get_research_agent
from ..schemas import QueryRequest, QueryResponse, CitationInfo, SourcesResponse, SourceInfo
from ...retrieval.vector_store import get_vector_store

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user_id: str = Depends(get_user_id)):
    try:
        agent = get_research_agent(user_id)
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        result = agent.query(request.query, chat_history=history, filters=request.filters)

        return QueryResponse(
            answer=result.answer,
            citations=[CitationInfo(**c.to_dict()) for c in result.citations],
            query=result.query,
            contextualized_query=result.contextualized_query,
            num_chunks_retrieved=len(result.context_chunks),
        )

    except Exception as e:
        traceback.print_exc()
        status, detail = _gemini_friendly(e)
        raise HTTPException(status_code=status, detail=detail)


@router.post("/query/stream")
async def query_stream(request: QueryRequest, user_id: str = Depends(get_user_id)):
    agent = get_research_agent(user_id)
    history = [{"role": m.role, "content": m.content} for m in request.chat_history]

    async def event_generator():
        try:
            async for token in agent.stream_query(request.query, chat_history=history, filters=request.filters):
                if token.startswith("\n\n__CITATIONS_JSON__"):
                    json_str = token.replace("\n\n__CITATIONS_JSON__", "")
                    payload = json.loads(json_str)
                    yield f"event: citations\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            traceback.print_exc()
            _, friendly = _gemini_friendly(e)
            yield f"event: error\ndata: {json.dumps({'error': friendly})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sources", response_model=SourcesResponse)
async def list_sources(user_id: str = Depends(get_user_id)):
    try:
        vs = get_vector_store(user_id)
        source_ids = vs.get_source_ids()
        sources = []
        for sid in source_ids:
            source_type = sid.split("_")[0] if "_" in sid else "unknown"
            sources.append(SourceInfo(
                source_id=sid,
                source_name=sid,
                source_type=source_type,
            ))
        return SourcesResponse(sources=sources, total=len(sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str, user_id: str = Depends(get_user_id)):
    try:
        get_vector_store(user_id).delete_source(source_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
