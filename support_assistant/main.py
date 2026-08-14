"""
FastAPI wrapper for the Zepto support-assistant LangGraph pipeline.

Run locally (after `python ingest.py` has populated ./chroma_db once):
    uvicorn main:app --host 0.0.0.0 --port 7860

MOCK_LLM defaults to "1" (mock/deterministic baseline, no LLM/API key needed).
Set MOCK_LLM=0 to opt into the optional real-LLM extension.
"""

from fastapi import FastAPI, HTTPException

from schemas import AskRequest, AskResponse
from graph import run_query

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based support assistant over Zepto's policy corpus, orchestrated with LangGraph.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return run_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
