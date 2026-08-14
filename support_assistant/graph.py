"""
LangGraph orchestration for the Zepto support assistant.

State machine:

    classify_intent
          |
          |  (conditional edge on state["intent"])
          |
    +-----+------------------+
    |                        |
    v                        v
retrieve_and_answer     direct_answer
    |                        |
    +-----------+------------+
                v
               END

Every node's *generation* step branches on the MOCK_LLM env var:
  - MOCK_LLM unset or "1"  -> deterministic, rule-based / templated logic (graded baseline)
  - MOCK_LLM == "0"        -> real LLM call (optional, ungraded extension)

The retrieval step inside retrieve_and_answer always runs "for real" in both
modes (local embedding + ChromaDB query, no API key needed).
"""

import os
from typing import List, TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from prompts import build_prompt
from schemas import AskResponse

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

DIRECT_ANSWER_FALLBACK = "I can only answer questions about Zepto policies right now."


def _mock_llm_enabled() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). Only '0' switches to real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


# --- lazy singletons so importing this module doesn't require the model/DB to
#     already be loaded, and so main.py / tests can import quickly ---
_embedder = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


class GraphState(TypedDict):
    query: str
    intent: str          # "policy_question" | "general_question"
    retrieved_ids: List[str]
    retrieved_texts: List[str]
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if _mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call the LLM to classify instead.
        intent = _llm_classify_intent(query)

    return {**state, "intent": intent}


def _llm_classify_intent(query: str) -> str:
    """
    Optional real-LLM classification path (MOCK_LLM=0).
    Placeholder: wire up your LLM client of choice (e.g. Groq) here and
    prompt it to return exactly "policy_question" or "general_question".
    """
    raise NotImplementedError(
        "Real-LLM classification is an optional extension. "
        "Set MOCK_LLM=1 (default) to use the graded mock baseline, or "
        "implement an LLM call here for the optional extension."
    )


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes (local embedding + ChromaDB).
    embedder = _get_embedder()
    collection = _get_collection()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
    retrieved_ids: List[str] = results["ids"][0]
    retrieved_texts: List[str] = results["documents"][0]

    if _mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk_snippet = retrieved_texts[0][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        sources = retrieved_ids
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM, grounded only
        # in the retrieved chunks, with schema-validation retry.
        context_str = "\n".join(
            f'[{doc_id}] "{text}"' for doc_id, text in zip(retrieved_ids, retrieved_texts)
        )
        prompt = build_prompt(query=query, retrieved_context=context_str)
        parsed = _call_llm_with_schema_retry(prompt, retrieved_ids)
        answer = parsed.answer
        sources = parsed.sources
        confidence = parsed.confidence

    return {
        **state,
        "retrieved_ids": retrieved_ids,
        "retrieved_texts": retrieved_texts,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


def _call_llm_with_schema_retry(prompt: str, retrieved_ids: List[str], max_retries: int = 2) -> AskResponse:
    """
    Optional real-LLM answer path (MOCK_LLM=0).
    Calls the LLM, parses its JSON output against AskResponse, and retries up
    to `max_retries` additional times with a corrective instruction if
    validation fails, before giving up and returning a clearly marked error.
    """
    import json

    last_error: Exception | None = None
    attempt_prompt = prompt

    for attempt in range(max_retries + 1):
        try:
            raw_output = _llm_generate(attempt_prompt)  # implement your LLM client call here
            data = json.loads(raw_output)
            # sources must only ever contain ids that were actually retrieved
            data["sources"] = [s for s in data.get("sources", []) if s in retrieved_ids]
            return AskResponse(**data)
        except Exception as e:  # JSON decode error or Pydantic ValidationError
            last_error = e
            attempt_prompt = (
                f"{prompt}\n\n"
                f"# CORRECTION\n"
                f"Your previous response was invalid ({e}). "
                f"Respond again with ONLY a valid JSON object matching the required schema."
            )

    return AskResponse(
        answer=f"[ERROR] LLM failed to produce a schema-valid response after {max_retries + 1} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )


def _llm_generate(prompt: str) -> str:
    """
    Optional real-LLM call (MOCK_LLM=0 extension).
    Placeholder: wire up an LLM API client here (e.g. Groq's OpenAI-compatible
    client) and return the raw text completion.
    """
    raise NotImplementedError(
        "Real-LLM generation is an optional extension. "
        "Set MOCK_LLM=1 (default) to use the graded mock baseline, or "
        "implement an LLM call here for the optional extension."
    )


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if _mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer = DIRECT_ANSWER_FALLBACK
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        answer = _llm_direct_answer(query)

    return {
        **state,
        "retrieved_ids": [],
        "retrieved_texts": [],
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


def _llm_direct_answer(query: str) -> str:
    """Optional real-LLM direct-answer path (MOCK_LLM=0), no retrieval."""
    raise NotImplementedError(
        "Real-LLM direct answering is an optional extension. "
        "Set MOCK_LLM=1 (default) to use the graded mock baseline, or "
        "implement an LLM call here for the optional extension."
    )


# ---------------------------------------------------------------------------
# Conditional routing edge (does not depend on MOCK_LLM)
# ---------------------------------------------------------------------------
def _route_from_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_from_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str) -> AskResponse:
    graph = get_graph()
    initial_state: GraphState = {
        "query": query,
        "intent": "",
        "retrieved_ids": [],
        "retrieved_texts": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }
    final_state = graph.invoke(initial_state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"],
    )
