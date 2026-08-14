# Module 3 — Support Assistant

A RAG-based customer support assistant for Zepto's policy corpus, orchestrated with
LangGraph and served via FastAPI. Graded baseline runs fully offline (`MOCK_LLM`
left unset/`1`) — no signup, no API key, no network call to any LLM provider.

## Setup

```bash
pip install -r requirements.txt
python ingest.py          # builds ./chroma_db from docs/, one-time step
uvicorn main:app --host 0.0.0.0 --port 7860
```

> **Note on this repo's provenance:** every file here (`docs/*.txt`, `ingest.py`,
> `schemas.py`, `prompts.py`, `graph.py`, `main.py`, `Dockerfile`) was written from
> the module spec and is believed correct against it, but it has **not been executed
> end-to-end** in the environment that produced this README — that sandbox has no
> network access, so `sentence-transformers`, `chromadb`, `langgraph`, and `fastapi`
> could not be installed there. Run the two commands above yourself, then replace the
> `<PASTE ACTUAL JSON HERE>` placeholders in the "Example calls" section below with
> the real output before submitting. Do not submit this file with the placeholders
> still in it.

## Architecture

**Ingestion** (`ingest.py`): loads the 8 files in `docs/doc_01.txt` … `doc_08.txt`.
Given how short each policy document is, chunking is one-chunk-per-document — each
file becomes exactly one chunk, tagged with its filename stem (`doc_01` … `doc_08`)
as its id.

**Embedding** (`ingest.py`, function `build_index`): each chunk's text is embedded
locally with `sentence-transformers`' `all-MiniLM-L6-v2` model — no API key, no
network call at inference time. The 8 resulting vectors are written into a
persistent ChromaDB collection named `zepto_policies`, stored on disk under
`./chroma_db`.

**Retrieval** (`graph.py`, node `retrieve_and_answer`): when a query is classified
as `policy_question`, the same `all-MiniLM-L6-v2` model embeds the query, and
`collection.query(...)` retrieves the top-3 most similar chunks from the
`zepto_policies` ChromaDB collection via cosine similarity. This step always runs
for real, in both `MOCK_LLM` modes, since it needs no LLM API.

**Generation** (`graph.py`, node `retrieve_and_answer`, and node `direct_answer`):
this is the only stage that branches on `MOCK_LLM`:
  - `MOCK_LLM` unset or `1` (**graded baseline**): no LLM call anywhere.
    - `retrieve_and_answer` returns a canned string,
      `f"Based on the retrieved context: {top_chunk_snippet}"`, built from the
      first ~200 characters of the single most similar retrieved chunk.
    - `direct_answer` returns the fixed string
      `"I can only answer questions about Zepto policies right now."`
    - The `AskResponse` schema (`schemas.py`) is populated deterministically by
      code: `sources` = the ids of the retrieved chunks (empty for
      `direct_answer`), `confidence` = a fixed `1.0`.
  - `MOCK_LLM=0` (**optional, ungraded extension**): `retrieve_and_answer` builds
    the structured prompt from `prompts.py` (role/context/task/format/length +
    negative constraint + few-shot example), sends it to a real LLM, and parses
    the LLM's JSON output against `AskResponse`, retrying up to 2 additional
    times with a corrective instruction if validation fails before returning a
    clearly marked error. `direct_answer` prompts the LLM directly with no
    retrieval. `classify_intent` also switches from the keyword heuristic to an
    LLM call for classification.

**Routing** (`graph.py`, `_route_from_intent`): a conditional edge out of
`classify_intent` sends `policy_question` queries to `retrieve_and_answer` and
`general_question` queries to `direct_answer`. This routing decision itself does
not depend on `MOCK_LLM` — only the generation step inside each node does.

**Data flow:**
```
docs/*.txt --(ingest.py: chunk + embed)--> ChromaDB "zepto_policies" collection
                                                     |
POST /ask {"query": "..."} --> classify_intent --(conditional edge)--+
                                                                      |
                                        +-----------------------------+
                                        |                             |
                                retrieve_and_answer              direct_answer
                                (embeds query, queries           (canned string,
                                 ChromaDB top-3, then             mock mode)
                                 canned template in mock
                                 mode / LLM in MOCK_LLM=0)
                                        |                             |
                                        +--------------+--------------+
                                                       v
                                        AskResponse{answer, sources, confidence}
```

## Structured prompt template

See `prompts.py` (`SYSTEM_PROMPT` / `USER_PROMPT_TEMPLATE` / `build_prompt`) for
the full role–context–task–format–length template, negative constraint, and
few-shot example as actual text. This is used only by the optional `MOCK_LLM=0`
extension.

## Example calls (MOCK_LLM left at its default)

Run these once the server is up:

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does it take to get a refund?"}'
```
Expected routing: `policy_question` → `retrieve_and_answer` (keyword `"refund"` matches).

```
<PASTE ACTUAL JSON HERE>
```

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```
Expected routing: `general_question` → `direct_answer` (no policy keyword present).

```
<PASTE ACTUAL JSON HERE>
```

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then POST to http://localhost:7860/ask as above
```

The Dockerfile runs `python ingest.py` at build time (needs no LLM API key —
only local embedding), then serves the FastAPI app with uvicorn on port 7860.

## Optional extensions (not required for grading)

- Real LLM (`MOCK_LLM=0`): implement `_llm_classify_intent`, `_llm_generate`, and
  `_llm_direct_answer` in `graph.py` using an LLM client (e.g. Groq's
  OpenAI-compatible client), reading the API key from an environment variable —
  never hardcode or commit it.
- Hugging Face Spaces deployment: push this Dockerfile to a Space on the free
  community CPU tier, storing the LLM API key as a Space secret, and record the
  live URL here.
