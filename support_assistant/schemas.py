"""
Pydantic models for the /ask endpoint.

AskRequest  -> what the client sends: {"query": "..."}
AskResponse -> the schema every answer (mock or real-LLM) must validate against:
    answer     : str            - the natural-language answer
    sources    : list[str]      - chunk/document ids used to ground the answer
                                   (empty list for general_question answers)
    confidence : float in [0,1] - how confident the pipeline is in the answer
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer.")
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk/document ids used to ground the answer. Empty for general_question.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )
