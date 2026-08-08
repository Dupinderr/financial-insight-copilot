"""Shared state for the multi-agent research graph."""

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    # Input
    question: str
    strip_unverified: bool

    # Planner
    plan: dict[str, Any]

    # Data Agent (existing text-to-SQL, wrapped as a tool)
    sql_result: dict[str, Any] | None
    sql_evidence: str | None

    # Research Agent (Chroma retrieval)
    doc_result: dict[str, Any] | None
    doc_evidence: str | None

    # Synthesizer
    draft: str

    # Critic
    critique: dict[str, Any]
    verified_answer: str
    confidence: str
