"""Typed models for the Ask-Act-Explore environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AaxObservation(BaseModel):
    task_id: str = ""
    task: str = ""
    difficulty: str = ""
    scenario: str = ""
    screen: str = ""
    logs: str = ""
    revealed_info: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    steps_taken: int = 0
    steps_left: int = 8
    ask_count: int = 0
    reward: Optional[float] = None
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AaxAction(BaseModel):
    type: Literal["act", "explore", "ask"]
    target: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    solved: bool
    efficient: bool
    minimal_ask: bool
    breakdown: dict
    summary: str
