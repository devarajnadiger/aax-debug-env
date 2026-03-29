"""Typed models for the Ask-Act-Explore environment (OpenEnv compliant)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""

    task_id: str
    task: str = ""           # human-readable task description (alias for scenario)
    difficulty: Literal["easy", "medium", "hard"]
    scenario: str
    screen: str
    logs: str
    revealed_info: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)
    steps_taken: int = 0
    steps_left: int = 8
    ask_count: int = 0


class Action(BaseModel):
    """An action the agent can take."""

    type: Literal["act", "explore", "ask"]
    # For explore: name of what to examine (e.g. "stack_trace", "source_code")
    # For act:     short description of the fix attempt
    # For ask:     optional question to the oracle (free text)
    target: Optional[str] = None
    content: Optional[str] = None


class Reward(BaseModel):
    """Reward signal returned by step()."""

    value: float
    reason: str


class StepResult(BaseModel):
    """Full result of a step() call."""

    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)


class GradeResult(BaseModel):
    """Final evaluation score from the grader."""

    score: float = Field(ge=0.0, le=1.0)
    solved: bool
    efficient: bool
    minimal_ask: bool
    breakdown: dict
    summary: str
