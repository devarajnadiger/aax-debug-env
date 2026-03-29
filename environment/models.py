"""Typed models for the Ask-Act-Explore environment (OpenEnv compliant).

Inherits from openenv_core base classes so the framework can auto-generate
/schema, /reset, and /step endpoints with correct validation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from openenv_core import Action as BaseAction
from openenv_core import Observation as BaseObservation


class AaxObservation(BaseObservation):
    """Full observation returned by reset() and step()."""

    task_id: str = ""
    task: str = ""                          # human-readable scenario (required by OpenEnv spec)
    difficulty: str = ""
    scenario: str = ""
    screen: str = ""
    logs: str = ""
    revealed_info: List[str] = []
    history: List[str] = []
    steps_taken: int = 0
    steps_left: int = 8
    ask_count: int = 0
    # done + reward + metadata inherited from BaseObservation


class AaxAction(BaseAction):
    """Action the agent takes at each step."""

    type: Literal["act", "explore", "ask"]
    target: Optional[str] = None    # explore: which source; act: unused
    content: Optional[str] = None   # ask: question text; act: fix description
    # metadata inherited from BaseAction


# Keep GradeResult as a plain Pydantic model (not an OpenEnv type)
from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    solved: bool
    efficient: bool
    minimal_ask: bool
    breakdown: dict
    summary: str
