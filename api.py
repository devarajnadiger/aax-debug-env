"""FastAPI server — exposes AaxDebugEnv over HTTP for Hugging Face Spaces / Docker.

Endpoints:
    GET  /tasks              → list available task IDs and metadata
    GET  /tasks/{task_id}    → metadata for one task (no ground truth)
    POST /reset              → start a new episode
    POST /step               → advance the episode one step
    GET  /state              → current observation (read-only)
    GET  /grade              → final score for the current episode

Sessions:
    Each HTTP session is tracked by a session_id cookie.  A single in-process
    dict stores active environments — suitable for a demo / hackathon context.
    For production use, replace with Redis or a proper session store.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional

from fastapi import Cookie, FastAPI, HTTPException, Response
from pydantic import BaseModel

from environment import AaxDebugEnv
from environment.models import Action, GradeResult, Observation

app = FastAPI(
    title="Ask–Act–Explore Debug Environment",
    description="Cost-aware mobile debugging environment for AI agents.",
    version="1.0.0",
)

# In-memory session store: session_id → AaxDebugEnv instance
_sessions: Dict[str, AaxDebugEnv] = {}


# ------------------------------------------------------------------ #
# Request/Response schemas                                             #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward_value: float
    reward_reason: str
    done: bool
    info: dict


class SessionInfo(BaseModel):
    session_id: str
    message: str


# ------------------------------------------------------------------ #
# Session helpers                                                      #
# ------------------------------------------------------------------ #

def _get_env(session_id: Optional[str]) -> AaxDebugEnv:
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return _sessions[session_id]


def _new_session() -> tuple[str, AaxDebugEnv]:
    sid = str(uuid.uuid4())
    env = AaxDebugEnv()
    _sessions[sid] = env
    return sid, env


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "Ask–Act–Explore Debug Environment",
        "version": "1.0.0",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/grade"],
    }


@app.get("/tasks", tags=["tasks"])
def list_tasks():
    """List all available tasks."""
    tmp_env = AaxDebugEnv()
    return {
        "tasks": [
            tmp_env.task_info(tid) for tid in tmp_env.available_tasks()
        ]
    }


@app.get("/tasks/{task_id}", tags=["tasks"])
def get_task(task_id: str):
    """Get metadata for a specific task (ground truth not exposed)."""
    tmp_env = AaxDebugEnv()
    try:
        return tmp_env.task_info(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")


@app.post("/reset", tags=["environment"], response_model=Observation)
def reset(body: ResetRequest, response: Response, session_id: Optional[str] = Cookie(default=None)):
    """Start a new episode.  Returns the initial observation.

    Creates a new session (or reuses existing) and sets a session_id cookie.
    """
    # Always create a fresh env on reset (allows re-running same session)
    sid, env = _new_session()
    try:
        obs = env.reset(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    return obs


@app.post("/step", tags=["environment"], response_model=StepResponse)
def step(
    body: StepRequest,
    session_id: Optional[str] = Cookie(default=None),
):
    """Advance the episode by one action."""
    env = _get_env(session_id)
    obs, reward, done, info = env.step(body.action)
    return StepResponse(
        observation=obs,
        reward_value=reward.value,
        reward_reason=reward.reason,
        done=done,
        info=info,
    )


@app.get("/state", tags=["environment"], response_model=Observation)
def get_state(session_id: Optional[str] = Cookie(default=None)):
    """Return current observation without advancing the episode."""
    env = _get_env(session_id)
    return env.state()


@app.get("/grade", tags=["environment"], response_model=GradeResult)
def grade(session_id: Optional[str] = Cookie(default=None)):
    """Compute and return the final grade for the current episode."""
    env = _get_env(session_id)
    return env.grade()


# ------------------------------------------------------------------ #
# Dev server entry point                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
