"""FastAPI server — exposes AaxDebugEnv over HTTP for Hugging Face Spaces / Docker.

Endpoints:
    GET  /tasks              → list available task IDs and metadata
    GET  /tasks/{task_id}    → metadata for one task (no ground truth)
    POST /reset              → start a new episode
    POST /step               → advance the episode one step
    GET  /state              → current observation (read-only)
    GET  /grade              → final score for the current episode

Session ID:
    /reset returns a session_id in the response body AND sets a cookie.
    All subsequent calls accept session_id via:
      1. Cookie         (session_id=<id>)
      2. Header         (X-Session-ID: <id>)
      3. Query param    (?session_id=<id>)
    This ensures compatibility with automated checkers that don't handle cookies.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional

from fastapi import Cookie, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import AaxDebugEnv
from environment.models import Action, GradeResult, Observation, Reward

app = FastAPI(
    title="Ask–Act–Explore Debug Environment",
    description="Cost-aware mobile debugging environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → AaxDebugEnv instance
_sessions: Dict[str, AaxDebugEnv] = {}


# ------------------------------------------------------------------ #
# Request / Response schemas                                           #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    action: Action
    session_id: Optional[str] = None   # callers may embed session_id in body


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward           # {"value": float, "reason": str}
    done: bool
    info: dict


# ------------------------------------------------------------------ #
# Session helpers                                                      #
# ------------------------------------------------------------------ #

def _resolve_session(
    body_sid: Optional[str] = None,
    cookie_sid: Optional[str] = None,
    header_sid: Optional[str] = None,
    query_sid: Optional[str] = None,
) -> AaxDebugEnv:
    """Resolve session_id from any source and return the env."""
    sid = body_sid or header_sid or query_sid or cookie_sid
    if not sid or sid not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )
    return _sessions[sid]


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
    tmp = AaxDebugEnv()
    return {"tasks": [tmp.task_info(tid) for tid in tmp.available_tasks()]}


@app.get("/tasks/{task_id}", tags=["tasks"])
def get_task(task_id: str):
    tmp = AaxDebugEnv()
    try:
        return tmp.task_info(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")


@app.post("/reset", tags=["environment"], response_model=ResetResponse)
def reset(
    response: Response,
    body: Optional[ResetRequest] = None,
):
    """Start a new episode. Returns session_id + initial observation.

    Body is fully optional — defaults to task_easy when omitted.
    """
    task_id = (body.task_id if body else None) or "task_easy"
    sid, env = _new_session()

    try:
        obs = env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Set cookie for browser clients; return in body for API clients
    response.set_cookie("session_id", sid, httponly=True, samesite="lax")
    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step", tags=["environment"], response_model=StepResponse)
def step(
    body: StepRequest,
    session_id: Optional[str] = Cookie(default=None),
    x_session_id: Optional[str] = Header(default=None),
    sid: Optional[str] = Query(default=None, alias="session_id"),
):
    """Advance the episode by one action."""
    env = _resolve_session(
        body_sid=body.session_id,
        cookie_sid=session_id,
        header_sid=x_session_id,
        query_sid=sid,
    )
    obs, reward, done, info = env.step(body.action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", tags=["environment"], response_model=Observation)
def get_state(
    session_id: Optional[str] = Cookie(default=None),
    x_session_id: Optional[str] = Header(default=None),
    sid: Optional[str] = Query(default=None, alias="session_id"),
):
    env = _resolve_session(cookie_sid=session_id, header_sid=x_session_id, query_sid=sid)
    return env.state()


@app.get("/grade", tags=["environment"], response_model=GradeResult)
def grade(
    session_id: Optional[str] = Cookie(default=None),
    x_session_id: Optional[str] = Header(default=None),
    sid: Optional[str] = Query(default=None, alias="session_id"),
):
    env = _resolve_session(cookie_sid=session_id, header_sid=x_session_id, query_sid=sid)
    return env.grade()


# ------------------------------------------------------------------ #
# Dev server entry point                                               #
# ------------------------------------------------------------------ #

def main():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
