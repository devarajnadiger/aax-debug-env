"""OpenEnv server entry point (server/app.py — required by multi-mode spec).

Plain FastAPI app. openenv-core is declared in pyproject.toml for the
validator's static dependency check but is NOT imported at runtime, avoiding
its heavy transitive deps (gradio, openai, etc.) in the Docker image.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import AaxDebugEnv
from environment.models import AaxAction, AaxObservation, GradeResult

app = FastAPI(
    title="Ask–Act–Explore Debug Environment",
    description="Cost-aware mobile debugging environment for AI agents (OpenEnv compliant).",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Module-level env — persists session state across requests
_env = AaxDebugEnv()


# ------------------------------------------------------------------ #
# Request schemas                                                      #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"


class StepRequest(BaseModel):
    action: AaxAction


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"name": "Ask–Act–Explore Debug Environment", "version": "1.0.0",
            "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health", "/schema"]}

@app.get("/schema")
def schema():
    return {"action": AaxAction.model_json_schema(), "observation": AaxObservation.model_json_schema()}

@app.get("/tasks")
def list_tasks():
    return {"tasks": [_env.task_info(t) for t in _env.available_tasks()]}

@app.post("/reset", response_model=AaxObservation)
def reset(body: Optional[ResetRequest] = None):
    task_id = (body.task_id if body else None) or "task_easy"
    return _env.reset(task_id=task_id)

@app.post("/step", response_model=AaxObservation)
def step(body: StepRequest):
    return _env.step(body.action)

@app.get("/state", response_model=AaxObservation)
def state():
    return _env.get_state()

@app.get("/grade", response_model=GradeResult)
def grade():
    return _env.grade()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
