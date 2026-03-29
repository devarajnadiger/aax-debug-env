"""AaxDebugEnv — core environment logic (no openenv_core runtime dependency).

State lives in a module-level session store so it persists across the
per-request instances that FastAPI creates.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from .grader import Grader
from .models import AaxAction, AaxObservation, GradeResult
from .reward_engine import RewardEngine
from .state_manager import StateManager

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tasks.json")


def _load_tasks() -> Dict[str, Dict[str, Any]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)
    return {t["id"]: t for t in raw}


_TASKS: Dict[str, Dict[str, Any]] = _load_tasks()
_SESSIONS: Dict[str, StateManager] = {}
_SESSION_TASKS: Dict[str, Dict[str, Any]] = {}
_CURRENT_SESSION_ID: Optional[str] = None

_reward_engine = RewardEngine()
_grader = Grader()


class AaxDebugEnv:
    """Ask–Act–Explore mobile debugging environment."""

    def __init__(self) -> None:
        self._session_id: Optional[str] = None

    def reset(self, task_id: Optional[str] = None, **kwargs: Any) -> AaxObservation:
        global _CURRENT_SESSION_ID
        tid = task_id or "task_easy"
        if tid not in _TASKS:
            tid = "task_easy"
        task = _TASKS[tid]
        sid = str(uuid.uuid4())
        self._session_id = sid
        _CURRENT_SESSION_ID = sid
        sm = StateManager(task, max_steps=task.get("max_steps", 8))
        _SESSIONS[sid] = sm
        _SESSION_TASKS[sid] = task
        return self._build_obs(sm, task, reward=None, done=False)

    def step(self, action: AaxAction) -> AaxObservation:
        sid = self._session_id or _CURRENT_SESSION_ID
        if sid is None or sid not in _SESSIONS:
            return self.reset()
        sm = _SESSIONS[sid]
        task = _SESSION_TASKS[sid]
        if sm.is_done():
            obs = self._build_obs(sm, task, reward=0.0, done=True)
            obs.metadata["warning"] = "Episode already finished."
            return obs
        if action.type == "explore":
            target = action.target or ""
            valid = sm.explore_target_valid(target)
            seen = sm.explore_already_seen(target)
            sm.apply_explore(target)
            rwd = _reward_engine.compute(action, explore_target_valid=valid, explore_already_seen=seen)
        elif action.type == "ask":
            sm.apply_ask(action.content)
            rwd = _reward_engine.compute(action)
        elif action.type == "act":
            correct = sm.apply_act(action.content)
            rwd = _reward_engine.compute(action, act_correct=correct)
        else:
            rwd = _reward_engine.compute(action)
        done = sm.is_done()
        obs = self._build_obs(sm, task, reward=rwd.value, done=done)
        obs.metadata["reward_reason"] = rwd.reason
        obs.metadata["solved"] = sm.solved
        return obs

    def get_state(self) -> AaxObservation:
        sid = self._session_id or _CURRENT_SESSION_ID
        if sid and sid in _SESSIONS:
            sm = _SESSIONS[sid]
            task = _SESSION_TASKS[sid]
            return self._build_obs(sm, task, reward=None, done=sm.is_done())
        return AaxObservation(task_id="none", task="No active session.")

    def grade(self) -> GradeResult:
        sid = self._session_id or _CURRENT_SESSION_ID
        if sid and sid in _SESSIONS:
            sm = _SESSIONS[sid]
            task = _SESSION_TASKS[sid]
            return _grader.grade(task, solved=sm.solved, steps_taken=sm.steps_taken, ask_count=sm.ask_count)
        return GradeResult(score=0.0, solved=False, efficient=False, minimal_ask=True, breakdown={}, summary="No active session.")

    def available_tasks(self) -> List[str]:
        order = {"easy": 0, "medium": 1, "hard": 2}
        return sorted(_TASKS.keys(), key=lambda t: order.get(_TASKS[t]["difficulty"], 9))

    def task_info(self, task_id: str) -> Dict[str, Any]:
        t = _TASKS[task_id]
        return {"id": t["id"], "difficulty": t["difficulty"], "title": t["title"], "scenario": t["scenario"], "max_steps": t["max_steps"]}

    @staticmethod
    def _build_obs(sm: StateManager, task: Dict[str, Any], reward: Optional[float], done: bool) -> AaxObservation:
        return AaxObservation(
            task_id=task["id"], task=task["scenario"], difficulty=task["difficulty"],
            scenario=task["scenario"], screen=sm._screen, logs=sm._logs,
            revealed_info=list(sm._revealed), history=list(sm._history),
            steps_taken=sm._steps_taken, steps_left=sm.steps_left,
            ask_count=sm._ask_count, reward=reward, done=done,
        )
