"""AaxDebugEnv — OpenEnv-compliant main environment.

The framework creates a fresh env instance per HTTP request, so all
mutable episode state lives in a module-level session store keyed by
task_id. This works correctly for single-session use (the validator)
and for sequential multi-user use.

Public API (OpenEnv):
    env.reset(task_id="task_easy", seed=None, episode_id=None)  → AaxObservation
    env.step(action)                                             → AaxObservation
    env.state                                                    → AaxObservation
    env.grade()                                                  → GradeResult
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from openenv_core import Environment

from .grader import Grader
from .models import AaxAction, AaxObservation, GradeResult
from .oracle import HumanOracle
from .reward_engine import RewardEngine
from .state_manager import StateManager

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tasks.json")


def _load_tasks() -> Dict[str, Dict[str, Any]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)
    return {t["id"]: t for t in raw}


_TASKS: Dict[str, Dict[str, Any]] = _load_tasks()

# Module-level session store — persists across the per-request env instances
# that the OpenEnv HTTP framework creates.
_SESSIONS: Dict[str, StateManager] = {}
_SESSION_TASKS: Dict[str, Dict[str, Any]] = {}
_CURRENT_SESSION_ID: Optional[str] = None   # points to the most recently reset session

_reward_engine = RewardEngine()
_grader = Grader()


class AaxDebugEnv(Environment[AaxAction, AaxObservation, AaxObservation]):
    """Ask–Act–Explore mobile debugging environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # OpenEnv API                                                          #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AaxObservation:
        """Initialise a new episode and return the initial observation."""
        tid = task_id or "task_easy"
        if tid not in _TASKS:
            tid = "task_easy"

        task = _TASKS[tid]
        sid = str(episode_id or uuid.uuid4())
        self._session_id = sid

        global _CURRENT_SESSION_ID
        state_mgr = StateManager(task, max_steps=task.get("max_steps", 8))
        _SESSIONS[sid] = state_mgr
        _SESSION_TASKS[sid] = task
        _CURRENT_SESSION_ID = sid

        return self._build_obs(state_mgr, task, reward=None, done=False)

    def step(
        self,
        action: AaxAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AaxObservation:
        """Apply action, update state, return new observation."""
        # Resolve session: prefer instance-local, fall back to global current
        sid = self._session_id or _CURRENT_SESSION_ID
        if sid is None or sid not in _SESSIONS:
            return self.reset()

        sm = _SESSIONS[sid]
        task = _SESSION_TASKS[sid]

        if sm.is_done():
            obs = self._build_obs(sm, task, reward=0.0, done=True)
            obs.metadata["warning"] = "Episode already finished."
            return obs

        # Dispatch action
        if action.type == "explore":
            target = action.target or ""
            valid = sm.explore_target_valid(target)
            seen = sm.explore_already_seen(target)
            sm.apply_explore(target)
            reward_obj = _reward_engine.compute(
                action, explore_target_valid=valid, explore_already_seen=seen
            )

        elif action.type == "ask":
            sm.apply_ask(action.content)
            reward_obj = _reward_engine.compute(action)

        elif action.type == "act":
            correct = sm.apply_act(action.content)
            reward_obj = _reward_engine.compute(action, act_correct=correct)

        else:
            reward_obj = _reward_engine.compute(action)

        done = sm.is_done()

        obs = self._build_obs(sm, task, reward=reward_obj.value, done=done)
        obs.metadata["reward_reason"] = reward_obj.reason
        obs.metadata["solved"] = sm.solved
        obs.metadata["ask_count"] = sm.ask_count
        return obs

    @property
    def state(self) -> AaxObservation:
        """Return the current observation without advancing the episode."""
        if self._session_id and self._session_id in _SESSIONS:
            sm = _SESSIONS[self._session_id]
            task = _SESSION_TASKS[self._session_id]
            return self._build_obs(sm, task, reward=None, done=sm.is_done())
        return AaxObservation(task_id="none", task="No active session.")

    def grade(self) -> GradeResult:
        """Compute the final score for the current episode."""
        if self._session_id and self._session_id in _SESSIONS:
            sm = _SESSIONS[self._session_id]
            task = _SESSION_TASKS[self._session_id]
            return _grader.grade(task, solved=sm.solved, steps_taken=sm.steps_taken, ask_count=sm.ask_count)
        return GradeResult(score=0.0, solved=False, efficient=False, minimal_ask=True, breakdown={}, summary="No active session.")

    def close(self) -> None:
        """No-op: session state lives in module-level store, not the instance."""
        pass

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_obs(
        sm: StateManager,
        task: Dict[str, Any],
        reward: Optional[float],
        done: bool,
    ) -> AaxObservation:
        return AaxObservation(
            task_id=task["id"],
            task=task["scenario"],
            difficulty=task["difficulty"],
            scenario=task["scenario"],
            screen=sm._screen,
            logs=sm._logs,
            revealed_info=list(sm._revealed),
            history=list(sm._history),
            steps_taken=sm._steps_taken,
            steps_left=sm.steps_left,
            ask_count=sm._ask_count,
            reward=reward,
            done=done,
        )
