"""AaxDebugEnv — OpenEnv-compliant main environment.

Public API:
    env.reset(task_id)                → Observation
    env.step(action)                  → (Observation, Reward, done, info)
    env.state()                       → Observation
    env.grade()                       → GradeResult
    env.available_tasks()             → List[str]
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .grader import Grader
from .models import Action, GradeResult, Observation, Reward, StepResult
from .reward_engine import RewardEngine
from .state_manager import StateManager

# Path to bundled task data
_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tasks.json")


def _load_tasks() -> Dict[str, Dict[str, Any]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)
    return {task["id"]: task for task in raw}


class AaxDebugEnv:
    """Ask–Act–Explore mobile debugging environment.

    Usage
    -----
    >>> env = AaxDebugEnv()
    >>> obs = env.reset("task_easy")
    >>> action = Action(type="explore", target="stack_trace")
    >>> obs, reward, done, info = env.step(action)
    >>> result = env.grade()
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = _load_tasks()
        self._reward_engine = RewardEngine()
        self._grader = Grader()
        self._state_mgr: Optional[StateManager] = None
        self._current_task: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Initialise a new episode.

        Args:
            task_id: One of the task IDs from tasks.json.  If None, defaults
                     to "task_easy".

        Returns:
            Initial observation.
        """
        if task_id is None:
            task_id = "task_easy"

        if task_id not in self._tasks:
            available = list(self._tasks.keys())
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {available}")

        self._current_task = self._tasks[task_id]
        max_steps = self._current_task.get("max_steps", 8)
        self._state_mgr = StateManager(self._current_task, max_steps=max_steps)
        return self._state_mgr.observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Apply an action and advance the environment one step.

        Args:
            action: An Action instance.

        Returns:
            (observation, reward, done, info)
        """
        self._assert_active()

        sm = self._state_mgr  # type: ignore[union-attr]

        # If already done, return terminal state without changing anything
        if sm.is_done():
            obs = sm.observation()
            reward = Reward(value=0.0, reason="Episode already finished.")
            return obs, reward, True, {"warning": "Episode already done."}

        # Dispatch to state manager
        if action.type == "explore":
            target = action.target or ""
            valid = sm.explore_target_valid(target)
            seen = sm.explore_already_seen(target)
            sm.apply_explore(target)
            reward = self._reward_engine.compute(
                action,
                explore_target_valid=valid,
                explore_already_seen=seen,
            )

        elif action.type == "ask":
            sm.apply_ask(action.content)
            reward = self._reward_engine.compute(action)

        elif action.type == "act":
            correct = sm.apply_act(action.content)
            reward = self._reward_engine.compute(action, act_correct=correct)

        else:
            reward = Reward(value=0.0, reason="Unknown action type — no-op.")

        # Check terminal conditions
        done = sm.is_done()

        # If steps exhausted without solving, apply timeout penalty
        if done and not sm.solved and sm.steps_left == 0:
            timeout_reward = self._reward_engine.compute(
                action, timed_out=True
            )
            reward = Reward(
                value=reward.value + timeout_reward.value,
                reason=f"{reward.reason} {timeout_reward.reason}",
            )

        obs = sm.observation()
        info: dict = {
            "solved": sm.solved,
            "steps_taken": sm.steps_taken,
            "ask_count": sm.ask_count,
        }
        return obs, reward, done, info

    def state(self) -> Observation:
        """Return the current observation without advancing the environment."""
        self._assert_active()
        return self._state_mgr.observation()  # type: ignore[union-attr]

    def grade(self) -> GradeResult:
        """Compute the final score for the completed episode.

        Can be called at any point, but most meaningful after done=True.
        """
        self._assert_active()
        sm = self._state_mgr  # type: ignore[union-attr]
        return self._grader.grade(
            self._current_task,  # type: ignore[arg-type]
            solved=sm.solved,
            steps_taken=sm.steps_taken,
            ask_count=sm.ask_count,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def available_tasks(self) -> List[str]:
        """Return list of task IDs sorted by difficulty."""
        order = {"easy": 0, "medium": 1, "hard": 2}
        return sorted(
            self._tasks.keys(),
            key=lambda tid: order.get(self._tasks[tid]["difficulty"], 99),
        )

    def task_info(self, task_id: str) -> Dict[str, Any]:
        """Return public metadata for a task (no ground truth exposed)."""
        task = self._tasks[task_id]
        return {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "title": task["title"],
            "scenario": task["scenario"],
            "max_steps": task["max_steps"],
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _assert_active(self) -> None:
        if self._state_mgr is None or self._current_task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
