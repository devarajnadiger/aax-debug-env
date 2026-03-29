"""State Manager — tracks all mutable state for one episode.

Responsibilities:
  - Initialise state from a task definition
  - Apply explore / ask / act transitions
  - Expose current Observation
  - Track what has been revealed (prevents infinite explore reward)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .oracle import HumanOracle


class StateManager:
    """Manages the full mutable state of one debugging episode."""

    def __init__(self, task: Dict[str, Any], max_steps: int) -> None:
        self._task = task
        self._max_steps = max_steps

        # Build a lookup: target_name -> revealed text
        self._explore_map: Dict[str, str] = {
            step["target"]: step["reveals"]
            for step in task.get("explore_steps", [])
        }

        self._oracle = HumanOracle(task.get("oracle_hints", {}))

        # Mutable state
        self._screen: str = task["initial_screen"]
        self._logs: str = task["initial_logs"]
        self._revealed: List[str] = []         # explored targets (in order)
        self._history: List[str] = []          # human-readable action log
        self._steps_taken: int = 0
        self._ask_count: int = 0
        self._solved: bool = False

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def steps_taken(self) -> int:
        return self._steps_taken

    @property
    def ask_count(self) -> int:
        return self._ask_count

    @property
    def steps_left(self) -> int:
        return self._max_steps - self._steps_taken

    def is_done(self) -> bool:
        return self._solved or self._steps_taken >= self._max_steps

    def explore_target_valid(self, target: str) -> bool:
        return target in self._explore_map

    def explore_already_seen(self, target: str) -> bool:
        return target in self._revealed

    def observation(self):
        return Observation(
            task_id=self._task["id"],
            task=self._task["scenario"],
            difficulty=self._task["difficulty"],
            scenario=self._task["scenario"],
            screen=self._screen,
            logs=self._logs,
            revealed_info=list(self._revealed),
            history=list(self._history),
            steps_taken=self._steps_taken,
            steps_left=self.steps_left,
            ask_count=self._ask_count,
        )

    # ------------------------------------------------------------------ #
    # Transitions                                                          #
    # ------------------------------------------------------------------ #

    def apply_explore(self, target: str) -> Optional[str]:
        """Reveal information for `target`. Returns the revealed text or None."""
        self._steps_taken += 1
        if target in self._explore_map and target not in self._revealed:
            self._revealed.append(target)
            text = self._explore_map[target]
            self._logs += f"\n\n[EXPLORE: {target}]\n{text}"
            self._history.append(f"explore:{target} → new info revealed")
            return text
        if target in self._revealed:
            self._history.append(f"explore:{target} → already seen (no new info)")
        else:
            self._history.append(f"explore:{target} → target not found")
        return None

    def apply_ask(self, question: Optional[str] = None) -> str:
        """Query the oracle. Returns the hint."""
        self._steps_taken += 1
        self._ask_count += 1
        hint = self._oracle.ask(self._ask_count, question)
        q_text = f'"{question}"' if question else "(no question)"
        self._history.append(f"ask {q_text} → hint: {hint}")
        self._logs += f"\n\n[ORACLE HINT #{self._ask_count}]\n{hint}"
        return hint

    def apply_act(self, content: Optional[str]) -> bool:
        """Attempt to solve the task. Returns True if correct."""
        self._steps_taken += 1
        correct = self._is_correct_act(content)
        if correct:
            self._solved = True
            self._screen = "Bug fixed. App is running correctly."
            self._history.append(f"act → CORRECT: {content}")
        else:
            self._history.append(f"act → WRONG: {content}")
        return correct

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _is_correct_act(self, content: Optional[str]) -> bool:
        """Determine if the act content matches the ground truth.

        Matching strategy (deterministic):
          1. Check for key phrases from the root_cause in the content.
          2. Check for the fix description keywords.
          3. Any single key term from ground_truth is enough to count.

        This is intentionally forgiving — the grader cares about the act
        being meaningful, not about exact string matching.
        """
        if not content:
            return False

        gt = self._task["ground_truth"]
        content_lower = content.lower()

        # Extract key tokens from root_cause and fix (words > 4 chars)
        key_tokens = set()
        for field in ("root_cause", "fix"):
            for word in gt[field].lower().split():
                word = word.strip(".,;:()")
                if len(word) > 4:
                    key_tokens.add(word)

        # Also accept the filename or line number as a signal
        key_tokens.add(gt["file"].lower())

        matches = sum(1 for tok in key_tokens if tok in content_lower)
        # Require at least 2 key term matches for correctness
        return matches >= 2
