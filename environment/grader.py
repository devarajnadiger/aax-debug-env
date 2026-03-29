"""Grader — deterministic final scorer (0.0 → 1.0).

Score breakdown:
  solved          → 0.6  (task was correctly fixed)
  efficient_steps → 0.2  (completed within min_steps_for_bonus)
  minimal_ask     → 0.2  (asked oracle 0 or 1 times)

Properties:
  - Deterministic
  - Bounded [0.0, 1.0]
  - Reproducible across identical episodes
"""

from __future__ import annotations

from typing import Any, Dict

from .models import GradeResult


class Grader:
    """Scores a completed episode against the task definition."""

    SOLVED_WEIGHT: float = 0.6
    EFFICIENT_WEIGHT: float = 0.2
    MINIMAL_ASK_WEIGHT: float = 0.2

    def grade(
        self,
        task: Dict[str, Any],
        *,
        solved: bool,
        steps_taken: int,
        ask_count: int,
    ) -> GradeResult:
        """Compute final score.

        Args:
            task:        The task definition dict (from tasks.json).
            solved:      Whether the agent correctly fixed the bug.
            steps_taken: Total steps used in the episode.
            ask_count:   How many times the agent queried the oracle.
        """
        min_steps: int = task.get("min_steps_for_bonus", task["max_steps"])

        efficient = solved and steps_taken <= min_steps
        minimal_ask = ask_count <= 1

        score = 0.0
        if solved:
            score += self.SOLVED_WEIGHT
        if efficient:
            score += self.EFFICIENT_WEIGHT
        if minimal_ask:
            score += self.MINIMAL_ASK_WEIGHT

        breakdown = {
            "solved": {"earned": self.SOLVED_WEIGHT if solved else 0.0, "max": self.SOLVED_WEIGHT},
            "efficient_steps": {"earned": self.EFFICIENT_WEIGHT if efficient else 0.0, "max": self.EFFICIENT_WEIGHT, "steps_taken": steps_taken, "threshold": min_steps},
            "minimal_ask": {"earned": self.MINIMAL_ASK_WEIGHT if minimal_ask else 0.0, "max": self.MINIMAL_ASK_WEIGHT, "ask_count": ask_count},
        }

        summary = self._build_summary(solved, efficient, minimal_ask, score, steps_taken, ask_count)

        return GradeResult(
            score=round(score, 4),
            solved=solved,
            efficient=efficient,
            minimal_ask=minimal_ask,
            breakdown=breakdown,
            summary=summary,
        )

    @staticmethod
    def _build_summary(
        solved: bool,
        efficient: bool,
        minimal_ask: bool,
        score: float,
        steps_taken: int,
        ask_count: int,
    ) -> str:
        parts = []
        if solved:
            parts.append("Task solved.")
        else:
            parts.append("Task NOT solved.")
        if efficient:
            parts.append(f"Solved efficiently in {steps_taken} steps (bonus awarded).")
        if not minimal_ask:
            parts.append(f"Asked oracle {ask_count} time(s) — efficiency bonus lost.")
        parts.append(f"Final score: {score:.2f}/1.00")
        return " ".join(parts)
