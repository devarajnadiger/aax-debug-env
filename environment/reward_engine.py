"""Reward Engine — dense, cost-aware reward computation.

Reward table (per design spec):
  Useful explore (new info revealed)  : +0.2
  Redundant explore (already seen)    : -0.1   (penalise repetition)
  Ask                                 : -0.2
  Correct act (solved)                : +0.5
  Wrong act                           : -0.3
  Timeout (steps exhausted)           : -0.1   (applied at terminal step)

Properties:
  - Dense (signal at every step)
  - No infinite explore reward: each explore target can only be rewarded once
  - Clipped to [-1.0, +1.0] cumulatively — individual step values can exceed 0.5
"""

from __future__ import annotations

from .models import Action, Reward


class RewardEngine:
    """Stateless reward calculator — all context passed in per call."""

    # ------------------------------------------------------------------ #
    # Constants                                                            #
    # ------------------------------------------------------------------ #
    EXPLORE_NEW: float = 0.2
    EXPLORE_REPEAT: float = -0.1
    ASK: float = -0.2
    ACT_CORRECT: float = 0.5
    ACT_WRONG: float = -0.3
    TIMEOUT: float = -0.1

    def compute(
        self,
        action: Action,
        *,
        explore_target_valid: bool = False,
        explore_already_seen: bool = False,
        act_correct: bool = False,
        timed_out: bool = False,
    ) -> Reward:
        """Compute the immediate reward for one step.

        Args:
            action:               The action taken.
            explore_target_valid: Whether the explore target exists in the task.
            explore_already_seen: Whether this target was already explored.
            act_correct:          Whether the act action solved the task.
            timed_out:            Whether the step budget was exhausted.
        """
        if timed_out:
            return Reward(value=self.TIMEOUT, reason="Step budget exhausted without solving.")

        if action.type == "explore":
            if explore_already_seen:
                return Reward(
                    value=self.EXPLORE_REPEAT,
                    reason=f"Already explored '{action.target}' — no new information.",
                )
            if explore_target_valid:
                return Reward(
                    value=self.EXPLORE_NEW,
                    reason=f"Explored '{action.target}' — new information revealed.",
                )
            # Target doesn't exist in this task
            return Reward(
                value=self.EXPLORE_REPEAT,
                reason=f"No information found for '{action.target}'.",
            )

        if action.type == "ask":
            return Reward(value=self.ASK, reason="Asked oracle — hint provided (costly).")

        if action.type == "act":
            if act_correct:
                return Reward(value=self.ACT_CORRECT, reason="Correct fix applied — task solved!")
            return Reward(value=self.ACT_WRONG, reason="Incorrect fix — problem persists.")

        # Should never reach here with valid Action type, but safe fallback
        return Reward(value=0.0, reason="Unknown action type.")
