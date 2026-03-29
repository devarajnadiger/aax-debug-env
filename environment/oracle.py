"""Human Oracle — simulates a senior engineer giving deterministic hints.

The oracle never reveals the full solution; it provides progressively more
specific hints keyed by the number of times it has been asked.
"""

from __future__ import annotations

from typing import Dict


class HumanOracle:
    """Deterministic hint provider for a single task instance.

    Hints are indexed by ask_count (1-based). Once all hints are exhausted
    the oracle repeats its final hint, indicating it has no more to add.
    """

    def __init__(self, hints: Dict[str, str]) -> None:
        # hints keys are string integers: "1", "2", ...
        self._hints = hints
        self._max_hint = max(int(k) for k in hints) if hints else 0

    def ask(self, ask_count: int, question: str | None = None) -> str:
        """Return the hint for this ask attempt.

        Args:
            ask_count: How many times the agent has asked (including this one).
            question:  Optional free-text question from the agent (ignored for
                       determinism, but logged in the observation history).
        """
        key = str(min(ask_count, self._max_hint))
        hint = self._hints.get(key, "I have no more hints for you.")
        return hint
