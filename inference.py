"""Baseline inference loop — runs an AI agent against the AaxDebugEnv.

The baseline agent uses the Anthropic Claude API.  It receives the current
observation as a structured prompt and must reply with a valid JSON action.

Environment variable required:
    ANTHROPIC_API_KEY

Usage:
    python inference.py [--task task_easy|task_medium|task_hard] [--model MODEL]

Flags:
    --task    Which task to run (default: task_easy)
    --model   Claude model ID (default: claude-haiku-4-5-20251001)
    --verbose Show full prompts and raw responses
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

import anthropic

from environment import AaxDebugEnv
from environment.models import Action, Observation


# ------------------------------------------------------------------ #
# Prompt helpers                                                       #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert mobile app debugger.  You are interacting with a debugging
environment.  At each step you must choose exactly ONE action by responding
with valid JSON matching the schema below.  No prose, no markdown — only JSON.

Action schema:
{
  "type": "explore" | "act" | "ask",
  "target": "<string or null>",
  "content": "<string or null>"
}

Action semantics:
  explore  — examine a specific information source.
             Set "target" to one of the available explore targets listed in the
             observation.  Costs a step but may reveal new clues.
  act      — attempt to fix the bug.
             Set "content" to a clear description of the fix (file, line,
             what to change).  If correct, the task ends with a high reward.
             If wrong, you lose points but can try again.
  ask      — query the human oracle for a hint.
             Set "content" to your question (optional).  The oracle will give
             a partial hint.  Costs a step AND reduces your efficiency score.

Strategy:
  - Explore first to gather evidence before acting.
  - Only ask when genuinely stuck.
  - Act as soon as you have enough evidence — don't over-explore.
""").strip()


def build_user_prompt(obs: Observation, last_reward_reason: Optional[str] = None) -> str:
    parts = [
        f"## Task: {obs.task_id} ({obs.difficulty})",
        f"**Scenario:** {obs.scenario}",
        "",
        f"**Screen:** {obs.screen}",
        "",
        "**Logs:**",
        "```",
        obs.logs,
        "```",
    ]

    if obs.revealed_info:
        parts += ["", "**Already explored:** " + ", ".join(obs.revealed_info)]

    if obs.history:
        parts += ["", "**Action history:**"]
        for h in obs.history[-5:]:  # last 5 to keep context tight
            parts.append(f"  - {h}")

    parts += [
        "",
        f"**Steps left:** {obs.steps_left}  |  **Oracle asks used:** {obs.ask_count}",
    ]

    if last_reward_reason:
        parts += ["", f"**Last reward:** {last_reward_reason}"]

    parts += ["", "What is your next action? Respond with JSON only."]
    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Agent loop                                                           #
# ------------------------------------------------------------------ #

def parse_action(raw: str) -> Optional[Action]:
    """Parse a JSON string into an Action, returning None on failure."""
    try:
        data = json.loads(raw.strip())
        return Action(**data)
    except Exception:
        # Try to extract JSON from a response that has surrounding text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return Action(**data)
            except Exception:
                pass
    return None


def run_episode(
    task_id: str,
    model: str = "claude-haiku-4-5-20251001",
    verbose: bool = False,
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    env = AaxDebugEnv()

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    obs = env.reset(task_id)
    last_reward_reason: Optional[str] = None
    total_reward: float = 0.0
    step_num = 0

    while True:
        step_num += 1
        user_prompt = build_user_prompt(obs, last_reward_reason)

        if verbose:
            print(f"\n--- Step {step_num} | Prompt ---")
            print(user_prompt)

        # Call the model
        message = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,          # deterministic
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_response = message.content[0].text

        if verbose:
            print(f"\n--- Step {step_num} | Raw response ---")
            print(raw_response)

        # Parse action
        action = parse_action(raw_response)
        if action is None:
            print(f"Step {step_num}: Could not parse action — using no-op explore.")
            action = Action(type="explore", target="__invalid__")

        print(f"Step {step_num}: {action.type.upper()}", end="")
        if action.target:
            print(f" → {action.target}", end="")
        if action.content:
            print(f" | {action.content[:80]}", end="")
        print()

        # Advance environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        last_reward_reason = reward.reason

        print(f"         Reward: {reward.value:+.1f}  ({reward.reason})")

        if done:
            break

    # Final grade
    result = env.grade()
    print(f"\n{'='*60}")
    print(f"EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f"Cumulative reward : {total_reward:+.2f}")
    print(f"Final score       : {result.score:.2f} / 1.00")
    print(f"Summary           : {result.summary}")
    print(f"\nBreakdown:")
    for key, val in result.breakdown.items():
        print(f"  {key}: {val}")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run AaxDebugEnv baseline agent")
    parser.add_argument("--task", default="task_easy", choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_episode(task_id=args.task, model=args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
