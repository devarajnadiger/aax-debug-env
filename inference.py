"""
Inference Script — Ask-Act-Explore Debug Environment
=====================================================

Mandatory env vars:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key for the LLM router

Optional:
    ENV_URL        HF Space base URL (default: https://dev-nadiger-aax-debug-env.hf.space)

stdout format (strict):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_URL      = os.getenv("ENV_URL",      "https://dev-nadiger-aax-debug-env.hf.space").rstrip("/")

BENCHMARK  = "aax-debug-env"
TASKS      = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS  = 8

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert mobile app debugger interacting with a debugging environment.
At each step respond with a single JSON object — no prose, no markdown.

Schema:
{"type": "explore"|"act"|"ask", "target": "<str|null>", "content": "<str|null>"}

Rules:
- explore: set "target" to the log source you want to inspect.
- act:     set "content" to a precise fix description (file, line, what to change).
- ask:     set "content" to your question for the oracle.

Strategy: explore to gather evidence, act when confident, ask only if stuck.
""").strip()


# ------------------------------------------------------------------ #
# Structured log helpers (mandatory format)                           #
# ------------------------------------------------------------------ #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ------------------------------------------------------------------ #
# Environment HTTP client                                             #
# ------------------------------------------------------------------ #

def env_call(method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
    url  = f"{ENV_URL}{path}"
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def env_reset(task_id: str) -> Dict[str, Any]:
    return env_call("POST", "/reset", {"task_id": task_id})


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    return env_call("POST", "/step", {"action": action})


def env_grade() -> Dict[str, Any]:
    return env_call("GET", "/grade")


# ------------------------------------------------------------------ #
# LLM action selection                                                #
# ------------------------------------------------------------------ #

def build_prompt(obs: Dict[str, Any], last_reward: float, last_reason: str) -> str:
    parts = [
        f"Task: {obs.get('task_id')} ({obs.get('difficulty')})",
        f"Scenario: {obs.get('scenario')}",
        f"Screen: {obs.get('screen')}",
        "",
        "Logs:",
        obs.get("logs", ""),
    ]
    if obs.get("revealed_info"):
        parts += ["", "Already explored: " + ", ".join(obs["revealed_info"])]
    if obs.get("history"):
        parts += ["", "History (last 4):"]
        for h in obs["history"][-4:]:
            parts.append(f"  {h}")
    parts += [
        "",
        f"Steps left: {obs.get('steps_left')}  |  Oracle asks used: {obs.get('ask_count')}",
    ]
    if last_reason:
        parts += [f"Last reward: {last_reward:+.2f}  ({last_reason})"]
    parts += ["", "Respond with JSON action only."]
    return "\n".join(parts)


def get_llm_action(client: Any, prompt: str) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action. Returns a fallback on any error."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = (resp.choices[0].message.content or "").strip()
        # strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)
    except Exception:
        return {"type": "explore", "target": "stack_trace"}


# ------------------------------------------------------------------ #
# Heuristic fallback (used when LLM client is unavailable)            #
# ------------------------------------------------------------------ #

_EXPLORE_TARGETS = {
    "task_easy":   ["stack_trace", "source_code"],
    "task_medium": ["network_logs", "auth_flow", "source_code"],
    "task_hard":   ["sync_logs", "conflict_resolver", "database_schema"],
}

_ACT_FIXES = {
    "task_easy":   "Initialize SharedPreferences with getSharedPreferences in MainActivity.java before line 42",
    "task_medium": "Await token refresh before login request in AuthService.kt line 88 to fix race condition",
    "task_hard":   "Use server timestamps instead of device-local timestamps in ConflictResolver.java line 201",
}


def get_heuristic_action(task_id: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    revealed = set(obs.get("revealed_info", []))
    targets  = _EXPLORE_TARGETS.get(task_id, ["stack_trace"])
    for t in targets:
        if t not in revealed:
            return {"type": "explore", "target": t}
    return {"type": "act", "content": _ACT_FIXES.get(task_id, "Fix the root cause")}


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def run_episode(client: Any, task_id: str) -> float:
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0
    obs:         Dict        = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)

        last_reward: float = 0.0
        last_reason: str   = ""

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break

            prompt = build_prompt(obs, last_reward, last_reason)

            if client is not None:
                action = get_llm_action(client, prompt)
            else:
                action = get_heuristic_action(task_id, obs)

            # Validate action schema
            if not isinstance(action, dict) or action.get("type") not in ("act", "explore", "ask"):
                action = {"type": "explore", "target": "stack_trace"}

            action_str = json.dumps(action, separators=(",", ":"))

            obs = env_step(action)

            reward      = float(obs.get("reward") or 0.0)
            done        = bool(obs.get("done", False))
            last_reason = str(obs.get("metadata", {}).get("reward_reason", ""))
            last_reward = reward
            error_msg   = str(obs.get("metadata", {}).get("error", "")) or None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        grade   = env_grade()
        score   = float(grade.get("score", 0.0))
        success = bool(grade.get("solved", False))

    except Exception as exc:
        print(f"[DEBUG] episode error: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    # Build OpenAI client (uses HF router by default)
    client = None
    if OpenAI is not None and HF_TOKEN and HF_TOKEN != "dummy":
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            print(f"[DEBUG] Could not init OpenAI client: {exc}", flush=True)
    else:
        print("[DEBUG] OpenAI client not available — using heuristic agent.", flush=True)

    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
