---
title: Ask Act Explore Debug Env
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cost-aware OpenEnv for mobile debugging agents
---

<div align="center">

# 🐛 Ask–Act–Explore
### A Cost-Aware Mobile Debugging Environment for AI Agents

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An OpenEnv-compliant environment where AI agents debug mobile apps by intelligently deciding when to **Act**, **Explore**, or **Ask** — evaluated under a cost-aware reward system.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Why This Matters](#-why-this-matters)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tasks](#-tasks)
- [Action System](#-action-system)
- [Reward Design](#-reward-design)
- [Grader](#-grader-00--10)
- [Typed Models](#-typed-models-openenv-compliant)
- [Quickstart](#-quickstart)
- [API Reference](#-api-reference)
- [Running the Baseline Agent](#-running-the-baseline-agent)
- [Deployment](#-deployment)

---

## 🎯 Overview

**Ask–Act–Explore** is a real-world simulation environment for evaluating AI agent *decision intelligence* in mobile app debugging.

Instead of asking:
> *"Can the agent solve the problem?"*

It asks:
> *"Can the agent solve the problem **efficiently**, with **minimal cost**?"*

The environment rewards agents that gather just enough evidence before acting, and penalises over-reliance on the human oracle.

---

## 💡 Why This Matters

In real-world debugging, engineers don't immediately solve problems. They:
- Inspect logs and traces
- Explore system state step by step
- Ask teammates only when truly stuck

This environment replicates that workflow and evaluates **decision quality**, not just final correctness.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                             │
│              (explores, asks, acts strategically)           │
└────────────────────────┬────────────────────────────────────┘
                         │  Action(type, target, content)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AaxDebugEnv  (env.py)                    │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │StateManager │  │RewardEngine  │  │   HumanOracle     │  │
│  │             │  │              │  │                   │  │
│  │ • screen    │  │ +0.2 explore │  │ deterministic     │  │
│  │ • logs      │  │ -0.2 ask     │  │ progressive hints │  │
│  │ • history   │  │ +0.5 correct │  │                   │  │
│  │ • revealed  │  │ -0.3 wrong   │  └───────────────────┘  │
│  └─────────────┘  └──────────────┘                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Grader                            │   │
│  │     score = solved(0.6) + efficient(0.2) + ask(0.2) │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
              (Observation, Reward, done, info)
```

**Episode flow:**

```
reset(task_id)
    └─► initial Observation
         └─► Agent picks Action
              └─► step(action)
                   └─► (Observation, Reward, done, info)
                        └─► loop until done
                             └─► grade() → score 0.0–1.0
```

---

## 📁 Project Structure

```
aax-debug-env/
│
├── data/
│   └── tasks.json              # 3 task scenarios with ground truth
│
├── environment/
│   ├── __init__.py
│   ├── env.py                  # OpenEnv core: reset / step / state / grade
│   ├── models.py               # Pydantic types: Observation, Action, Reward, GradeResult
│   ├── state_manager.py        # Stateful transitions (explore / ask / act)
│   ├── oracle.py               # Deterministic human oracle (progressive hints)
│   ├── reward_engine.py        # Dense, cost-aware reward computation
│   └── grader.py               # Final scorer (0.0 → 1.0)
│
├── api.py                      # FastAPI server (/reset /step /state /grade)
├── inference.py                # Baseline Claude agent loop
├── requirements.txt
└── Dockerfile                  # HF Spaces compatible (port 7860)
```

---

## 🧪 Tasks

Three tasks of increasing difficulty, each requiring a different debugging strategy.

### 🟢 Task 1 — Easy: NullPointerException on App Launch
| Property | Value |
|----------|-------|
| **ID** | `task_easy` |
| **Scenario** | App crashes immediately on startup with a clear stack trace |
| **Root cause** | `SharedPreferences` accessed before initialization in `MainActivity.java:42` |
| **Optimal strategy** | Explore stack trace → Act |
| **Max steps** | 6 |

### 🟡 Task 2 — Medium: Intermittent Login Failure
| Property | Value |
|----------|-------|
| **ID** | `task_medium` |
| **Scenario** | Login fails ~30% of the time silently; logs are noisy |
| **Root cause** | Race condition — login dispatched before token refresh completes in `AuthService.kt:88` |
| **Optimal strategy** | Explore network logs → Explore auth flow → Act |
| **Max steps** | 8 |

### 🔴 Task 3 — Hard: Silent Data Corruption in Offline Sync
| Property | Value |
|----------|-------|
| **ID** | `task_hard` |
| **Scenario** | Offline edits silently overwritten after sync; logs show "no conflicts detected" |
| **Root cause** | Conflict resolver uses unsynchronized device-local timestamps; Device B clock was skewed ahead, causing stale records to win in `ConflictResolver.java:201` |
| **Optimal strategy** | Explore sync logs → Explore conflict resolver → Ask oracle → Act |
| **Max steps** | 8 |

> **Key design**: Each task requires a different number of explore steps and has a different threshold for when to ask the oracle. This forces the agent to develop a generalised strategy rather than memorising a fixed pattern.

---

## ⚡ Action System

The agent has exactly three action types at each step:

| Action | When to use | Cost |
|--------|-------------|------|
| `explore` | Gather information from a specific source (logs, code, network trace, etc.) | Low |
| `ask` | Query the human oracle for a hint | Medium |
| `act` | Attempt to fix the bug | Terminal if correct |

### Action Schema

```python
class Action(BaseModel):
    type: Literal["act", "explore", "ask"]
    target: Optional[str]   # explore: what to examine; act: what to fix
    content: Optional[str]  # ask: your question; act: fix description
```

### Explore Targets per Task

| Task | Available targets |
|------|-------------------|
| `task_easy` | `stack_trace`, `source_code` |
| `task_medium` | `network_logs`, `auth_flow`, `source_code` |
| `task_hard` | `sync_logs`, `conflict_resolver`, `database_schema`, `network_logs` |

---

## 🧮 Reward Design

Rewards are **dense** — the agent receives a signal at every step.

| Condition | Reward |
|-----------|--------|
| Explore new target (reveals info) | **+0.2** |
| Explore already-seen target | **−0.1** |
| Ask oracle | **−0.2** |
| Correct act (task solved) | **+0.5** |
| Wrong act | **−0.3** |
| Timeout (steps exhausted) | **−0.1** |

**Anti-exploit properties:**
- Each explore target can only yield `+0.2` once — no infinite farming
- Repetitive exploration is penalised
- Over-asking is penalised at grading time even if the episode is solved

---

## 🏆 Grader (0.0 → 1.0)

The grader produces a deterministic, bounded final score from three independent bonuses:

```
score = solved_bonus + efficiency_bonus + ask_bonus
```

| Bonus | Condition | Value |
|-------|-----------|-------|
| **Solved** | Bug correctly identified and fixed | +0.6 |
| **Efficient** | Solved within `min_steps_for_bonus` threshold | +0.2 |
| **Minimal ask** | Oracle queried ≤ 1 time | +0.2 |

| Score | Meaning |
|-------|---------|
| `1.00` | Perfect — solved efficiently, minimal oracle use |
| `0.80` | Solved, one of the two bonuses earned |
| `0.60` | Solved, but neither efficiency nor ask bonus |
| `0.00` | Not solved |

---

## 📐 Typed Models (OpenEnv Compliant)

```python
class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    scenario: str
    screen: str
    logs: str
    revealed_info: List[str]   # explored targets so far
    history: List[str]         # human-readable action log
    steps_taken: int
    steps_left: int
    ask_count: int

class Action(BaseModel):
    type: Literal["act", "explore", "ask"]
    target: Optional[str]
    content: Optional[str]

class Reward(BaseModel):
    value: float
    reason: str

class GradeResult(BaseModel):
    score: float               # 0.0 – 1.0
    solved: bool
    efficient: bool
    minimal_ask: bool
    breakdown: dict
    summary: str
```

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/devarajnadiger/aax-debug-env.git
cd aax-debug-env
pip install -r requirements.txt
```

### Use the environment directly

```python
from environment import AaxDebugEnv
from environment.models import Action

env = AaxDebugEnv()

# List available tasks
print(env.available_tasks())
# ['task_easy', 'task_medium', 'task_hard']

# Start an episode
obs = env.reset("task_medium")
print(obs.scenario)
print(obs.logs)

# Explore
obs, reward, done, info = env.step(Action(type="explore", target="network_logs"))
print(reward.value, reward.reason)   # 0.2  Explored 'network_logs' — new info revealed

# Ask the oracle
obs, reward, done, info = env.step(Action(type="ask", content="Why does login fail?"))
print(reward.value)   # -0.2

# Attempt a fix
obs, reward, done, info = env.step(Action(
    type="act",
    content="Await token refresh before dispatching login in AuthService.kt:88 — race condition"
))
print(done, reward.value)   # True  0.5

# Grade the episode
result = env.grade()
print(result.score)    # 0.8
print(result.summary)
```

---

## 📡 API Reference

Start the server:

```bash
python api.py
# Listening on http://0.0.0.0:7860
```

Or with Docker:

```bash
docker build -t aax-debug-env .
docker run -p 7860:7860 aax-debug-env
```

### Endpoints

#### `GET /tasks`
List all available tasks.

```json
{
  "tasks": [
    { "id": "task_easy", "difficulty": "easy", "title": "NullPointerException on App Launch", "max_steps": 6 },
    { "id": "task_medium", "difficulty": "medium", "title": "Intermittent Login Failure", "max_steps": 8 },
    { "id": "task_hard", "difficulty": "hard", "title": "Silent Data Corruption in Offline Mode", "max_steps": 8 }
  ]
}
```

#### `POST /reset`
Start a new episode. Sets a `session_id` cookie.

```json
// Request
{ "task_id": "task_easy" }

// Response — Observation
{
  "task_id": "task_easy",
  "difficulty": "easy",
  "scenario": "A mobile app crashes immediately on startup...",
  "screen": "App crashes on launch. Splash screen never appears.",
  "logs": "E/AndroidRuntime: FATAL EXCEPTION...",
  "revealed_info": [],
  "history": [],
  "steps_taken": 0,
  "steps_left": 6,
  "ask_count": 0
}
```

#### `POST /step`
Advance the episode by one action.

```json
// Request
{
  "action": {
    "type": "explore",
    "target": "stack_trace",
    "content": null
  }
}

// Response
{
  "observation": { ... },
  "reward_value": 0.2,
  "reward_reason": "Explored 'stack_trace' — new information revealed.",
  "done": false,
  "info": { "solved": false, "steps_taken": 1, "ask_count": 0 }
}
```

#### `GET /state`
Get current observation without advancing the episode.

#### `GET /grade`
Get the final score for the current episode.

```json
{
  "score": 1.0,
  "solved": true,
  "efficient": true,
  "minimal_ask": true,
  "breakdown": {
    "solved":          { "earned": 0.6, "max": 0.6 },
    "efficient_steps": { "earned": 0.2, "max": 0.2, "steps_taken": 2, "threshold": 3 },
    "minimal_ask":     { "earned": 0.2, "max": 0.2, "ask_count": 0 }
  },
  "summary": "Task solved. Solved efficiently in 2 steps (bonus awarded). Final score: 1.00/1.00"
}
```

---

## 🤖 Running the Baseline Agent

The baseline uses Claude to play through an episode autonomously.

```bash
export ANTHROPIC_API_KEY=sk-...

# Easy task (default)
python inference.py --task task_easy

# Hard task with verbose prompt/response logging
python inference.py --task task_hard --verbose

# Use a different model
python inference.py --task task_medium --model claude-opus-4-6
```

**Example output:**
```
============================================================
Task: task_hard
Model: claude-haiku-4-5-20251001
============================================================

Step 1: EXPLORE → sync_logs
         Reward: +0.2  (Explored 'sync_logs' — new information revealed.)
Step 2: EXPLORE → conflict_resolver
         Reward: +0.2  (Explored 'conflict_resolver' — new information revealed.)
Step 3: ASK | Why does 'no conflicts detected' appear when data is corrupted?
         Reward: -0.2  (Asked oracle — hint provided (costly).)
Step 4: ACT | Fix timestamp conflict — use server timestamps, disable TRUST_CLIENT_TIMESTAMPS in ConflictResolver.java
         Reward: +0.5  (Correct fix applied — task solved!)

============================================================
EPISODE COMPLETE
============================================================
Cumulative reward : +0.70
Final score       : 0.80 / 1.00
Summary           : Task solved. Solved efficiently in 4 steps (bonus awarded). Final score: 0.80/1.00
```

---

## 🐳 Deployment

### Docker

```bash
docker build -t aax-debug-env .
docker run -p 7860:7860 aax-debug-env
```

### Hugging Face Spaces

The repo is ready to deploy as a Docker Space. The `README.md` frontmatter configures it automatically.

Push to your Space:
```bash
git remote add hf https://huggingface.co/spaces/<your-username>/aax-debug-env
git push hf main
```

The Space will be live at `https://huggingface.co/spaces/<your-username>/aax-debug-env`.

---

## 📄 License

MIT
