---
title: Ask Act Explore Debug Env
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ask–Act–Explore: A Cost-Aware Mobile Debugging Environment for AI Agents

An OpenEnv-compliant environment where an AI agent debugs mobile application issues by deciding **when to act, explore, or ask** under a cost-aware reward system.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/tasks` | List all available tasks |
| POST | `/reset` | Start a new episode (`{"task_id": "task_easy"}`) |
| POST | `/step` | Advance one step (`{"action": {"type": "explore", "target": "stack_trace"}}`) |
| GET | `/state` | Current observation (read-only) |
| GET | `/grade` | Final score for the episode |

## Action Schema

```json
{
  "type": "explore" | "act" | "ask",
  "target": "<string or null>",
  "content": "<string or null>"
}
```

## Tasks

| ID | Difficulty | Scenario |
|----|-----------|----------|
| `task_easy` | Easy | NullPointerException on app launch |
| `task_medium` | Medium | Intermittent login failure (race condition) |
| `task_hard` | Hard | Silent data corruption in offline sync |

## Reward Design

| Action | Reward |
|--------|--------|
| Useful explore (new info) | +0.2 |
| Redundant explore | -0.1 |
| Ask oracle | -0.2 |
| Correct fix | +0.5 |
| Wrong fix | -0.3 |

## Grader (0.0 → 1.0)

- **Solved** → +0.6
- **Efficient steps** → +0.2
- **Minimal oracle asks (≤1)** → +0.2

## Run Baseline Agent

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python inference.py --task task_hard --verbose
```
