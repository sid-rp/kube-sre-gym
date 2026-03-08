---
title: Kube SRE Environment Server
emoji: 🔧
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Kube SRE Environment

A Kubernetes SRE training environment where an RL agent diagnoses and fixes real GKE cluster incidents. Features curriculum-driven difficulty, LLM-based judging, and dynamic scenario generation.

## Quick Start

```python
from kube_sre_env import K8sSREAction, K8sSREEnv

with K8sSREEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    print(result.observation.command_output)

    # Investigate the incident
    result = client.step(K8sSREAction(command="kubectl get pods -A"))
    print(result.observation.cluster_status_summary)

    # Diagnose and fix
    result = client.step(K8sSREAction(command="diagnose: OOMKill due to low memory limits"))
    result = client.step(K8sSREAction(command="fix: kubectl set resources deployment/payment-api --limits=memory=256Mi -n payments"))
```

## Building the Docker Image

```bash
docker build -t kube_sre_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Environment Details

### Action
**K8sSREAction**: A kubectl command or diagnosis/fix statement
- `command` (str) - e.g., `kubectl get pods -n payments`, `diagnose: OOM`, `fix: kubectl set resources ...`

### Observation
**K8sSREObservation**: What the agent sees after each action
- `command_output` (str) - Output from the last command
- `cluster_status_summary` (str) - Current cluster pod status
- `active_alerts` (list[str]) - Active PagerDuty-style alerts
- `steps_taken` / `max_steps` (int) - Episode progress
- `hint` (str) - Hint from junior judge persona

### State
**K8sSREState**: Episode metadata
- `incident_type`, `root_cause`, `correct_fix` - Scenario details
- `difficulty` (float 0-1) - Curriculum-driven difficulty
- `judge_persona` - junior/senior/principal
- `curriculum_stats` - Agent skill profile

### Failure Types
| Type | Description |
|------|-------------|
| `oom_kill` | Memory limit too low, pod OOMKills |
| `crashloop` | Bad container command, CrashLoopBackOff |
| `image_pull` | Nonexistent image tag, ImagePullBackOff |
| `bad_config` | Wrong DB_HOST env var, connection errors |
| `liveness_probe` | Wrong probe path, restart loop |
| `resource_quota` | Tight quota blocks pod creation |
| `cascading_db` | Redis OOM cascades to payment-api and frontend |

## Configuration (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `K8S_TOKEN` | Bearer token for GKE | - |
| `K8S_ENDPOINT` | GKE API endpoint | - |
| `K8S_CA_CERT` | Base64 CA cert | - |
| `LLM_BACKEND` | `hf` or `openai` | `hf` |
| `LLM_MODEL` | Model name | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace token | - |
| `GENERATOR_MODE` | `simple` or `llm` | `simple` |

## Project Structure

```
kube_sre_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── client.py              # K8sSREEnv client
├── models.py              # Action, Observation, State models
└── server/
    ├── __init__.py
    ├── kube_sre_env_environment.py  # Core environment (reset/step)
    ├── app.py             # FastAPI application
    ├── k8s_backend.py     # Kubernetes API client
    ├── llm_client.py      # HF/OpenAI LLM wrapper
    ├── scenario_generator.py  # Failure scenario generation
    ├── curriculum.py      # Difficulty & persona controller
    ├── judge.py           # LLM-based action evaluator
    ├── requirements.txt   # Docker dependencies
    └── Dockerfile         # Container image
```
