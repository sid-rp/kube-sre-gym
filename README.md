---
title: Kube SRE Gym
emoji: 🔧
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Kube SRE Gym

A Kubernetes SRE training environment where an RL agent diagnoses and fixes real GKE cluster incidents. Features curriculum-driven difficulty, LLM-based judging, and dynamic scenario generation.

## Quick Start

```python
from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv

with KubeSreGymEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    print(result.observation.command_output)

    result = client.step(KubeSreGymAction(command="kubectl get pods -A"))
    print(result.observation.cluster_status_summary)

    result = client.step(KubeSreGymAction(command="diagnose: OOMKill due to low memory limits"))
    result = client.step(KubeSreGymAction(command="fix: kubectl set resources deployment/payment-api --limits=memory=256Mi -n payments"))
```

## Architecture

```
HF Space (Docker)              GKE Cluster
┌──────────────────┐          ┌──────────────────┐
│ OpenEnv Server   │          │ hackathon namespace│
│                  │  Python  │                   │
│ reset() ─────────┼──k8s────►│ deploy broken app │
│ step(action) ────┼──client──►│ run kubectl cmd   │
│ _score() ────────┼──────────│ return real output │
└──────────────────┘          └──────────────────┘
```

## Failure Types

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
kube_sre_gym/               (this repo root)
├── __init__.py             # Module exports
├── models.py               # Action, Observation, State models
├── client.py               # KubeSreGymEnv client
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container image (HF Spaces)
├── train.py                # GRPO training script (runs on H100)
└── server/
    ├── __init__.py
    ├── kube_sre_gym_environment.py  # Core environment (reset/step)
    ├── app.py              # FastAPI application
    ├── k8s_backend.py      # Kubernetes API client
    ├── llm_client.py       # HF/OpenAI LLM wrapper
    ├── scenario_generator.py  # Failure scenario generation
    ├── curriculum.py       # Difficulty & persona controller
    └── judge.py            # LLM-based action evaluator
```
