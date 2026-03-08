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

Install the environment client:

```bash
pip install git+https://huggingface.co/spaces/openenv-community/kube-sre-gym
```

Use the environment:

```python
from kube_sre_gym import KubeSreGymAction, KubeSreGymEnv

with KubeSreGymEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    print(result.observation.command_output)

    result = client.step(KubeSreGymAction(command="kubectl get pods -A"))
    print(result.observation.cluster_status_summary)

    result = client.step(KubeSreGymAction(command="diagnose: OOMKill due to low memory limits"))
    result = client.step(KubeSreGymAction(command="fix: kubectl set resources deployment/payment-api --limits=memory=256Mi -n payments"))
```

## Training (H100)

```bash
# 1. Clone and install with training deps
git clone https://huggingface.co/spaces/openenv-community/kube-sre-gym
cd kube-sre-gym
pip install -e ".[train]"

# 2. Set env vars
export K8S_TOKEN=<gke-token>
export K8S_ENDPOINT=<gke-api-url>
export K8S_CA_CERT=<base64-ca-cert>
export HF_TOKEN=<hf-token>

# 3. Start judge model (Terminal 1)
trl vllm-serve --model Qwen/Qwen3-14B --host 0.0.0.0 --port 8001

# 4. Start environment server (Terminal 2)
LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8001/v1 \
  uv run server

# 5. Run GRPO training (Terminal 3)
python train.py --vllm-mode colocate
```

## Development

```bash
# Install in editable mode
cd kube-sre-gym
pip install -e .

# Run server locally
uv run server
```

## Architecture

Everything runs on the H100. HF Hub is just for code + model weights.

```
H100 (all-in-one)                              GKE Cluster
┌──────────────────────────────────┐          ┌──────────────┐
│ OpenEnv server  :8000            │  k8s     │ hackathon ns │
│  reset/step/state                │──client──►│ payment-api  │
│  Judge ──► vLLM :8001            │          │ redis        │
│                                  │          └──────────────┘
│ vLLM :8001  Qwen3-14B (judge)    │
│                                  │
│ train.py  GRPO (TRL+vLLM)       │
│  Qwen3-8B agent, G=4            │
└──────────────────────────────────┘
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
| `LLM_BACKEND` | `openai` or `hf` | `openai` |
| `LLM_BASE_URL` | vLLM judge endpoint | `http://localhost:8001/v1` |
| `LLM_MODEL` | Judge model name | `Qwen/Qwen3-14B` |
| `HF_TOKEN` | HuggingFace token (model push) | - |
| `GENERATOR_MODE` | `simple` or `llm` | `simple` |

## Project Structure

```
kube_sre_gym/               (this repo root)
├── __init__.py             # Module exports
├── models.py               # Action, Observation, State models
├── client.py               # KubeSreGymEnv client
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container image
├── train.py                # GRPO training (TRL + vLLM, runs on H100)
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
