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

with KubeSreGymEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    print(result.observation.command_output)

    result = client.step(KubeSreGymAction(command="kubectl get pods -A"))
    print(result.observation.cluster_status_summary)

    result = client.step(KubeSreGymAction(command="diagnose: OOMKill due to low memory limits"))
    result = client.step(KubeSreGymAction(command="fix: kubectl set resources deployment/payment-api --limits=memory=256Mi -n payments"))
```

## Training (H100)

You need 3 terminals. The env server talks to your GKE cluster, the judge model scores agent actions, and train.py runs GRPO.

**Step 1 — Install**
```bash
git clone https://huggingface.co/spaces/openenv-community/kube-sre-gym && cd kube-sre-gym
pip install -e ".[train]"
```

**Step 2 — Set credentials** (add to your `.bashrc` or export in each terminal)
```bash
export K8S_TOKEN=<gke-token>           # GKE service account bearer token
export K8S_ENDPOINT=<gke-api-url>      # e.g. https://35.x.x.x
export K8S_CA_CERT=<base64-ca-cert>    # base64-encoded cluster CA
export HF_TOKEN=<hf-token>             # for pushing model checkpoints
```

**Step 3 — Launch** (3 terminals)
```bash
# Terminal 1: Judge model (~32GB VRAM)
trl vllm-serve --model Qwen/Qwen3-14B --host 0.0.0.0 --port 8001 --gpu_memory_utilization 0.4

# Terminal 2: Environment server
LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8001/v1 uv run server

# Terminal 3: GRPO training (~48GB VRAM)
python train.py --vllm-mode colocate
```

The curriculum starts with easy faults (OOM, crashloop, bad image) and automatically progresses to harder ones as the agent improves. No manual difficulty tuning needed.

## Adversarial Mode (External Judge)

Uses Claude as judge + scenario designer instead of the self-hosted Qwen model. Frees up the full 80GB for the training agent.

```bash
# Terminal 1: Environment server (no vLLM judge needed)
export ANTHROPIC_API_KEY=sk-ant-...
GYM_MODE=adversarial LLM_BACKEND=anthropic uv run server

# Terminal 2: GRPO training (full GPU)
python train.py --vllm-mode colocate
```

In adversarial mode, Claude designs multi-step incidents with cascading failures and red herrings, injects them into the cluster, and scores agent actions using phase-aware SRE workflow evaluation.

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
┌──────────────────────────────────┐          ┌─────────────────────┐
│ OpenEnv server  :8000            │  k8s     │ payments ns         │
│  reset/step/state                │──client──►│ frontend ns         │
│  Curriculum → Judge → vLLM :8001 │          │ auth ns             │
│                                  │          │ hackathon ns        │
│ vLLM :8001  Qwen3-14B (judge)    │          └─────────────────────┘
│                                  │
│ train.py  GRPO (TRL+vLLM)       │
│  Qwen3-8B agent, BF16+LoRA, G=4 │
└──────────────────────────────────┘
```

Each episode: reset deploys clean healthy pods → curriculum picks one fault → injects it → agent investigates and fixes → judge scores → curriculum tracks mastery.

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
| `GYM_MODE` | `standard` or `adversarial` | `standard` |
| `LLM_BACKEND` | `openai`, `hf`, or `anthropic` | `openai` |
| `LLM_BASE_URL` | vLLM judge endpoint | `http://localhost:8001/v1` |
| `LLM_MODEL` | Judge model name | `Qwen/Qwen3-14B` |
| `ANTHROPIC_API_KEY` | Anthropic API key (adversarial mode) | - |
| `HF_TOKEN` | HuggingFace token (model push) | - |
| `GENERATOR_MODE` | `simple` or `llm` (standard mode only) | `simple` |

## Project Structure

```
kube-sre-gym/
├── __init__.py             # Module exports
├── models.py               # Action, Observation, State models
├── client.py               # KubeSreGymEnv client
├── train.py                # GRPO training (TRL + vLLM, runs on H100)
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container image
├── sample_app/
│   ├── namespaces.yaml     # payments, frontend, auth, hackathon
│   ├── base/               # Healthy manifests (deployed on reset)
│   ├── hackathon/          # Training/eval/complex broken scenarios
│   ├── deploy_all.sh
│   └── cleanup.sh
└── server/
    ├── kube_sre_gym_environment.py  # Core environment (reset/step)
    ├── k8s_backend.py      # K8s auth, command dispatch, reset, health checks
    ├── k8s_commands.py     # kubectl command handlers (get/describe/logs/set/patch)
    ├── k8s_injectors.py    # Failure injectors (oom, crashloop, image_pull, etc.)
    ├── constants.py        # Shared constants (topology, healthy state, timeouts)
    ├── curriculum.py       # Progressive difficulty + mastery tracking
    ├── scenario_generator.py  # Fault scenario pool + LLM generation
    ├── adversarial_designer.py  # LLM-designed compound incidents
    ├── judge.py            # LLMJudge + AdversarialJudge (phase-aware)
    ├── llm_client.py       # OpenAI/HF/Anthropic LLM wrapper
    └── app.py              # FastAPI application
```
