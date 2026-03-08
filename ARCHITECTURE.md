# Kube SRE Gym — Architecture

An RL agent learns to diagnose and fix **real** Kubernetes incidents via GRPO.

```
H100 (all-in-one)                                          GKE Cluster
┌──────────────────────────────────────┐                  ┌──────────────┐
│                                      │      k8s         │ hackathon ns │
│  OpenEnv server  :8000               │──── client ─────►│ payment-api  │
│   reset/step/state                   │                  │ redis        │
│   Judge ──► vLLM :8001               │                  │ web-frontend │
│                                      │                  │ auth-service │
│  vLLM :8001  Qwen3-14B (judge)       │                  └──────────────┘
│                                      │
│  train.py  GRPO (TRL + vLLM colocate)│
│   Qwen3-8B agent  BF16               │
│   G=4 generations, rollout_func      │
└──────────────────────────────────────┘

HF Hub: code repo + trained model weights
```

## One Episode

```
1. train.py calls env.reset() → OpenEnv injects failure into GKE
2. Agent sees: alert + cluster status
3. Agent generates: kubectl get pods → kubectl describe → diagnose → fix
4. Each command runs on real cluster via OpenEnv server
5. Judge (Qwen3-14B via vLLM) scores each action
6. Reward flows back → GRPO updates agent weights
```

## H100 Setup (3 terminals)

```bash
# Terminal 1: Judge model (vLLM OpenAI-compatible server)
trl vllm-serve --model Qwen/Qwen3-14B --host 0.0.0.0 --port 8001

# Terminal 2: OpenEnv server (environment + k8s backend + judge client)
LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8001/v1 \
  uv run server

# Terminal 3: GRPO training (agent model with TRL's built-in vLLM)
python train.py --vllm-mode colocate
```

## Tokens

| Where | Secret | Purpose |
|-------|--------|---------|
| H100 | `K8S_TOKEN` | Authenticate to GKE cluster |
| H100 | `K8S_ENDPOINT` | GKE API URL |
| H100 | `K8S_CA_CERT` | SSL cert for GKE |
| H100 | `HF_TOKEN` | Push trained model to Hub |
