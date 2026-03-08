# Kube SRE Gym — Architecture

An RL agent learns to diagnose and fix **real** Kubernetes incidents via GRPO.

```
H100 (Northflank)              HF Space (Docker)              GKE Cluster
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ train.py         │          │ OpenEnv Server   │          │ payments/        │
│                  │  HTTP    │                  │  Python  │   payment-api    │
│ Qwen3-8B        │ ────────►│ reset() ─────────┼──k8s────►│   redis          │
│ BF16 + LoRA     │ ◄────────│ step(action) ────┼──client──►│ frontend/        │
│                  │ obs+reward│ state()         │          │   web-frontend   │
│ GRPO training   │          │                  │          │ auth/            │
│ 4 generations   │          │ Judge ───────────┼──HF API──►│   auth-service   │
│ reinforce best  │          │ Qwen3-14B scores │          │                  │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

## One Episode

```
1. train.py calls POST /reset
2. Environment injects real failure into GKE (OOMKill, CrashLoop, etc.)
3. Agent sees: alert + cluster status
4. Agent generates: kubectl get pods → kubectl describe → diagnose → fix
5. Each command runs on real cluster, judge scores it
6. Reward flows back → GRPO updates agent weights
```

## Tokens

| Where | Secret | Purpose |
|-------|--------|---------|
| HF Space | `K8S_TOKEN` | Authenticate to GKE cluster |
| HF Space | `K8S_ENDPOINT` | GKE API URL |
| HF Space | `K8S_CA_CERT` | SSL cert for GKE |
| HF Space | `HF_TOKEN` | Judge calls HF Inference API |
| H100 | `OPENENV_URL` | Connect to HF Space |
| H100 | `HF_TOKEN` | Push trained model to Hub |
