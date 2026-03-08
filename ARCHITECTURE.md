# Kube SRE Gym — Architecture

## Overview

An RL agent learns to be a Kubernetes SRE by diagnosing and fixing **real** GKE cluster incidents. A larger LLM judges each action. The agent improves through GRPO (Group Relative Policy Optimization).

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING (Northflank H100)                       │
│                                                                         │
│  train.py                                                               │
│  ┌───────────────────────────────────────────┐                          │
│  │  Qwen3-8B (BF16 + LoRA)                  │                          │
│  │                                           │                          │
│  │  1. See alert + cluster state             │                          │
│  │  2. Generate kubectl commands      ◄──────┼── GRPO updates weights   │
│  │  3. Get reward from environment           │   using reward signal    │
│  │  4. Repeat × 4 generations (G=4)         │                          │
│  │  5. Reinforce best response               │                          │
│  └──────────────┬────────────────────────────┘                          │
│                 │                                                        │
│    Env vars:    │  HTTP                                                  │
│    OPENENV_URL  │  POST /reset                                           │
│    HF_TOKEN     │  POST /step {command: "kubectl get pods -n payments"}  │
│                 │  GET  /state                                           │
└─────────────────┼───────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT (HF Space — Docker)                       │
│                    openenv-community/kube-sre-gym                        │
│                                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ FastAPI      │  │ K8s Backend  │  │ Scenario    │  │ Curriculum   │  │
│  │ server       │  │              │  │ Generator   │  │ Controller   │  │
│  │              │  │ execute()    │  │             │  │              │  │
│  │ /reset ──────┼─►│ inject_fail()├─►│ pick failure│  │ difficulty   │  │
│  │ /step ───────┼─►│ get_pods()  │  │ based on    │◄─┤ 0.2 → 0.95  │  │
│  │ /state       │  │ describe()  │  │ weak spots  │  │              │  │
│  │ /ws          │  │ logs()      │  │             │  │ persona:     │  │
│  └──────┬───────┘  └──────┬──────┘  └─────────────┘  │ jr/sr/princ  │  │
│         │                 │                           └──────────────┘  │
│         │                 │ Python kubernetes client                     │
│         ▼                 ▼                                              │
│  ┌─────────────────────────────┐                                        │
│  │ LLM Judge                   │        Env vars (HF Space Secrets):    │
│  │                             │        K8S_TOKEN                       │
│  │ Evaluates each agent action │        K8S_ENDPOINT                    │
│  │ Returns reward: -1.0 to 1.0 │        K8S_CA_CERT                     │
│  │ Persona-based (jr/sr/princ) │        HF_TOKEN                       │
│  └──────────┬──────────────────┘        LLM_MODEL                      │
│             │                           GENERATOR_MODE                  │
│             │ HF Inference API                                          │
└─────────────┼───────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐    ┌──────────────────────────────────────┐
│  HF Inference (Serverless)  │    │  GKE Cluster (Google Cloud)          │
│                             │    │                                      │
│  Qwen3-14B                  │    │  Namespaces:                         │
│                             │    │  ├── payments/                       │
│  Scores agent actions:      │    │  │   ├── payment-api                 │
│  "Good diagnostic step"     │    │  │   ├── redis                       │
│  → reward: +0.6             │    │  │   └── postgres                    │
│                             │    │  ├── frontend/                       │
│  "Random command"           │    │  │   ├── web-frontend                │
│  → reward: -0.3             │    │  │   └── nginx-proxy                 │
│                             │    │  └── auth/                           │
│  "Correct fix!"             │    │      ├── auth-service                │
│  → reward: +1.0             │    │      └── token-store                 │
│                             │    │                                      │
│  Cost: ~$0.01/call          │    │  Real failures injected:             │
│  Budget: ~2000 calls        │    │  OOMKill, CrashLoop, ImagePull,     │
│                             │    │  BadConfig, LivenessProbe,           │
│                             │    │  ResourceQuota, CascadingDB          │
└─────────────────────────────┘    └──────────────────────────────────────┘
```

## Training Flow (One GRPO Step)

```
1. COLLECT         train.py calls POST /reset × 50
                   → Environment injects real failure into GKE
                   → Returns: alert + cluster status as observation
                   → 50 different incident scenarios collected

2. GENERATE        Qwen3-8B generates 4 responses per scenario (G=4)
                   Each response = sequence of kubectl commands
                   Example:
                     "kubectl get pods -n payments"
                     "kubectl describe pod payment-api -n payments"
                     "diagnose: OOMKill due to memory limit 4Mi"
                     "fix: kubectl set resources deployment/payment-api --limits=memory=256Mi"

3. EVALUATE        For each response, train.py calls:
                     POST /reset  (fresh failure)
                     POST /step   (for each command in response)
                   Environment executes real kubectl, judge scores each step
                   Returns cumulative reward per response

4. UPDATE          GRPO compares 4 responses:
                     Response A: reward  0.8
                     Response B: reward -0.2
                     Response C: reward  1.2  ← best
                     Response D: reward -0.3
                   Update model weights toward Response C's behavior

5. REPEAT          Next batch of scenarios, agent gets better over time
                   Curriculum auto-increases difficulty as agent improves
```

## Tokens & Secrets

| Where | Variable | What | Who needs it |
|-------|----------|------|-------------|
| **HF Space** | `K8S_TOKEN` | GKE bearer token (`eyJ...`) | K8s Backend → authenticate to cluster |
| **HF Space** | `K8S_ENDPOINT` | GKE API URL (`https://34.169.10.97`) | K8s Backend → cluster address |
| **HF Space** | `K8S_CA_CERT` | Base64 CA cert (`LS0tLS1...`) | K8s Backend → SSL verification |
| **HF Space** | `HF_TOKEN` | HuggingFace token | LLM Judge → call Inference API |
| **HF Space** | `LLM_MODEL` | Judge model (default: `Qwen/Qwen3-14B`) | LLM Judge → which model to use |
| **HF Space** | `GENERATOR_MODE` | `simple` or `llm` | Scenario Generator → how to pick failures |
| **H100** | `OPENENV_URL` | Space URL (`https://openenv-community-kube-sre-gym.hf.space`) | train.py → connect to environment |
| **H100** | `HF_TOKEN` | HuggingFace token | train.py → push trained model to Hub |
| **H100** | `AGENT_MODEL` | Agent model (default: `Qwen/Qwen3-8B`) | train.py → which model to fine-tune |
| **H100** | `HF_PUSH_REPO` | e.g. `your-name/k8s-sre-agent` | train.py → where to upload final model |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Environment framework | [OpenEnv](https://github.com/meta-pytorch/OpenEnv) |
| Environment hosting | HuggingFace Spaces (Docker) |
| Cluster | Google Kubernetes Engine (GKE) |
| K8s client | Python `kubernetes` library (not kubectl binary) |
| Judge LLM | Qwen3-14B via HF Inference API |
| Agent LLM | Qwen3-8B (BF16 + LoRA) |
| Training | GRPO via TRL + Unsloth |
| GPU | H100 80GB (Northflank) |
