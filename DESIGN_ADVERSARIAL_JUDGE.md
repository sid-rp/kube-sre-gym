# Design: External Judge + Adversarial Incident Mode

## Overview

Two new modes that build on the existing architecture:

1. **External Judge** — Use Claude (or any API) instead of self-hosted Qwen3-14B, freeing ~28GB GPU memory
2. **Adversarial Mode** — The judge LLM *designs and injects* complex multi-step incidents, teaching the agent real long-horizon SRE behavior

Both modes are additive. The existing self-hosted judge + simple scenarios remain the default.

---

## Mode 1: External Judge

### Problem
Serving Qwen3-14B on the same H100 as training wastes ~28GB. An external judge (Claude, GPT-4, etc.) is stronger, costs pennies per episode, and frees GPU for the agent.

### Changes

**`server/llm_client.py`** — Add `anthropic` backend:

```python
# New env vars:
#   LLM_BACKEND=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
#   LLM_MODEL=claude-sonnet-4-20250514  (default)

class LLMClient:
    def __init__(self):
        self.backend = os.environ.get("LLM_BACKEND", "openai")
        if self.backend == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
            self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

    def _chat_anthropic(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text
```

**No other files change.** The judge, scenario generator, and environment all call `llm.chat()` / `llm.chat_json()` — the backend is transparent.

### Setup (replaces Terminal 1)

```bash
# Instead of: trl vllm-serve --model Qwen/Qwen3-14B ...
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Terminal 1: OpenEnv server (no judge vLLM needed)
uv run server

# Terminal 2: GRPO training (full 80GB for agent)
python train.py --vllm-mode colocate
```

---

## Mode 2: Adversarial Judge (Multi-Step Incidents)

### Problem
Current scenarios are single-fault (one injection, one fix). Real SRE incidents are multi-step: cascading failures, misleading symptoms, multiple components broken, requires triage → investigate → mitigate → fix → verify.

### Concept
The external judge LLM designs a full **incident plan** — a sequence of k8s mutations that create a realistic, multi-step incident. The agent must learn the full SRE loop:

```
broad triage → targeted investigation → mitigate blast radius → root cause fix → verify
```

### New Data Model

**`models.py`** — Add `AdversarialScenarioSpec`:

```python
@dataclass
class IncidentStep:
    """One mutation in a multi-step incident."""
    action: str              # k8s mutation command
    effect: str              # What this causes (for judge context)
    order: int               # Injection sequence
    is_root_cause: bool      # True for the primary fault
    depends_on: list[int]    # Steps that must be injected first

@dataclass
class AdversarialScenarioSpec:
    """Multi-step incident designed by the adversarial judge."""
    title: str                          # e.g. "Cascading auth failure"
    narrative: str                      # What happened (for alert)
    steps: list[IncidentStep]           # Ordered mutations
    expected_triage_phase: list[str]    # Commands agent should run first
    expected_investigation: list[str]   # Targeted diagnostic commands
    expected_mitigation: str            # Immediate blast radius control
    expected_fix_sequence: list[str]    # Ordered fix commands
    expected_verification: list[str]    # Post-fix health checks
    difficulty: float                   # 0.0-1.0
    alert_message: str                  # PagerDuty alert
    root_cause_explanation: str         # Ground truth
    red_herrings: list[str]             # Misleading symptoms the agent must see through
```

### New Component: Adversarial Scenario Designer

**`server/adversarial_designer.py`**:

```python
class AdversarialDesigner:
    """Uses an external LLM to design multi-step incidents."""

    def __init__(self, llm: LLMClient, backend: K8sBackend):
        self.llm = llm
        self.backend = backend

    def design(self, skill_profile: dict, difficulty: float) -> AdversarialScenarioSpec:
        """
        Ask the LLM to design a complex incident using available k8s mutations.

        The LLM receives:
        - Available namespaces, deployments, and their current state
        - Available mutation primitives (the inject_failure types + raw kubectl mutations)
        - Agent's skill profile (what it's good/bad at)
        - Target difficulty

        The LLM returns a structured incident plan.
        """
        cluster_state = self.backend.check_health()
        available_mutations = self._get_mutation_primitives()

        scenario = self.llm.chat_json(
            system=ADVERSARIAL_DESIGNER_PROMPT,
            user=json.dumps({
                "cluster_state": cluster_state,
                "available_mutations": available_mutations,
                "skill_profile": skill_profile,
                "difficulty": difficulty,
                "topology": TOPOLOGY,
            }),
        )
        return AdversarialScenarioSpec(**scenario)

    def inject(self, scenario: AdversarialScenarioSpec) -> str:
        """Execute the incident plan step by step."""
        results = []
        for step in sorted(scenario.steps, key=lambda s: s.order):
            result = self.backend.execute(step.action)
            results.append(f"Step {step.order}: {step.effect} -> {result}")
        return "\n".join(results)

    def _get_mutation_primitives(self) -> list[dict]:
        """Return available k8s mutations the LLM can compose."""
        return [
            {"name": "set_memory_limit", "cmd": "kubectl set resources ...", "effect": "OOMKill"},
            {"name": "set_bad_command", "cmd": "kubectl set command ...", "effect": "CrashLoop"},
            {"name": "set_bad_image", "cmd": "kubectl set image ...", "effect": "ImagePullBackOff"},
            {"name": "set_bad_env", "cmd": "kubectl set env ...", "effect": "App error"},
            {"name": "patch_probe", "cmd": "kubectl patch ...", "effect": "Probe failure"},
            {"name": "apply_quota", "cmd": "kubectl apply quota ...", "effect": "Blocks scaling"},
            {"name": "delete_pod", "cmd": "kubectl delete pod ...", "effect": "Temporary disruption"},
            {"name": "scale_down", "cmd": "kubectl scale --replicas=0 ...", "effect": "Service down"},
            {"name": "patch_service", "cmd": "kubectl patch svc ...", "effect": "Misrouted traffic"},
            {"name": "corrupt_configmap", "cmd": "kubectl patch cm ...", "effect": "Bad config"},
        ]
```

### Adversarial Judge Prompt (core of the design)

```python
ADVERSARIAL_DESIGNER_PROMPT = """You are a Kubernetes chaos engineer designing realistic
production incidents for SRE training.

Given the cluster topology and available mutation primitives, design a multi-step incident that:

1. Has a clear ROOT CAUSE (one primary fault)
2. Creates CASCADING EFFECTS (secondary symptoms from the root cause)
3. Includes RED HERRINGS (symptoms that look related but aren't the root cause)
4. Requires a SPECIFIC RESOLUTION ORDER:
   a. Triage: broad commands to understand scope (kubectl get pods -A, kubectl get events)
   b. Investigation: targeted commands to find root cause (kubectl describe, kubectl logs)
   c. Mitigation: stop the bleeding (scale up healthy replicas, restart unrelated pods)
   d. Fix: address root cause (correct the bad config, fix resource limits)
   e. Verification: confirm resolution (check pod status, test endpoints)

RULES:
- Only use the provided mutation primitives
- The incident must be resolvable with kubectl commands the agent can run
- Design incidents that teach SYSTEMATIC debugging, not lucky guessing
- Higher difficulty = more cascading effects, more red herrings, longer resolution path

Return JSON with the AdversarialScenarioSpec schema.
"""
```

### Multi-Step Reward Function

**`server/judge.py`** — Add phase-aware scoring:

```python
class AdversarialJudge(LLMJudge):
    """Scores agent actions based on SRE phase progression."""

    PHASES = ["triage", "investigation", "mitigation", "fix", "verification"]

    def evaluate(self, command, output, scenario, history, persona):
        current_phase = self._detect_phase(command, history, scenario)
        phase_order_correct = self._is_phase_order_correct(current_phase, history)

        # Base score from LLM evaluation
        base_score, feedback = super().evaluate(command, output, scenario, history, persona)

        # Bonus: following correct SRE workflow order
        if phase_order_correct:
            base_score += 0.2

        # Penalty: skipping phases (e.g. jumping to fix without investigation)
        if self._skipped_phases(current_phase, history):
            base_score -= 0.3
            feedback += " Warning: skipping diagnostic steps before fix."

        # Bonus: discovering red herrings and dismissing them
        if self._identified_red_herring(command, output, scenario):
            base_score += 0.15

        return clamp(base_score, -1.0, 1.0), feedback

    def _detect_phase(self, command, history, scenario):
        """Classify command into SRE phase."""
        if command.startswith("fix:"):
            return "fix"
        if command.startswith("diagnose:"):
            return "investigation"
        # Broad commands = triage
        if any(cmd in command for cmd in ["get pods -A", "get events", "top pods"]):
            return "triage"
        # Targeted commands = investigation
        if any(cmd in command for cmd in ["describe", "logs"]):
            return "investigation"
        # Scale/restart non-root = mitigation
        if any(cmd in command for cmd in ["scale", "rollout restart"]):
            return "mitigation"
        # Post-fix checks = verification
        if self._is_after_fix(history) and "get" in command:
            return "verification"
        return "triage"
```

### Modified Environment Flow

**`server/kube_sre_gym_environment.py`**:

```python
class KubeSreGymEnvironment(Environment):
    def __init__(self):
        # ... existing init ...
        self.mode = os.environ.get("GYM_MODE", "standard")  # "standard" or "adversarial"
        if self.mode == "adversarial":
            self.designer = AdversarialDesigner(self.llm, self.backend)
            self.judge = AdversarialJudge(self.llm)

    def reset(self):
        self.backend.reset()
        difficulty = self.curriculum.get_difficulty()
        skill_profile = self.curriculum.get_skill_profile()

        if self.mode == "adversarial":
            # LLM designs the incident
            self.scenario = self.designer.design(skill_profile, difficulty)
            self.designer.inject(self.scenario)
        else:
            # Existing simple mode
            self.scenario = self.generator.generate(skill_profile, difficulty)
            self.backend.inject_failure(self.scenario.failure_type, self.scenario.params)

        # ... rest of reset unchanged ...

    def step(self, action):
        output = self.backend.execute(action.command)

        if self.mode == "adversarial":
            reward, feedback = self.judge.evaluate(
                action.command, output, self.scenario,
                self.history, self.persona,
            )
            # Multi-step resolution: check if ALL faults are resolved
            if action.command.startswith("fix:"):
                health = self.backend.check_health()
                all_healthy = all(
                    status in ("Running", "Completed")
                    for ns in health.values() for status in ns.values()
                )
                if all_healthy:
                    reward += 0.5
                    self.done = True
        else:
            # Existing standard flow
            reward, feedback = self.judge.evaluate(...)

        # ... rest of step unchanged ...
```

---

## Example: Adversarial Incident

**Designed by Claude:**

```json
{
  "title": "Cascading auth failure with misleading OOM",
  "narrative": "Auth service config update broke token validation. Payment-api retries exhaust memory. Frontend shows 502s.",
  "steps": [
    {
      "action": "kubectl set env deployment/auth-service -n auth DB_HOST=wrong-host.invalid",
      "effect": "Auth service cannot connect to token-store, returns 500",
      "order": 1,
      "is_root_cause": true,
      "depends_on": []
    },
    {
      "action": "kubectl set resources deployment/payment-api -n payments --limits=memory=32Mi",
      "effect": "Payment-api OOMKills due to retry storm from auth failures",
      "order": 2,
      "is_root_cause": false,
      "depends_on": [1]
    }
  ],
  "expected_triage_phase": [
    "kubectl get pods -A",
    "kubectl get events -A --sort-by=.lastTimestamp"
  ],
  "expected_investigation": [
    "kubectl logs deployment/auth-service -n auth",
    "kubectl describe pod -l app=payment-api -n payments",
    "kubectl logs deployment/payment-api -n payments --tail=50"
  ],
  "expected_mitigation": "kubectl scale deployment/payment-api -n payments --replicas=0",
  "expected_fix_sequence": [
    "kubectl set env deployment/auth-service -n auth DB_HOST=token-store.auth.svc.cluster.local",
    "kubectl set resources deployment/payment-api -n payments --limits=memory=256Mi",
    "kubectl scale deployment/payment-api -n payments --replicas=2"
  ],
  "expected_verification": [
    "kubectl get pods -A",
    "kubectl logs deployment/auth-service -n auth --tail=5"
  ],
  "red_herrings": [
    "payment-api OOMKilled looks like a memory issue but it's caused by retry storm"
  ],
  "difficulty": 0.7,
  "root_cause_explanation": "auth-service DB_HOST misconfigured, causing cascading failures"
}
```

**Agent learns the workflow:**

```
Step 1: kubectl get pods -A                     → triage    (+0.1)
Step 2: kubectl get events -A                   → triage    (+0.1)
Step 3: kubectl describe pod payment-api-xxx    → investigate (+0.15)
  (sees OOMKilled — red herring!)
Step 4: kubectl logs auth-service-xxx -n auth   → investigate (+0.2)
  (sees "connection refused to wrong-host.invalid" — root cause!)
Step 5: diagnose: auth-service DB_HOST wrong    → investigate (+0.3)
Step 6: kubectl scale payment-api --replicas=0  → mitigate   (+0.2)
Step 7: fix: kubectl set env auth-service DB_HOST=token-store.auth.svc  → fix (+0.4)
Step 8: fix: kubectl set resources payment-api --limits=memory=256Mi    → fix (+0.2)
Step 9: kubectl get pods -A                     → verify     (+0.15)
  (all healthy → done, +0.5 bonus)
```

---

## File Changes Summary

| File | Change | Effort |
|------|--------|--------|
| `server/llm_client.py` | Add `anthropic` backend | Small |
| `models.py` | Add `IncidentStep`, `AdversarialScenarioSpec` | Small |
| `server/adversarial_designer.py` | **New file** — LLM designs + injects incidents | Medium |
| `server/judge.py` | Add `AdversarialJudge` with phase-aware scoring | Medium |
| `server/kube_sre_gym_environment.py` | Add `GYM_MODE=adversarial` branch in reset/step | Small |
| `pyproject.toml` | Add `anthropic` to optional deps | Tiny |
| `README.md` | Document new modes | Small |

### New Env Vars

| Variable | Description | Default |
|----------|-------------|---------|
| `GYM_MODE` | `standard` or `adversarial` | `standard` |
| `LLM_BACKEND` | `openai`, `hf`, or `anthropic` | `openai` |
| `ANTHROPIC_API_KEY` | API key for Claude | — |

---

## Implementation Order

1. **External judge** (`llm_client.py` anthropic backend) — unblocks everything, frees GPU
2. **Data models** (`IncidentStep`, `AdversarialScenarioSpec`)
3. **Adversarial designer** (new file, the core LLM prompt)
4. **Phase-aware judge** (extends existing `LLMJudge`)
5. **Environment wiring** (mode switch in reset/step)
6. **Testing** — run a few adversarial episodes manually before training

---

## Key Design Decisions

- **Adversarial mode is opt-in** via `GYM_MODE=adversarial`. Default stays `standard`.
- **External judge reuses the same `LLMClient` interface** — no new abstractions. Just a new backend.
- **Phase detection is heuristic-based** (not LLM-called per step) to keep latency low. The LLM only scores quality, not phase classification.
- **Resolution requires ALL pods healthy**, not just fixing one thing. This forces the agent to handle cascading effects.
- **Red herrings are explicitly tracked** so the judge can reward agents that investigate but correctly dismiss misleading symptoms.
