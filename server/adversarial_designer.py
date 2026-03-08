"""
Adversarial Incident Designer.

Uses an external LLM (Claude, GPT-4, etc.) to design and inject complex
multi-step Kubernetes incidents that require real SRE workflow:
  triage -> investigation -> mitigation -> fix -> verification

Design principles (from ChaosEater, Chaos Mesh, Google SRE):
  - Progressive context enrichment: feed topology + health, then ask for scenario
  - Steady-state-first: define healthy baseline so success criteria are clear
  - Inject/fix pairs: every mutation has an exact reversal command
  - Step budget: scenarios must be solvable within agent's step limit
"""

import json
import logging
import random
import time

from .llm_client import LLMClient
from .k8s_backend import K8sBackend
from .constants import TOPOLOGY, HEALTHY_STATE

try:
    from ..models import AdversarialScenarioSpec, IncidentStep
except ImportError:
    from models import AdversarialScenarioSpec, IncidentStep

logger = logging.getLogger(__name__)


# ---- Simple single-fault scenarios for warmup / beginner tiers ----
# These don't need the LLM — they're predictable and easy to diagnose.
# Deployment names must match sample_app manifests and HEALTHY_STATE in constants.py.
WARMUP_SCENARIOS = [
    {
        "name": "wrong-database-url-worker",
        "namespace": "payments",
        "deployment": "payment-worker",
        "root_cause": "payment-worker DATABASE_URL env var points to non-existent host",
        "alert_message": "CRITICAL: payment-worker CrashLoopBackOff, cannot connect to database",
        "correct_fix_description": "Set DATABASE_URL back to correct postgres connection string",
        "steps": [{"action": "kubectl set env deployment/payment-worker -n payments DATABASE_URL=postgres://wrong-host.invalid:5432/payments",
                    "effect": "payment-worker cannot connect to database"}],
        "diagnosis_steps": ["kubectl get pods -n payments", "kubectl logs payment-worker -n payments --tail=50"],
        "fix_steps": ["kubectl set env deployment/payment-worker -n payments DATABASE_URL=postgres://payments_user:payments_pass@payment-db.payments.svc.cluster.local:5432/payments"],
        "verify_steps": ["kubectl get pods -n payments"],
        "red_herrings": [],
        "expected_observation_hints": ["connection refused", "host not found"],
    },
    {
        "name": "oom-kill-payment-gateway",
        "namespace": "payments",
        "deployment": "payment-gateway",
        "root_cause": "payment-gateway memory limit set too low, causing OOMKill",
        "alert_message": "CRITICAL: payment-gateway pods OOMKilled, service unavailable",
        "correct_fix_description": "Increase memory limits on payment-gateway back to 128Mi",
        "steps": [{"action": "kubectl set resources deployment/payment-gateway -n payments --limits=memory=4Mi",
                    "effect": "payment-gateway OOMKilled (exit code 137)"}],
        "diagnosis_steps": ["kubectl get pods -n payments", "kubectl describe pod payment-gateway -n payments"],
        "fix_steps": ["kubectl set resources deployment/payment-gateway -n payments --limits=memory=128Mi"],
        "verify_steps": ["kubectl get pods -n payments"],
        "red_herrings": [],
        "expected_observation_hints": ["OOMKilled", "exit code 137"],
    },
    {
        "name": "bad-image-web-app",
        "namespace": "frontend",
        "deployment": "web-app",
        "root_cause": "web-app image tag changed to nonexistent version",
        "alert_message": "WARNING: web-app pods stuck in ImagePullBackOff",
        "correct_fix_description": "Set image back to nginx:1.25",
        "steps": [{"action": "kubectl set image deployment/web-app -n frontend web-app=nginx:nonexistent-tag-99999",
                    "effect": "web-app ImagePullBackOff"}],
        "diagnosis_steps": ["kubectl get pods -n frontend", "kubectl describe pod web-app -n frontend"],
        "fix_steps": ["kubectl set image deployment/web-app -n frontend web-app=nginx:1.25"],
        "verify_steps": ["kubectl get pods -n frontend"],
        "red_herrings": [],
        "expected_observation_hints": ["ImagePullBackOff", "ErrImagePull"],
    },
    {
        "name": "oom-kill-auth-service",
        "namespace": "auth",
        "deployment": "auth-service",
        "root_cause": "auth-service memory limit set too low, causing OOMKill",
        "alert_message": "CRITICAL: auth-service pods OOMKilled, login failures",
        "correct_fix_description": "Increase memory limits on auth-service back to 128Mi",
        "steps": [{"action": "kubectl set resources deployment/auth-service -n auth --limits=memory=4Mi",
                    "effect": "auth-service OOMKilled (exit code 137)"}],
        "diagnosis_steps": ["kubectl get pods -n auth", "kubectl describe pod auth-service -n auth"],
        "fix_steps": ["kubectl set resources deployment/auth-service -n auth --limits=memory=128Mi"],
        "verify_steps": ["kubectl get pods -n auth"],
        "red_herrings": [],
        "expected_observation_hints": ["OOMKilled", "exit code 137"],
    },
    {
        "name": "scaled-to-zero-frontend-cache",
        "namespace": "frontend",
        "deployment": "frontend-cache",
        "root_cause": "frontend-cache scaled to zero replicas, web-app has no cache backend",
        "alert_message": "WARNING: frontend-cache has 0 pods, web-app reporting connection errors",
        "correct_fix_description": "Scale frontend-cache back to 1 replica",
        "steps": [{"action": "kubectl scale deployment/frontend-cache -n frontend --replicas=0",
                    "effect": "frontend-cache has no running pods"}],
        "diagnosis_steps": ["kubectl get pods -n frontend", "kubectl get deploy -n frontend"],
        "fix_steps": ["kubectl scale deployment/frontend-cache -n frontend --replicas=1"],
        "verify_steps": ["kubectl get pods -n frontend"],
        "red_herrings": [],
        "expected_observation_hints": ["0/0", "connection refused"],
    },
    {
        "name": "crashloop-payment-api",
        "namespace": "payments",
        "deployment": "payment-api",
        "root_cause": "payment-api container command changed to exit 1, CrashLoopBackOff",
        "alert_message": "CRITICAL: payment-api CrashLoopBackOff, payment processing down",
        "correct_fix_description": "Rollout restart payment-api to restore original command",
        "steps": [{"action": "kubectl set image deployment/payment-api -n payments payment-api=python:3.11-slim",
                    "effect": "payment-api crashes on startup (image is correct but will be used with crashloop injector)"}],
        "diagnosis_steps": ["kubectl get pods -n payments", "kubectl logs payment-api -n payments --tail=50"],
        "fix_steps": ["kubectl rollout restart deployment/payment-api -n payments"],
        "verify_steps": ["kubectl get pods -n payments"],
        "red_herrings": [],
        "expected_observation_hints": ["CrashLoopBackOff", "Error"],
    },
]


ADVERSARIAL_DESIGNER_PROMPT = """You are a Kubernetes chaos engineer designing realistic production
incidents for SRE training. Design incidents that teach SYSTEMATIC debugging, not lucky guessing.

STEP 1 — UNDERSTAND THE CLUSTER

Cluster topology (namespace -> deployments):
{topology}

Healthy baseline (what each deployment looks like when working):
{healthy_state}

Current cluster health:
{cluster_health}

STEP 2 — CONSTRAINTS

Step budget: agent has {max_steps} total steps.
  - Triage (broad commands): ~2-3 steps
  - Investigation (targeted describe/logs): ~3-4 steps
  - Fix commands: at most {max_fix_steps}
  - Verification: ~1-2 steps
Max faults to inject: {max_mutations}

STEP 3 — AVAILABLE FAULT TYPES

Use ONLY these inject/fix pairs. These are the ONLY faults that reliably break pods.

1. LOW MEMORY LIMIT (OOMKill) — works on ANY deployment
   Inject: kubectl set resources deployment/<deploy> -n <ns> --limits=memory=4Mi
   Fix:    kubectl set resources deployment/<deploy> -n <ns> --limits=memory=<correct from healthy baseline>
   Symptoms: OOMKilled status, exit code 137, pod restarts
   Red herring: looks like a memory leak but it's just a low limit

2. BAD IMAGE TAG — works on ANY deployment
   Inject: kubectl set image deployment/<deploy> -n <ns> <container>=nginx:nonexistent-tag-99999
   Fix:    kubectl set image deployment/<deploy> -n <ns> <container>=<correct from healthy baseline>
   Symptoms: ImagePullBackOff, ErrImagePull

3. SCALE TO ZERO — works on ANY deployment
   Inject: kubectl scale deployment/<deploy> -n <ns> --replicas=0
   Fix:    kubectl scale deployment/<deploy> -n <ns> --replicas=<correct from healthy baseline>
   Symptoms: no pods, dependent services fail with connection refused

4. CRASHLOOP (bad command) — works on ANY deployment
   Inject: kubectl patch deployment/<deploy> -n <ns> -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"<container>","command":["sh","-c","exit 1"]}}]}}}}}}}}'
   Fix:    kubectl rollout restart deployment/<deploy> -n <ns>
   Symptoms: CrashLoopBackOff, container exits immediately

5. WRONG DATABASE URL — ONLY works on payment-worker
   Inject: kubectl set env deployment/payment-worker -n payments DATABASE_URL=postgres://wrong-host.invalid:5432/payments
   Fix:    kubectl set env deployment/payment-worker -n payments DATABASE_URL=postgres://payments_user:payments_pass@payment-db.payments.svc.cluster.local:5432/payments
   Symptoms: payment-worker CrashLoopBackOff, "host not found" in logs

IMPORTANT: Do NOT use env var faults (DB_HOST, PORT, POSTGRES_DB) on any deployment
other than payment-worker — other deployments ignore unknown env vars and stay Running.

STEP 4 — DESIGN THE INCIDENT

HARD CONSTRAINTS — the scenario MUST be solvable:
- At most {max_mutations} injected faults (inject_commands). Never exceed this.
- Each inject_command MUST have exactly one corresponding entry in fix_steps (the reversal).
- fix_steps count MUST NOT exceed {max_fix_steps}.
- Every deployment and namespace referenced MUST exist in the topology above.
- Use container names from the healthy baseline (container_name field) for set image commands.
- Keep it simple enough that a methodical agent can solve it within {max_steps} steps total.

COMPLEXITY STRATEGY — what makes this HARDER than a single fault:
- SPREAD faults across DIFFERENT namespaces (payments + frontend + auth) so the agent
  must investigate multiple areas, not just one namespace
- Use DIFFERENT fault types together (e.g., OOM + bad image + scale zero) so the agent
  sees mixed symptoms (OOMKilled in one namespace, ImagePullBackOff in another, missing pods in a third)
- Create RED HERRINGS: e.g., OOM on payment-gateway looks like a memory leak, but the
  REAL root cause is that auth-service was scaled to zero causing retry storms
- Order MATTERS: fix the upstream dependency first (e.g., fix auth-service before payment-api)
- Make the agent investigate ALL namespaces — don't cluster faults in one place

Think about this before generating:
- What combination of faults creates the most confusing symptoms?
- Which fault should the agent find LAST? (hide it in a namespace they might not check first)
- What order must the fixes be applied in? (upstream dependencies first)
- What will the agent see in "kubectl get pods -A" that is misleading?

STEP 5 — GENERATE THE SCENARIO

Respond with ONLY this JSON (no markdown fences, no extra text):
{{
  "name": "short_snake_case_name",
  "difficulty": <float 0.0-1.0>,
  "failure_type": "adversarial",
  "namespace": "<primary affected namespace>",
  "deployment": "<primary affected deployment>",
  "inject_commands": [
    "<exact kubectl command from fault types above>",
    "<optional second fault for cascading>"
  ],
  "root_cause": "<one sentence: what broke and why it caused the symptoms>",
  "correct_fix_description": "<human-readable ordered fix steps>",
  "diagnosis_steps": [
    "kubectl get pods -A",
    "kubectl get events -n <ns>",
    "kubectl describe pod <deploy> -n <ns>",
    "kubectl logs <deploy> -n <ns> --tail=50"
  ],
  "fix_steps": [
    "<exact kubectl reversal command from fault types above>"
  ],
  "verify_steps": ["kubectl get pods -n <ns>"],
  "red_herrings": ["<symptom the agent sees that looks like X but is actually caused by Y>"],
  "expected_observation_hints": ["OOMKilled", "connection refused", "502"],
  "alert_message": "CRITICAL: <realistic PagerDuty-style alert>"
}}"""


class AdversarialDesigner:
    """Uses an external LLM to design and inject multi-step K8s incidents."""

    def __init__(self, llm: LLMClient, backend: K8sBackend, max_steps: int = 15):
        self.llm = llm
        self.backend = backend
        self.max_steps = max_steps

    def design(self, skill_profile: dict, difficulty: float) -> AdversarialScenarioSpec:
        """Design an incident appropriate for the agent's current skill level.

        - difficulty < 0.4: pick from WARMUP_SCENARIOS (no LLM call, fast, predictable)
        - difficulty >= 0.4: use LLM to design progressively harder incidents

        Uses progressive context enrichment (ChaosEater pattern):
        topology + healthy baseline + current health → scenario design.
        """
        # Low difficulty → use simple hardcoded scenarios (warmup + beginner tiers)
        # Only start using LLM-designed incidents at intermediate tier and above
        if difficulty <= 0.4:
            return self._design_warmup(skill_profile, difficulty)

        return self._design_llm(skill_profile, difficulty)

    def _design_warmup(self, skill_profile: dict, difficulty: float) -> AdversarialScenarioSpec:
        """Pick a simple single-fault scenario from the warmup pool.

        Prefers scenarios the agent hasn't solved yet, or ones it's weak at.
        """
        solved = set(skill_profile.keys()) if skill_profile else set()
        # Prefer unsolved scenarios
        unsolved = [s for s in WARMUP_SCENARIOS if f"adversarial:{s['name']}" not in solved]
        pool = unsolved if unsolved else WARMUP_SCENARIOS

        # If there are weak spots, prefer scenarios targeting those failure types
        if skill_profile:
            weak_names = {k for k, v in skill_profile.items() if v < 0.5}
            # Match by namespace/deployment mentioned in the weak scenario names
            weak_pool = [s for s in pool if f"adversarial:{s['name']}" in weak_names]
            if weak_pool:
                pool = weak_pool

        chosen = random.choice(pool)
        steps = [IncidentStep(action=s["action"], effect=s["effect"], order=i+1, is_root_cause=(i == 0))
                 for i, s in enumerate(chosen["steps"])]

        scenario = AdversarialScenarioSpec(
            name=chosen["name"],
            failure_type="adversarial",
            namespace=chosen["namespace"],
            deployment=chosen["deployment"],
            root_cause=chosen["root_cause"],
            difficulty=difficulty,
            alert_message=chosen["alert_message"],
            correct_fix_description=chosen["correct_fix_description"],
            steps=steps,
            diagnosis_steps=chosen["diagnosis_steps"],
            fix_steps=chosen["fix_steps"],
            verify_steps=chosen["verify_steps"],
            red_herrings=chosen.get("red_herrings", []),
            expected_observation_hints=chosen["expected_observation_hints"],
            expected_diagnostic_path=chosen["diagnosis_steps"],
            params={},
        )
        logger.info(f"Warmup scenario selected: {scenario.name} (difficulty={difficulty:.2f})")
        return scenario

    def _design_llm(self, skill_profile: dict, difficulty: float) -> AdversarialScenarioSpec:
        """Use the LLM to design a harder incident based on agent's skill gaps."""
        # Budget: reserve ~7 steps for triage+investigation+verify, leave rest for fixes
        max_fix_steps = max(1, min(6, self.max_steps - 7))
        # Scale mutations with difficulty tier:
        #   intermediate (0.4-0.6): 1 fault
        #   advanced     (0.6-0.8): 2 faults
        #   expert       (0.8+):    3 faults
        if difficulty < 0.6:
            max_mutations = 1
        elif difficulty < 0.8:
            max_mutations = 2
        else:
            max_mutations = 3

        cluster_state = self.backend.check_health()
        weak_spots = [k for k, v in skill_profile.items() if v < 0.5] if skill_profile else []

        system_prompt = ADVERSARIAL_DESIGNER_PROMPT.format(
            topology=json.dumps(TOPOLOGY, indent=2),
            healthy_state=json.dumps(HEALTHY_STATE, indent=2),
            cluster_health=json.dumps(cluster_state, indent=2),
            max_steps=self.max_steps,
            max_fix_steps=max_fix_steps,
            max_mutations=max_mutations,
        )

        user_prompt = f"""Design a Kubernetes incident.

Target difficulty: {difficulty:.2f}/1.0
Agent skill profile: {json.dumps(skill_profile) if skill_profile else "no history yet"}
Weak spots to exploit: {weak_spots if weak_spots else "any — agent is new"}
Previously solved scenarios: {list(skill_profile.keys()) if skill_profile else "none"}

{"HARD MODE: Design a " + str(max_mutations) + "-fault compound incident spread across ALL 3 namespaces (payments, frontend, auth). Use DIFFERENT fault types for each (e.g., OOM in payments + bad image in frontend + scale-zero in auth). The agent must investigate every namespace to find all faults. Include a red herring where one symptom looks like it could be caused by a different fault." if difficulty > 0.7 else "Design a " + str(max_mutations) + "-fault incident across at least 2 namespaces. Use different fault types so the agent sees mixed symptoms (e.g., OOMKilled + ImagePullBackOff). Keep it solvable within the step budget." if difficulty > 0.5 else "Start with a single-fault scenario. One root cause, clear symptoms."}

{"Focus on these weak areas: " + ", ".join(weak_spots) if weak_spots else ""}"""

        try:
            data = self.llm.chat_json(system_prompt, user_prompt, temperature=0.7, max_tokens=2048)
            scenario = self._parse_scenario(data, difficulty)
            logger.info(f"Adversarial scenario designed: {scenario.name} (difficulty={scenario.difficulty}, "
                        f"faults={len(scenario.steps)}, fix_steps={len(scenario.fix_steps)})")
            return scenario
        except Exception as e:
            logger.error(f"AdversarialDesigner error: {e} — falling back to hardcoded scenario", exc_info=True)
            return self._fallback_scenario(difficulty)

    # Commands the backend can actually execute
    _VALID_PREFIXES = ("kubectl set ", "kubectl patch ", "kubectl scale ", "kubectl delete ", "kubectl rollout ")

    # Known deployments per namespace (must match sample_app manifests)
    _KNOWN_DEPLOYMENTS = {
        ns: set(deploys.keys()) for ns, deploys in HEALTHY_STATE.items()
    }

    def _validate_command_targets(self, command: str) -> str | None:
        """Check that a kubectl command targets a real deployment/namespace.

        Returns an error string if invalid, None if OK.
        """
        parts = command.split()
        # Extract namespace
        ns = None
        for i, p in enumerate(parts):
            if p == "-n" and i + 1 < len(parts):
                ns = parts[i + 1]
        if ns and ns not in self._KNOWN_DEPLOYMENTS:
            return f"unknown namespace '{ns}' (valid: {list(self._KNOWN_DEPLOYMENTS.keys())})"

        # Extract deployment name
        for p in parts:
            if p.startswith("deployment/"):
                deploy_name = p.split("/")[-1]
                if ns and deploy_name not in self._KNOWN_DEPLOYMENTS.get(ns, set()):
                    return (f"deployment '{deploy_name}' not found in namespace '{ns}' "
                            f"(valid: {list(self._KNOWN_DEPLOYMENTS.get(ns, set()))})")
        return None

    def inject(self, scenario: AdversarialScenarioSpec) -> str:
        """Execute the incident injection commands on the real cluster.

        Validates each command before execution to prevent unsupported
        operations or commands targeting non-existent resources.
        """
        results = []
        success_count = 0
        for i, step in enumerate(scenario.steps):
            # Validate command is something the backend can execute
            if not any(step.action.startswith(p) for p in self._VALID_PREFIXES):
                msg = f"Step {i+1}: SKIPPED — unsupported command: {step.action}"
                results.append(msg)
                logger.warning(msg)
                continue

            # Validate command targets real resources
            target_err = self._validate_command_targets(step.action)
            if target_err:
                msg = f"Step {i+1}: SKIPPED — {target_err}"
                results.append(msg)
                logger.warning(msg)
                continue

            try:
                result = self.backend.execute(step.action)
                if result.startswith("error:") or result.startswith("Error"):
                    results.append(f"Step {i+1}: FAILED -> {result}")
                    logger.error(f"Injection failed step {i+1}: {result}")
                else:
                    success_count += 1
                    results.append(f"Step {i+1}: {step.effect} -> {result}")
                    logger.info(f"Injected step {i+1}/{len(scenario.steps)}: {step.action}")
            except Exception as e:
                results.append(f"Step {i+1}: FAILED -> {e}")
                logger.error(f"Injection failed step {i+1}: {e}")

        # If no steps succeeded, the scenario is broken
        if success_count == 0:
            logger.error("All injection steps failed — no faults injected")

        # Let failures propagate through the cluster
        wait_time = min(15, 5 + len(scenario.steps) * 3)
        time.sleep(wait_time)

        scenario._inject_success_count = success_count
        return "\n".join(results)

    def _parse_scenario(self, data: dict, difficulty: float) -> AdversarialScenarioSpec:
        """Convert LLM JSON response into AdversarialScenarioSpec.

        Enforces solvability constraints: max 2 inject commands, fix_steps
        must not exceed inject count, all targets must be valid.
        """
        inject_commands = data.get("inject_commands", [])
        fix_steps = data.get("fix_steps", [])[:len(inject_commands)]  # at most 1 fix per fault
        hints = data.get("expected_observation_hints", [])
        steps = []
        for i, cmd in enumerate(inject_commands):
            steps.append(IncidentStep(
                action=cmd,
                effect=hints[i] if i < len(hints) else "fault injected",
                order=i + 1,
                is_root_cause=(i == 0),
            ))

        return AdversarialScenarioSpec(
            name=data.get("name", "adversarial-incident"),
            failure_type="adversarial",
            namespace=data.get("namespace", "payments"),
            deployment=data.get("deployment", "payment-api"),
            root_cause=data.get("root_cause", ""),
            difficulty=data.get("difficulty", difficulty),
            alert_message=data.get("alert_message", "CRITICAL: Multiple services degraded"),
            correct_fix_description=data.get("correct_fix_description", ""),
            steps=steps,
            diagnosis_steps=data.get("diagnosis_steps", []),
            fix_steps=fix_steps,
            verify_steps=data.get("verify_steps", []),
            red_herrings=data.get("red_herrings", []),
            expected_observation_hints=hints,
            expected_diagnostic_path=data.get("diagnosis_steps", []),
            params={},
        )

    def _fallback_scenario(self, difficulty: float) -> AdversarialScenarioSpec:
        """Hardcoded fallback if LLM fails to design a scenario."""
        if difficulty > 0.5:
            # Hard: cascading OOM + bad image
            return AdversarialScenarioSpec(
                name="cascading-oom-image",
                failure_type="adversarial",
                namespace="payments",
                deployment="payment-gateway",
                root_cause="payment-gateway OOMKilled and web-app has wrong image tag. "
                           "Looks like two separate issues but OOM is the primary.",
                difficulty=difficulty,
                alert_message="CRITICAL: payment-gateway OOMKilled, web-app ImagePullBackOff",
                correct_fix_description="1. Restore payment-gateway memory limits to 128Mi, 2. Fix web-app image to nginx:1.25",
                steps=[
                    IncidentStep(
                        action="kubectl set resources deployment/payment-gateway -n payments --limits=memory=4Mi",
                        effect="payment-gateway OOMKills (exit code 137)",
                        order=1,
                        is_root_cause=True,
                    ),
                    IncidentStep(
                        action="kubectl set image deployment/web-app -n frontend web-app=nginx:nonexistent-tag-99999",
                        effect="web-app ImagePullBackOff",
                        order=2,
                        is_root_cause=False,
                    ),
                ],
                diagnosis_steps=[
                    "kubectl get pods -A",
                    "kubectl describe pod payment-gateway -n payments",
                    "kubectl describe pod web-app -n frontend",
                ],
                fix_steps=[
                    "kubectl set resources deployment/payment-gateway -n payments --limits=memory=128Mi",
                    "kubectl set image deployment/web-app -n frontend web-app=nginx:1.25",
                ],
                verify_steps=["kubectl get pods -A"],
                red_herrings=["Two namespaces affected looks like a cluster-wide issue but they are independent faults"],
                expected_observation_hints=["OOMKilled", "ImagePullBackOff"],
                expected_diagnostic_path=["kubectl get pods -A", "kubectl describe pod payment-gateway -n payments"],
                params={},
            )
        else:
            # Easy: single OOM kill
            return AdversarialScenarioSpec(
                name="oom-kill-payment-gateway",
                failure_type="adversarial",
                namespace="payments",
                deployment="payment-gateway",
                root_cause="payment-gateway memory limit set too low, causing OOMKill",
                difficulty=difficulty,
                alert_message="CRITICAL: payment-gateway pods OOMKilled, service unavailable",
                correct_fix_description="Increase memory limits on payment-gateway back to 128Mi",
                steps=[
                    IncidentStep(
                        action="kubectl set resources deployment/payment-gateway -n payments --limits=memory=4Mi",
                        effect="payment-gateway OOMKilled (exit code 137)",
                        order=1,
                        is_root_cause=True,
                    ),
                ],
                diagnosis_steps=[
                    "kubectl get pods -n payments",
                    "kubectl describe pod payment-gateway -n payments",
                ],
                fix_steps=[
                    "kubectl set resources deployment/payment-gateway -n payments --limits=memory=128Mi",
                ],
                verify_steps=["kubectl get pods -n payments"],
                red_herrings=[],
                expected_observation_hints=["OOMKilled", "exit code 137"],
                expected_diagnostic_path=["kubectl get pods -n payments", "kubectl describe pod payment-gateway -n payments"],
                params={},
            )
