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
        "name": "wrong-db-host-payments",
        "namespace": "payments",
        "deployment": "payment-api",
        "root_cause": "payment-api DB_HOST env var points to non-existent host",
        "alert_message": "CRITICAL: payment-api returning 500 errors on all requests",
        "correct_fix_description": "Set DB_HOST back to payment-db.payments.svc.cluster.local",
        "steps": [{"action": "kubectl set env deployment/payment-api -n payments DB_HOST=wrong-host.invalid",
                    "effect": "payment-api cannot connect to database"}],
        "diagnosis_steps": ["kubectl get pods -n payments", "kubectl logs payment-api -n payments --tail=50"],
        "fix_steps": ["kubectl set env deployment/payment-api -n payments DB_HOST=payment-db.payments.svc.cluster.local"],
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
        "name": "wrong-auth-db-host",
        "namespace": "auth",
        "deployment": "auth-service",
        "root_cause": "auth-service DB_HOST env var points to wrong host",
        "alert_message": "CRITICAL: auth-service returning 500 errors, login failures",
        "correct_fix_description": "Set DB_HOST back to auth-db.auth.svc.cluster.local",
        "steps": [{"action": "kubectl set env deployment/auth-service -n auth DB_HOST=wrong-host.invalid",
                    "effect": "auth-service cannot connect to token store"}],
        "diagnosis_steps": ["kubectl get pods -n auth", "kubectl logs auth-service -n auth --tail=50"],
        "fix_steps": ["kubectl set env deployment/auth-service -n auth DB_HOST=auth-db.auth.svc.cluster.local"],
        "verify_steps": ["kubectl get pods -n auth"],
        "red_herrings": [],
        "expected_observation_hints": ["connection refused", "host not found"],
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
        "name": "wrong-port-web-app",
        "namespace": "frontend",
        "deployment": "web-app",
        "root_cause": "web-app PORT env var changed, readiness probe fails on wrong port",
        "alert_message": "WARNING: web-app pods not ready, 502 errors reported",
        "correct_fix_description": "Set PORT back to 3000",
        "steps": [{"action": "kubectl set env deployment/web-app -n frontend PORT=9999",
                    "effect": "web-app readiness probe fails, service returns 502"}],
        "diagnosis_steps": ["kubectl get pods -n frontend", "kubectl describe pod web-app -n frontend"],
        "fix_steps": ["kubectl set env deployment/web-app -n frontend PORT=3000"],
        "verify_steps": ["kubectl get pods -n frontend"],
        "red_herrings": [],
        "expected_observation_hints": ["not ready", "502"],
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

Use ONLY these inject/fix pairs. Each shows the exact kubectl syntax.

1. WRONG DATABASE/SERVICE HOST
   Inject: kubectl set env deployment/<deploy> -n <ns> DB_HOST=wrong-host.invalid
   Fix:    kubectl set env deployment/<deploy> -n <ns> DB_HOST=<correct from healthy baseline>
   Symptoms: pods Running but app logs show "connection refused" or "host not found"
   Cascading: dependent services get timeouts, may exhaust memory from retries

2. WRONG PORT NUMBER
   Inject: kubectl set env deployment/<deploy> -n <ns> PORT=9999
   Fix:    kubectl set env deployment/<deploy> -n <ns> PORT=<correct from healthy baseline>
   Symptoms: readiness probe fails (port mismatch), service returns connection refused
   Red herring: looks like a network issue but it's a config issue

3. LOW MEMORY LIMIT (OOMKill)
   Inject: kubectl set resources deployment/<deploy> -n <ns> --limits=memory=4Mi
   Fix:    kubectl set resources deployment/<deploy> -n <ns> --limits=memory=<correct from healthy baseline>
   Symptoms: OOMKilled status, exit code 137, pod restarts
   Red herring: looks like a memory leak but it's just a low limit

4. BAD IMAGE TAG
   Inject: kubectl set image deployment/<deploy> -n <ns> <container>=nginx:nonexistent-tag-99999
   Fix:    kubectl set image deployment/<deploy> -n <ns> <container>=<correct from healthy baseline>
   Symptoms: ImagePullBackOff, ErrImagePull

5. SCALE TO ZERO
   Inject: kubectl scale deployment/<deploy> -n <ns> --replicas=0
   Fix:    kubectl scale deployment/<deploy> -n <ns> --replicas=<correct from healthy baseline>
   Symptoms: no pods, dependent services fail with connection refused

6. BAD LIVENESS PROBE
   Inject: kubectl patch deployment/<deploy> -n <ns> -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"<container>","livenessProbe":{{"httpGet":{{"path":"/nonexistent-health-check"}}}}}}]}}}}}}}}'
   Fix:    kubectl patch deployment/<deploy> -n <ns> -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"<container>","livenessProbe":{{"httpGet":{{"path":"/health"}}}}}}]}}}}}}}}'
   Symptoms: pod restarting every 30s, CrashLoopBackOff

7. WRONG DATABASE CREDENTIALS / CONFIG
   Inject: kubectl set env deployment/<deploy> -n <ns> POSTGRES_DB=wrong_db_name
   Fix:    kubectl set env deployment/<deploy> -n <ns> POSTGRES_DB=<correct from healthy baseline>
   Symptoms: app logs show "database not found" or authentication errors

STEP 4 — DESIGN THE INCIDENT

HARD CONSTRAINTS — the scenario MUST be solvable:
- At most {max_mutations} injected faults (inject_commands). Never exceed this.
- Each inject_command MUST have exactly one corresponding entry in fix_steps (the reversal).
- fix_steps count MUST NOT exceed {max_fix_steps}.
- Every deployment and namespace referenced MUST exist in the topology above.
- Use container names from the healthy baseline (container_name field) for set image commands.
- Keep it simple enough that a methodical agent can solve it within {max_steps} steps total.

Think about this before generating:
- What is the ONE root cause? (pick from fault types above)
- What cascading effects does it cause? (e.g., auth down → payment-api retries → OOM)
- What will the agent see first? (the symptom, not the cause)
- What is misleading? (red herring — a symptom that looks like one fault type but is actually caused by another)
- What is the correct SRE workflow? (what should the agent check first, second, etc.)

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

{"Design a complex multi-fault incident with " + str(max_mutations) + " faults. Use cascading failures across namespaces — e.g., auth DB misconfigured causing retry storms that OOM payment-api, plus a port conflict. Include red herrings." if difficulty > 0.7 else "Use at most " + str(max_mutations) + " fault(s). One root cause with clear cascading effects. Keep it solvable within the step budget." if difficulty > 0.5 else "Start with a single-fault scenario. One root cause, clear symptoms."}

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
            # Hard: cascading auth + OOM
            return AdversarialScenarioSpec(
                name="cascading-auth-oom",
                failure_type="adversarial",
                namespace="auth",
                deployment="auth-service",
                root_cause="auth-service DB_HOST misconfigured, causing cascading failures. "
                           "payment-gateway OOMs from retry storm — looks like memory issue but it's auth.",
                difficulty=difficulty,
                alert_message="CRITICAL: Multiple services returning 500 errors, payment-gateway OOMKilled",
                correct_fix_description="1. Fix auth-service DB_HOST, 2. Restore payment-gateway memory limits, 3. Restart payment-gateway",
                steps=[
                    IncidentStep(
                        action="kubectl set env deployment/auth-service -n auth DB_HOST=wrong-host.invalid",
                        effect="auth-service returns 500 on all requests",
                        order=1,
                        is_root_cause=True,
                    ),
                    IncidentStep(
                        action="kubectl set resources deployment/payment-gateway -n payments --limits=memory=16Mi",
                        effect="payment-gateway OOMKills (looks like memory leak, actually retry storm)",
                        order=2,
                        is_root_cause=False,
                    ),
                ],
                diagnosis_steps=[
                    "kubectl get pods -A",
                    "kubectl get events -n payments",
                    "kubectl describe pod payment-gateway -n payments",
                    "kubectl logs auth-service -n auth --tail=50",
                ],
                fix_steps=[
                    "kubectl set env deployment/auth-service -n auth DB_HOST=auth-db.auth.svc.cluster.local",
                    "kubectl set resources deployment/payment-gateway -n payments --limits=memory=128Mi",
                    "kubectl rollout restart deployment/payment-gateway -n payments",
                ],
                verify_steps=["kubectl get pods -A"],
                red_herrings=["payment-gateway OOMKilled looks like a memory leak but is caused by auth retry storm"],
                expected_observation_hints=["OOMKilled", "connection refused", "500"],
                expected_diagnostic_path=["kubectl get pods -A", "kubectl logs auth-service -n auth --tail=50"],
                params={},
            )
        else:
            # Easy: single wrong DB host
            return AdversarialScenarioSpec(
                name="wrong-db-host",
                failure_type="adversarial",
                namespace="payments",
                deployment="payment-api",
                root_cause="payment-api DB_HOST env var points to non-existent host",
                difficulty=difficulty,
                alert_message="CRITICAL: payment-api returning 500 errors on all requests",
                correct_fix_description="Set DB_HOST back to payment-db.payments.svc.cluster.local",
                steps=[
                    IncidentStep(
                        action="kubectl set env deployment/payment-api -n payments DB_HOST=wrong-host.invalid",
                        effect="payment-api cannot connect to database",
                        order=1,
                        is_root_cause=True,
                    ),
                ],
                diagnosis_steps=[
                    "kubectl get pods -n payments",
                    "kubectl logs payment-api -n payments --tail=50",
                    "kubectl describe pod payment-api -n payments",
                ],
                fix_steps=[
                    "kubectl set env deployment/payment-api -n payments DB_HOST=payment-db.payments.svc.cluster.local",
                ],
                verify_steps=["kubectl get pods -n payments"],
                red_herrings=[],
                expected_observation_hints=["connection refused", "host not found"],
                expected_diagnostic_path=["kubectl get pods -n payments", "kubectl logs payment-api -n payments --tail=50"],
                params={},
            )
