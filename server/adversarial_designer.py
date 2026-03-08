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
import time

from .llm_client import LLMClient
from .k8s_backend import K8sBackend
from .scenario_generator import TOPOLOGY

try:
    from ..models import AdversarialScenarioSpec, IncidentStep
except ImportError:
    from models import AdversarialScenarioSpec, IncidentStep

logger = logging.getLogger(__name__)


# ---- Steady-state: what "healthy" looks like ----
HEALTHY_STATE = {
    "payments": {
        "payment-api": {"image": "nginx:latest", "env": {"DB_HOST": "postgres.payments.svc.cluster.local", "REDIS_URL": "redis://redis.payments.svc.cluster.local:6379", "PORT": "8080"}, "memory_limit": "256Mi", "replicas": 2},
        "redis": {"image": "redis:7-alpine", "env": {}, "memory_limit": "128Mi", "replicas": 1},
        "postgres": {"image": "postgres:15-alpine", "env": {"POSTGRES_DB": "payments", "PGPORT": "5432"}, "memory_limit": "256Mi", "replicas": 1},
    },
    "frontend": {
        "web-frontend": {"image": "nginx:latest", "env": {"API_URL": "http://payment-api.payments.svc.cluster.local:8080", "PORT": "3000"}, "memory_limit": "128Mi", "replicas": 2},
        "nginx-proxy": {"image": "nginx:latest", "env": {"UPSTREAM_PORT": "3000"}, "memory_limit": "64Mi", "replicas": 1},
    },
    "auth": {
        "auth-service": {"image": "nginx:latest", "env": {"DB_HOST": "token-store.auth.svc.cluster.local", "TOKEN_SECRET": "supersecret", "PORT": "8081"}, "memory_limit": "128Mi", "replicas": 2},
        "token-store": {"image": "redis:7-alpine", "env": {}, "memory_limit": "64Mi", "replicas": 1},
    },
}


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
   Cascading: upstream services (nginx-proxy, web-frontend) get 502 errors
   Red herring: looks like a network issue but it's a config issue

3. PORT CONFLICT (two services on same port)
   Inject: kubectl set env deployment/<deploy_A> -n <ns> PORT=8080
          (when another service already listens on 8080)
   Fix:    kubectl set env deployment/<deploy_A> -n <ns> PORT=<original_port>
   Symptoms: one service crashes, the other works fine. Confusing partial outage.

4. LOW MEMORY LIMIT (OOMKill)
   Inject: kubectl set resources deployment/<deploy> -n <ns> --limits=memory=4Mi
   Fix:    kubectl set resources deployment/<deploy> -n <ns> --limits=memory=<correct from healthy baseline>
   Symptoms: OOMKilled status, exit code 137, pod restarts
   Red herring: looks like a memory leak but it's just a low limit

5. BAD IMAGE TAG
   Inject: kubectl set image deployment/<deploy> -n <ns> <container>=nginx:nonexistent-tag-99999
   Fix:    kubectl set image deployment/<deploy> -n <ns> <container>=<correct from healthy baseline>
   Symptoms: ImagePullBackOff, ErrImagePull

6. SCALE TO ZERO
   Inject: kubectl scale deployment/<deploy> -n <ns> --replicas=0
   Fix:    kubectl scale deployment/<deploy> -n <ns> --replicas=<correct from healthy baseline>
   Symptoms: no pods, dependent services fail with connection refused

7. BAD LIVENESS PROBE
   Inject: kubectl patch deployment/<deploy> -n <ns> -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"<container>","livenessProbe":{{"httpGet":{{"path":"/nonexistent-health-check"}}}}}}]}}}}}}}}'
   Fix:    kubectl patch deployment/<deploy> -n <ns> -p '{{"spec":{{"template":{{"spec":{{"containers":[{{"name":"<container>","livenessProbe":{{"httpGet":{{"path":"/health"}}}}}}]}}}}}}}}'
   Symptoms: pod restarting every 30s, CrashLoopBackOff

8. WRONG DATABASE CREDENTIALS / CONFIG
   Inject: kubectl set env deployment/<deploy> -n <ns> POSTGRES_DB=wrong_db_name
   Fix:    kubectl set env deployment/<deploy> -n <ns> POSTGRES_DB=<correct from healthy baseline>
   Symptoms: app logs show "database not found" or authentication errors

STEP 4 — DESIGN THE INCIDENT

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
        """Ask the LLM to design a complex incident based on agent's skill gaps.

        Uses progressive context enrichment (ChaosEater pattern):
        topology + healthy baseline + current health → scenario design.
        """
        # Budget: reserve ~7 steps for triage+investigation+verify
        max_fix_steps = max(1, self.max_steps - 7)
        # Scale mutations with difficulty: easy=1, medium=2, hard=3-4
        max_mutations = min(4, max(1, int(1 + difficulty * 3)))

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

{"IMPORTANT: Make it HARDER than basic single-fault scenarios. Use compound failures — e.g., wrong DB host causing retry storm that OOMs another service, or port conflict between two services." if difficulty > 0.5 else "Start with a clear single-fault scenario. One root cause, one cascading effect."}

{"Focus on these weak areas: " + ", ".join(weak_spots) if weak_spots else ""}"""

        try:
            data = self.llm.chat_json(system_prompt, user_prompt, temperature=0.7, max_tokens=2048)
            scenario = self._parse_scenario(data, difficulty)
            logger.info(f"Adversarial scenario designed: {scenario.name} (difficulty={scenario.difficulty}, "
                        f"faults={len(scenario.steps)}, fix_steps={len(scenario.fix_steps)})")
            return scenario
        except Exception as e:
            logger.error(f"AdversarialDesigner error: {e} — falling back to hardcoded scenario")
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
        """Convert LLM JSON response into AdversarialScenarioSpec."""
        inject_commands = data.get("inject_commands", [])
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
            fix_steps=data.get("fix_steps", []),
            verify_steps=data.get("verify_steps", []),
            red_herrings=data.get("red_herrings", []),
            expected_observation_hints=hints,
            expected_diagnostic_path=data.get("diagnosis_steps", []),
            params={},
        )

    def _fallback_scenario(self, difficulty: float) -> AdversarialScenarioSpec:
        """Hardcoded fallback if LLM fails to design a scenario."""
        if difficulty > 0.5:
            # Hard: cascading auth + port conflict
            return AdversarialScenarioSpec(
                name="cascading-auth-port-conflict",
                failure_type="adversarial",
                namespace="auth",
                deployment="auth-service",
                root_cause="auth-service DB_HOST misconfigured, causing cascading failures. "
                           "payment-api OOMs from retry storm — looks like memory issue but it's auth.",
                difficulty=difficulty,
                alert_message="CRITICAL: Multiple services returning 500 errors, payment-api OOMKilled",
                correct_fix_description="1. Fix auth-service DB_HOST, 2. Restore payment-api memory limits, 3. Restart payment-api",
                steps=[
                    IncidentStep(
                        action="kubectl set env deployment/auth-service -n auth DB_HOST=wrong-host.invalid",
                        effect="auth-service returns 500 on all requests",
                        order=1,
                        is_root_cause=True,
                    ),
                    IncidentStep(
                        action="kubectl set resources deployment/payment-api -n payments --limits=memory=16Mi",
                        effect="payment-api OOMKills (looks like memory leak, actually retry storm)",
                        order=2,
                        is_root_cause=False,
                    ),
                ],
                diagnosis_steps=[
                    "kubectl get pods -A",
                    "kubectl get events -n payments",
                    "kubectl describe pod payment-api -n payments",
                    "kubectl logs auth-service -n auth --tail=50",
                ],
                fix_steps=[
                    "kubectl set env deployment/auth-service -n auth DB_HOST=token-store.auth.svc.cluster.local",
                    "kubectl set resources deployment/payment-api -n payments --limits=memory=256Mi",
                    "kubectl rollout restart deployment/payment-api -n payments",
                ],
                verify_steps=["kubectl get pods -A"],
                red_herrings=["payment-api OOMKilled looks like a memory leak but is caused by auth retry storm"],
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
                correct_fix_description="Set DB_HOST back to postgres.payments.svc.cluster.local",
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
                    "kubectl set env deployment/payment-api -n payments DB_HOST=postgres.payments.svc.cluster.local",
                ],
                verify_steps=["kubectl get pods -n payments"],
                red_herrings=[],
                expected_observation_hints=["connection refused", "host not found"],
                expected_diagnostic_path=["kubectl get pods -n payments", "kubectl logs payment-api -n payments --tail=50"],
                params={},
            )
