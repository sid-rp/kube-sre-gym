import json
import logging
import random
from .llm_client import LLMClient
from models import ScenarioSpec

logger = logging.getLogger(__name__)

# What the backend can actually inject into GKE
INJECTABLE_FAILURES = {
    "oom_kill": "Sets memory limit to 4Mi — pod OOMKills (exit code 137)",
    "crashloop": "Changes container command to 'exit 1' — CrashLoopBackOff",
    "image_pull": "Sets image to nonexistent tag — ImagePullBackOff/ErrImagePull",
    "bad_config": "Sets DB_HOST env var to invalid host — app connection errors",
    "liveness_probe": "Sets liveness probe to wrong path — pod restart loop",
    "resource_quota": "Applies tight ResourceQuota — new pods blocked from scheduling",
    "cascading_db": "OOMs redis — payment-api loses backend — frontend 502s",
}

TOPOLOGY = {
    "payments": ["payment-api", "redis", "postgres"],
    "frontend": ["web-frontend", "nginx-proxy"],
    "auth": ["auth-service", "token-store"],
}

# Simple scenario pool — used in simple mode and as LLM fallback
SCENARIO_POOL = [
    ScenarioSpec(
        failure_type="oom_kill", namespace="payments", deployment="payment-api",
        params={"namespace": "payments", "deployment": "payment-api"},
        root_cause="Pod memory limit too low, causing OOMKill",
        difficulty=0.2, alert_message="CRITICAL: payment-api pods OOMKilled",
        correct_fix_description="Increase memory limits on payment-api",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl describe pod payment-api -n payments"],
    ),
    ScenarioSpec(
        failure_type="crashloop", namespace="payments", deployment="payment-api",
        params={"namespace": "payments", "deployment": "payment-api"},
        root_cause="Container crashes on startup due to bad command",
        difficulty=0.2, alert_message="CRITICAL: payment-api CrashLoopBackOff",
        correct_fix_description="Fix container command, rollout restart",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl logs payment-api -n payments"],
    ),
    ScenarioSpec(
        failure_type="image_pull", namespace="frontend", deployment="web-frontend",
        params={"namespace": "frontend", "deployment": "web-frontend"},
        root_cause="Deployment references non-existent image tag",
        difficulty=0.2, alert_message="WARNING: web-frontend ImagePullBackOff",
        correct_fix_description="Set image to valid tag",
        expected_diagnostic_path=["kubectl get pods -n frontend", "kubectl describe pod web-frontend -n frontend"],
    ),
    ScenarioSpec(
        failure_type="bad_config", namespace="payments", deployment="payment-api",
        params={"namespace": "payments", "deployment": "payment-api"},
        root_cause="DB_HOST env var changed to invalid host",
        difficulty=0.4, alert_message="CRITICAL: payment-api returning 500 errors",
        correct_fix_description="Set DB_HOST back to correct postgres host",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl logs payment-api -n payments"],
    ),
    ScenarioSpec(
        failure_type="liveness_probe", namespace="frontend", deployment="web-frontend",
        params={"namespace": "frontend", "deployment": "web-frontend"},
        root_cause="Liveness probe checking wrong path, pod keeps restarting",
        difficulty=0.5, alert_message="WARNING: web-frontend pods restarting frequently",
        correct_fix_description="Patch liveness probe to correct path",
        expected_diagnostic_path=["kubectl get pods -n frontend", "kubectl describe pod web-frontend -n frontend"],
    ),
    ScenarioSpec(
        failure_type="resource_quota", namespace="payments", deployment="payment-api",
        params={"namespace": "payments"},
        root_cause="ResourceQuota too tight, blocking new pod creation",
        difficulty=0.5, alert_message="WARNING: payment-api unable to scale",
        correct_fix_description="Delete or increase the restrictive ResourceQuota",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl get events -n payments"],
    ),
    ScenarioSpec(
        failure_type="cascading_db", namespace="payments", deployment="redis",
        params={"namespace": "payments"},
        root_cause="Redis OOMKilled causing cascading failures across services",
        difficulty=0.8, alert_message="CRITICAL: Multiple services degraded",
        correct_fix_description="Fix redis memory limits, restart dependent services",
        expected_diagnostic_path=["kubectl get pods -A", "kubectl describe pod redis -n payments"],
    ),
]

SYSTEM_PROMPT = f"""You are an SRE chaos engineering expert designing Kubernetes failure scenarios
to train an AI incident-response agent.

Available injectable failure types:
{json.dumps(INJECTABLE_FAILURES, indent=2)}

Cluster topology:
{json.dumps(TOPOLOGY, indent=2)}

Generate realistic, educational scenarios. Always return valid JSON."""


class ScenarioGenerator:
    """
    Two modes:
      simple (default) — cycles through SCENARIO_POOL based on difficulty + weak spots
      llm — uses LLM to generate novel scenarios dynamically
    """

    def __init__(self, llm: LLMClient = None, mode: str = "simple"):
        self.llm = llm
        self.mode = mode

    def generate(self, skill_profile: dict, difficulty: float) -> ScenarioSpec:
        if self.mode == "llm" and self.llm:
            return self._generate_llm(skill_profile, difficulty)
        return self._generate_simple(skill_profile, difficulty)

    def _generate_simple(self, skill_profile: dict, difficulty: float) -> ScenarioSpec:
        candidates = [s for s in SCENARIO_POOL if s.difficulty <= difficulty + 0.2]
        if not candidates:
            candidates = SCENARIO_POOL[:3]

        if skill_profile:
            weak_types = {ft for ft, rate in skill_profile.items() if rate < 0.5}
            weak_candidates = [s for s in candidates if s.failure_type in weak_types]
            if weak_candidates:
                candidates = weak_candidates

        return random.choice(candidates)

    def _generate_llm(self, skill_profile: dict, difficulty: float) -> ScenarioSpec:
        weak_spots = [k for k, v in skill_profile.items() if v < 0.5] if skill_profile else []

        user_prompt = f"""Generate a Kubernetes failure scenario.

Target difficulty: {difficulty:.2f}/1.0
Agent skill profile: {json.dumps(skill_profile) if skill_profile else "no history yet"}
Weak spots to target: {weak_spots if weak_spots else "any"}

Difficulty guide:
  < 0.4: single service, obvious symptoms
  0.4-0.7: requires correlating multiple signals
  > 0.7: cascading failures, misleading symptoms

Return JSON:
{{
  "failure_type": "<one of: {', '.join(INJECTABLE_FAILURES.keys())}>",
  "namespace": "<payments|frontend|auth>",
  "deployment": "<valid deployment from topology>",
  "params": {{"namespace": "...", "deployment": "..."}},
  "root_cause": "<one-sentence root cause>",
  "difficulty": {difficulty},
  "alert_message": "<PagerDuty-style alert>",
  "correct_fix_description": "<how to fix it>",
  "expected_diagnostic_path": ["<kubectl cmd 1>", "..."]
}}"""

        try:
            data = self.llm.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.8)
            return ScenarioSpec(**data)
        except Exception as e:
            logger.error(f"ScenarioGenerator LLM error: {e} — falling back to simple")
            return self._generate_simple(skill_profile, difficulty)
