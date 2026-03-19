import json
import logging
import os
import random
from .llm_client import LLMClient
from .constants import INJECTABLE_FAILURES, TOPOLOGY, EVAL_HELD_OUT_COMBOS

try:
    from ..models import ScenarioSpec
except ImportError:
    from models import ScenarioSpec

logger = logging.getLogger(__name__)

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
        failure_type="image_pull", namespace="frontend", deployment="web-app",
        params={"namespace": "frontend", "deployment": "web-app"},
        root_cause="Deployment references non-existent image tag",
        difficulty=0.2, alert_message="WARNING: web-app ImagePullBackOff",
        correct_fix_description="Set image to valid tag",
        expected_diagnostic_path=["kubectl get pods -n frontend", "kubectl describe pod web-app -n frontend"],
    ),
    ScenarioSpec(
        failure_type="bad_config", namespace="payments", deployment="payment-worker",
        params={"namespace": "payments", "deployment": "payment-worker"},
        root_cause="DATABASE_URL env var changed to invalid host, worker can't connect",
        difficulty=0.4, alert_message="CRITICAL: payment-worker CrashLoopBackOff, connection errors in logs",
        correct_fix_description="Set DATABASE_URL back to correct postgres connection string",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl logs payment-worker -n payments"],
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
        failure_type="scale_zero", namespace="payments", deployment="payment-gateway",
        params={"namespace": "payments", "deployment": "payment-gateway"},
        root_cause="payment-gateway scaled to 0 replicas — no pods running",
        difficulty=0.35, alert_message="CRITICAL: payment-gateway has 0 available replicas",
        correct_fix_description="Scale payment-gateway back to 2 replicas",
        expected_diagnostic_path=["kubectl get pods -n payments", "kubectl get deployment payment-gateway -n payments"],
    ),
    ScenarioSpec(
        failure_type="scale_zero", namespace="auth", deployment="auth-service",
        params={"namespace": "auth", "deployment": "auth-service"},
        root_cause="auth-service scaled to 0 replicas — authentication unavailable",
        difficulty=0.35, alert_message="CRITICAL: auth-service has 0 available replicas, login failures",
        correct_fix_description="Scale auth-service back to 2 replicas",
        expected_diagnostic_path=["kubectl get pods -n auth", "kubectl get deployment auth-service -n auth"],
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

    def generate(self, skill_profile: dict, difficulty: float,
                 fault_type_hint: str = None) -> ScenarioSpec:
        """Generate a scenario, optionally targeting a specific fault type.

        Args:
            skill_profile: {fault_type: success_rate} from curriculum
            difficulty: 0.0-1.0 from curriculum
            fault_type_hint: if set, prefer this fault type (from curriculum.pick_fault_type())
        """
        if self.mode == "llm" and self.llm:
            return self._generate_llm(skill_profile, difficulty)
        return self._generate_simple(skill_profile, difficulty, fault_type_hint)

    def _generate_simple(self, skill_profile: dict, difficulty: float,
                         fault_type_hint: str = None) -> ScenarioSpec:
        # Train/eval split by fault+deployment combos (not entire deployments).
        # Training uses all deployments for diversity but excludes held-out combos.
        # Eval (EVAL_SPLIT=1) uses ONLY the held-out combos.
        pool = SCENARIO_POOL
        if os.environ.get("EVAL_SPLIT"):
            pool = [s for s in SCENARIO_POOL
                    if (s.failure_type, s.namespace, s.deployment) in EVAL_HELD_OUT_COMBOS]
            if not pool:
                pool = SCENARIO_POOL  # fallback
        else:
            pool = [s for s in SCENARIO_POOL
                    if (s.failure_type, s.namespace, s.deployment) not in EVAL_HELD_OUT_COMBOS]
            if not pool:
                pool = SCENARIO_POOL  # fallback if everything is held out

        candidates = [s for s in pool if s.difficulty <= difficulty + 0.2]
        if not candidates:
            candidates = pool[:3] if len(pool) >= 3 else pool

        # If curriculum gave us a specific fault type, prefer it
        if fault_type_hint:
            hint_candidates = [s for s in candidates if s.failure_type == fault_type_hint]
            if hint_candidates:
                return random.choice(hint_candidates)

        # Otherwise fall back to weak-spot targeting
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
