"""
Kube SRE Gym Environment Implementation.

Agent diagnoses and fixes real GKE incidents with curriculum-driven difficulty.

Modes (set via GYM_MODE env var):
  standard     — single-fault scenarios from pool or LLM generator (default)
  adversarial  — multi-step incidents designed by external LLM judge
"""

import os
import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState
except ImportError:
    from models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState

from .llm_client import LLMClient
from .k8s_backend import K8sBackend
from .scenario_generator import ScenarioGenerator
from .curriculum import CurriculumController
from .judge import LLMJudge, AdversarialJudge
from .adversarial_designer import AdversarialDesigner

logger = logging.getLogger(__name__)


class KubeSreGymEnvironment(Environment):
    """
    K8s SRE OpenEnv Environment — agent diagnoses and fixes real GKE incidents.

    Config via env vars:
      GYM_MODE       - "standard" (default) or "adversarial"
      LLM_BACKEND    - "openai" (default), "hf", or "anthropic"
      LLM_MODEL      - model name
      HF_TOKEN       - HuggingFace token
      ANTHROPIC_API_KEY - Anthropic API key (for adversarial mode)
      K8S_ENDPOINT   - GKE API endpoint
      K8S_TOKEN      - Bearer token for GKE
      K8S_CA_CERT    - Base64 CA cert
      GENERATOR_MODE - "simple" (default) or "llm"
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        llm = LLMClient()
        self.backend = K8sBackend()
        self.curriculum = CurriculumController()
        self.mode = os.environ.get("GYM_MODE", "standard")

        self.scenario = None
        self._step_count = 0
        self.max_steps = int(os.environ.get("MAX_STEPS", "15"))
        self.history = []
        self._state = KubeSreGymState(episode_id=str(uuid4()), step_count=0)

        if self.mode == "adversarial":
            self.designer = AdversarialDesigner(llm, self.backend, max_steps=self.max_steps)
            self.judge = AdversarialJudge(llm)
            self.generator = None
            logger.info("GYM_MODE=adversarial — LLM designs multi-step incidents")
        else:
            self.designer = None
            self.generator = ScenarioGenerator(llm, mode=os.environ.get("GENERATOR_MODE", "simple"))
            self.judge = LLMJudge(llm)
            logger.info("GYM_MODE=standard — using scenario pool/generator")

    def reset(self) -> KubeSreGymObservation:
        self.backend.reset()

        skill_profile = self.curriculum.get_skill_profile()
        difficulty = self.curriculum.get_difficulty()

        if self.mode == "adversarial":
            self.scenario = self.designer.design(skill_profile, difficulty)
            self.designer.inject(self.scenario)
        else:
            self.scenario = self.generator.generate(skill_profile, difficulty)
            self.backend.inject_failure(self.scenario.failure_type, self.scenario.params)

        self._step_count = 0
        self.history = []
        self._state = KubeSreGymState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            incident_type=self.scenario.failure_type,
            root_cause=self.scenario.root_cause,
            correct_fix=self.scenario.correct_fix_description,
            judge_persona=self.curriculum.get_judge_persona(),
            curriculum_stats=self.curriculum.get_stats(),
        )

        persona = self.curriculum.get_judge_persona()

        # Initial observation: richer snapshot so agent can decide where to dig
        pods_output = self.backend.execute("kubectl get pods --all-namespaces")
        events_output = self.backend.execute("kubectl get events --all-namespaces")
        cluster_summary = f"=== POD STATUS ===\n{pods_output}\n\n=== RECENT EVENTS ===\n{events_output}"

        return KubeSreGymObservation(
            command_output=(
                f"PAGERDUTY ALERT: {self.scenario.alert_message}\n\n"
                f"You are the on-call SRE. Investigate and resolve this incident.\n"
                f"Use kubectl commands to diagnose, then submit:\n"
                f"  'diagnose: <root cause>'\n"
                f"  'fix: kubectl <fix command>'"
            ),
            cluster_status_summary=cluster_summary,
            active_alerts=[self.scenario.alert_message],
            steps_taken=0,
            max_steps=self.max_steps,
            hint="Start by checking pod status in the affected namespace." if persona == "junior" else "",
            done=False,
            reward=0.0,
        )

    def step(self, action: KubeSreGymAction) -> KubeSreGymObservation:
        self._step_count += 1
        self._state.step_count = self._step_count

        output = self.backend.execute(action.command)

        # Penalize repeated commands (same command run before)
        repeat_count = sum(1 for h in self.history if h["command"] == action.command)

        persona = self.curriculum.get_judge_persona()
        reward, feedback = self.judge.evaluate(
            action.command, output, self.scenario, self.history, persona
        )

        if repeat_count > 0:
            penalty = min(0.5, repeat_count * 0.15)
            reward -= penalty
            feedback += f" Repeated command ({repeat_count + 1}x)."

        done = False

        if action.command.startswith("fix:"):
            health = self.backend.check_health()
            healthy_count = sum(
                1 for ns_pods in health.values() for s in ns_pods.values()
                if s in ("Running", "Completed")
            )
            total_count = sum(len(ns_pods) for ns_pods in health.values())
            all_healthy = healthy_count == total_count and total_count > 0

            if all_healthy:
                done = True
                reward += 0.5
                feedback = "Incident resolved! All pods healthy."
            else:
                # Partial progress feedback — tell agent how many pods are healthy
                feedback += f" Fix applied. {healthy_count}/{total_count} pods healthy."

        if self._step_count >= self.max_steps:
            done = True
            reward -= 0.2
            feedback = "Timeout -- incident not resolved."

        self.history.append({
            "step": self._step_count,
            "command": action.command,
            "output": output[:200],
            "reward": reward,
            "feedback": feedback,
        })

        if done:
            # Track adversarial scenarios by name for curriculum granularity
            track_type = self.scenario.failure_type
            if hasattr(self.scenario, "name") and self.scenario.name:
                track_type = f"adversarial:{self.scenario.name}"

            self.curriculum.record(
                failure_type=track_type,
                success="resolved" in feedback.lower(),
                steps=self._step_count,
                reward=sum(h["reward"] for h in self.history),
            )
            self._state.is_resolved = "resolved" in feedback.lower()
            self._state.cumulative_reward = sum(h["reward"] for h in self.history)

        # Only auto-fetch cluster summary after fix attempts or on done
        # Otherwise the agent should run its own diagnostic commands
        if action.command.startswith("fix:") or done:
            cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")
        else:
            cluster_summary = ""

        return KubeSreGymObservation(
            command_output=output,
            cluster_status_summary=cluster_summary,
            active_alerts=[self.scenario.alert_message] if not done else [],
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            hint=feedback if persona != "principal" else "",
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> KubeSreGymState:
        return self._state
