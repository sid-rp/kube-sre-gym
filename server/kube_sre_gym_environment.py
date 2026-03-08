"""
Kube SRE Gym Environment Implementation.

Agent diagnoses and fixes real GKE incidents with curriculum-driven difficulty.
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
from .judge import LLMJudge

logger = logging.getLogger(__name__)


class KubeSreGymEnvironment(Environment):
    """
    K8s SRE OpenEnv Environment — agent diagnoses and fixes real GKE incidents.

    Config via env vars:
      LLM_BACKEND    - "hf" (default) or "openai"
      LLM_MODEL      - model name (default: Qwen/Qwen2.5-72B-Instruct)
      HF_TOKEN       - HuggingFace token
      K8S_ENDPOINT   - GKE API endpoint
      K8S_TOKEN      - Bearer token for GKE
      K8S_CA_CERT    - Base64 CA cert
      GENERATOR_MODE - "simple" (default) or "llm"
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        llm = LLMClient()
        self.backend = K8sBackend()
        self.generator = ScenarioGenerator(llm, mode=os.environ.get("GENERATOR_MODE", "simple"))
        self.curriculum = CurriculumController()
        self.judge = LLMJudge(llm)

        self.scenario = None
        self._step_count = 0
        self.max_steps = 15
        self.history = []
        self._state = KubeSreGymState(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> KubeSreGymObservation:
        self.backend.reset()

        skill_profile = self.curriculum.get_skill_profile()
        difficulty = self.curriculum.get_difficulty()

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

        cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")
        persona = self.curriculum.get_judge_persona()

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

        persona = self.curriculum.get_judge_persona()
        reward, feedback = self.judge.evaluate(
            action.command, output, self.scenario, self.history, persona
        )

        done = False

        if action.command.startswith("fix:"):
            health = self.backend.check_health()
            all_healthy = all(
                s in ("Running", "Completed")
                for ns_pods in health.values()
                for s in ns_pods.values()
            )
            if all_healthy:
                done = True
                reward += 0.5
                feedback = "Incident resolved! All pods healthy."
            else:
                feedback += " Fix applied but cluster not fully healthy yet."

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
            self.curriculum.record(
                failure_type=self.scenario.failure_type,
                success="resolved" in feedback.lower(),
                steps=self._step_count,
                reward=sum(h["reward"] for h in self.history),
            )
            self._state.is_resolved = "resolved" in feedback.lower()
            self._state.cumulative_reward = sum(h["reward"] for h in self.history)

        cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")

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
