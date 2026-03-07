import os
import logging
from openenv.core.env_server import Environment
from openenv.core.types import StepResult
from ..models import K8sSREAction, K8sSREObservation, K8sSREState
from .llm_client import LLMClient
from .k8s_backend import K8sBackend
from .scenario_generator import ScenarioGenerator
from .curriculum import CurriculumController
from .judge import LLMJudge

logger = logging.getLogger(__name__)


class K8sEnvironment(Environment):
    """
    Self-contained OpenEnv environment for K8s SRE training.

    Spins up its own LLM client (judge + scenario generator) internally.
    Connects to a real GKE cluster for failure injection and observation.

    Config via env vars:
      # LLM (picks HF Inference by default — zero infra)
      LLM_BACKEND    - "hf" (default) or "openai"
      LLM_MODEL      - model name (default: Qwen/Qwen2.5-72B-Instruct)
      HF_TOKEN       - HuggingFace token (for hf backend)

      # GKE cluster
      GCP_SA_KEY_JSON    - SA key JSON string (for HF Spaces)
      GKE_CLUSTER_ENDPOINT - e.g. https://34.169.10.97
      GKE_CA_CERT        - base64 CA cert

      # Scenario generation
      GENERATOR_MODE - "simple" (default) or "llm"
    """

    def __init__(self):
        # Single LLM client used by both judge and generator
        llm = LLMClient()

        self.backend = K8sBackend()
        self.generator = ScenarioGenerator(llm, mode=os.environ.get("GENERATOR_MODE", "simple"))
        self.curriculum = CurriculumController()
        self.judge = LLMJudge(llm)

        self.scenario = None
        self.step_count = 0
        self.max_steps = 15
        self.history = []

    async def reset(self):
        self.backend.reset()

        skill_profile = self.curriculum.get_skill_profile()
        difficulty = self.curriculum.get_difficulty()

        self.scenario = self.generator.generate(skill_profile, difficulty)
        self.backend.inject_failure(self.scenario.failure_type, self.scenario.params)

        self.step_count = 0
        self.history = []

        cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")
        persona = self.curriculum.get_judge_persona()

        return K8sSREObservation(
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
        )

    async def step(self, action: K8sSREAction):
        self.step_count += 1

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

        if self.step_count >= self.max_steps:
            done = True
            reward -= 0.2
            feedback = "Timeout -- incident not resolved."

        self.history.append({
            "step": self.step_count,
            "command": action.command,
            "output": output[:200],
            "reward": reward,
            "feedback": feedback,
        })

        if done:
            self.curriculum.record(
                failure_type=self.scenario.failure_type,
                success="resolved" in feedback.lower(),
                steps=self.step_count,
                reward=sum(h["reward"] for h in self.history),
            )

        cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")

        return StepResult(
            observation=K8sSREObservation(
                command_output=output,
                cluster_status_summary=cluster_summary,
                active_alerts=[self.scenario.alert_message] if not done else [],
                steps_taken=self.step_count,
                max_steps=self.max_steps,
                hint=feedback if persona != "principal" else "",
            ),
            reward=reward,
            done=done,
        )

    async def state(self):
        return K8sSREState(
            incident_id=f"INC-{id(self.scenario) % 10000:04d}" if self.scenario else "",
            difficulty=self.curriculum.get_difficulty(),
            incident_type=self.scenario.failure_type if self.scenario else "",
            root_cause=self.scenario.root_cause if self.scenario else "",
            correct_fix=self.scenario.correct_fix_description if self.scenario else "",
            is_resolved=bool(self.history and "resolved" in self.history[-1].get("feedback", "").lower()),
            cumulative_reward=sum(h["reward"] for h in self.history),
            judge_persona=self.curriculum.get_judge_persona(),
            curriculum_stats=self.curriculum.get_stats(),
        )
