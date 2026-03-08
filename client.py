"""Kube SRE Gym Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from .models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState


class KubeSreGymEnv(
    EnvClient[KubeSreGymAction, KubeSreGymObservation, KubeSreGymState]
):
    """
    Client for the K8s SRE Gym Environment.

    OpenEnv v0.2.1 uses sync WebSocket calls — no .sync() needed.

    Example:
        >>> with KubeSreGymEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.command_output)
        ...     result = client.step(KubeSreGymAction(command="kubectl get pods -A"))
        ...     print(result.observation.command_output)
    """

    def __init__(self, base_url: str, **kwargs):
        # K8s operations + LLM calls can be slow (especially adversarial reset)
        kwargs.setdefault("message_timeout_s", 300.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: KubeSreGymAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[KubeSreGymObservation]:
        obs_data = payload.get("observation", {})
        observation = KubeSreGymObservation(
            command_output=obs_data.get("command_output", ""),
            cluster_status_summary=obs_data.get("cluster_status_summary", ""),
            active_alerts=obs_data.get("active_alerts", []),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 15),
            hint=obs_data.get("hint", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> KubeSreGymState:
        return KubeSreGymState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            incident_id=payload.get("incident_id", ""),
            difficulty=payload.get("difficulty", 0.2),
            incident_type=payload.get("incident_type", ""),
            root_cause=payload.get("root_cause", ""),
            correct_fix=payload.get("correct_fix", ""),
            is_resolved=payload.get("is_resolved", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            judge_persona=payload.get("judge_persona", "junior"),
            curriculum_stats=payload.get("curriculum_stats", {}),
        )
