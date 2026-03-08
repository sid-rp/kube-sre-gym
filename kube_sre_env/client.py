"""Kube SRE Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient

from .models import K8sSREAction, K8sSREObservation, K8sSREState


class K8sSREEnv(
    EnvClient[K8sSREAction, K8sSREObservation, K8sSREState]
):
    """
    Client for the K8s SRE Environment.

    Example:
        >>> with K8sSREEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.command_output)
        ...     result = client.step(K8sSREAction(command="kubectl get pods -A"))
        ...     print(result.observation.command_output)

    Example with Docker:
        >>> client = K8sSREEnv.from_docker_image("kube_sre_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(K8sSREAction(command="kubectl get pods -A"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: K8sSREAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[K8sSREObservation]:
        obs_data = payload.get("observation", {})
        observation = K8sSREObservation(
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

    def _parse_state(self, payload: Dict) -> K8sSREState:
        return K8sSREState(
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
