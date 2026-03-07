from openenv.core.http_env_client import HTTPEnvClient
from openenv.core.types import StepResult
from .models import K8sSREAction, K8sSREObservation, K8sSREState


class K8sSREEnv(HTTPEnvClient[K8sSREAction, K8sSREObservation]):
    def _step_payload(self, action):
        return {"command": action.command}

    def _parse_result(self, payload):
        return StepResult(
            observation=K8sSREObservation(**payload["observation"]),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False))

    def _parse_state(self, payload):
        return K8sSREState(**payload)
