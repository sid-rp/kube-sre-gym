from dataclasses import dataclass, field
from pydantic import Field as PydanticField
from openenv.core.env_server.types import Action, Observation, State


class K8sSREAction(Action):
    """Agent's action — a kubectl command or diagnosis/fix statement."""
    command: str = PydanticField(..., min_length=1, description="kubectl command or diagnose:/fix: statement")


class K8sSREObservation(Observation):
    """What the agent sees after each action."""
    command_output: str = PydanticField(default="", description="Output from the last command")
    cluster_status_summary: str = PydanticField(default="", description="Current cluster pod status")
    active_alerts: list[str] = PydanticField(default_factory=list, description="Active PagerDuty alerts")
    steps_taken: int = PydanticField(default=0, ge=0, description="Steps taken this episode")
    max_steps: int = PydanticField(default=15, description="Max steps per episode")
    hint: str = PydanticField(default="", description="Hint for junior persona")


class K8sSREState(State):
    """Episode metadata."""
    incident_id: str = ""
    difficulty: float = 0.2
    incident_type: str = ""
    root_cause: str = ""
    correct_fix: str = ""
    is_resolved: bool = False
    cumulative_reward: float = 0.0
    judge_persona: str = "junior"
    curriculum_stats: dict = {}


@dataclass
class ScenarioSpec:
    """A dynamically generated failure scenario."""
    failure_type: str
    namespace: str
    deployment: str
    params: dict
    root_cause: str
    difficulty: float
    alert_message: str
    correct_fix_description: str
    expected_diagnostic_path: list[str] = field(default_factory=list)
