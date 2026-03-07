from dataclasses import dataclass, field
from openenv.core.env_server import Action, Observation, State


@dataclass
class K8sSREAction(Action):
    """Agent's action — a kubectl command or diagnosis/fix statement."""
    command: str


@dataclass
class K8sSREObservation(Observation):
    """What the agent sees after each action."""
    command_output: str
    cluster_status_summary: str
    active_alerts: list[str] = field(default_factory=list)
    steps_taken: int = 0
    max_steps: int = 15
    hint: str = ""


@dataclass
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
    curriculum_stats: dict = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    """A dynamically generated failure scenario."""
    failure_type: str           # injectable primitive
    namespace: str
    deployment: str
    params: dict                # passed to backend.inject_failure
    root_cause: str
    difficulty: float           # 0.0-1.0
    alert_message: str
    correct_fix_description: str
    expected_diagnostic_path: list[str] = field(default_factory=list)
