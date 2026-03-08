"""
Data models for the Kube SRE Gym Environment.

Defines Action, Observation, State for K8s incident response training.
"""

from dataclasses import dataclass, field
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class KubeSreGymAction(Action):
    """Agent's action — a kubectl command or diagnosis/fix statement."""
    command: str = Field(..., min_length=1, description="kubectl command or diagnose:/fix: statement")


class KubeSreGymObservation(Observation):
    """What the agent sees after each action."""
    command_output: str = Field(default="", description="Output from the last command")
    cluster_status_summary: str = Field(default="", description="Current cluster pod status")
    active_alerts: list[str] = Field(default_factory=list, description="Active PagerDuty alerts")
    steps_taken: int = Field(default=0, ge=0, description="Steps taken this episode")
    max_steps: int = Field(default=15, description="Max steps per episode")
    hint: str = Field(default="", description="Hint for junior persona")


class KubeSreGymState(State):
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


@dataclass
class IncidentStep:
    """One mutation in a multi-step adversarial incident."""
    action: str
    effect: str
    order: int
    is_root_cause: bool = False


@dataclass
class AdversarialScenarioSpec:
    """Multi-step incident designed by an external LLM judge.

    Compatible with ScenarioSpec interface (same fields used by judge.evaluate).
    """
    # Fields matching ScenarioSpec interface (used by LLMJudge.evaluate)
    failure_type: str
    namespace: str
    deployment: str
    root_cause: str
    difficulty: float
    alert_message: str
    correct_fix_description: str

    # Multi-step incident fields
    name: str = ""
    steps: list[IncidentStep] = field(default_factory=list)
    diagnosis_steps: list[str] = field(default_factory=list)
    fix_steps: list[str] = field(default_factory=list)
    verify_steps: list[str] = field(default_factory=list)
    red_herrings: list[str] = field(default_factory=list)
    expected_observation_hints: list[str] = field(default_factory=list)

    # Kept for ScenarioSpec compat
    params: dict = field(default_factory=dict)
    expected_diagnostic_path: list[str] = field(default_factory=list)
