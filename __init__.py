"""Kube SRE Gym Environment."""

from .client import KubeSreGymEnv
from .models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState

__all__ = [
    "KubeSreGymAction",
    "KubeSreGymObservation",
    "KubeSreGymState",
    "KubeSreGymEnv",
]


def get_training_utils():
    """Lazy-import training utilities from train.py.

    Returns a dict with: SYSTEM_PROMPT, rollout_once, format_observation,
    format_history, parse_commands, apply_chat_template, reward_total,
    reward_diagnosis, reward_fix, plot_rewards.

    Usage in Colab:
        from kube_sre_gym import get_training_utils
        tu = get_training_utils()
        SYSTEM_PROMPT = tu["SYSTEM_PROMPT"]
        rollout_once = tu["rollout_once"]
    """
    from . import train as _train
    return {
        "SYSTEM_PROMPT": _train.SYSTEM_PROMPT,
        "rollout_once": _train.rollout_once,
        "format_observation": _train.format_observation,
        "format_history": _train.format_history,
        "parse_commands": _train.parse_commands,
        "apply_chat_template": _train.apply_chat_template,
        "reward_total": _train.reward_total,
        "reward_diagnosis": _train.reward_diagnosis,
        "reward_fix": _train.reward_fix,
        "plot_rewards": _train.plot_rewards,
    }
