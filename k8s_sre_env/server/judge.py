import logging
from .llm_client import LLMClient
from ..models import ScenarioSpec

logger = logging.getLogger(__name__)

PERSONAS = {
    "junior": """You are a Junior SRE mentor evaluating a trainee's incident response.
Be encouraging and lenient. Give partial credit for partially correct approaches.
Provide hints in feedback. Accept approximate answers.""",

    "senior": """You are a Senior SRE evaluating an engineer's incident response.
Apply standard expectations. Reward systematic diagnosis.
Penalize repeated commands and irrelevant actions.""",

    "principal": """You are a Principal SRE evaluating incident response with high standards.
Be strict. Demand root cause diagnosis before fix attempts.
Penalize inefficient investigation. Reward precise, targeted commands.""",
}


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        command: str,
        output: str,
        scenario: ScenarioSpec,
        history: list,
        persona: str = "junior",
    ) -> tuple[float, str]:

        history_summary = "\n".join(
            f"  Step {h['step']}: {h['command']} -> reward {h['reward']:.2f}"
            for h in history[-5:]
        ) or "  (first step)"

        user_prompt = f"""Evaluate this SRE action during a Kubernetes incident.

INCIDENT:
- Alert: {scenario.alert_message}
- Root cause: {scenario.root_cause}
- Correct fix: {scenario.correct_fix_description}
- Difficulty: {scenario.difficulty:.1f}/1.0

AGENT ACTION:
- Command: {command}
- Output (truncated): {output[:500]}

RECENT HISTORY:
{history_summary}
- Total steps taken: {len(history) + 1}/15

Return JSON only: {{"score": <float -1.0 to 1.0>, "feedback": "<1-2 sentence evaluation>"}}"""

        try:
            result = self.llm.chat_json(PERSONAS[persona], user_prompt, temperature=0.3)
            score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
            feedback = result.get("feedback", "")
            return score, feedback
        except Exception as e:
            logger.error(f"Judge LLM error: {e}")
            return 0.0, "Judge unavailable."
