import logging
from .llm_client import LLMClient

try:
    from ..models import ScenarioSpec, AdversarialScenarioSpec
except ImportError:
    from models import ScenarioSpec, AdversarialScenarioSpec

logger = logging.getLogger(__name__)

PERSONAS = {
    "junior": """You are a Junior SRE mentor evaluating a trainee's incident response.
Be encouraging and lenient. Give partial credit for partially correct approaches.
Provide hints in feedback. Accept approximate answers.""",

    "senior": """You are a Senior SRE evaluating an engineer's incident response.
Apply standard expectations. Reward systematic diagnosis.
Penalize repeated commands and irrelevant actions.""",

    "principal": """You are a Principal SRE evaluating incident response with high standards.
Reward efficiency — if the agent correctly identifies and fixes the issue quickly, that's GOOD.
Direct fixes are acceptable if they target the right resource. Penalize WRONG fixes, not fast ones.
For multi-fault scenarios, reward fixing ALL faults, not just one. Penalize incomplete fixes.""",
}


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify_resolution(
        self,
        scenario: ScenarioSpec,
        history: list,
        cluster_snapshot: str,
    ) -> tuple[bool, str]:
        """Ask the judge to verify if the incident is actually resolved.

        Runs after the programmatic health check passes. The judge looks at
        the full cluster state and action history to decide if ALL faults
        from the scenario were actually fixed.
        """
        history_summary = "\n".join(
            f"  Step {h['step']}: {h['command']} -> {h.get('output', '')[:100]}"
            for h in history
        )

        fix_desc = scenario.correct_fix_description
        # For adversarial scenarios, include fix_steps if available
        fix_steps = ""
        if hasattr(scenario, "fix_steps") and scenario.fix_steps:
            fix_steps = "\n".join(f"  - {s}" for s in scenario.fix_steps)

        user_prompt = f"""You are verifying whether a Kubernetes incident was ACTUALLY resolved.

INCIDENT:
- Alert: {scenario.alert_message}
- Root cause: {scenario.root_cause}
- Required fix: {fix_desc}
{f'- Expected fix commands:{chr(10)}{fix_steps}' if fix_steps else ''}

AGENT'S ACTIONS:
{history_summary}

CURRENT CLUSTER STATE (kubectl get pods -A):
{cluster_snapshot[:4000]}

QUESTION: Did the agent actually fix ALL the issues described above?
Look for:
- Are the specific broken deployments/pods now Running with correct images/config?
- Did the agent use the RIGHT namespace (not just any namespace)?
- If there were multiple faults, were ALL of them addressed?
- Is the fix command output showing success (not "Error: Not Found")?

Return JSON only: {{"resolved": true/false, "reason": "<1-2 sentence explanation>"}}"""

        try:
            result = self.llm.chat_json(
                "You are a strict Kubernetes incident verification system. Only confirm resolution if ALL faults were genuinely fixed.",
                user_prompt,
                temperature=0.1,
                max_tokens=256,
            )
            resolved = bool(result.get("resolved", False))
            reason = result.get("reason", "")
            logger.info(f"    Judge verification: resolved={resolved} | {reason}")
            return resolved, reason
        except Exception as e:
            logger.error(f"Judge verify error: {e}", exc_info=True)
            # On error, fall back to programmatic check (assume resolved)
            return True, f"Verification error: {type(e).__name__}, defaulting to resolved"

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
- Total steps taken: {len(history) + 1}

Return JSON only: {{"score": <float -1.0 to 1.0>, "feedback": "<1-2 sentence evaluation>"}}"""

        try:
            result = self.llm.chat_json(PERSONAS[persona], user_prompt, temperature=0.3, max_tokens=256)
            score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
            feedback = result.get("feedback", "")
            return score, feedback
        except Exception as e:
            logger.error(f"Judge LLM error: {e}", exc_info=True)
            return 0.0, f"Judge error: {type(e).__name__}"


# ---- SRE phase detection (heuristic, no LLM call) ----

_TRIAGE_PATTERNS = ("get pods", "get pod", "get po", "get events", "get ev", "top pods", "top nodes", "-A", "--all-namespaces")
_INVESTIGATE_PATTERNS = ("describe", "logs")
_MITIGATE_PATTERNS = ("scale", "rollout restart")


def _detect_phase(command: str, history: list) -> str:
    """Classify a command into an SRE workflow phase."""
    if command.startswith("fix:"):
        return "fix"
    if command.startswith("diagnose:"):
        return "investigation"

    cmd_lower = command.lower()

    # Post-fix verification: any read command after a fix was applied
    has_fix = any(h["command"].startswith("fix:") for h in history)
    if has_fix and any(p in cmd_lower for p in ("get pods", "get po", "get events", "logs")):
        return "verification"

    if any(p in cmd_lower for p in _MITIGATE_PATTERNS):
        return "mitigation"
    if any(p in cmd_lower for p in _INVESTIGATE_PATTERNS):
        return "investigation"
    if any(p in cmd_lower for p in _TRIAGE_PATTERNS):
        return "triage"

    return "triage"


# Phase order: lower number = earlier in correct SRE workflow
_PHASE_ORDER = {"triage": 0, "investigation": 1, "mitigation": 2, "fix": 3, "verification": 4}


class AdversarialJudge(LLMJudge):
    """Extends LLMJudge with phase-aware scoring for multi-step incidents.

    Rewards:
      - Following the correct SRE workflow order (triage -> investigate -> fix -> verify)
      - Identifying and dismissing red herrings
    Penalties:
      - Skipping phases (e.g. jumping straight to fix without investigation)
    """

    def evaluate(
        self,
        command: str,
        output: str,
        scenario,
        history: list,
        persona: str = "senior",
    ) -> tuple[float, str]:
        # Get base score from LLM
        base_score, feedback = super().evaluate(command, output, scenario, history, persona)

        current_phase = _detect_phase(command, history)

        # Bonus: correct phase ordering
        if self._is_phase_order_correct(current_phase, history):
            base_score += 0.2
        else:
            # Penalty: skipping phases
            skipped = self._get_skipped_phases(current_phase, history)
            if skipped:
                base_score -= 0.3
                feedback += f" Skipped {', '.join(skipped)} before {current_phase}."

        # Bonus: red herring awareness (investigating something misleading
        # but not trying to fix it)
        if isinstance(scenario, AdversarialScenarioSpec) and scenario.red_herrings:
            if self._touches_red_herring(command, output, scenario) and not command.startswith("fix:"):
                base_score += 0.15
                feedback += " Good investigation of a misleading symptom."

        score = max(-1.0, min(1.0, base_score))
        return score, feedback

    def _is_phase_order_correct(self, current_phase: str, history: list) -> bool:
        """Check if the current phase follows the expected SRE order."""
        if not history:
            return current_phase == "triage"

        current_order = _PHASE_ORDER.get(current_phase, 0)
        past_phases = [_detect_phase(h["command"], history[:i]) for i, h in enumerate(history)]
        if not past_phases:
            return True

        max_past_order = max(_PHASE_ORDER.get(p, 0) for p in past_phases)
        # Allow same phase, going back, or advancing by at most 1 phase
        return current_order <= max_past_order + 1

    def _get_skipped_phases(self, current_phase: str, history: list) -> list[str]:
        """Return list of phases that were skipped."""
        current_order = _PHASE_ORDER.get(current_phase, 0)
        if current_order <= 1:
            return []

        past_phases = {_detect_phase(h["command"], history[:i]) for i, h in enumerate(history)}
        past_phases.add(current_phase)

        skipped = []
        for phase, order in _PHASE_ORDER.items():
            if order < current_order and phase not in past_phases:
                skipped.append(phase)
        return skipped

    # K8s-specific terms that indicate a red herring symptom in command output
    _RED_HERRING_TERMS = {
        "oomkilled", "oomkill", "exit code 137", "memory leak",
        "imagepullbackoff", "errimagepull", "crashloopbackoff",
        "connection refused", "host not found", "502", "503", "500",
        "timeout", "retry storm", "port conflict",
    }

    def _touches_red_herring(self, command: str, output: str, scenario: AdversarialScenarioSpec) -> bool:
        """Check if the command output relates to a known red herring.

        Uses two-tier matching: first checks for K8s-specific symptom terms,
        then requires at least 2 meaningful words from the herring description
        to match in the output, avoiding false positives from common words.
        """
        output_lower = output.lower()
        for herring in scenario.red_herrings:
            herring_lower = herring.lower()
            # Tier 1: check for known K8s symptom terms in both herring and output
            for term in self._RED_HERRING_TERMS:
                if term in herring_lower and term in output_lower:
                    return True
            # Tier 2: require at least 2 meaningful (6+ char) words to match
            herring_keywords = [w for w in herring_lower.split() if len(w) >= 6]
            matches = sum(1 for kw in herring_keywords if kw in output_lower)
            if matches >= 2:
                return True
        return False
