"""
Kube SRE Gym Environment Implementation.

Agent diagnoses and fixes real GKE incidents with curriculum-driven difficulty.

Modes (set via GYM_MODE env var):
  standard     — single-fault scenarios from pool or LLM generator (default)
  adversarial  — multi-step incidents designed by external LLM judge
"""

import json
import os
import logging
import time
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from .constants import MAX_STEPS, INJECT_VISIBILITY_MAX_POLLS, INJECT_VISIBILITY_INTERVAL

try:
    from ..models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState
except ImportError:
    from models import KubeSreGymAction, KubeSreGymObservation, KubeSreGymState

from .llm_client import LLMClient
from .k8s_backend import K8sBackend
from .scenario_generator import ScenarioGenerator
from .curriculum import CurriculumController
from .judge import LLMJudge, AdversarialJudge
from .adversarial_designer import AdversarialDesigner

logger = logging.getLogger(__name__)


class KubeSreGymEnvironment(Environment):
    """
    K8s SRE OpenEnv Environment — agent diagnoses and fixes real GKE incidents.

    Config via env vars:
      GYM_MODE       - "standard" (default) or "adversarial"
      LLM_BACKEND    - "openai" (default), "hf", or "anthropic"
      LLM_MODEL      - model name
      HF_TOKEN       - HuggingFace token
      ANTHROPIC_API_KEY - Anthropic API key (for adversarial mode)
      K8S_ENDPOINT   - GKE API endpoint
      K8S_TOKEN      - Bearer token for GKE
      K8S_CA_CERT    - Base64 CA cert
      GENERATOR_MODE - "simple" (default) or "llm"
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        try:
            logger.info("Initializing KubeSreGymEnvironment...")
            llm = LLMClient()
            logger.info("LLMClient initialized")
            self.backend = K8sBackend()
            logger.info("K8sBackend initialized")
            self.curriculum = CurriculumController()
            self.mode = os.environ.get("GYM_MODE", "standard")

            self.scenario = None
            self._step_count = 0
            self._base_max_steps = int(os.environ.get("MAX_STEPS", str(MAX_STEPS)))
            self.max_steps = self._base_max_steps
            self.history = []
            self._state = KubeSreGymState(episode_id=str(uuid4()), step_count=0)

            # Always initialize both paths — curriculum may switch mid-training
            self.designer = AdversarialDesigner(llm, self.backend, max_steps=self.max_steps)
            self.generator = ScenarioGenerator(llm, mode=os.environ.get("GENERATOR_MODE", "simple"))

            if self.mode == "adversarial":
                self.judge = AdversarialJudge(llm)
                logger.info("GYM_MODE=adversarial — LLM designs multi-step incidents")
            else:
                self.judge = LLMJudge(llm)
                logger.info("GYM_MODE=standard — curriculum-driven single-fault injection")
            logger.info("KubeSreGymEnvironment initialized successfully")
        except Exception as e:
            logger.error(f"FATAL: KubeSreGymEnvironment.__init__ failed: {e}", exc_info=True)
            raise

    def reset(self) -> KubeSreGymObservation:
        logger.info("reset() called — resetting cluster...")
        try:
            return self._do_reset()
        except Exception as e:
            logger.error(f"FATAL: reset() failed: {e}", exc_info=True)
            raise

    def _do_reset(self) -> KubeSreGymObservation:
        # Step 1: Deploy clean healthy cluster (base manifests only)
        self.backend.reset()
        logger.info("Cluster reset complete")

        skill_profile = self.curriculum.get_skill_profile()
        difficulty = self.curriculum.get_difficulty()

        # Step 2: Curriculum decides what fault to inject
        use_adversarial = (self.mode == "adversarial"
                           or self.curriculum.should_use_adversarial())

        if use_adversarial:
            # Adversarial: LLM designs a targeted compound incident
            logger.info(f"Episode {self.curriculum.episode_count + 1}: "
                        f"adversarial mode (difficulty={difficulty:.2f}, "
                        f"weak_spots={self.curriculum.get_weak_spots()})")
            self.scenario = self.designer.design(skill_profile, difficulty)
            self.designer.inject(self.scenario)
            if getattr(self.scenario, '_inject_success_count', 0) == 0:
                logger.warning("LLM scenario injection failed — using fallback")
                self.scenario = self.designer._fallback_scenario(difficulty)
                self.designer.inject(self.scenario)
            # Switch to stricter judge for adversarial
            if not isinstance(self.judge, AdversarialJudge):
                self.judge = AdversarialJudge(LLMClient())
        else:
            # Standard: curriculum picks ONE fault type, inject into clean cluster
            fault_type = self.curriculum.pick_fault_type()
            logger.info(f"Episode {self.curriculum.episode_count + 1}: "
                        f"standard mode, fault='{fault_type}' "
                        f"(difficulty={difficulty:.2f}, "
                        f"graduated={self.curriculum.get_graduated()})")
            self.scenario = self.generator.generate(
                skill_profile, difficulty, fault_type_hint=fault_type
            )
            self.backend.inject_failure(self.scenario.failure_type, self.scenario.params)

        # Step 3: Wait for fault to actually manifest before snapshotting
        self._wait_for_fault_visible()

        # Scale max steps with difficulty — harder scenarios need more investigation
        # Easy (0.15): 15 steps, Hard (0.80): 20 steps, Expert (0.95): 25 steps
        self.max_steps = int(self._base_max_steps + 10 * difficulty)

        self._step_count = 0
        self.history = []
        self._state = KubeSreGymState(
            episode_id=str(uuid4()),
            step_count=0,
            difficulty=difficulty,
            incident_type=self.scenario.failure_type,
            root_cause=self.scenario.root_cause,
            correct_fix=self.scenario.correct_fix_description,
            judge_persona=self.curriculum.get_judge_persona(),
            curriculum_stats=self.curriculum.get_stats(),
        )

        persona = self.curriculum.get_judge_persona()

        # Initial observation: richer snapshot so agent can decide where to dig
        pods_output = self.backend.execute("kubectl get pods --all-namespaces")
        events_output = self.backend.execute("kubectl get events --all-namespaces")
        cluster_summary = f"=== POD STATUS ===\n{pods_output}\n\n=== RECENT EVENTS ===\n{events_output}"

        return KubeSreGymObservation(
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
            done=False,
            reward=0.0,
        )

    def _wait_for_fault_visible(self):
        """Poll until at least one pod in the affected namespace shows unhealthy status."""
        fault_type = self.scenario.failure_type if self.scenario else ""
        affected_ns = self.scenario.namespace if self.scenario else None

        # resource_quota doesn't crash existing pods, just blocks new ones
        # scale_zero removes all pods — check for 0 pods instead of unhealthy
        if fault_type in ("resource_quota", "scale_zero", "adversarial"):
            time.sleep(3)
            return

        for i in range(INJECT_VISIBILITY_MAX_POLLS):
            health = self.backend.check_health()
            # Only check the affected namespace to avoid false positives
            # from stale broken pods in other namespaces
            if affected_ns and affected_ns in health:
                check_health = {affected_ns: health[affected_ns]}
            else:
                check_health = health
            unhealthy = [
                f"{ns}/{pod}"
                for ns, pods in check_health.items()
                for pod, status in pods.items()
                if status not in ("Running", "Completed")
            ]
            if unhealthy:
                logger.info(f"Fault visible after {i + 1} polls: {unhealthy[:3]}")
                return
            time.sleep(INJECT_VISIBILITY_INTERVAL)

        logger.warning(
            f"Fault '{fault_type}' not visible after {INJECT_VISIBILITY_MAX_POLLS} polls "
            f"({INJECT_VISIBILITY_MAX_POLLS * INJECT_VISIBILITY_INTERVAL}s) — proceeding anyway"
        )

    def step(self, action: KubeSreGymAction) -> KubeSreGymObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        logger.info(f"  Step {self._step_count}/{self.max_steps}: {action.command}")

        # Strip "fix:" or "diagnose:" prefix before executing
        raw_cmd = action.command.strip()
        is_fix = raw_cmd.lower().startswith("fix:")
        is_diagnose = raw_cmd.lower().startswith("diagnose:")

        if is_fix:
            exec_cmd = raw_cmd[4:].strip()
        elif is_diagnose:
            exec_cmd = raw_cmd[9:].strip()
        else:
            exec_cmd = raw_cmd

        # Detect mutation commands as implicit fixes even without "fix:" prefix
        # This ensures the post-fix health check runs when the agent applies a fix
        _MUTATION_PATTERNS = ("kubectl set ", "kubectl patch ", "kubectl scale ",
                              "kubectl rollout restart")
        if not is_fix and any(exec_cmd.startswith(p) for p in _MUTATION_PATTERNS):
            is_fix = True

        # Execute the kubectl command (if any remains after stripping prefix)
        if exec_cmd and exec_cmd.startswith("kubectl"):
            output = self.backend.execute(exec_cmd)
        elif is_diagnose:
            # "diagnose: <root cause>" — no command to execute, just the diagnosis text
            output = f"Diagnosis submitted: {exec_cmd}"
        elif is_fix and not exec_cmd.startswith("kubectl"):
            output = f"Fix submitted but no kubectl command found. Use: fix: kubectl <command>"
        else:
            output = self.backend.execute(exec_cmd)

        # Penalize repeated commands (same command run before)
        repeat_count = sum(1 for h in self.history if h["command"] == action.command)

        persona = self.curriculum.get_judge_persona()
        reward, feedback = self.judge.evaluate(
            action.command, output, self.scenario, self.history, persona
        )

        logger.info(f"    -> reward={reward:.2f} | {feedback[:80]}")
        if output:
            logger.info(f"    -> output: {output[:120].replace(chr(10), ' ')}")

        if repeat_count > 0:
            penalty = min(0.5, repeat_count * 0.15)
            reward -= penalty
            feedback += f" Repeated command ({repeat_count + 1}x)."

        done = False

        if is_fix:
            # Allow rollout to progress before judging health (set/rollout/patch are async)
            # ContainerCreating/PodInitializing = fix is working, pod still starting
            # Only check the affected namespace — other namespaces may have stale pods
            STARTING_STATES = ("ContainerCreating", "PodInitializing")
            affected_ns = self.scenario.namespace if self.scenario else None
            for poll in range(12):
                time.sleep(5)
                health = self.backend.check_health()
                if affected_ns and affected_ns in health:
                    check_health = {affected_ns: health[affected_ns]}
                else:
                    check_health = health
                statuses = [s for ns_pods in check_health.values() for s in ns_pods.values()]
                total_count = len(statuses)
                healthy_count = sum(1 for s in statuses if s in ("Running", "Completed"))
                starting_count = sum(1 for s in statuses if s in STARTING_STATES)
                all_healthy = healthy_count == total_count and total_count > 0
                if all_healthy:
                    break
                # If pods are still starting, keep waiting (up to 60s)
                if starting_count == 0 and poll >= 3:
                    # No pods starting and none healthy after 15s — fix didn't work
                    break

            if all_healthy:
                done = True
                # Efficiency bonus scaled by difficulty:
                # - Easy (0.15): max 3.0, rewards fast solves
                # - Hard (0.80): max 5.0, rewards solving at all
                difficulty = self.curriculum.get_difficulty()
                base_bonus = 1.0 + difficulty * 2.0  # 1.0 at easy, 3.0 at hard
                efficiency = base_bonus + 2.0 * (1.0 - self._step_count / self.max_steps)
                reward += efficiency
                feedback = f"Incident resolved! All pods healthy. (bonus: +{efficiency:.1f})"
            else:
                # Partial progress feedback — tell agent how many pods are healthy
                feedback += f" Fix applied. {healthy_count}/{total_count} pods healthy."

        if self._step_count >= self.max_steps:
            done = True
            reward -= 1.0  # significant timeout penalty
            feedback = "Timeout -- incident remains unresolved."

        self.history.append({
            "step": self._step_count,
            "command": action.command,
            "output": output[:200],
            "reward": reward,
            "feedback": feedback,
        })

        if done:
            # Track adversarial scenarios by name for curriculum granularity
            track_type = self.scenario.failure_type
            if hasattr(self.scenario, "name") and self.scenario.name:
                track_type = f"adversarial:{self.scenario.name}"

            # Normalize total reward by number of steps to avoid rewarding
            # long episodes over efficient ones
            raw_sum = sum(h["reward"] for h in self.history)
            total_reward = raw_sum / self._step_count if self._step_count > 0 else 0.0
            resolved = "Incident resolved!" in feedback
            self.curriculum.record(
                failure_type=track_type,
                success=resolved,
                steps=self._step_count,
                reward=total_reward,
            )
            self._state.is_resolved = resolved
            self._state.cumulative_reward = total_reward

            logger.info(f"  === EPISODE DONE: {'RESOLVED' if resolved else 'FAILED'} | "
                        f"fault={track_type} | steps={self._step_count} | "
                        f"total_reward={total_reward:.2f} | "
                        f"tier={self.curriculum.get_tier_name()} | "
                        f"difficulty={self.curriculum.get_difficulty():.2f} ===")

            # Save episode transcript to JSONL
            try:
                transcript = {
                    "episode": self.curriculum.episode_count,
                    "fault_type": track_type,
                    "resolved": resolved,
                    "steps": self._step_count,
                    "total_reward": total_reward,
                    "difficulty": self.curriculum.get_difficulty(),
                    "tier": self.curriculum.get_tier_name(),
                    "alert": self.scenario.alert_message,
                    "root_cause": self.scenario.root_cause,
                    "correct_fix": self.scenario.correct_fix_description,
                    "initial_observation": {
                        "cluster_status": self._state.curriculum_stats,
                        "incident_type": self._state.incident_type,
                    },
                    "history": self.history,
                }
                log_path = os.environ.get("EPISODE_LOG", "episode_transcripts.jsonl")
                with open(log_path, "a") as f:
                    f.write(json.dumps(transcript) + "\n")
            except Exception as e:
                logger.warning(f"Failed to save episode transcript: {e}")

        # Only auto-fetch cluster summary after fix attempts or on done
        # Otherwise the agent should run its own diagnostic commands
        if is_fix or done:
            cluster_summary = self.backend.execute("kubectl get pods --all-namespaces")
        else:
            cluster_summary = ""

        return KubeSreGymObservation(
            command_output=output,
            cluster_status_summary=cluster_summary,
            active_alerts=[self.scenario.alert_message] if not done else [],
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            hint=feedback if persona != "principal" else "",
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> KubeSreGymState:
        return self._state
