"""
Curriculum Controller — drives progressive difficulty across training.

Training flow:
  1. Agent starts with easy single-fault scenarios (oom_kill, crashloop, image_pull)
  2. As it masters each fault type (>= MASTERY_THRESHOLD over MASTERY_WINDOW),
     that type is "graduated" and appears less frequently
  3. Harder fault types unlock as overall difficulty rises
  4. Judge persona scales: junior (lenient) → senior → principal (strict)
  5. In adversarial mode, the curriculum feeds weak spots to the LLM designer
     so it generates targeted, progressively harder incidents

Key design choice: clean state + ONE injected fault per episode.
  - Gives clean reward signal for GRPO (action → outcome → reward attribution)
  - Avoids noisy multi-fault starts that dilute learning signal
"""

from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# ---- Fault type difficulty tiers ----
FAULT_TIERS = {
    # Tier 1 (easy, difficulty 0.0-0.3): obvious symptoms, single step fix
    "oom_kill":        {"tier": 1, "min_difficulty": 0.0},
    "crashloop":       {"tier": 1, "min_difficulty": 0.0},
    "image_pull":      {"tier": 1, "min_difficulty": 0.0},
    # Tier 2 (medium, difficulty 0.3-0.6): requires investigation
    "bad_config":       {"tier": 2, "min_difficulty": 0.3},
    "liveness_probe":   {"tier": 2, "min_difficulty": 0.3},
    "scale_zero":       {"tier": 2, "min_difficulty": 0.3},
    # Tier 3 (hard, difficulty 0.6-0.8): multi-step diagnosis
    "resource_quota":  {"tier": 3, "min_difficulty": 0.6},
    "cascading_db":    {"tier": 3, "min_difficulty": 0.6},
    # Tier 4 (adversarial, difficulty 0.8+): LLM-designed compound faults
    "adversarial":     {"tier": 4, "min_difficulty": 0.8},
}

MASTERY_THRESHOLD = 0.7   # 70% success rate = mastered
MASTERY_WINDOW = 10       # look at last N episodes per fault type
MIN_EPISODES_FOR_MASTERY = 3  # need at least N attempts before graduating


# Difficulty tiers — agent must earn its way through each tier
# Each tier defines: max difficulty, min episodes to stay, min success rate to advance
DIFFICULTY_TIERS = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "beginner",     "max_diff": 0.40, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 8,  "advance_rate": 0.65},
    {"name": "advanced",     "max_diff": 0.80, "min_episodes": 10, "advance_rate": 0.7},
    {"name": "expert",       "max_diff": 0.95, "min_episodes": 0,  "advance_rate": 1.0},
]


class CurriculumController:
    """
    Tracks agent skill across failure types.
    Drives: scenario difficulty, fault type selection, judge persona, weak-spot targeting.

    Progression: warmup (single easy faults) -> beginner -> intermediate ->
    advanced (compound faults) -> expert. Agent must sustain a success rate
    over multiple episodes before advancing to the next tier.
    """

    def __init__(self):
        self.history = defaultdict(list)       # fault_type → [bool, bool, ...]
        self.step_counts = defaultdict(list)    # fault_type → [int, int, ...]
        self.episode_rewards = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0  # episodes spent in current tier
        self._graduated = set()                # fault types the agent has mastered

    def record(self, failure_type: str, success: bool, steps: int, reward: float):
        """Record episode outcome and check for mastery graduation."""
        self.history[failure_type].append(success)
        self.step_counts[failure_type].append(steps)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self._tier_episodes += 1
        self._maybe_advance_tier()

        # Check if this fault type is now mastered
        recent = self.history[failure_type][-MASTERY_WINDOW:]
        if (len(recent) >= MIN_EPISODES_FOR_MASTERY
                and sum(recent) / len(recent) >= MASTERY_THRESHOLD):
            if failure_type not in self._graduated:
                self._graduated.add(failure_type)
                logger.info(
                    f"Curriculum: agent MASTERED '{failure_type}' "
                    f"({sum(recent)}/{len(recent)} success rate) — graduating"
                )

    def _maybe_advance_tier(self):
        """Advance to the next difficulty tier if the agent is ready.

        Fast-track: if success rate >= 90% after at least 3 episodes, skip
        the min_episodes requirement. Agents that ace easy tiers shouldn't
        be held back.
        """
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return  # already at max tier
        tier = DIFFICULTY_TIERS[self._tier_index]
        recent_rate = self._recent_success_rate()

        # Fast-track: 90%+ success after 3 episodes → advance immediately
        fast_track = (self._tier_episodes >= 3 and recent_rate >= 0.9)

        if not fast_track and self._tier_episodes < tier["min_episodes"]:
            return  # not enough episodes in this tier yet

        if recent_rate >= tier["advance_rate"]:
            logger.info(f"Curriculum: advancing from {tier['name']} "
                        f"(rate={recent_rate:.0%}, episodes={self._tier_episodes}"
                        f"{', FAST-TRACK' if fast_track else ''})")
            self._tier_index += 1
            self._tier_episodes = 0

    def _recent_success_rate(self, window: int = 10) -> float:
        """Success rate over the last `window` episodes across all failure types."""
        all_results = [r for results in self.history.values() for r in results[-window:]]
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def get_skill_profile(self) -> dict:
        """Success rate per failure type over last MASTERY_WINDOW episodes."""
        return {
            ft: round(sum(results[-MASTERY_WINDOW:]) / len(results[-MASTERY_WINDOW:]), 2)
            for ft, results in self.history.items()
            if results
        }

    def get_weak_spots(self) -> list[str]:
        """Fault types where agent success rate is below mastery threshold."""
        profile = self.get_skill_profile()
        return [ft for ft, rate in profile.items() if rate < MASTERY_THRESHOLD]

    def get_graduated(self) -> set[str]:
        """Fault types the agent has mastered."""
        return set(self._graduated)

    def get_unlocked_fault_types(self) -> list[str]:
        """Fault types available at current difficulty level."""
        difficulty = self.get_difficulty()
        return [
            ft for ft, meta in FAULT_TIERS.items()
            if meta["min_difficulty"] <= difficulty
        ]

    def get_difficulty(self) -> float:
        """Continuous difficulty within the current tier.

        Within each tier, difficulty scales with success rate but is capped
        by the tier's max. This prevents sudden jumps.
        """
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self.episode_count < 3:
            return 0.15  # first few episodes are always easy
        rate = self._recent_success_rate()
        # Scale within the tier: low success = tier floor, high success = tier ceiling
        if self._tier_index == 0:
            tier_floor = 0.1
        else:
            tier_floor = DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"]
        return min(tier["max_diff"], tier_floor + rate * (tier["max_diff"] - tier_floor))

    def get_tier_name(self) -> str:
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    def get_judge_persona(self) -> str:
        """
        Judge strictness scales with difficulty:
          junior   (< 0.4) — lenient scoring, gives hints
          senior   (0.4-0.7) — standard scoring
          principal (> 0.7) — strict, penalizes inefficiency
        """
        d = self.get_difficulty()
        if d < 0.4:
            return "junior"
        elif d < 0.7:
            return "senior"
        return "principal"

    def should_use_adversarial(self) -> bool:
        """Switch to adversarial LLM-designed scenarios when agent is ready."""
        return (self.get_difficulty() >= 0.8
                and len(self._graduated) >= 3)

    def pick_fault_type(self) -> str | None:
        """
        Pick next fault type using curriculum logic.

        Priority:
          1. Weak spots (fault types below mastery) — target weaknesses
          2. Unlocked but untried fault types — encourage exploration
          3. Random from unlocked pool (excluding over-graduated) — maintain breadth

        Returns None if adversarial mode should take over.
        """
        import random

        if self.should_use_adversarial():
            return None  # signal: let adversarial designer handle it

        unlocked = self.get_unlocked_fault_types()
        weak_spots = self.get_weak_spots()

        # Count how many times each fault type has been used
        ft_counts = {ft: len(results) for ft, results in self.history.items()}

        # Max repeats: simple (tier 1) = 2x, medium+ = 3x before all unlocked are tried
        tried = set(self.history.keys())
        untried = [ft for ft in unlocked if ft not in tried and ft != "adversarial"]

        def _over_limit(ft):
            tier = FAULT_TIERS.get(ft, {}).get("tier", 1)
            limit = 2 if tier == 1 else 3
            return ft_counts.get(ft, 0) >= limit and untried

        # Priority 1: untried fault types that are now unlocked — explore first
        if untried:
            return random.choice(untried)

        # Priority 2: target weak spots that are unlocked and not over-repeated
        weak_and_unlocked = [ft for ft in weak_spots
                             if ft in unlocked and ft != "adversarial"
                             and not _over_limit(ft)]
        if weak_and_unlocked:
            return random.choice(weak_and_unlocked)

        # Priority 3: sample from unlocked, weighting non-graduated higher
        candidates = [ft for ft in unlocked
                      if ft != "adversarial" and not _over_limit(ft)]
        if not candidates:
            # All over limit — allow any unlocked
            candidates = [ft for ft in unlocked if ft != "adversarial"]
        if not candidates:
            return "oom_kill"  # fallback

        # Graduated types get 1x weight, non-graduated get 3x
        weights = [1 if ft in self._graduated else 3 for ft in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def get_stats(self) -> dict:
        """Full curriculum state for logging/debugging."""
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "tier_episodes": self._tier_episodes,
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "skill_profile": self.get_skill_profile(),
            "graduated": sorted(self._graduated),
            "weak_spots": self.get_weak_spots(),
            "unlocked_faults": self.get_unlocked_fault_types(),
            "use_adversarial": self.should_use_adversarial(),
            "avg_reward_last_10": round(
                sum(self.episode_rewards[-10:]) / max(1, len(self.episode_rewards[-10:])), 3
            ),
        }
