from collections import defaultdict


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
    Drives: scenario difficulty, judge persona, weak-spot targeting.

    Progression: warmup (single easy faults) -> beginner -> intermediate ->
    advanced (compound faults) -> expert. Agent must sustain a success rate
    over multiple episodes before advancing to the next tier.
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.episode_rewards = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0  # episodes spent in current tier

    def record(self, failure_type: str, success: bool, steps: int, reward: float):
        self.history[failure_type].append(success)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self._tier_episodes += 1
        self._maybe_advance_tier()

    def _maybe_advance_tier(self):
        """Advance to the next difficulty tier if the agent is ready."""
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return  # already at max tier
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self._tier_episodes < tier["min_episodes"]:
            return  # not enough episodes in this tier yet
        recent_rate = self._recent_success_rate()
        if recent_rate >= tier["advance_rate"]:
            self._tier_index += 1
            self._tier_episodes = 0

    def _recent_success_rate(self, window: int = 10) -> float:
        """Success rate over the last `window` episodes across all failure types."""
        all_results = [r for results in self.history.values() for r in results[-window:]]
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def get_skill_profile(self) -> dict:
        """Success rate per failure type over last 10 episodes."""
        return {
            ft: round(sum(results[-10:]) / len(results[-10:]), 2)
            for ft, results in self.history.items()
            if results
        }

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
        """junior → lenient, senior → standard, principal → strict."""
        d = self.get_difficulty()
        if d < 0.4:
            return "junior"
        elif d < 0.7:
            return "senior"
        return "principal"

    def get_stats(self) -> dict:
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "tier_episodes": self._tier_episodes,
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "skill_profile": self.get_skill_profile(),
        }
