from collections import defaultdict


class CurriculumController:
    """
    Tracks agent skill across failure types.
    Drives: scenario difficulty, judge persona, weak-spot targeting.
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.episode_rewards = []
        self.episode_count = 0

    def record(self, failure_type: str, success: bool, steps: int, reward: float):
        self.history[failure_type].append(success)
        self.episode_rewards.append(reward)
        self.episode_count += 1

    def get_skill_profile(self) -> dict:
        """Success rate per failure type over last 10 episodes."""
        return {
            ft: round(sum(results[-10:]) / len(results[-10:]), 2)
            for ft, results in self.history.items()
            if results
        }

    def get_difficulty(self) -> float:
        """Continuous difficulty 0.0-1.0. Starts easy, grows with agent success rate."""
        all_results = [r for results in self.history.values() for r in results[-10:]]
        if len(all_results) < 3:
            return 0.2
        rate = sum(all_results) / len(all_results)
        return min(0.95, rate + 0.15)

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
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "skill_profile": self.get_skill_profile(),
        }
