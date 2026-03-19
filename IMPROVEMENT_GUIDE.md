# Kube SRE Gym — Deep Research & Improvement Guide

**Goal**: Elevate this hackathon project to HuggingFace blog-post quality with rigorous validation, better reward shaping, curriculum design, and evaluation methodology.

---

## Table of Contents

1. [Reward Function Improvements](#1-reward-function-improvements)
2. [Evaluation & Validation Methodology](#2-evaluation--validation-methodology)
3. [Curriculum Difficulty Improvements](#3-curriculum-difficulty-improvements)
4. [Hyperparameter Recommendations](#4-hyperparameter-recommendations)
5. [GRPO / DAPO Algorithm Improvements](#5-grpo--dapo-algorithm-improvements)
6. [Adversarial Design Improvements](#6-adversarial-design-improvements)
7. [Judge System Improvements](#7-judge-system-improvements)
8. [Blog Post Structure & Rigor](#8-blog-post-structure--rigor)
9. [Code-Level Bug Fixes & Gaps](#9-code-level-bug-fixes--gaps)
10. [Priority-Ranked Action Items](#10-priority-ranked-action-items)

---

## 1. Reward Function Improvements

### Current State

Your reward has 6 components: per-step LLM score (-1 to +1), repeat penalty (-0.15 to -0.5), resolution bonus (+1 to +5), timeout penalty (-2.0), judge verification, and phase-order bonus (+0.2 / -0.3). Total episode range is roughly -2 to +8.

### Issues Found

**A. Resolution bonus doesn't scale enough with difficulty.**

```python
# Current (line ~376 of environment.py):
base_bonus = 1.0 + difficulty * 2.0   # Easy: 1.6, Hard: 2.6
efficiency = base_bonus + 2.0 * (1 - steps_taken / max_steps)
```

At difficulty 0.3 the total bonus is ~1.6–3.6. At difficulty 0.8 it's ~2.6–4.6. The difference is only ~1.0, which means GRPO doesn't strongly differentiate between solving easy vs hard problems.

**Fix**: Use quadratic efficiency scaling and widen the difficulty multiplier:

```python
base_bonus = 1.0 + difficulty * 3.0   # Easy: 1.9, Hard: 3.4
efficiency_ratio = 1.0 - (steps_taken / max_steps)
efficiency_bonus = 3.0 * efficiency_ratio ** 2   # Quadratic: fast solves rewarded exponentially
resolution_bonus = base_bonus + efficiency_bonus
```

This gives a range of ~1.9–4.9 for easy and ~3.4–6.4 for hard, with a steep curve for fast resolution.

**B. Repeat command penalty is too uniform.**

Currently, `kubectl get pods -A` (a diagnostic read-only command) gets the same repeat penalty as a mutation command. Diagnostic commands are legitimately repeated (to check cluster state after a fix).

**Fix**: Split penalties by command type:

```python
DIAGNOSTIC_CMDS = {"get", "describe", "logs", "top", "rollout status"}
MUTATION_CMDS = {"set", "patch", "scale", "delete", "rollout restart"}

# Diagnostic: lighter penalty, no circuit breaker
if is_diagnostic:
    penalty = -0.1 * (repeat_count - 1)  # 2nd: -0.1, 3rd: -0.2, never blocked
# Mutation: strict penalty + circuit breaker
else:
    if repeat_count == 2: penalty = -0.3
    elif repeat_count >= 3: penalty = -0.5; blocked = True
```

**C. No partial credit for multi-fault scenarios.**

In adversarial mode with 2-3 faults, the agent gets the full resolution bonus only when ALL faults are fixed. If it fixes 2 of 3, it gets zero resolution credit. This is too sparse for GRPO to learn from.

**Fix**: Track per-fault resolution and award partial bonuses:

```python
faults_fixed = count_resolved_faults(scenario, cluster_state)
total_faults = len(scenario.fix_steps)
partial_ratio = faults_fixed / total_faults

if partial_ratio == 1.0:
    resolution_bonus = full_bonus  # existing calculation
elif partial_ratio > 0:
    resolution_bonus = full_bonus * partial_ratio * 0.5  # 50% of proportional credit
    # Intentionally less than proportional to still incentivize full resolution
```

**D. Timeout penalty is a hard cliff.**

The current `-2.0` wipe is binary — if the agent times out at step 15 having fixed 2 of 3 faults, it gets the same penalty as an agent that did nothing. This destroys useful gradient signal.

**Fix**: Graduated timeout penalty:

```python
if timed_out:
    progress_ratio = max(0, cumulative_reward) / expected_full_reward
    timeout_penalty = -2.0 * (1.0 - progress_ratio * 0.5)
    # Made progress: -1.0 to -2.0
    # No progress: -2.0 (same as before)
```

**E. Consider adding a format/syntax reward.**

Currently, if the model outputs valid kubectl syntax but the wrong command, it gets the same penalty as outputting gibberish. A small format reward helps the model learn syntax faster in early training.

```python
def format_reward(command_text: str) -> float:
    """Small reward for syntactically valid kubectl commands."""
    if command_text.startswith("kubectl ") and len(command_text.split()) >= 3:
        return 0.05  # Tiny nudge toward valid syntax
    return 0.0
```

### Research Context

Latest research (2025) strongly recommends **hybrid dense+sparse rewards**. Your per-step LLM score provides dense signal, and the resolution bonus provides sparse signal — this is architecturally correct. The improvements above refine the balance. Dense rewards improve credit assignment; sparse rewards define the true objective.

---

## 2. Evaluation & Validation Methodology

### Current State

`eval.py` runs 5 episodes per model (base vs trained), reports resolution rate, average reward, and steps-to-fix. No statistical testing, no held-out scenarios, no confidence intervals.

### What Top HuggingFace Blog Posts Include

Based on analysis of Open-R1, DeepSeek-R1, Mini-R1, and TRL GRPO tutorials, respected blog posts include:

1. Training curves with rolling averages
2. Comparison against 2-3 baselines with confidence intervals
3. Ablation studies (remove one component, measure impact)
4. Qualitative examples (successful traces + failure analysis)
5. Computational cost reporting
6. Reproducibility details (seeds, versions, hardware)

### Required Improvements

**A. Increase eval episodes to 30-50 minimum (100 ideal).**

5 episodes is far too few for statistical significance. With high-variance RL episodes, you need at least 30 to get meaningful confidence intervals.

```python
parser.add_argument("--num-eval-episodes", type=int, default=50)
```

**B. Add confidence intervals and statistical tests.**

```python
import numpy as np
from scipy import stats

def compute_ci(values, confidence=0.95):
    """Bootstrap 95% confidence interval."""
    n_bootstrap = 10000
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return lower, upper

def compare_models(base_rewards, trained_rewards):
    """Statistical comparison with effect size."""
    t_stat, p_value = stats.ttest_ind(trained_rewards, base_rewards)
    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(base_rewards)**2 + np.std(trained_rewards)**2) / 2)
    cohens_d = (np.mean(trained_rewards) - np.mean(base_rewards)) / pooled_std
    return {"t_stat": t_stat, "p_value": p_value, "cohens_d": cohens_d}
```

**C. Create held-out validation scenarios.**

This is the most critical missing piece. Currently, training and eval use the same scenario distribution — you can't distinguish generalization from memorization.

**Implementation**: Split your scenario space:

```python
# In scenario_generator.py or adversarial_designer.py
TRAIN_DEPLOYMENTS = {"payment-api", "payment-gateway", "web-app"}
EVAL_DEPLOYMENTS = {"payment-worker", "frontend-cache", "auth-service"}

# Or split by fault combinations:
TRAIN_COMBOS = [("oom_kill", "payments"), ("image_pull", "frontend"), ...]
EVAL_COMBOS = [("oom_kill", "auth"), ("crashloop", "payments"), ...]  # unseen combos
```

**D. Add out-of-distribution (OOD) evaluation.**

Test on scenarios the model has never seen during training:
- New fault types (e.g., disk pressure, network partition simulation)
- New namespace names
- Different numbers of deployments
- Compound faults with 3+ simultaneous issues

**E. Add ablation studies.**

For the blog, run eval with these ablations:
1. **No curriculum** (random difficulty) — shows curriculum value
2. **No adversarial mode** (standard scenarios only) — shows adversarial value
3. **No repeat penalty** — shows penalty value
4. **No phase-order bonus** — shows workflow guidance value
5. **Base model zero-shot** — shows that learning happened

**F. Report per-fault-type performance.**

```python
# In eval results:
{
    "by_fault_type": {
        "oom_kill": {"resolve_rate": 0.85, "avg_reward": 4.2, "avg_steps": 5.1},
        "crashloop": {"resolve_rate": 0.60, "avg_reward": 2.1, "avg_steps": 8.3},
        "multi_fault": {"resolve_rate": 0.40, "avg_reward": 1.5, "avg_steps": 12.7}
    }
}
```

**G. Run multiple seeds.**

Train with 3 different random seeds and report mean ± std across seeds. This is table stakes for a respected HuggingFace blog.

---

## 3. Curriculum Difficulty Improvements

### Current Issues

1. **Sudden tier transitions**: Jumping from warmup to beginner happens all-at-once when success rate crosses threshold, causing difficulty spikes
2. **No demotion mechanism**: If agent fails 5 multi-fault episodes in a row, curriculum doesn't step back
3. **Success rate is global, not per-tier**: One good easy episode affects advancement to hard tier
4. **Per-fault mastery doesn't track combinations**: Mastering OOM and image_pull individually ≠ mastering them together

### Recommended Fixes

**A. Add smooth difficulty transitions (soft tier boundaries).**

Instead of hard tier boundaries, blend difficulty:

```python
def compute_difficulty(self) -> float:
    """Smooth difficulty that ramps within and across tiers."""
    base_diff = self._current_tier_floor()
    tier_range = self._current_tier_ceiling() - base_diff

    # Use sigmoid for smooth transition instead of linear
    success = self._recent_success_rate()
    sigmoid_factor = 1.0 / (1.0 + math.exp(-10 * (success - 0.5)))

    return base_diff + tier_range * sigmoid_factor
```

**B. Add demotion (step-back) mechanism.**

```python
DEMOTION_THRESHOLD = 0.25  # If success drops below 25% for 5 episodes
DEMOTION_WINDOW = 5

def check_demotion(self):
    recent = self._episode_history[-DEMOTION_WINDOW:]
    if len(recent) >= DEMOTION_WINDOW:
        success_rate = sum(1 for r in recent if r["resolved"]) / len(recent)
        if success_rate < DEMOTION_THRESHOLD and self.current_tier_idx > 0:
            self.current_tier_idx -= 1
            logger.info(f"Demoted to tier {self.tiers[self.current_tier_idx].name}")
```

**C. Track combination mastery, not just individual faults.**

```python
# Instead of just tracking "oom_kill" mastery:
self.combo_mastery = {
    "oom_kill": 0.8,
    "image_pull": 0.7,
    "oom_kill+image_pull": 0.3,  # Track combinations separately
    "oom_kill+crashloop+bad_config": 0.1,
}
```

**D. Use per-tier success rate for advancement.**

```python
def _tier_success_rate(self) -> float:
    """Only count episodes at current difficulty tier."""
    tier = self.tiers[self.current_tier_idx]
    tier_episodes = [e for e in self._episode_history
                     if tier.min_diff <= e["difficulty"] <= tier.max_diff]
    if len(tier_episodes) < tier.min_episodes:
        return 0.0
    return sum(1 for e in tier_episodes[-10:] if e["resolved"]) / min(10, len(tier_episodes))
```

**E. Add replay buffer for catastrophic forgetting prevention.**

Periodically replay easy scenarios even at higher tiers:

```python
REPLAY_RATIO = 0.15  # 15% of episodes are replayed from earlier tiers

def should_replay(self) -> bool:
    return random.random() < REPLAY_RATIO and self.current_tier_idx > 0

def get_replay_scenario(self):
    """Sample a scenario from an earlier tier."""
    replay_tier_idx = random.randint(0, self.current_tier_idx - 1)
    replay_difficulty = self.tiers[replay_tier_idx].min_diff + 0.1
    return self._generate_scenario(difficulty=replay_difficulty)
```

### Research Context

Latest curriculum RL research (2025) recommends training at the **50% success rate sweet spot** — where information gain is maximized. Your current system pushes for 60-70% success before advancing, which may be too conservative. Consider lowering advancement thresholds to 50-55% and relying on smooth difficulty transitions instead.

---

## 4. Hyperparameter Recommendations

### Current Values vs Recommended

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `learning_rate` | 2e-6 | **1e-6** | Lower LR is more stable for small models with GRPO. Research shows 1e-6 is the sweet spot for <2B params. |
| `beta` (KL penalty) | 0.01 | **0.0** (with DAPO) | DAPO removes KL penalty entirely. For verifiable rewards (kubectl succeeds/fails), KL hurts. |
| `num_generations` | 8 | **16-32** | 8 is minimum viable. 16+ gives stabler advantage estimation. 64 is ideal but expensive. |
| `temperature` | 1.0 | **1.0** (correct) | T=1.0 is validated optimal for GRPO exploration. |
| `gradient_accumulation_steps` | 8 | **8** (correct) | Good effective batch size. |
| `max_grad_norm` | 1.0 | **1.0** (correct) | Standard gradient clipping. |
| `lora_r` | 16 | **32** | Higher rank gives more capacity for learning complex SRE patterns. |
| `lora_alpha` | 32 | **64** | Maintain 2× rank ratio. |
| `lora_dropout` | 0.05 | **0.1** | Slightly higher dropout for regularization with small model. |
| `lr_scheduler_type` | cosine | **cosine** (correct) | Standard for GRPO. |
| `warmup_steps` | 2 | **5-10** | 2 is very aggressive. 5-10 steps lets the optimizer stabilize. |
| `max_completion_length` | 512 | **256** | SRE commands are short. 512 wastes tokens. Shorter = faster training. |
| `loss_type` | "dapo" | **"dapo"** (correct) | Already using DAPO — good choice. |
| `mask_truncated_completions` | True | **True** (correct) | Prevents noise from truncated episodes. |
| `dataset_size` | 50 | **200-500** | 50 episodes is very few. 200+ needed for convergence. |

### LoRA Target Modules

```python
# Current:
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Recommended — add MLP layers for richer representation:
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Adding MLP projections (gate/up/down) lets the model learn more complex command-generation patterns. This is standard for TRL GRPO training in 2025.

### Clip Range

```python
# Consider adding asymmetric clipping (DAPO recommendation):
# In GRPOConfig, if supported:
clip_range = 0.2       # Standard lower bound
clip_range_high = 0.28  # Higher upper bound for small models
```

---

## 5. GRPO / DAPO Algorithm Improvements

### You're Already Using DAPO — Good

Your `loss_type="dapo"` is the right call. DAPO (Decoupled clip and dynamic sampling) has been shown to outperform vanilla GRPO with 50% fewer training steps on verifiable-reward tasks.

### Additional DAPO Optimizations

**A. Consider removing KL penalty entirely.**

```python
# Current:
beta=0.01  # KL penalty

# Recommended for verifiable rewards:
beta=0.0   # Let DAPO's clipping handle stability
```

Research shows KL penalties actively hurt performance on tasks with objective correctness signals (like kubectl command success/failure).

**B. Dynamic sampling for advantage estimation.**

DAPO's dynamic sampling filters out prompts where all generations succeed or all fail (zero variance = no learning signal). Verify this is active in your TRL version:

```python
# In GRPOConfig, if available:
filter_zero_variance=True  # Drop prompts with no reward variance
```

**C. Token-level reward credit assignment.**

Currently, your reward is assigned to the entire episode sequence. For multi-turn episodes, this means early tokens get the same credit as the final fix command. Consider per-turn reward assignment:

```python
# Instead of one reward for entire sequence:
# Assign per-turn rewards to each turn's token span
for turn_idx, (start_pos, end_pos) in enumerate(turn_boundaries):
    turn_reward = step_rewards[turn_idx]
    token_rewards[start_pos:end_pos] = turn_reward
```

---

## 6. Adversarial Design Improvements

### Current Issues

1. LLM can design unsolvable scenarios (3 faults, 15 steps)
2. Injection failures are silent
3. Weak-spot targeting uses scenario names as keys, not fault types

### Recommended Fixes

**A. Add scenario solvability validation.**

Before injecting, estimate whether the scenario is solvable within the step budget:

```python
def validate_solvability(scenario, max_steps):
    """Check if scenario is solvable within step budget."""
    # Each fault needs: 1 triage + 1-2 investigate + 1 fix + 1 verify = 4-5 steps
    min_steps_needed = len(scenario.fix_steps) * 5 + 2  # +2 for initial discovery
    if min_steps_needed > max_steps * 0.8:  # 80% budget margin
        return False, f"Needs ~{min_steps_needed} steps, budget is {max_steps}"
    return True, "OK"
```

**B. Track injection success rate.**

```python
def inject_and_verify(self, scenario):
    """Inject faults and verify they took effect."""
    injection_results = []
    for step in scenario.steps:
        success = self._inject_single(step)
        injection_results.append({"step": step, "success": success})

    success_rate = sum(r["success"] for r in injection_results) / len(injection_results)
    if success_rate < 0.5:
        logger.warning(f"Only {success_rate:.0%} of injections succeeded — using fallback")
        return self._use_fallback_scenario()
    return injection_results
```

**C. Implement self-play style adversarial design (SWE-RL pattern).**

Instead of a separate LLM designing scenarios, use the agent itself:
1. Agent generates a scenario it thinks is hard
2. A second rollout of the same agent tries to solve it
3. If the solver fails, the scenario-generator gets a positive reward
4. This creates an arms race without needing Claude API calls

This is cutting-edge (2025 SWE-RL research) and would be a major differentiator for the blog.

---

## 7. Judge System Improvements

### Current Issues

1. Base LLMJudge doesn't enforce multi-fault resolution
2. Phase detection is keyword-based (can misclassify)
3. Red herring matching is brittle
4. Cluster snapshot truncated at 4000 chars

### Recommended Fixes

**A. Multi-fault resolution enforcement.**

```python
def verify_multi_fault_resolution(self, scenario, cluster_state, action_history):
    """Verify each injected fault was specifically addressed."""
    prompt = f"""Given these {len(scenario.fix_steps)} injected faults:
{json.dumps(scenario.fix_steps, indent=2)}

And these agent actions:
{json.dumps(action_history[-10:], indent=2)}

Current cluster state:
{cluster_state[:3000]}

For EACH fault, determine:
1. Was it specifically fixed (not just masked)?
2. What command fixed it?

Return JSON: {{"faults": [{{"fault": "...", "fixed": true/false, "fix_command": "..."}}]}}"""

    result = self.llm.chat_json(prompt)
    faults_fixed = sum(1 for f in result["faults"] if f["fixed"])
    return faults_fixed, len(result["faults"])
```

**B. Stateful phase tracking instead of keyword matching.**

```python
class PhaseTracker:
    PHASES = ["triage", "investigation", "mitigation", "fix", "verification"]

    def __init__(self):
        self.current_phase_idx = 0
        self.phase_history = []

    def update(self, command: str, output: str) -> str:
        """Determine phase from command + context, not just keywords."""
        if self.current_phase_idx == 0 and "get pods" in command:
            detected = "triage"
        elif "describe" in command or "logs" in command:
            detected = "investigation"
        elif any(m in command for m in ["set ", "patch ", "scale "]):
            # Only "fix" if we've investigated first
            if self.current_phase_idx >= 1:
                detected = "fix"
            else:
                detected = "mitigation"  # Premature fix attempt
        elif self.current_phase_idx >= 3 and "get pods" in command:
            detected = "verification"  # get pods AFTER a fix = verification
        else:
            detected = self.PHASES[self.current_phase_idx]

        self.phase_history.append(detected)
        self.current_phase_idx = max(self.current_phase_idx,
                                      self.PHASES.index(detected))
        return detected
```

**C. Structured cluster summary instead of truncation.**

```python
def summarize_cluster(self, raw_output: str, max_chars: int = 4000) -> str:
    """Structured summary by namespace instead of blind truncation."""
    lines = raw_output.strip().split("\n")
    by_namespace = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            ns = parts[0]
            by_namespace.setdefault(ns, []).append(line)

    summary = []
    for ns in sorted(by_namespace.keys()):
        pods = by_namespace[ns]
        unhealthy = [p for p in pods if "Running" not in p]
        if unhealthy:
            summary.append(f"NAMESPACE {ns} ({len(unhealthy)} unhealthy):")
            summary.extend(f"  {p}" for p in unhealthy)
        else:
            summary.append(f"NAMESPACE {ns}: {len(pods)} pods, all Running")

    return "\n".join(summary)[:max_chars]
```

---

## 8. Blog Post Structure & Rigor

### Recommended Structure (Based on Top HF Posts)

Based on analysis of Open-R1, Mini-R1, and DeepSeek-R1 blog posts:

```
1. Hook / TL;DR (the "0.6B model learns SRE" pitch — you have this, it's great)
2. Problem Motivation
   - Why RL for SRE? Why not SFT? (multi-step exploration, no golden trajectories)
   - Why a live cluster? (simulators miss real failure modes)
3. Method
   - Architecture diagram (you have this)
   - Reward function with mathematical notation
   - Curriculum design with progression chart
   - Adversarial self-play mechanism
4. Training
   - Hyperparameters table (EVERY value, for reproducibility)
   - Training curves with rolling averages + confidence bands
   - Per-tier learning dynamics (show curriculum stages)
   - Computational cost (GPU hours, API costs, cluster costs)
5. Evaluation (THIS IS THE WEAKEST SECTION — NEEDS MOST WORK)
   - Baselines: zero-shot, SFT on golden trajectories, random policy
   - Held-out scenarios (in-distribution + OOD)
   - Statistical tests with p-values and confidence intervals
   - Per-fault-type breakdown
   - Ablation table (curriculum ON/OFF, adversarial ON/OFF, etc.)
6. Qualitative Analysis
   - 2-3 successful episode traces (show the "aha moment")
   - 2-3 failure cases with analysis
   - Emergent behaviors (agent discovering kubectl patterns)
7. What We Learned / Failure Analysis
   - Environment co-evolution story (you have this — it's compelling)
   - Reward shaping mistakes and fixes
   - What didn't work
8. Limitations & Future Work
9. Reproducibility Section
   - Exact environment versions
   - Random seeds used
   - Hardware specs
   - Training cost breakdown
```

### Specific Numbers to Include

| Metric | Minimum for Respectability | Ideal |
|--------|---------------------------|-------|
| Eval episodes | 30 per model | 100+ |
| Training episodes | 50 | 200-500 |
| Random seeds | 1 (acknowledged limitation) | 3-5 |
| Baselines | 2 (base + trained) | 4 (zero-shot, SFT, curriculum-only, full) |
| Ablations | 0 (current) | 3-5 component ablations |
| Statistical tests | None (current) | Bootstrap CI + t-test |
| OOD eval | None (current) | Separate held-out scenario set |

### Mathematical Notation for Reward

For the blog, formalize your reward:

```
R_episode = Σ(r_step) + R_resolution + R_timeout

where:
  r_step = r_judge + r_repeat + r_phase
  r_judge ∈ [-1.0, +1.0]  (LLM judge per-step score)
  r_repeat = -0.15 × I(repeated)  (repeat penalty)
  r_phase = +0.2 × I(correct_order) - 0.3 × I(skipped_phase)

  R_resolution = (1 + 3d) + 3(1 - t/T)²   if resolved
  R_timeout = -2(1 - 0.5 × progress)        if timed out

  d = difficulty ∈ [0, 1]
  t = steps taken, T = max steps
```

---

## 9. Code-Level Bug Fixes & Gaps

### Critical

1. **Judge doesn't enforce multi-fault resolution** (`judge.py`): LLM checks "is cluster healthy" but not "was each specific fault addressed." Agent could fix wrong fault and get credit if another pod auto-heals.

2. **Scenario solvability unchecked** (`adversarial_designer.py`): LLM can design 3-fault scenarios with only 15-step budget. Agent can't possibly triage + investigate + fix all 3 in time.

3. **Silent injection failures** (`adversarial_designer.py`): If an injected fault command fails (wrong resource name, API error), it's silently ignored. The scenario proceeds with fewer faults than intended, corrupting the reward signal.

### High Priority

4. **`kubectl get pods -A` penalized for repetition** (`kube_sre_gym_environment.py`): This diagnostic command is legitimately needed multiple times (post-fix verification). Circuit breaker should only apply to mutation commands.

5. **Reset timeout is silent** (`k8s_backend.py`): If deployment reconciliation stalls, `reset_cluster()` returns without error. Next episode starts with a partially broken cluster.

6. **Hardcoded scenario pool** (`scenario_generator.py`): Only 7 scenarios. Should dynamically parameterize (namespace, deployment, resource values).

7. **MAX_STEPS scaling is linear** (`kube_sre_gym_environment.py`): Multi-fault scenarios need exponential step budget. Current: `15 + 10 * difficulty`. Recommended: `15 + 15 * difficulty^2`.

### Medium Priority

8. **Phase detection misclassifies post-fix verification** (`judge.py`): Running `kubectl get pods -A` after a fix should be "verification" phase, but keyword matcher calls it "triage."

9. **Cluster snapshot truncation by character count** (`judge.py`): 4000-char limit cuts off pods alphabetically. Should use structured namespace-by-namespace summary.

10. **No metrics for adversarial scenario quality**: Can't tell if LLM is generating good scenarios. Should log scenario JSON + injection results + actual vs intended difficulty.

11. **`bad_config` injection only works on `payment-worker`** (`k8s_injectors.py`): Other deployments silently ignore the injected env var, making that fault type undetectable.

---

## 10. Priority-Ranked Action Items

### P0 — Must Do Before Blog Publication

- [ ] **Increase eval episodes to 50+** and add confidence intervals
- [ ] **Create held-out validation scenarios** (split by deployment or fault combo)
- [ ] **Add ablation studies** (at minimum: curriculum ON/OFF, adversarial ON/OFF)
- [ ] **Add per-fault-type eval breakdown**
- [ ] **Formalize reward function** with mathematical notation in blog
- [ ] **Include hyperparameter table** with every value and rationale
- [ ] **Add computational cost section** (GPU hours, API costs)

### P1 — High Impact Improvements

- [ ] **Fix repeat penalty to exempt diagnostic commands**
- [ ] **Add partial credit for multi-fault scenarios**
- [ ] **Add curriculum demotion mechanism**
- [ ] **Increase `num_generations` to 16+**
- [ ] **Lower `learning_rate` to 1e-6**
- [ ] **Add MLP layers to LoRA targets**
- [ ] **Increase `dataset_size` to 200+**
- [ ] **Set `beta=0.0`** (remove KL penalty with DAPO)
- [ ] **Add 2-3 qualitative episode traces** to blog (successful + failed)

### P2 — Differentiators for an Outstanding Blog

- [ ] **Multi-seed training** (3 seeds, report mean ± std)
- [ ] **OOD evaluation** on unseen fault types / namespace configurations
- [ ] **Self-play adversarial design** (agent designs scenarios for itself — SWE-RL style)
- [ ] **Combination mastery tracking** in curriculum
- [ ] **Smooth difficulty transitions** (sigmoid instead of hard tier boundaries)
- [ ] **Scenario solvability validation** before injection
- [ ] **Stateful phase tracking** in judge (not keyword-based)

### P3 — Polish

- [ ] **Add replay buffer** for catastrophic forgetting prevention
- [ ] **Structured cluster summaries** instead of truncation in judge
- [ ] **Injection success rate tracking** and logging
- [ ] **Training dynamics visualization** (per-tier reward curves, curriculum progression)
- [ ] **Graduated timeout penalty** instead of hard -2.0 cliff

---

## Appendix: Research Sources

### GRPO / DAPO
- JustRL: Simple single-stage GRPO recipe outperforming complex pipelines
- DAPO paper: 50% fewer training steps than GRPO with decoupled clipping
- Unsloth RL guide: Practical single-GPU GRPO implementation
- OpenPipe GRPO guide: Using GRPO to beat O1/O3-mini

### Reward Design
- Hybrid dense+sparse rewards (2025): Better credit assignment
- Auto MC-Reward: LLM-automated reward design
- DRLC: Dense Rewards from LLM Critic

### Curriculum Learning
- Easy-to-Hard scheduling with probabilistic difficulty ramp
- Adaptive mastery at 50% success rate sweet spot
- Self-Evolving Curriculum for automatic difficulty adjustment
- Catastrophic forgetting prevention via replay + LoRA isolation

### Evaluation
- 100+ episodes minimum for reliable RL evaluation
- Bootstrap confidence intervals + paired t-tests
- Cohen's d for practical significance
- Multiple random seeds (3-5)

### Top HF Blog Posts
- Open-R1: Multi-part series with full transparency
- Mini-R1: Simplified task showing core RL concepts
- DeepSeek-R1: Two-stage RL approach explanation
