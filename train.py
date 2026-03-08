"""
GRPO Training Script — K8s SRE Agent
Follows the standard OpenEnv + TRL pattern (same as wordle.py example).

Everything runs on the H100:
  - vLLM (colocate mode) handles agent inference during GRPO — no separate process
  - OpenEnv server runs on port 8000 (talks to GKE + external judge)
  - External judge (Claude via Anthropic API) scores actions — no GPU needed

Setup (2 terminals on H100):

  # Install
  pip install -e ".[train]"

  # Terminal 1: OpenEnv server (adversarial mode with Claude judge)
  GYM_MODE=adversarial LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... uv run server

  # Terminal 2: GRPO training (full 80GB for agent)
  python train.py --vllm-mode colocate
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

# Silence TRL experimental warning for rollout_func
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import AutoTokenizer

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

# ---- TRL 0.29.0 / vLLM 0.11.x compatibility ----
# TRL 0.29.0 expects vLLM logprobs as list-of-lists (top-k per token),
# but vLLM 0.11.x returns plain floats. Patch until TRL releases a fix.
# See: https://github.com/huggingface/trl/issues/4159
_orig_gen = GRPOTrainer._generate_single_turn

def _compat_generate_single_turn(self, prompts):
    prompt_ids, completion_ids, logprobs, extra = _orig_gen(self, prompts)
    return prompt_ids, completion_ids, logprobs, extra

# The issue is in the line AFTER vllm_generation.generate() returns:
#   logprobs = [[lp[0] for lp in seq] for seq in logprobs]
# When vLLM 0.11.x already returns floats, lp[0] fails.
# We patch vllm_generation.generate to wrap floats in lists.
import trl.trainer.grpo_trainer as _grpo_mod

_orig_vllm_gen = None

def _patch_vllm_generate(trainer):
    """Wrap vLLM generate to ensure logprobs are in top-k list format."""
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, 'vllm_generation'):
        return
    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        # If logprobs are plain floats, wrap them in lists for TRL's lp[0]
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate

# Patch trainer.train() to apply the fix before first generation
_orig_train = GRPOTrainer.train

def _patched_train(self, *args, **kwargs):
    _patch_vllm_generate(self)
    return _orig_train(self, *args, **kwargs)

GRPOTrainer.train = _patched_train

# Requires: pip install -e ".[train]"  (from repo root)
from kube_sre_gym import KubeSreGymEnv, KubeSreGymAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are an expert Kubernetes SRE (Site Reliability Engineer).
You receive PagerDuty alerts about Kubernetes incidents and must diagnose and fix them.

IMPORTANT RULES:
- Always specify namespace with -n <namespace> (pods are in: payments, frontend, auth)
- Start with: kubectl get pods -n <namespace> to see actual pod names and status
- Never guess pod names — always list pods first, then use exact names from the output
- NEVER repeat a command you already ran — check your previous commands first
- Output exactly ONE command. No explanations, no extra text. Just the command.

CLUSTER TOPOLOGY:
- payments namespace: payment-gateway (container: payment-gateway, image: nginx:1.25), payment-worker (container: payment-worker, image: busybox:1.36), payment-api (container: payment-api, image: python:3.11-slim)
- frontend namespace: web-app (container: web-app, image: nginx:1.25), frontend-cache (container: frontend-cache, image: redis:7)
- auth namespace: auth-service (container: auth-service, image: nginx:1.25)

AVAILABLE COMMANDS:
- kubectl get pods/deployments/events/services -n <ns>
- kubectl describe pod/deployment <name> -n <ns>
- kubectl logs <pod-name> -n <ns> [--previous]
- kubectl set image deployment/<name> <container>=<image> -n <ns>
- kubectl set resources deployment/<name> -c <container> --limits=memory=<val> -n <ns>
- kubectl set env deployment/<name> KEY=VALUE -n <ns>
- kubectl patch deployment <name> -n <ns> -p '{"spec":...}'
- kubectl rollout restart deployment/<name> -n <ns>
- kubectl scale deployment/<name> --replicas=N -n <ns>
- kubectl delete pod <name> -n <ns>

COMMON FIXES (use the exact container names from CLUSTER TOPOLOGY above):
- CrashLoopBackOff (bad command): kubectl patch deployment <name> -n <ns> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","command":null,"args":null}]}}}}'
- OOMKilled (exit code 137): kubectl set resources deployment/<name> -c <container> --limits=memory=256Mi -n <ns>
- ImagePullBackOff: kubectl set image deployment/<name> <container>=<correct-image-from-topology> -n <ns>
- Bad env config: kubectl set env deployment/<name> KEY=CORRECT_VALUE -n <ns>
- Liveness probe wrong path: kubectl patch deployment <name> -n <ns> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","livenessProbe":{"httpGet":{"path":"/","port":80}}}]}}}}'

WORKFLOW: list pods → describe/logs the broken pod → diagnose: <root cause> → fix: kubectl <command>
After applying a fix, STOP. Do not repeat the fix or run more commands."""


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for K8s SRE agent")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B", help="Agent model to fine-tune")
    parser.add_argument("--env-url", default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--dataset-size", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-turns", type=int, default=15, help="Max commands per episode")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens per agent response")
    parser.add_argument("--num-generations", type=int, default=4, help="G for GRPO")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Max GRPO training steps (-1 = auto)")
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    parser.add_argument("--hub-repo", default=None, help="HF Hub repo, e.g. your-name/k8s-sre-agent")
    parser.add_argument(
        "--vllm-mode", choices=("colocate", "server"), default="colocate",
        help="vLLM mode: colocate (1 GPU) or server (separate vLLM process)",
    )
    parser.add_argument("--vllm-server-url", default="http://localhost:8001", help="vLLM server URL (server mode)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--report-to", default="none", choices=("tensorboard", "wandb", "none"),
                        help="Logging backend for reward curves (default: none, uses CSV instead)")
    parser.add_argument("--reward-log", default="reward_log.csv",
                        help="CSV file for per-episode reward logging")
    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def format_observation(obs) -> str:
    """Format observation into agent-readable text."""
    command_output = getattr(obs, "command_output", "") or ""
    cluster_status = getattr(obs, "cluster_status_summary", "") or ""
    hint = getattr(obs, "hint", "") or ""
    steps = getattr(obs, "steps_taken", 0)
    max_steps = getattr(obs, "max_steps", 15)

    text = f"""{command_output}

CURRENT CLUSTER STATUS:
{cluster_status}"""

    if hint:
        text += f"\n\nHINT: {hint}"

    text += f"\n\nStep {steps}/{max_steps}. Diagnose and fix this incident."
    return text


def format_history(history: list[dict]) -> str:
    """Format conversation history into a condensed summary for the agent."""
    if not history:
        return ""
    lines = ["PREVIOUS COMMANDS AND RESULTS:"]
    for entry in history:
        cmd = entry["command"]
        output = entry["output"]
        reward = entry.get("reward", 0.0)
        feedback = entry.get("feedback", "")
        # Truncate long outputs but keep enough context
        if len(output) > 300:
            output = output[:300] + "... (truncated)"
        lines.append(f"$ {cmd}")
        lines.append(f"  Output: {output}")
        if feedback:
            lines.append(f"  Feedback: {feedback}")
    return "\n".join(lines)


def parse_commands(text: str) -> list[str]:
    """Extract kubectl/diagnose/fix commands from agent response.

    Returns at most 2 commands per turn to prevent the agent from spamming
    duplicate commands in a single response. The system prompt tells the agent
    to output one command at a time.
    """
    commands = []
    seen = set()
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(("kubectl ", "diagnose:", "fix:")):
            if line not in seen:
                commands.append(line)
                seen.add(line)
        elif line.startswith(("- kubectl", "* kubectl", "> kubectl")):
            cmd = line.lstrip("-*> ")
            if cmd not in seen:
                commands.append(cmd)
                seen.add(cmd)
        if len(commands) >= 2:
            break
    return commands


def apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback if enable_thinking is not supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


# ============================================================
# Rollout — one full SRE episode
# ============================================================

def rollout_once(
    trainer: GRPOTrainer,
    env: KubeSreGymEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
) -> dict[str, list]:
    """
    Run one full K8s incident episode.

    The agent builds a conversation history across turns so it can do
    multi-step diagnosis (triage -> investigate -> fix -> verify).
    Each turn, the full history is included in the prompt so the agent
    knows what it already tried.

    Token accumulation: prompt_ids and completion_ids are extended across
    turns. This matches the TRL OpenEnv pattern (see wordle example) —
    GRPO assigns episode-level reward to the full token sequence.
    """
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    diagnosis_rewards: list[float] = []
    fix_rewards: list[float] = []

    # Conversation history — agent needs this to avoid repeating commands
    # and to build on previous investigation results
    conversation_history: list[dict] = []

    for _turn in range(max_turns):
        if result.done:
            break

        # Build prompt with full history so agent has context
        history_text = format_history(conversation_history)
        obs_text = format_observation(observation)

        if history_text:
            user_prompt = f"{history_text}\n\n---\n\nCURRENT OBSERVATION:\n{obs_text}"
        else:
            user_prompt = obs_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate with vLLM via TRL
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse and execute commands on real cluster
        commands = parse_commands(completion_text)
        if not commands:
            step_rewards.append(-0.5)
            conversation_history.append({
                "agent_text": completion_text[:500],
                "command": completion_text[:100].strip(),
                "output": "(no valid command parsed)",
                "reward": -0.5,
                "feedback": "Invalid output — expected kubectl/diagnose:/fix: command.",
            })
            continue

        for cmd in commands:
            try:
                result = env.step(KubeSreGymAction(command=cmd))
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                observation = result.observation

                # Record in history for next turn's context
                cmd_output = getattr(observation, "command_output", "") or ""
                hint = getattr(observation, "hint", "") or ""
                conversation_history.append({
                    "agent_text": completion_text[:500],
                    "command": cmd,
                    "output": cmd_output[:500],
                    "reward": reward,
                    "feedback": hint,
                })

                if cmd.startswith("diagnose:"):
                    diagnosis_rewards.append(reward)
                elif cmd.startswith("fix:"):
                    fix_rewards.append(reward)

                if result.done:
                    break
            except Exception as e:
                logger.warning(f"Step error: {e}")
                step_rewards.append(-0.1)
                conversation_history.append({
                    "command": cmd,
                    "output": f"ERROR: {e}",
                    "reward": -0.1,
                    "feedback": "",
                })
                break

    # Aggregate rewards
    total_reward = sum(step_rewards) if step_rewards else -1.0
    diagnosis_score = diagnosis_rewards[-1] if diagnosis_rewards else 0.0
    fix_score = fix_rewards[-1] if fix_rewards else 0.0

    # Save detailed agent transcript for eval/SFT
    try:
        import json
        transcript_path = os.environ.get("AGENT_TRANSCRIPT_LOG", "agent_transcripts.jsonl")
        initial_obs = getattr(observation, "command_output", "") or ""
        agent_transcript = {
            "total_reward": total_reward,
            "diagnosis_reward": diagnosis_score,
            "fix_reward": fix_score,
            "num_steps": len(conversation_history),
            "resolved": result.done and total_reward > 0,
            "conversation": conversation_history,
        }
        with open(transcript_path, "a") as f:
            f.write(json.dumps(agent_transcript) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save agent transcript: {e}")

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "diagnosis_reward": diagnosis_score,
        "fix_reward": fix_score,
    }


# ============================================================
# Reward functions (TRL convention)
# ============================================================

def reward_total(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("total_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def reward_diagnosis(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("diagnosis_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def reward_fix(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("fix_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


# ============================================================
# Reward visualization
# ============================================================

def plot_rewards(csv_path: Path, out_path: Path = None):
    """Plot reward curves from the CSV log. Works without tensorboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, totals, diags, fixes = [], [], [], []
    with open(csv_path) as f:
        reader = __import__("csv").reader(f)
        next(reader)  # skip header
        for row in reader:
            episodes.append(int(row[0]))
            totals.append(float(row[1]))
            diags.append(float(row[2]))
            fixes.append(float(row[3]))

    if not episodes:
        logger.warning("No episodes to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Rolling average
    window = min(10, len(episodes))
    def rolling_avg(vals):
        return [sum(vals[max(0,i-window):i+1]) / min(i+1, window) for i in range(len(vals))]

    # Total reward
    ax1.plot(episodes, totals, alpha=0.3, color="blue", label="Per episode")
    ax1.plot(episodes, rolling_avg(totals), color="blue", linewidth=2, label=f"Rolling avg ({window})")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("K8s SRE Agent — GRPO Training Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Diagnosis vs Fix
    ax2.plot(episodes, rolling_avg(diags), color="orange", linewidth=2, label="Diagnosis (rolling)")
    ax2.plot(episodes, rolling_avg(fixes), color="green", linewidth=2, label="Fix (rolling)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_path or csv_path.with_suffix(".png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("K8s SRE Agent — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:    {args.model_id}")
    logger.info(f"Env URL:        {args.env_url}")
    logger.info(f"Episodes:       {args.dataset_size}")
    logger.info(f"Generations/G:  {args.num_generations}")
    logger.info(f"vLLM mode:      {args.vllm_mode}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Connect to OpenEnv server ----
    env = KubeSreGymEnv(base_url=args.env_url)

    # ---- Dataset (each entry triggers one episode) ----
    dataset_prompt = "Diagnose and fix this Kubernetes incident."
    dataset = Dataset.from_dict({"prompt": [dataset_prompt] * args.dataset_size})

    # ---- GRPO Config (matches wordle.py pattern) ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"k8s-sre-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",  # cosine/linear hurts GRPO
        warmup_steps=2,
        max_grad_norm=0.2,  # lower than default 1.0 for GRPO stability
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        save_total_limit=3,  # keep last 3 checkpoints to save disk
    )

    # ---- Reward CSV logger ----
    import csv
    reward_log_path = output_dir / args.reward_log
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_counter = [0]  # mutable counter for closure
    all_rewards = []  # track all episode rewards for running stats

    with open(reward_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "diagnosis_reward", "fix_reward", "timestamp"])

    def _log_episode(total_r: float, diag_r: float, fix_r: float):
        episode_counter[0] += 1
        all_rewards.append(total_r)
        with open(reward_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_counter[0], total_r, diag_r, fix_r,
                             datetime.now().isoformat()])

        n = len(all_rewards)
        mean_all = sum(all_rewards) / n
        last10 = all_rewards[-10:]
        mean_10 = sum(last10) / len(last10)
        best = max(all_rewards)

        logger.info(
            f"Episode {episode_counter[0]}: reward={total_r:.2f} "
            f"(diag={diag_r:.2f}, fix={fix_r:.2f}) | "
            f"mean={mean_all:.2f}, mean(10)={mean_10:.2f}, best={best:.2f}"
        )

    # ---- Rollout function (called by GRPOTrainer each step) ----
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        total_rewards: list[float] = []
        diagnosis_rewards: list[float] = []
        fix_rewards: list[float] = []

        for prompt_text in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT,
                max_turns=args.max_turns,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            diagnosis_rewards.append(episode["diagnosis_reward"])
            fix_rewards.append(episode["fix_reward"])
            _log_episode(episode["total_reward"], episode["diagnosis_reward"], episode["fix_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
            "diagnosis_reward": diagnosis_rewards,
            "fix_reward": fix_rewards,
        }

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_total,
            reward_diagnosis,
            reward_fix,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    # ---- Train ----
    logger.info("Starting GRPO training...")
    logger.info(f"Using {args.num_generations} rollouts per episode")

    try:
        trainer.train()
    finally:
        env.close()

    # ---- Save ----
    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_log_path}")

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to https://huggingface.co/{args.hub_repo}")

    # ---- Plot rewards ----
    try:
        plot_rewards(reward_log_path, output_dir / "reward_plot.png")
    except Exception as e:
        logger.warning(f"Could not generate reward plot: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
