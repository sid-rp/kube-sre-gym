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

# Requires: pip install -e ".[train]"  (from repo root)
from kube_sre_gym import KubeSreGymEnv, KubeSreGymAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are an expert Kubernetes SRE (Site Reliability Engineer).
You receive PagerDuty alerts about Kubernetes incidents and must diagnose and fix them.

You can run kubectl commands to investigate. After diagnosis, submit:
- diagnose: <your root cause analysis>
- fix: kubectl <the fix command>

Be systematic: check pod status, read logs, describe resources, then diagnose and fix.
Be efficient: minimize unnecessary commands.
Output one command per line. No explanations, just commands."""


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
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    parser.add_argument("--hub-repo", default=None, help="HF Hub repo, e.g. your-name/k8s-sre-agent")
    parser.add_argument(
        "--vllm-mode", choices=("colocate", "server"), default="colocate",
        help="vLLM mode: colocate (1 GPU) or server (separate vLLM process)",
    )
    parser.add_argument("--vllm-server-url", default="http://localhost:8000", help="vLLM server URL (server mode)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
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


def parse_commands(text: str) -> list[str]:
    """Extract kubectl/diagnose/fix commands from agent response."""
    commands = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(("kubectl ", "diagnose:", "fix:")):
            commands.append(line)
        elif line.startswith(("- kubectl", "* kubectl", "> kubectl")):
            commands.append(line.lstrip("-*> "))
    return commands


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
    Agent generates commands, environment executes them on real cluster,
    judge scores each action.
    """
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    diagnosis_rewards: list[float] = []
    fix_rewards: list[float] = []

    for _turn in range(max_turns):
        if result.done:
            break

        # Build prompt from current observation
        user_prompt = format_observation(observation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

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
            continue

        for cmd in commands:
            try:
                result = env.step(KubeSreGymAction(command=cmd))
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                observation = result.observation

                # Track specific reward types
                if cmd.startswith("diagnose:"):
                    diagnosis_rewards.append(reward)
                elif cmd.startswith("fix:"):
                    fix_rewards.append(reward)

                if result.done:
                    break
            except Exception as e:
                logger.warning(f"Step error: {e}")
                step_rewards.append(-0.1)
                break

    # Aggregate rewards
    total_reward = sum(step_rewards) if step_rewards else -1.0
    diagnosis_score = diagnosis_rewards[-1] if diagnosis_rewards else 0.0
    fix_score = fix_rewards[-1] if fix_rewards else 0.0

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
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
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

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to https://huggingface.co/{args.hub_repo}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
