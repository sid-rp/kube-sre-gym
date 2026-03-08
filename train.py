"""
GRPO Training Script — K8s SRE Agent
Runs on Northflank H100 (or anywhere with a GPU).

The OpenEnv environment runs on HF Spaces. This script connects to it over HTTP.
That's the whole point of OpenEnv — environment is a remote API.

  ┌─────────────────────────┐         ┌──────────────────────────┐
  │ This script (H100)      │  HTTP   │ HF Spaces                │
  │ Agent + GRPO training   │ ──────→ │ OpenEnv server           │
  │                         │ ←────── │ ├── K8sBackend → GKE     │
  │                         │ obs/rwd │ ├── Judge (HF Inference)  │
  └─────────────────────────┘         └──────────────────────────┘

Setup:
  pip install unsloth trl datasets huggingface_hub openenv-core requests

Required env vars:
  OPENENV_URL        - HF Spaces URL, e.g. https://your-name-k8s-sre-env.hf.space
  HF_TOKEN           - HuggingFace token (for model push)

Optional env vars:
  AGENT_MODEL        - default: Qwen/Qwen2.5-7B-Instruct
  NUM_EPISODES       - default: 50
  NUM_GENERATIONS    - G for GRPO, default: 4
  HF_PUSH_REPO      - push trained model, e.g. your-name/k8s-sre-agent
"""

import os
import sys
import logging
import requests
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================

OPENENV_URL = os.environ.get("OPENENV_URL", "http://localhost:7860")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "Qwen/Qwen3-8B")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "50"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LEARNING_RATE = 5e-6
OUTPUT_DIR = "./sre_agent_checkpoints"
HF_PUSH_REPO = os.environ.get("HF_PUSH_REPO", None)

# ============================================================
# Agent system prompt
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
# OpenEnv HTTP client (thin wrapper)
# ============================================================

class OpenEnvClient:
    """Talks to the OpenEnv server over HTTP. Runs anywhere."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Verify connection
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            logger.info(f"Connected to OpenEnv at {self.base_url}")
        except Exception:
            logger.warning(f"Could not reach {self.base_url}/health — continuing anyway")

    def reset(self) -> dict:
        """Reset environment, get initial observation."""
        r = requests.post(f"{self.base_url}/reset", timeout=120)
        r.raise_for_status()
        return r.json()

    def step(self, command: str) -> dict:
        """Send action, get observation + reward + done."""
        r = requests.post(
            f"{self.base_url}/step",
            json={"command": command},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        """Get current environment state."""
        r = requests.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()


# ============================================================
# Observation formatting
# ============================================================

def format_observation(obs: dict) -> str:
    """Format raw observation dict into agent prompt."""
    command_output = obs.get("command_output", "")
    cluster_status = obs.get("cluster_status_summary", "")
    hint = obs.get("hint", "")
    steps = obs.get("steps_taken", 0)
    max_steps = obs.get("max_steps", 15)

    text = f"""{command_output}

CURRENT CLUSTER STATUS:
{cluster_status}"""

    if hint:
        text += f"\n\nHINT: {hint}"

    text += f"\n\nStep {steps}/{max_steps}. Diagnose and fix this incident."
    return text


# ============================================================
# Episode runner
# ============================================================

def run_episode_with_commands(env: OpenEnvClient, response: str) -> float:
    """
    Parse agent response into commands, step through environment.
    Returns cumulative reward.
    """
    commands = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith(("kubectl ", "diagnose:", "fix:")):
            commands.append(line)
        elif line.startswith(("- kubectl", "* kubectl", "> kubectl")):
            commands.append(line.lstrip("-*> "))

    if not commands:
        return -0.5

    total_reward = 0.0
    for cmd in commands:
        try:
            result = env.step(cmd)
            total_reward += result.get("reward", 0.0)
            if result.get("done", False):
                break
        except Exception as e:
            logger.warning(f"Step error: {e}")
            total_reward -= 0.1
            break

    return total_reward


# ============================================================
# Collect training observations
# ============================================================

def collect_observations(env: OpenEnvClient, n: int) -> list[dict]:
    """Reset env n times, collect observations as training prompts."""
    observations = []
    for i in range(n):
        try:
            data = env.reset()
            obs = data.get("observation", data)
            prompt_text = format_observation(obs)
            observations.append({"prompt_text": prompt_text})
            alerts = obs.get("active_alerts", [])
            logger.info(f"  [{i+1}/{n}] {alerts[0] if alerts else 'collected'}")
        except Exception as e:
            logger.error(f"  [{i+1}/{n}] Reset error: {e}")
    return observations


# ============================================================
# GRPO reward function
# ============================================================

def make_reward_fn(env: OpenEnvClient):
    """Reward function compatible with TRL GRPOTrainer."""

    def reward_fn(completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                # Reset env for fresh episode per completion
                env.reset()

                # Extract text
                if isinstance(completion, list):
                    text = completion[-1].get("content", "") if completion else ""
                elif isinstance(completion, dict):
                    text = completion.get("content", "")
                else:
                    text = str(completion)

                reward = run_episode_with_commands(env, text)
            except Exception as e:
                logger.warning(f"Reward error: {e}")
                reward = -0.5
            rewards.append(reward)
        return rewards

    return reward_fn


# ============================================================
# Main
# ============================================================

def main():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    logger.info("=" * 60)
    logger.info("K8s SRE Agent — GRPO Training")
    logger.info("=" * 60)
    logger.info(f"OpenEnv URL:    {OPENENV_URL}")
    logger.info(f"Agent model:    {AGENT_MODEL}")
    logger.info(f"Episodes:       {NUM_EPISODES}")
    logger.info(f"Generations/G:  {NUM_GENERATIONS}")
    if HF_PUSH_REPO:
        logger.info(f"Push to:        {HF_PUSH_REPO}")

    # ---- Connect to OpenEnv ----
    env = OpenEnvClient(OPENENV_URL)

    # ---- Load agent model ----
    logger.info("Loading agent model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=AGENT_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype="bfloat16",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_R,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    logger.info("Agent model loaded.")

    # ---- Collect observations ----
    logger.info(f"Collecting {NUM_EPISODES} observations...")
    obs_data = collect_observations(env, NUM_EPISODES)

    if not obs_data:
        logger.error("No observations collected. Is the OpenEnv server running?")
        logger.error(f"Tried: {OPENENV_URL}")
        return

    logger.info(f"Collected {len(obs_data)} observations.")

    # Build HF dataset in chat format
    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt_text"]},
            ]
        }
        for item in obs_data
    ])

    # ---- GRPO training ----
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        max_completion_length=MAX_SEQ_LENGTH // 2,
        report_to="none",
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=grpo_config,
        reward_funcs=make_reward_fn(env),
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # ---- Save + push ----
    logger.info(f"Saving to {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    if HF_PUSH_REPO:
        logger.info(f"Pushing to HF Hub: {HF_PUSH_REPO}")
        model.push_to_hub(HF_PUSH_REPO)
        tokenizer.push_to_hub(HF_PUSH_REPO)
        logger.info(f"Model live at https://huggingface.co/{HF_PUSH_REPO}")

    # ---- Print stats ----
    try:
        state = env.state()
        logger.info(f"Final curriculum stats: {state.get('curriculum_stats', {})}")
    except Exception:
        pass

    logger.info("Done!")


if __name__ == "__main__":
    main()
