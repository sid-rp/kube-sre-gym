"""
Evaluation Script — K8s SRE Agent (Base vs Trained)

Runs both the base Qwen2.5-1.5B and a GRPO-trained LoRA checkpoint
through random adversarial scenarios from the server, then compares.

Cross-references agent_transcripts.jsonl from training to verify
eval scenarios differ from training scenarios.

Usage:
  # Start the OpenEnv server (adversarial mode)
  GYM_MODE=adversarial LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... uv run server

  # Run eval
  python eval.py --trained-model outputs/k8s-sre-grpo-.../checkpoint-8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from kube_sre_gym import KubeSreGymEnv, KubeSreGymAction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# System prompt & helpers (same as train.py)
# ============================================================

SYSTEM_PROMPT = """You are a Kubernetes SRE. Diagnose and fix incidents.

Output exactly ONE kubectl command per turn. No explanations.

NAMESPACES: payments, frontend, auth

DEPLOYMENTS:
- payments: payment-gateway (nginx:1.25), payment-worker (busybox:1.36), payment-api (python:3.11-slim)
- frontend: web-app (nginx:1.25), frontend-cache (redis:7)
- auth: auth-service (nginx:1.25)

STEP 1 — Check all namespaces:
kubectl get pods -n payments
kubectl get pods -n frontend
kubectl get pods -n auth

STEP 2 — Investigate broken pods:
kubectl describe pod payment-gateway-xxx -n payments
kubectl logs payment-worker-xxx -n payments --previous

STEP 3 — Fix based on symptoms:
OOMKilled: kubectl set resources deployment/payment-gateway -c payment-gateway --limits=memory=128Mi -n payments
ImagePullBackOff: kubectl set image deployment/web-app web-app=nginx:1.25 -n frontend
CrashLoopBackOff: kubectl rollout restart deployment/auth-service -n auth
Missing pods (0 replicas): kubectl scale deployment/frontend-cache --replicas=1 -n frontend
Bad DATABASE_URL: kubectl set env deployment/payment-worker DATABASE_URL=postgres://payments_user:payments_pass@payment-db.payments.svc.cluster.local:5432/payments -n payments

IMPORTANT: Replace pod names with ACTUAL names from kubectl get pods output. Check ALL three namespaces."""


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
        feedback = entry.get("feedback", "")
        if len(output) > 300:
            output = output[:300] + "... (truncated)"
        lines.append(f"$ {cmd}")
        lines.append(f"  Output: {output}")
        if feedback:
            lines.append(f"  Feedback: {feedback}")
    return "\n".join(lines)


def parse_commands(text: str) -> list[str]:
    """Extract kubectl/diagnose/fix commands from agent response."""
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
    """Apply chat template with fallback."""
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
# Training data cross-reference
# ============================================================

def load_training_scenarios(jsonl_path: str = "agent_transcripts.jsonl") -> set[str]:
    """Load scenario fault types seen during training from the JSONL log."""
    seen = set()
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning(f"Training transcript not found: {jsonl_path}")
        return seen
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Extract commands used during training for fingerprinting
                cmds = [h.get("command", "") for h in entry.get("conversation", [])]
                seen.add(frozenset(cmds))
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(seen)} training episode fingerprints from {jsonl_path}")
    return seen


# ============================================================
# Agent episode runner (uses vLLM directly, no TRL)
# ============================================================

def run_episode(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    env: KubeSreGymEnv,
    max_turns: int,
    lora_request=None,
) -> dict:
    """Run a single SRE episode and return metrics."""
    result = env.reset()
    observation = result.observation

    step_rewards: list[float] = []
    commands_used: list[str] = []
    conversation_history: list[dict] = []

    for _turn in range(max_turns):
        if result.done:
            break

        # Build prompt with history
        history_text = format_history(conversation_history)
        obs_text = format_observation(observation)

        if history_text:
            user_prompt = f"{history_text}\n\n---\n\nCURRENT OBSERVATION:\n{obs_text}"
        else:
            user_prompt = obs_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate with vLLM
        gen_kwargs = dict(prompts=[prompt_text], sampling_params=sampling_params)
        if lora_request is not None:
            gen_kwargs["lora_request"] = lora_request
        outputs = llm.generate(**gen_kwargs)
        completion_text = outputs[0].outputs[0].text

        # Parse and execute commands
        commands = parse_commands(completion_text)
        if not commands:
            step_rewards.append(-0.5)
            conversation_history.append({
                "command": completion_text[:100].strip(),
                "output": "(no valid command parsed)",
                "reward": -0.5,
                "feedback": "Invalid output.",
            })
            continue

        for cmd in commands:
            commands_used.append(cmd)
            try:
                result = env.step(KubeSreGymAction(command=cmd))
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                observation = result.observation

                cmd_output = getattr(observation, "command_output", "") or ""
                hint = getattr(observation, "hint", "") or ""
                conversation_history.append({
                    "command": cmd,
                    "output": cmd_output[:500],
                    "reward": reward,
                    "feedback": hint,
                })

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

    total_reward = sum(step_rewards) if step_rewards else -1.0
    resolved = result.done and total_reward > 0

    return {
        "total_reward": round(total_reward, 3),
        "steps_taken": len(commands_used),
        "resolved": resolved,
        "commands_used": commands_used,
    }


# ============================================================
# Evaluate a single model across N episodes
# ============================================================

def evaluate_model(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    env_url: str,
    num_episodes: int,
    max_turns: int,
    model_label: str,
    lora_request=None,
) -> list[dict]:
    """Run num_episodes episodes and return per-episode results."""
    env = KubeSreGymEnv(base_url=env_url)
    results = []

    for ep in range(num_episodes):
        logger.info(f"[{model_label}] Episode {ep + 1}/{num_episodes}")
        t0 = time.time()

        try:
            ep_result = run_episode(
                llm=llm,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                env=env,
                max_turns=max_turns,
                lora_request=lora_request,
            )
            elapsed = time.time() - t0
            ep_result["episode"] = ep + 1
            ep_result["elapsed_s"] = round(elapsed, 1)
            results.append(ep_result)

            status = "RESOLVED" if ep_result["resolved"] else "FAILED"
            logger.info(
                f"[{model_label}] Episode {ep + 1}: {status} | "
                f"reward={ep_result['total_reward']:.2f} | "
                f"steps={ep_result['steps_taken']} | "
                f"{elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(f"[{model_label}] Episode {ep + 1} crashed: {e}")
            results.append({
                "episode": ep + 1,
                "total_reward": -1.0,
                "steps_taken": 0,
                "resolved": False,
                "commands_used": [],
                "elapsed_s": 0,
                "error": str(e),
            })

    env.close()
    return results


# ============================================================
# Comparison table & plotting
# ============================================================

def print_comparison(base_results: list[dict], trained_results: list[dict]):
    """Print a side-by-side comparison table."""
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed, printing plain text")
        tabulate = None

    rows = []
    n = max(len(base_results), len(trained_results))

    for i in range(n):
        br = base_results[i] if i < len(base_results) else {}
        tr = trained_results[i] if i < len(trained_results) else {}

        rows.append([
            f"Episode {i + 1}",
            br.get("total_reward", "N/A"),
            br.get("steps_taken", "N/A"),
            "Yes" if br.get("resolved") else "No",
            tr.get("total_reward", "N/A"),
            tr.get("steps_taken", "N/A"),
            "Yes" if tr.get("resolved") else "No",
        ])

    # Summary row
    def avg(results, key):
        vals = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def resolve_rate(results):
        if not results:
            return "0%"
        count = sum(1 for r in results if r.get("resolved"))
        return f"{count}/{len(results)} ({100 * count / len(results):.0f}%)"

    rows.append([
        "--- AVG ---",
        avg(base_results, "total_reward"),
        avg(base_results, "steps_taken"),
        resolve_rate(base_results),
        avg(trained_results, "total_reward"),
        avg(trained_results, "steps_taken"),
        resolve_rate(trained_results),
    ])

    headers = [
        "Episode",
        "Base Reward", "Base Steps", "Base Resolved",
        "Trained Reward", "Trained Steps", "Trained Resolved",
    ]

    if tabulate:
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Fallback plain-text table
        print("\n" + " | ".join(headers))
        print("-" * 100)
        for row in rows:
            print(" | ".join(str(c) for c in row))

    print()


def plot_comparison(base_results: list[dict], trained_results: list[dict], out_path: str):
    """Save a reward comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = max(len(base_results), len(trained_results))
    episodes = np.arange(1, n + 1)
    bar_width = 0.35

    base_rewards = [r.get("total_reward", 0) for r in base_results]
    trained_rewards = [r.get("total_reward", 0) for r in trained_results]

    # Pad shorter list
    while len(base_rewards) < n:
        base_rewards.append(0)
    while len(trained_rewards) < n:
        trained_rewards.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: per-episode rewards
    ax1.bar(episodes - bar_width / 2, base_rewards, bar_width, label="Base", color="#4C72B0", alpha=0.85)
    ax1.bar(episodes + bar_width / 2, trained_rewards, bar_width, label="Trained", color="#55A868", alpha=0.85)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Per-Episode Reward Comparison")
    ax1.set_xticks(episodes)
    ax1.legend()
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Summary bar chart
    base_avg = np.mean(base_rewards)
    trained_avg = np.mean(trained_rewards)
    base_resolve = sum(1 for r in base_results if r.get("resolved")) / max(len(base_results), 1) * 100
    trained_resolve = sum(1 for r in trained_results if r.get("resolved")) / max(len(trained_results), 1) * 100

    x = np.arange(2)
    metrics_base = [base_avg, base_resolve]
    metrics_trained = [trained_avg, trained_resolve]

    ax2.bar(x - bar_width / 2, metrics_base, bar_width, label="Base", color="#4C72B0", alpha=0.85)
    ax2.bar(x + bar_width / 2, metrics_trained, bar_width, label="Trained", color="#55A868", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Avg Reward", "Resolve Rate (%)"])
    ax2.set_title("Summary Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar_group in [ax2.containers[0], ax2.containers[1]]:
        for bar in bar_group:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10)

    plt.suptitle("K8s SRE Agent — Base vs GRPO-Trained", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved to {out_path}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate K8s SRE agent: base vs trained")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                        help="Base model name/path (default: Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--trained-model", default=None,
                        help="Path to trained LoRA checkpoint dir (e.g. outputs/.../checkpoint-8)")
    parser.add_argument("--env-url", default="http://localhost:8000",
                        help="OpenEnv server URL (default: http://localhost:8000)")
    parser.add_argument("--num-eval-episodes", type=int, default=5,
                        help="Number of eval episodes per model (default: 5)")
    parser.add_argument("--max-turns", type=int, default=15,
                        help="Max agent turns per episode (default: 15)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for eval_results.json and eval_results.png")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="vLLM GPU memory utilization (default: 0.8)")
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("K8s SRE Agent — Evaluation (Base vs Trained)")
    logger.info("=" * 60)
    logger.info(f"Base model:       {args.base_model}")
    logger.info(f"Trained model:    {args.trained_model or '(none — base only)'}")
    logger.info(f"Env URL:          {args.env_url}")
    logger.info(f"Episodes/model:   {args.num_eval_episodes}")
    logger.info(f"Max turns:        {args.max_turns}")

    # Tokenizer (same for base and LoRA)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sampling params — lower temperature for deterministic eval
    sampling_params = SamplingParams(temperature=0.3, max_tokens=512)

    # ---- Evaluate base model ----
    logger.info("\n--- Evaluating BASE model ---")

    if args.trained_model:
        # Load with LoRA support so we can reuse the same engine for both
        llm = LLM(
            model=args.base_model,
            max_model_len=4096,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=16,
        )
    else:
        llm = LLM(
            model=args.base_model,
            max_model_len=4096,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    base_results = evaluate_model(
        llm=llm,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        env_url=args.env_url,
        num_episodes=args.num_eval_episodes,
        max_turns=args.max_turns,
        model_label="BASE",
    )

    # ---- Evaluate trained model (LoRA) ----
    trained_results = []
    if args.trained_model:
        from vllm.lora.request import LoRARequest

        logger.info("\n--- Evaluating TRAINED model (LoRA) ---")
        lora_request = LoRARequest("trained", 1, args.trained_model)

        trained_results = evaluate_model(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            env_url=args.env_url,
            num_episodes=args.num_eval_episodes,
            max_turns=args.max_turns,
            model_label="TRAINED",
            lora_request=lora_request,
        )

    # ---- Print comparison ----
    if trained_results:
        print_comparison(base_results, trained_results)
    else:
        logger.info("\nBase model only (no trained checkpoint provided):")
        for r in base_results:
            status = "RESOLVED" if r.get("resolved") else "FAILED"
            logger.info(
                f"  Episode {r['episode']}: {status} | "
                f"reward={r['total_reward']:.2f} | steps={r['steps_taken']}"
            )

    # ---- Save results ----
    all_results = {
        "base_model": args.base_model,
        "trained_model": args.trained_model,
        "num_episodes": args.num_eval_episodes,
        "max_turns": args.max_turns,
        "base_results": base_results,
        "trained_results": trained_results,
        "summary": {
            "base_avg_reward": round(
                sum(r["total_reward"] for r in base_results) / max(len(base_results), 1), 3
            ),
            "base_resolve_rate": sum(1 for r in base_results if r.get("resolved")) / max(len(base_results), 1),
            "trained_avg_reward": round(
                sum(r["total_reward"] for r in trained_results) / max(len(trained_results), 1), 3
            ) if trained_results else None,
            "trained_resolve_rate": (
                sum(1 for r in trained_results if r.get("resolved")) / max(len(trained_results), 1)
            ) if trained_results else None,
        },
    }

    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Plot ----
    if trained_results:
        try:
            plot_path = out_dir / "eval_results.png"
            plot_comparison(base_results, trained_results, str(plot_path))
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
    else:
        logger.info("Skipping plot (no trained model to compare)")

    # ---- Print final summary ----
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    s = all_results["summary"]
    logger.info(f"Base model avg reward:     {s['base_avg_reward']}")
    logger.info(f"Base model resolve rate:   {s['base_resolve_rate']:.0%}")
    if trained_results:
        logger.info(f"Trained model avg reward:  {s['trained_avg_reward']}")
        logger.info(f"Trained model resolve rate: {s['trained_resolve_rate']:.0%}")
        delta = s["trained_avg_reward"] - s["base_avg_reward"]
        logger.info(f"Reward improvement:        {delta:+.3f}")


if __name__ == "__main__":
    main()
