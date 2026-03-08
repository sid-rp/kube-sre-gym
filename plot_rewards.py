#!/usr/bin/env python3
"""
Plot training rewards from CSV log.

Usage (while training is running or after):
  python plot_rewards.py                           # auto-find latest reward_log.csv
  python plot_rewards.py outputs/*/reward_log.csv  # specific file
  python plot_rewards.py --live                    # refresh every 30s
"""

import argparse
import csv
import sys
import time
from pathlib import Path


def find_latest_csv():
    """Find the most recent reward_log.csv in outputs/."""
    csvs = sorted(Path("outputs").glob("*/reward_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        return csvs[0]
    # Also check current dir
    if Path("reward_log.csv").exists():
        return Path("reward_log.csv")
    return None


def load_csv(path):
    episodes, totals, diags, fixes = [], [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            episodes.append(int(row[0]))
            totals.append(float(row[1]))
            diags.append(float(row[2]))
            fixes.append(float(row[3]))
    return episodes, totals, diags, fixes


def rolling_avg(vals, window=10):
    window = min(window, len(vals))
    return [sum(vals[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(vals))]


def plot(path, save_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, totals, diags, fixes = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    window = min(10, len(episodes))

    ax1.plot(episodes, totals, alpha=0.3, color="blue", label="Per episode")
    ax1.plot(episodes, rolling_avg(totals, window), color="blue", linewidth=2, label=f"Rolling avg ({window})")
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"K8s SRE Agent — GRPO Rewards ({len(episodes)} episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.plot(episodes, rolling_avg(diags, window), color="orange", linewidth=2, label="Diagnosis (rolling)")
    ax2.plot(episodes, rolling_avg(fixes, window), color="green", linewidth=2, label="Fix (rolling)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or path.with_suffix(".png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")

    # Print summary stats
    print(f"\nEpisodes: {len(episodes)}")
    print(f"Latest reward:  {totals[-1]:.2f}")
    print(f"Avg (last 10):  {sum(totals[-10:]) / min(10, len(totals)):.2f}")
    print(f"Best reward:    {max(totals):.2f}")
    print(f"Worst reward:   {min(totals):.2f}")


def print_table(path):
    """Print a simple ASCII table of rewards (no matplotlib needed)."""
    episodes, totals, diags, fixes = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    print(f"\n{'Ep':>4} | {'Total':>7} | {'Diag':>6} | {'Fix':>6} | {'Avg(10)':>8}")
    print("-" * 45)
    for i in range(len(episodes)):
        avg10 = sum(totals[max(0, i - 9):i + 1]) / min(i + 1, 10)
        marker = " *" if totals[i] == max(totals[:i + 1]) else ""
        print(f"{episodes[i]:>4} | {totals[i]:>7.2f} | {diags[i]:>6.2f} | {fixes[i]:>6.2f} | {avg10:>8.2f}{marker}")

    print(f"\nBest: {max(totals):.2f} (ep {episodes[totals.index(max(totals))]})")
    print(f"Avg:  {sum(totals) / len(totals):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training rewards")
    parser.add_argument("csv_path", nargs="?", help="Path to reward_log.csv")
    parser.add_argument("--live", action="store_true", help="Refresh every 30s")
    parser.add_argument("--table", action="store_true", help="Print ASCII table instead of plot")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    path = Path(args.csv_path) if args.csv_path else find_latest_csv()
    if not path or not path.exists():
        print("No reward_log.csv found. Run training first or specify path.")
        sys.exit(1)

    print(f"Reading: {path}")

    if args.table:
        print_table(path)
        return

    if args.live:
        while True:
            try:
                plot(path, Path(args.out) if args.out else None)
                time.sleep(30)
            except KeyboardInterrupt:
                break
    else:
        plot(path, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
