"""
Generate training curve PNG and summary stats.
Run after training completes (from Colab or locally).

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --log logs/training_log.csv --out training_curves.png
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_curves(log_path: str, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return

    df = pd.read_csv(log_path)
    if len(df) == 0:
        print("ERROR: empty log file")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0f")

    for ax in axes:
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#1e293b")
        ax.spines["left"].set_color("#1e293b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].plot(df["step"], df["total_reward"], color="#10b981", linewidth=2)
    axes[0].set_title("Total Reward", color="#e2e8f0", fontsize=14)
    axes[0].set_xlabel("Step", color="#94a3b8")
    axes[0].set_ylabel("Reward", color="#94a3b8")
    axes[0].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    axes[1].plot(df["step"], df["difficulty"], color="#f59e0b", linewidth=2)
    axes[1].set_title("Difficulty Escalation", color="#e2e8f0", fontsize=14)
    axes[1].set_xlabel("Step", color="#94a3b8")
    axes[1].set_ylabel("Tier", color="#94a3b8")
    axes[1].set_ylim(0.5, 3.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    print(f"Saved: {output_path}")

    # Summary stats
    first = df["total_reward"].iloc[0]
    last = df["total_reward"].iloc[-1]
    best = df["total_reward"].max()
    pct = ((last - first) / abs(first) * 100) if first != 0 else 0

    summary = f"""
=== Training Summary ===
Steps:        {len(df)}
First reward:  {first:+.3f}
Final reward:  {last:+.3f}
Best reward:   {best:+.3f}
Improvement:   {pct:+.0f}%
========================
HEADLINE: Reward improved from {first:+.2f} to {last:+.2f} ({pct:+.0f}%)
"""
    print(summary)

    summary_path = output_path.replace(".png", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/training_log.csv")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()
    plot_curves(args.log, args.out)
