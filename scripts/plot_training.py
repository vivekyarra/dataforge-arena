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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0a0a0f")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#1e293b")
        ax.spines["left"].set_color("#1e293b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    reward_ma = df["total_reward"].rolling(window=3, min_periods=1).mean()
    axes[0].plot(df["step"], df["total_reward"], color="#86efac", linewidth=1.2, alpha=0.45)
    axes[0].plot(df["step"], reward_ma, color="#10b981", linewidth=2.5)
    axes[0].set_title("Total Reward", color="#e2e8f0", fontsize=14)
    axes[0].set_xlabel("Step", color="#94a3b8")
    axes[0].set_ylabel("Reward", color="#94a3b8")
    axes[0].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    axes[1].plot(df["step"], df["accuracy_delta"], color="#38bdf8", linewidth=2)
    axes[1].set_title("Accuracy Delta Component", color="#e2e8f0", fontsize=14)
    axes[1].set_xlabel("Step", color="#94a3b8")
    axes[1].set_ylabel("Reward Component", color="#94a3b8")
    axes[1].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    parse_pct = df["parse_success_rate"] * 100.0
    axes[2].plot(df["step"], parse_pct, color="#fbbf24", linewidth=2)
    axes[2].set_title("Parse Success Rate", color="#e2e8f0", fontsize=14)
    axes[2].set_xlabel("Step", color="#94a3b8")
    axes[2].set_ylabel("Valid JSON (%)", color="#94a3b8")
    axes[2].set_ylim(0, 105)

    axes[3].plot(df["step"], df["efficiency"], color="#f472b6", linewidth=2)
    axes[3].set_title("Repair Targeting Efficiency", color="#e2e8f0", fontsize=14)
    axes[3].set_xlabel("Step", color="#94a3b8")
    axes[3].set_ylabel("Reward Component", color="#94a3b8")
    axes[3].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    print(f"Saved: {output_path}")

    # Summary stats
    first = df["total_reward"].iloc[0]
    last = df["total_reward"].iloc[-1]
    best = df["total_reward"].max()
    pct = ((last - first) / abs(first) * 100) if first != 0 else 0
    parse_first = df["parse_success_rate"].iloc[0] * 100.0
    parse_last = df["parse_success_rate"].iloc[-1] * 100.0
    acc_first = df["accuracy_delta"].iloc[0]
    acc_last = df["accuracy_delta"].iloc[-1]

    summary = f"""
=== Training Summary ===
Rows logged:    {len(df)}
Final step:     {df["step"].iloc[-1]}
First reward:  {first:+.3f}
Final reward:  {last:+.3f}
Best reward:   {best:+.3f}
Improvement:   {pct:+.0f}%
Parse rate:    {parse_first:.1f}% -> {parse_last:.1f}%
Accuracy comp: {acc_first:+.4f} -> {acc_last:+.4f}
========================
HEADLINE: Reward {first:+.2f} -> {last:+.2f}; parse {parse_first:.0f}% -> {parse_last:.0f}%
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
