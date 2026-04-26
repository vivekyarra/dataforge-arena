"""
Generate training curve PNG and summary stats.

Run:
    python scripts/plot_training.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOG_PATH = os.path.join(LOGS_DIR, "training_log.csv")
DEFAULT_OUTPUT_PATH = os.path.join(LOGS_DIR, "training_curves_final.png")
BACKWARD_COMPAT_OUTPUT = os.path.join(LOGS_DIR, "training_curve.png")

sys.path.insert(0, ROOT_DIR)


def _resolve_log_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def _resolve_output_path(path: str) -> str:
    filename = os.path.basename(path) if path else os.path.basename(DEFAULT_OUTPUT_PATH)
    return os.path.join(LOGS_DIR, filename)


def _series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df))


def plot_curves(log_path: str, output_path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return

    resolved_log_path = _resolve_log_path(log_path)
    resolved_output_path = _resolve_output_path(output_path)
    os.makedirs(LOGS_DIR, exist_ok=True)

    df = pd.read_csv(resolved_log_path)
    if len(df) == 0:
        print("ERROR: empty log file")
        return

    total_reward = _series_or_zero(df, "total_reward")
    constraint_alignment = _series_or_zero(df, "constraint_alignment")
    shaped_reward_total = _series_or_zero(df, "shaped_reward_total")
    steps = _series_or_zero(df, "step")
    reward_ma = total_reward.rolling(window=10, min_periods=1).mean()
    shaped_mean = float(shaped_reward_total.mean())

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

    axes[0].plot(steps, total_reward, color="#86efac", linewidth=1.2, alpha=0.25)
    axes[0].plot(steps, reward_ma, color="#10b981", linewidth=2.5)
    axes[0].set_title("Total Reward Learning Curve", color="#e2e8f0", fontsize=14)
    axes[0].set_xlabel("Step", color="#94a3b8")
    axes[0].set_ylabel("Reward", color="#94a3b8")
    axes[0].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    best_idx = int(total_reward.idxmax())
    final_idx = len(df) - 1
    best_step = steps.iloc[best_idx]
    best_reward = total_reward.iloc[best_idx]
    final_step = steps.iloc[final_idx]
    final_reward = total_reward.iloc[final_idx]
    axes[0].annotate(
        f"Best: {best_reward:.2f}",
        xy=(best_step, best_reward),
        xytext=(best_step, best_reward + 0.45),
        color="#e2e8f0",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#10b981"},
    )
    axes[0].annotate(
        f"Final: {final_reward:.2f}",
        xy=(final_step, final_reward),
        xytext=(final_step - 35, final_reward - 0.75),
        color="#e2e8f0",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#10b981"},
    )

    axes[1].plot(steps, constraint_alignment, color="#38bdf8", linewidth=2)
    axes[1].fill_between(steps, constraint_alignment, color="#38bdf8", alpha=0.12)
    axes[1].set_title("Constraint Alignment Signal", color="#e2e8f0", fontsize=14)
    axes[1].set_xlabel("Step", color="#94a3b8")
    axes[1].set_ylabel("Reward Component", color="#94a3b8")
    axes[1].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)

    # Constraint alignment trend on a secondary axis (honest zero-signal handling).
    ax2 = axes[0].twinx()
    ax2.tick_params(colors="#f59e0b")
    ax2.spines["right"].set_color("#f59e0b")
    constraint_raw = pd.to_numeric(df.get("constraint_alignment", pd.Series([0.0] * len(df))), errors="coerce").fillna(0.0)
    if (constraint_raw == 0.0).all():
        ax2.axhspan(-0.05, 0.05, alpha=0.15, color='orange',
                    label='constraint_alignment (0 — tool collapse v1.0, fixed v1.1)')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(loc='upper left', fontsize=8)
        # Add text annotation
        ax2.text(270, 0.3,
                 'Fixed in v1.1\n(see reward.py)',
                 color='orange', fontsize=8, style='italic')
        handles, labels = ax2.get_legend_handles_labels()
        labels = [
            "constraint_alignment: 0 throughout (tool collapse fixed in v1.1)"
            if label == "constraint_alignment (0 — tool collapse v1.0, fixed v1.1)"
            else label
            for label in labels
        ]
        ax2.legend(handles, labels, loc="upper left", fontsize=8)
    else:
        constraint_ma = constraint_raw.rolling(window=10, min_periods=1).mean()
        ax2.plot(
            steps,
            constraint_ma,
            color="orange",
            linestyle="--",
            linewidth=1.8,
            label="constraint_alignment (10-step avg)",
        )
        ax2.set_ylabel("Constraint Alignment (rolling)", color="#f59e0b")
        ax2.legend(loc="upper left", fontsize=8)

    ax3 = axes[2]
    # Tool usage distribution
    tool_names = {0:'IMPUTE_MED', 1:'IMPUTE_MODE', 2:'FWD_FILL',
                  3:'CORRECT_FMT', 4:'DELETE_ROW', 5:'MERGE_DUP',
                  6:'FLAG_UNC', 7:'NO_OP'}
    if 'dominant_tool' in df.columns:
        tool_counts = df['dominant_tool'].value_counts().sort_index()
        ax3.bar([tool_names.get(int(i), str(i)) for i in tool_counts.index],
                tool_counts.values, color='steelblue')
        ax3.set_title('Tool Usage Distribution')
        ax3.set_xlabel('Tool')
        ax3.set_ylabel('Steps as Dominant Tool')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.set_title("Tool Usage Distribution", color="#e2e8f0", fontsize=14)
        ax3.text(0.5, 0.5, "No dominant_tool column", ha="center", va="center", color="#94a3b8")

    axes[3].plot(steps, shaped_reward_total, color="#f472b6", linewidth=2.2)
    axes[3].fill_between(steps, shaped_reward_total, color="#f472b6", alpha=0.12)
    axes[3].axhline(
        shaped_mean,
        color="#f8fafc",
        linestyle="--",
        linewidth=1,
        label=f"mean={shaped_mean:.2f}",
    )
    axes[3].set_title(
        "Shaped Reward Total (constraint + schema + outlier + reasoning + parse)",
        color="#e2e8f0",
        fontsize=12,
    )
    axes[3].set_xlabel("Step", color="#94a3b8")
    axes[3].set_ylabel("Reward Component", color="#94a3b8")
    axes[3].axhline(y=0, color="#374151", linestyle="--", alpha=0.5)
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(resolved_output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.savefig(BACKWARD_COMPAT_OUTPUT, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)
    print(f"Saved: {resolved_output_path}")
    print(f"Saved: {BACKWARD_COMPAT_OUTPUT}")

    first_reward = float(total_reward.iloc[0])
    last_reward = float(total_reward.iloc[-1])
    best_reward_value = float(total_reward.max())
    pct = ((last_reward - first_reward) / abs(first_reward) * 100) if first_reward != 0 else 0.0
    parse_first = float(_series_or_zero(df, "parse_success_rate").iloc[0]) * 100.0
    parse_last = float(_series_or_zero(df, "parse_success_rate").iloc[-1]) * 100.0
    constraint_first = float(constraint_alignment.iloc[0]) if len(constraint_alignment) else 0.0
    constraint_last = float(constraint_alignment.iloc[-1]) if len(constraint_alignment) else 0.0

    summary = f"""
=== Training Summary ===
Rows logged:    {len(df)}
Final step:     {int(steps.iloc[-1])}
First reward:  {first_reward:+.3f}
Final reward:  {last_reward:+.3f}
Best reward:   {best_reward_value:+.3f}
Improvement:   {pct:+.0f}%
Parse rate:    {parse_first:.1f}% -> {parse_last:.1f}%
Constraint:    {constraint_first:+.4f} -> {constraint_last:+.4f}
Shaped total:  {shaped_reward_total.min():+.4f} -> {shaped_reward_total.max():+.4f}
========================
HEADLINE: Reward {first_reward:+.2f} -> {last_reward:+.2f}; parse {parse_first:.0f}% -> {parse_last:.0f}%
"""
    print(summary)

    summary_path = resolved_output_path.replace(".png", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/training_log.csv")
    parser.add_argument("--out", default="logs/training_curves_final.png")
    args = parser.parse_args()
    plot_curves(args.log, args.out)
