import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
OUTPUT_FIG_PATH = os.path.join(RESULTS_DIR, "eval_performance_figures.png")


def generate_plots():
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found. Please run eval_benchmark.py first.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_summary = data["baseline_temp_0_2"]["summary"]
    abl_summary = data["ablation_temp_0_8"]["summary"]

    base_details = data["baseline_temp_0_2"]["details"]

    # Category breakdown for Baseline
    categories = ["Tool Use", "Memory Carry-Over", "Refusal / Safety", "Edge Cases"]
    cat_latencies = []
    cat_success = []

    for cat in categories:
        cat_items = [item for item in base_details if item["category"] == cat]
        if cat_items:
            avg_lat = np.mean([item["telemetry"].get("total_latency_ms", 0) for item in cat_items])
            succ_pct = (sum(1 for item in cat_items if item["task_success"]) / len(cat_items)) * 100
        else:
            avg_lat = 0
            succ_pct = 0
        cat_latencies.append(avg_lat)
        cat_success.append(succ_pct)

    # Create 2x2 Plot Matrix
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("MetroCity Smart Infrastructure GPT Chatbot Performance Analysis", fontsize=14, fontweight="bold")

    # Figure 1: Task Success Rate Comparison (Baseline vs Ablation)
    configs = ["Baseline (Temp=0.2)", "Ablation (Temp=0.8)"]
    ts_rates = [base_summary["task_success_rate_pct"], abl_summary["task_success_rate_pct"]]
    bars1 = axs[0, 0].bar(configs, ts_rates, color=["#0284c7", "#f59e0b"], width=0.5)
    axs[0, 0].set_title("Overall Task Success Rate (%)")
    axs[0, 0].set_ylim(0, 110)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval}%", ha="center", va="bottom", fontweight="bold")

    # Figure 2: Average Latency Breakdown per Prompt Category (Baseline)
    axs[0, 1].barh(categories, cat_latencies, color="#38bdf8")
    axs[0, 1].set_title("Average Total Latency by Category (ms)")
    axs[0, 1].set_xlabel("Latency (ms)")
    for i, v in enumerate(cat_latencies):
        axs[0, 1].text(v + 10, i, f"{round(v, 1)} ms", va="center")

    # Figure 3: Token Usage Breakdown (Prompt vs Completion)
    token_types = ["Prompt Tokens", "Completion Tokens"]
    base_p_tokens = sum(item["telemetry"].get("prompt_tokens", 50) for item in base_details)
    base_c_tokens = sum(item["telemetry"].get("completion_tokens", 30) for item in base_details)
    axs[1, 0].pie([base_p_tokens, base_c_tokens], labels=token_types, autopct="%1.1f%%", colors=["#6366f1", "#10b981"], startangle=140)
    axs[1, 0].set_title("Total Token Volume Distribution")

    # Figure 4: Category Success Breakdown
    axs[1, 1].bar(categories, cat_success, color="#10b981", width=0.5)
    axs[1, 1].set_title("Task Success Rate by Category (%)")
    axs[1, 1].set_ylim(0, 110)
    axs[1, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(cat_success):
        axs[1, 1].text(i, v + 2, f"{round(v, 1)}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FIG_PATH, dpi=300)
    print(f"📈 Benchmarking plots successfully saved to: {OUTPUT_FIG_PATH}")


if __name__ == "__main__":
    generate_plots()