"""
visualize_results.py
===============================
依赖：matplotlib >= 3.5, seaborn, numpy, json
输出：D:\EEG\figures\ 下 PNG 文件（300 DPI）
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ======================== 全局设置 ========================
OUTPUT_DIR = r"D:\EEG\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# 颜色
ACC_COLOR = "#3f72af"
F1_COLOR  = "#d67c7c"
SHAP_COLOR = "#2e8b8b"

# ============================================================
# 1. 加载结果
# ============================================================
results_path = r"D:\EEG\results\results.json"
if not os.path.exists(results_path):
    raise FileNotFoundError(f"结果文件未找到: {results_path}")

with open(results_path, "r") as f:
    data = json.load(f)

# 汇总数据
valence_summary = data["valence"]["summary"]
arousal_summary = data["arousal"]["summary"]
models = list(valence_summary.keys())

# 提取度量函数
def extract_metric(summary, metric):
    means = [summary[m][metric]["mean"] * 100 for m in models]
    stds  = [summary[m][metric]["std"]  * 100 for m in models]
    return np.array(means), np.array(stds)

val_acc_m, val_acc_s = extract_metric(valence_summary, "accuracy")
val_f1_m,  val_f1_s  = extract_metric(valence_summary, "f1")
aro_acc_m, aro_acc_s = extract_metric(arousal_summary, "accuracy")
aro_f1_m,  aro_f1_s  = extract_metric(arousal_summary, "f1")

# ============================================================
# 2. 绘制模型性能对比柱状图（Figure 2）
# ============================================================
def plot_model_comparison():
    """绘制 Accuracy 与 F1 分任务对比的分组柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(models))
    width = 0.35

    for idx, (ax, title, acc_m, acc_s, f1_m, f1_s) in enumerate([
        (axes[0], "Valence Classification", val_acc_m, val_acc_s, val_f1_m, val_f1_s),
        (axes[1], "Arousal Classification", aro_acc_m, aro_acc_s, aro_f1_m, aro_f1_s)
    ]):
        # 柱状图
        bars1 = ax.bar(x - width/2, acc_m, width, yerr=acc_s, capsize=4,
                       label="Accuracy", color=ACC_COLOR, edgecolor="white",
                       error_kw={"elinewidth": 1.2, "ecolor": "gray"})
        bars2 = ax.bar(x + width/2, f1_m, width, yerr=f1_s, capsize=4,
                       label="F1-score", color=F1_COLOR, edgecolor="white",
                       error_kw={"elinewidth": 1.2, "ecolor": "gray"})

        # 在柱子上方标注数值（显示两位小数 + "%"）
        for bar, val in zip(bars1, acc_m):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
        for bar, val in zip(bars2, f1_m):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fancybox=False, edgecolor="gray")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)  # 网格在柱后面

        # 设定 y 轴范围（留出标注空间）
        max_val = max(acc_m.max(), f1_m.max()) + 5
        ax.set_ylim(0, max_val)

    fig.suptitle("Figure 2. Performance Comparison of Four Models under LOSO Protocol",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_model_performance.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 2: figure2_model_performance.png")

plot_model_comparison()

# ============================================================
# 3. SHAP 特征重要性条形图（Figure 3a & 3b）
# ============================================================
def get_shap_data_from_json(task="valence", fallback=None):
    """从 results.json 读取 SHAP 数据，若不存在或为空则使用 fallback"""
    shap_key = "valence" if task == "valence" else "arousal"
    shap_data = data.get("shap", {}).get(shap_key, None)

    if shap_data and shap_data.get("top_features") and shap_data.get("top_shap_mean"):
        names = shap_data["top_features"]
        values = shap_data["top_shap_mean"]
        return names, values
    elif fallback is not None:
        print(f"Warning: {task.upper()} SHAP data not found in JSON, using manual fallback.")
        names = [f[0] for f in fallback]
        values = [f[1] for f in fallback]
        return names, values
    else:
        raise ValueError(f"{task.upper()} task missing SHAP data and no fallback provided.")

def plot_shap_bars(names, values, title, filename, top_n=10):
    """绘制水平条形图，按重要程度降序（最重要在上方）"""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    # 取前 top_n（通常就是传入的所有）
    names = names[:top_n]
    values = values[:top_n]

    # 绘制条形（从下往上递增，使用 invert_yaxis 后最重要在上）
    bars = ax.barh(range(len(names)), values, height=0.6,
                   color=SHAP_COLOR, edgecolor="white", linewidth=0.5)

    # 在条形末端标注数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.002, i, f"{val:.4f}",
                va="center", fontsize=8, color="dimgray")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # 最重要的特征在顶部
    ax.set_xlabel("mean(|SHAP value|)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure: {filename}")

# ---- 尝试从 JSON 读取，若失败则使用手动输入 fallback ----
# Valence fallback
valence_fallback = [
    ("DE_ch21_beta", 0.08767),
    ("PSD_ch30_alpha", 0.08294),
    ("PSD_ch07_gamma", 0.07356),
    ("PSD_ch25_beta", 0.07246),
    ("Hjorth_ch15_Activity", 0.06040),
    ("Hjorth_ch22_Complexity", 0.05725),
    ("DE_ch14_gamma", 0.05546),
    ("Hjorth_ch22_Mobility", 0.04940),
    ("Hjorth_ch07_Mobility", 0.04890),
    ("PSD_ch32_gamma", 0.04607),
]

# Arousal fallback
arousal_fallback = [
    ("PSD_ch27_beta", 0.11373),
    ("Hjorth_ch28_Mobility", 0.09463),
    ("PSD_ch29_beta", 0.07621),
    ("PSD_ch04_beta", 0.07506),
    ("Hjorth_ch19_Mobility", 0.06125),
    ("Hjorth_ch05_Complexity", 0.06050),
    ("PSD_ch27_alpha", 0.05245),
    ("DE_ch08_beta", 0.04963),
    ("Hjorth_ch32_Complexity", 0.04907),
    ("Hjorth_ch28_Complexity", 0.04408),
]

try:
    # Valence
    v_names, v_vals = get_shap_data_from_json("valence", fallback=valence_fallback)
    plot_shap_bars(v_names, v_vals,
                   "Figure 3a. Top-10 Important Features (Valence)",
                   "figure3a_shap_valence.png")

    # Arousal
    a_names, a_vals = get_shap_data_from_json("arousal", fallback=arousal_fallback)
    plot_shap_bars(a_names, a_vals,
                   "Figure 3b. Top-10 Important Features (Arousal)",
                   "figure3b_shap_arousal.png")
except Exception as e:
    print(f"SHAP plotting failed: {e}")
    print("Please ensure results.json contains 'shap' field or fallback data is provided.")

# ============================================================
# 4. 可选：混淆矩阵与 ROC 曲线
# ============================================================
print("\nVisualization main process completed.")
print("\nNote: For confusion matrix and ROC curves, please modify main.py to save per-fold predictions")
print("      as numpy files and load them in this script. This script currently generates Figures 2 and 3 only.\n")