"""
generate_extra_figures.py
======================================
生成论文额外图表：PCA方差解释、特征组对比、箱线图、特征热力图、脑地形图。
依赖：mne, matplotlib, seaborn, numpy, scikit-learn, xgboost, shap

注意：运行此脚本前需确保以下文件存在：
  - D:\EEG\results\X_all.npy         (特征矩阵，形状 (1280, 352) 或 (1280, 480))
  - D:\EEG\results\y_val_all.npy     (Valence 标签)
  - D:\EEG\results\y_aro_all.npy     (Arousal 标签)
  - D:\EEG\results\subject_ids.npy   (被试编号)
  - D:\EEG\results\results.json      (LOSO 结果，包含各折准确率)

若某些文件缺失，程序会给出提示并跳过对应图表。
"""

import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import mne

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================== 全局设置 ========================
OUTPUT_DIR = r"D:\EEG\figures"
DATA_DIR = r"D:\EEG\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
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
})
sns.set_style("whitegrid")

# ======================== 数据加载 ========================
def load_data():
    """加载必需数据，若缺失则返回 None"""
    files = {
        "X_all": "X_all.npy",
        "y_val": "y_val_all.npy",
        "y_aro": "y_aro_all.npy",
        "subj_ids": "subject_ids.npy",
        "results": "results.json",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"缺失文件：{path}，跳过所有依赖该文件的图表。")
            return None
        if fname.endswith(".json"):
            with open(path, "r") as f:
                data[key] = json.load(f)
        else:
            data[key] = np.load(path)
    return data

data = load_data()
if data is None:
    print("数据加载失败，程序终止。")
    exit()

X_all = data["X_all"]
y_val = data["y_val"]
y_aro = data["y_aro"]
subj_ids = data["subj_ids"]
results_data = data["results"]

# 确认特征维度，并自动推断特征组边界
n_features = X_all.shape[1]
if n_features == 352:   # PSD(128) + DE(128) + Hjorth(96)
    N_PSD = 128
    N_DE  = 128
    N_HJ  = 96
    N_HFD = 0
elif n_features == 480: # PSD(128) + DE(128) + Hjorth(96) + HFD(128)
    N_PSD = 128
    N_DE  = 128
    N_HJ  = 96
    N_HFD = 128
else:
    raise ValueError(f"未知特征维度：{n_features}。请检查 X_all.npy。")

# 提取各模型32折准确率
models = ["KNN", "SVM", "RF", "XGB"]
val_acc_folds = {m: results_data["valence"]["folds"][m] for m in models}
aro_acc_folds = {m: results_data["arousal"]["folds"][m] for m in models}
val_acc_matrix = {m: [f["accuracy"] for f in folds] for m, folds in val_acc_folds.items()}
aro_acc_matrix = {m: [f["accuracy"] for f in folds] for m, folds in aro_acc_folds.items()}

# ============================================================
# 1. PCA 累积方差贡献图
# ============================================================
def plot_pca_variance():
    """绘制主成分累积方差曲线，并标记 95% 阈值"""
    print("绘制 PCA 累积方差图...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA()
    pca.fit(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumsum) + 1), cumsum * 100, marker='o', linestyle='-',
            color='steelblue', markersize=4, linewidth=1.5)
    ax.axhline(y=95, color='red', linestyle='--', linewidth=1.2, label='95% variance')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('Figure A. PCA Cumulative Explained Variance')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extra_pca_variance.png"), dpi=300)
    plt.close()
    print("  → 已保存 extra_pca_variance.png")

# ============================================================
# 2. 特征组对比柱状图
# ============================================================
def evaluate_group(X, y, groups, clf=SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')):
    """使用 LOSO 评估给定特征组的准确率"""
    logo = LeaveOneGroupOut()
    acc_list = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        acc_list.append(accuracy_score(y_test, clf.predict(X_test)))
    return np.mean(acc_list), np.std(acc_list)

def plot_feature_group_comparison():
    """对比不同特征组（PSD, DE, Hjorth, HFD, 融合）在 Valence 上的 LOSO 准确率"""
    print("绘制特征组对比柱状图...")
    # 切分特征组
    X_psd = X_all[:, :N_PSD]
    X_de  = X_all[:, N_PSD:N_PSD + N_DE]
    X_hj  = X_all[:, N_PSD + N_DE:N_PSD + N_DE + N_HJ]
    if N_HFD > 0:
        X_hfd = X_all[:, N_PSD + N_DE + N_HJ:]
    else:
        X_hfd = None

    feature_groups = {"PSD": X_psd, "DE": X_de, "Hjorth": X_hj}
    if X_hfd is not None:
        feature_groups["HFD"] = X_hfd
    feature_groups["Fusion"] = X_all

    # 评估各组的性能
    group_acc = {}
    for name, Xg in feature_groups.items():
        mean_acc, std_acc = evaluate_group(Xg, y_val, subj_ids)
        group_acc[name] = (mean_acc, std_acc)
        print(f"  {name}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    names = list(group_acc.keys())
    means = [group_acc[n][0] * 100 for n in names]
    stds  = [group_acc[n][1] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color='steelblue',
                  edgecolor='white', linewidth=0.8, error_kw={'elinewidth': 1.2, 'ecolor': 'gray'})
    # 数值标签
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Figure B. Performance of Different Feature Groups (Valence, LOSO)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extra_feature_group_comparison.png"), dpi=300)
    plt.close()
    print("  → 已保存 extra_feature_group_comparison.png")

# ============================================================
# 3. 单被试准确率箱线图
# ============================================================
def plot_boxplot():
    """绘制各模型在32个fold上准确率的箱线图"""
    print("绘制箱线图...")
    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [val_acc_matrix[m] for m in models]
    # 使用 tick_labels 替代 labels
    bp = ax.boxplot(data_to_plot, tick_labels=models, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    ax.set_ylabel('Accuracy')
    ax.set_title('Figure C. Distribution of 32-fold Accuracies for Each Model (Valence)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extra_boxplot.png"), dpi=300)
    plt.close()
    print("  → 已保存 extra_boxplot.png")

# ============================================================
# 4. 特征重要性热力图（按通道×频带）
# ============================================================
def plot_shap_heatmap():
    """
    使用 SHAP 计算 Valence 任务上的特征重要性，
    聚合为通道×频带的平均重要性，并以热力图展示。
    """
    print("绘制 SHAP 热力图...")
    try:
        import shap
    except ImportError:
        print("shap 未安装，跳过此图。")
        return

    # 标准化全部数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 训练 XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                        eval_metric="logloss", verbosity=0)
    xgb.fit(X_scaled, y_val)

    # SHAP 解释
    explainer = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_scaled)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)

    # 获取特征名称
    try:
        from feature_extract import get_feature_names
        feat_names = get_feature_names()
    except ImportError:
        # 若无法导入，则构造临时名称
        bands = ['theta', 'alpha', 'beta', 'gamma']
        feat_names = []
        for prefix in ['PSD', 'DE']:
            for ch in range(1, 33):
                for b in bands:
                    feat_names.append(f"{prefix}_ch{ch:02d}_{b}")
        for ch in range(1, 33):
            for p in ['Activity', 'Mobility', 'Complexity']:
                feat_names.append(f"Hjorth_ch{ch:02d}_{p}")
        if N_HFD > 0:
            for ch in range(1, 33):
                for b in bands:
                    feat_names.append(f"HFD_ch{ch:02d}_{b}")

    # 构建通道-频带重要性矩阵
    bands = ['theta', 'alpha', 'beta', 'gamma']
    n_channels = 32
    n_bands = 4

    importance_cb = np.zeros((n_channels, n_bands))
    for ch in range(n_channels):
        for b_idx, band in enumerate(bands):
            idx_psd = ch * n_bands + b_idx
            idx_de  = N_PSD + ch * n_bands + b_idx
            if N_HFD > 0:
                idx_hfd = N_PSD + N_DE + N_HJ + ch * n_bands + b_idx
                imp = (mean_abs_shap[idx_psd] + mean_abs_shap[idx_de] + mean_abs_shap[idx_hfd]) / 3
            else:
                imp = (mean_abs_shap[idx_psd] + mean_abs_shap[idx_de]) / 2
            importance_cb[ch, b_idx] = imp

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(importance_cb, xticklabels=bands, yticklabels=[f'CH{i+1}' for i in range(32)],
                cmap='viridis', ax=ax, cbar_kws={'label': 'Mean |SHAP value|'})
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Channel')
    ax.set_title('Figure D. Feature Importance (SHAP) by Channel and Band (Valence)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extra_shap_heatmap.png"), dpi=300)
    plt.close()
    print("  → 已保存 extra_shap_heatmap.png")

# ============================================================
# 5. 脑地形图
# ============================================================
def plot_topomap():
    """基于通道平均重要性绘制脑地形图（使用 MNE）"""
    print("绘制脑地形图...")
    # DEAP 32 通道名称（按标准10-20系统）
    ch_names = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5',
                'P7','P3','PO3','O1','Oz','Pz',
                'Fp2','AF4','F8','F4','FC2','FC6','T8','C4','CP2','CP6',
                'P8','P4','PO4','O2','Fz','Cz']

    mne.set_log_level('WARNING')

    # 创建 info 对象
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names_avail = [ch for ch in ch_names if ch in montage.ch_names]
    info = mne.pick_info(info, [ch_names.index(ch) for ch in ch_names_avail])

    # 计算每个通道的重要性（对所有频带和特征类型平均）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                        eval_metric="logloss", verbosity=0)
    xgb.fit(X_scaled, y_val)
    import shap
    explainer = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_scaled)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    n_channels = 32
    n_bands = 4
    ch_imp = np.zeros(n_channels)
    for ch in range(n_channels):
        idxs = []
        for b in range(n_bands):
            idxs.append(ch * n_bands + b)                          # PSD
            idxs.append(N_PSD + ch * n_bands + b)                  # DE
            if N_HFD > 0:
                idxs.append(N_PSD + N_DE + N_HJ + ch * n_bands + b)  # HFD
        ch_imp[ch] = np.mean(mean_abs_shap[idxs])

    # 选择可用通道
    channel_importance = ch_imp[[ch_names.index(ch) for ch in ch_names_avail]]
    # 创建 Evoked 对象
    evoked = mne.EvokedArray(channel_importance[:, np.newaxis], info, tmin=0)
    evoked.set_montage(montage)

    # 移除 title 参数，使用 fig.suptitle
    fig = evoked.plot_topomap(times=0, size=2.5, show=False)
    fig.suptitle('Figure E. Scalp Topography of Feature Importance (Valence)',
                 fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTPUT_DIR, "extra_topomap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  → 已保存 extra_topomap.png")

# ============================================================
# 执行所有图表生成
# ============================================================
if __name__ == "__main__":
    plot_pca_variance()
    plot_feature_group_comparison()
    plot_boxplot()
    plot_shap_heatmap()
    plot_topomap()
    print("\n所有额外图表已生成，保存在：", OUTPUT_DIR)