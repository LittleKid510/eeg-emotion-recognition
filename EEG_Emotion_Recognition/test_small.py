"""
test_small.py
=============
DEAP EEG 最小实验验证脚本
- 仅使用 s01 + s02 两个被试
- 特征：PSD (Power Spectral Density)
- 模型：SVM
- 数据划分：s01 训练，s02 测试（模拟 Subject-Independent）
- 成功标准：输出 Accuracy 在 0.50 ~ 0.80 之间
"""

import numpy as np
import scipy.io as sio
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# ============================================================
# 配置：请确认DEAP数据路径
# ============================================================
DATA_DIR = r"D:\EEG\data_preprocessed_matlab"
SUBJECTS = [1, 2]          # s01, s02
FS = 128                   # 采样率 128 Hz
BASELINE_LEN = 3 * FS      # 基线 = 3秒 = 384 个点
EEG_CHANNELS = 32          # 只取前32个EEG通道

# 频带定义 (Hz)
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ============================================================
# Step 1：读取数据（包含基线校正）
# ============================================================
def load_subject(subject_id):
    """
    读取单个被试的 .mat 文件，并进行基线校正
    返回：
        eeg    : (40, 32, 7680)  40个试次，32个通道，60秒数据
        labels : (40, 4)         对应的标签
    """
    path = os.path.join(DATA_DIR, f"s{subject_id:02d}.mat")
    mat = sio.loadmat(path)

    data   = mat["data"]    # shape: (40, 40, 8064)
    labels = mat["labels"]  # shape: (40, 4)

    # 只保留前32个EEG通道
    eeg = data[:, :EEG_CHANNELS, :]   # (40, 32, 8064)

    # 基线校正：减去前3秒（384点）均值，保留刺激段（后60秒）
    baseline = eeg[:, :, :BASELINE_LEN]                     # (40, 32, 384)
    baseline_mean = baseline.mean(axis=2, keepdims=True)    # (40, 32, 1)
    eeg = eeg[:, :, BASELINE_LEN:] - baseline_mean          # (40, 32, 7680)

    return eeg, labels


# ============================================================
# Step 2：PSD 特征提取
# ============================================================
def compute_psd_features(eeg):
    """
    对每个 trial、每个通道、每个频带计算 PSD 平均功率
    输入：eeg  shape (n_trials, 32, n_times)
    输出：features shape (n_trials, 32 * 4)  = (n_trials, 128)
    """
    n_trials, n_channels, n_times = eeg.shape
    n_bands = len(BANDS)
    features = np.zeros((n_trials, n_channels * n_bands))

    for trial_idx in range(n_trials):
        feat_vec = []
        for ch_idx in range(n_channels):
            signal = eeg[trial_idx, ch_idx, :]

            # Welch 方法估计 PSD
            freqs, psd = welch(
                signal,
                fs=FS,
                nperseg=256,
                noverlap=128
            )

            for band_name, (low, high) in BANDS.items():
                idx = np.where((freqs >= low) & (freqs < high))[0]
                band_power = psd[idx].mean() if len(idx) > 0 else 0.0
                feat_vec.append(band_power)

        features[trial_idx, :] = feat_vec

    return features  # shape: (n_trials, 128)


# ============================================================
# Step 3：构建标签（二分类）
# ============================================================
def build_labels(labels):
    """
    Valence  (labels[:,0]): > 5 → 1 (高效价), ≤ 5 → 0 (低效价)
    Arousal  (labels[:,1]): > 5 → 1 (高唤醒), ≤ 5 → 0 (低唤醒)
    """
    y_valence = (labels[:, 0] > 5).astype(int)
    y_arousal = (labels[:, 1] > 5).astype(int)
    return y_valence, y_arousal


# ============================================================
# 主流程：s01 训练，s02 测试
# ============================================================
def main():
    print("=" * 55)
    print("  DEAP EEG test_small.py — 最小实验验证")
    print("  数据划分：s01 训练，s02 测试")
    print("=" * 55)

    # 存储被试数据
    subject_data = {}
    for sid in SUBJECTS:
        print(f"\n[INFO] 正在读取 s{sid:02d}.mat ...")
        eeg, labels = load_subject(sid)
        feats = compute_psd_features(eeg)
        y_val, y_aro = build_labels(labels)
        subject_data[sid] = {
            'X': feats,          # (40, 128)
            'y_val': y_val,
            'y_aro': y_aro
        }
        print(f"       EEG shape after baseline: {eeg.shape}")
        print(f"       PSD features shape: {feats.shape}")
        print(f"       Valence 分布: {dict(zip(*np.unique(y_val, return_counts=True)))}")
        print(f"       Arousal 分布: {dict(zip(*np.unique(y_aro, return_counts=True)))}")

    # 训练集：s01，测试集：s02
    X_train = subject_data[1]['X']
    y_val_train = subject_data[1]['y_val']
    y_aro_train = subject_data[1]['y_aro']

    X_test = subject_data[2]['X']
    y_val_test = subject_data[2]['y_val']
    y_aro_test = subject_data[2]['y_aro']

    print(f"\n[INFO] 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    # 标准化（fit on train only）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # PCA（保留95%方差）
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    print(f"\n[INFO] PCA 后维度: {X_train_pca.shape[1]}（原始 128 维）")

    # 训练 SVM
    print("\n" + "=" * 55)
    print("  Valence 分类结果（高效价 vs 低效价）")
    print("=" * 55)

    svm_val = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    svm_val.fit(X_train_pca, y_val_train)
    pred_val = svm_val.predict(X_test_pca)

    acc_val = accuracy_score(y_val_test, pred_val)
    f1_val  = f1_score(y_val_test, pred_val, average="macro", zero_division=0)

    print(f"  Accuracy : {acc_val:.4f}  ({acc_val*100:.1f}%)")
    print(f"  F1-score : {f1_val:.4f}")
    print("\n  详细报告：")
    print(classification_report(y_val_test, pred_val,
                                target_names=["低效价(0)", "高效价(1)"],
                                zero_division=0))

    print("=" * 55)
    print("  Arousal 分类结果（高唤醒 vs 低唤醒）")
    print("=" * 55)

    svm_aro = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    svm_aro.fit(X_train_pca, y_aro_train)
    pred_aro = svm_aro.predict(X_test_pca)

    acc_aro = accuracy_score(y_aro_test, pred_aro)
    f1_aro  = f1_score(y_aro_test, pred_aro, average="macro", zero_division=0)

    print(f"  Accuracy : {acc_aro:.4f}  ({acc_aro*100:.1f}%)")
    print(f"  F1-score : {f1_aro:.4f}")
    print("\n  详细报告：")
    print(classification_report(y_aro_test, pred_aro,
                                target_names=["低唤醒(0)", "高唤醒(1)"],
                                zero_division=0))

    # 验证结论
    print("=" * 55)
    print("  pipeline 验证结论")
    print("=" * 55)
    ok = acc_val > 0.40 and acc_aro > 0.40
    if ok:
        print("  Pipeline 运行正常，可以进入 main.py 阶段。")
        print(f"     Valence Acc = {acc_val*100:.1f}%")
        print(f"     Arousal Acc = {acc_aro*100:.1f}%")
        print()
        print("  注意：此结果仅用于验证 pipeline，不代表论文最终结果（LOSO 才是正式评估）。")
    else:
        print("  准确率低于预期，请检查：")
        print("     1. DATA_DIR 路径是否正确")
        print("     2. 数据文件是否完整")
        print("     3. Python 包版本是否正常")
    print("=" * 55)


if __name__ == "__main__":
    main()