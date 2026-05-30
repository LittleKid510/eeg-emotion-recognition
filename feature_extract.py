"""
feature_extract.py
===================================
DEAP EEG 特征提取模块（PSD + DE + Hjorth）

"""

import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import svd
import os
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

# ============================================================
# 全局参数
# ============================================================
FS = 128
BASELINE_LEN = 3 * FS  # 384 点
EEG_CHANNELS = 32

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_NAMES = list(BANDS.keys())

# ============================================================
# 数据读取 & 基线校正
# ============================================================
def load_subject(data_dir: str, subject_id: int):
    """
    读取单个被试 .mat 文件，执行基线校正。
    返回：eeg (40,32,7680), labels (40,4)
    """
    path = os.path.join(data_dir, f"s{subject_id:02d}.mat")
    mat = sio.loadmat(path)
    data = mat["data"]          # (40, 40, 8064)
    labels = mat["labels"]      # (40, 4)
    eeg = data[:, :EEG_CHANNELS, :]
    baseline = eeg[:, :, :BASELINE_LEN]
    baseline_mean = baseline.mean(axis=2, keepdims=True)
    eeg = eeg[:, :, BASELINE_LEN:] - baseline_mean
    return eeg, labels

# ============================================================
# 标签构建
# ============================================================
def build_labels(labels: np.ndarray):
    y_valence = (labels[:, 0] > 5).astype(int)
    y_arousal = (labels[:, 1] > 5).astype(int)
    return y_valence, y_arousal

# ============================================================
# 辅助：带通滤波
# ============================================================
def _bandpass_filter(signal: np.ndarray, low: float, high: float,
                     fs: int = FS, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(low / nyq, 1e-6)
    high = min(high / nyq, 1 - 1e-6)
    if low >= high:
        # 频带无效，返回零数组
        return np.zeros_like(signal)
    b, a = butter(order, [low, high], btype="band")
    # filtfilt 可能导致边界 NaN，但一般不会；若出现则用 np.nan_to_num
    filtered = filtfilt(b, a, signal)
    return filtered

# ============================================================
# 特征 1：PSD (Welch)
# ============================================================
def _compute_psd(signal: np.ndarray, fs: int = FS) -> dict:
    freqs, psd = welch(signal, fs=fs, nperseg=256, noverlap=128,
                       window="hann", scaling='density')
    result = {}
    for band, (low, high) in BANDS.items():
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if len(idx) > 0:
            result[band] = psd[idx].mean()
        else:
            result[band] = 0.0
    return result

# ============================================================
# 特征 2：DE (Differential Entropy)
# ============================================================
def _compute_de(signal: np.ndarray, fs: int = FS) -> dict:
    result = {}
    for band, (low, high) in BANDS.items():
        filtered = _bandpass_filter(signal, low, high, fs)
        var = np.var(filtered, ddof=1)
        var = max(var, 1e-12)  # 避免 log(0)
        result[band] = 0.5 * np.log(2 * np.pi * np.e * var)
    return result

# ============================================================
# 特征 3：Hjorth 参数
# ============================================================
def _compute_hjorth(signal: np.ndarray):
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var_x = max(np.var(signal, ddof=1), 1e-12)
    var_d1 = max(np.var(d1, ddof=1), 1e-12)
    var_d2 = max(np.var(d2, ddof=1), 1e-12)

    activity = var_x
    mobility = np.sqrt(var_d1 / var_x)
    if mobility > 0:
        complexity = np.sqrt(var_d2 / var_d1) / mobility
    else:
        complexity = 0.0
    return activity, mobility, complexity

# ============================================================
# 单 trial 特征提取（用于并行）
# ============================================================
def _extract_trial(eeg_trial: np.ndarray, n_channels: int, n_bands: int):
    """
    提取单个 trial 的 352 维特征（PSD+DE+Hjorth）
    eeg_trial : (32, 7680)
    返回扁平数组
    """
    psd_vec = []
    de_vec = []
    hjorth_vec = []

    for ch in range(n_channels):
        sig = eeg_trial[ch, :]
        # PSD
        psd_dict = _compute_psd(sig)
        for b in BAND_NAMES:
            psd_vec.append(psd_dict[b])
        # DE
        de_dict = _compute_de(sig)
        for b in BAND_NAMES:
            de_vec.append(de_dict[b])
        # Hjorth
        act, mob, comp = _compute_hjorth(sig)
        hjorth_vec.extend([act, mob, comp])

    # 拼接：PSD(128) + DE(128) + Hjorth(96)
    return np.hstack([np.array(psd_vec),
                      np.array(de_vec),
                      np.array(hjorth_vec)])

# ============================================================
# 主函数：完整特征提取（352 维，并行加速）
# ============================================================
def extract_features(eeg: np.ndarray, verbose: bool = True,
                     n_jobs: int = -1) -> np.ndarray:
    """
    参数
    ----
    eeg     : ndarray (40, 32, 7680)
    verbose : 是否显示进度条
    n_jobs  : 并行核心数，-1 表示全部核心

    返回
    ----
    features : ndarray (40, 352)
    """
    n_trials, n_channels, _ = eeg.shape
    n_bands = len(BANDS)

    # 使用 joblib 并行处理所有 trial
    if verbose:
        iterator = tqdm(range(n_trials), desc="提取特征 (40 trials)", unit="trial")
    else:
        iterator = range(n_trials)

    # 并行或串行运行
    if n_jobs != 1 and n_trials > 1:
        # 预提取所有 trial
        trials_list = [eeg[t] for t in range(n_trials)]
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_extract_trial)(eeg[t], n_channels, n_bands)
            for t in iterator
        )
        features = np.array(results)
    else:
        # 串行（保留进度条）
        features = np.zeros((n_trials, n_channels * n_bands * 2 + n_channels * 3))
        for t in iterator:
            features[t] = _extract_trial(eeg[t], n_channels, n_bands)

    return features  # (40, 352)

# ============================================================
# 特征名称（352 个）
# ============================================================
def get_feature_names() -> list:
    names = []
    for ch in range(EEG_CHANNELS):
        for b in BAND_NAMES:
            names.append(f"PSD_ch{ch+1:02d}_{b}")
    for ch in range(EEG_CHANNELS):
        for b in BAND_NAMES:
            names.append(f"DE_ch{ch+1:02d}_{b}")
    for ch in range(EEG_CHANNELS):
        for param in ["Activity", "Mobility", "Complexity"]:
            names.append(f"Hjorth_ch{ch+1:02d}_{param}")
    return names

# ============================================================
# 自测（与之前相同）
# ============================================================
if __name__ == "__main__":
    import time
    DATA_DIR = r"D:\EEG\data_preprocessed_matlab"
    print("=" * 55)
    print("  feature_extract.py v4 自测（并行加速版）")
    print("=" * 55)
    print("\n[1] 读取 s01.mat ...")
    eeg, labels = load_subject(DATA_DIR, 1)
    print(f"    EEG shape : {eeg.shape}")
    print(f"    Labels    : {labels.shape}")
    print("\n[2] 提取 352 维特征（并行）...")
    t0 = time.time()
    feats = extract_features(eeg, verbose=True, n_jobs=-1)
    elapsed = time.time() - t0
    print(f"    特征 shape : {feats.shape}   （应为 (40, 352)）")
    print(f"    耗时       : {elapsed:.1f} 秒")
    nan_cnt = np.isnan(feats).sum()
    inf_cnt = np.isinf(feats).sum()
    print(f"    NaN 数量   : {nan_cnt}   （应为 0）")
    print(f"    Inf 数量   : {inf_cnt}   （应为 0）")
    print("\n[3] 各特征块数值范围检查 ...")
    psd_block = feats[:, :128]
    de_block = feats[:, 128:256]
    hjorth_block = feats[:, 256:352]
    for name, block in [("PSD   ", psd_block),
                        ("DE    ", de_block),
                        ("Hjorth", hjorth_block)]:
        print(f"    {name}  min={block.min():.4f}  max={block.max():.4f}"
              f"  mean={block.mean():.4f}")
    print("\n[4] 标签构建 ...")
    y_val, y_aro = build_labels(labels)
    print(f"    Valence : {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"    Arousal : {dict(zip(*np.unique(y_aro, return_counts=True)))}")
    names = get_feature_names()
    print(f"\n[5] 特征名称总数 : {len(names)}   （应为 352）")
    print(f"    示例（前5）  : {names[:5]}")
    print("\n" + "=" * 55)
    print("  自测完成。")
    print("=" * 55)