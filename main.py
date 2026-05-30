"""
main.py
============================
DEAP EEG 情绪识别

运行前确认：
    feature_extract.py 在同一目录下
    DATA_DIR 路径正确
    所有包已安装：sklearn / xgboost / shap / scipy / numpy
"""

import os
import json
import time
import re
import warnings
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier

# 导入特征提取模块（必须在同一目录）
from feature_extract import load_subject, build_labels, extract_features, get_feature_names

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
DATA_DIR    = r"D:\EEG\data_preprocessed_matlab"
RESULTS_DIR = r"D:\EEG\results"
CACHE_DIR   = r"D:\EEG\cache"
N_SUBJECTS  = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

# ============================================================
# 模型 & 超参数搜索空间（已大幅缩减）
# ============================================================
def get_models():
    """
    返回 {模型名: (estimator, param_grid)} 字典。
    GridSearchCV 内部分数折交叉验证（缩减为 3 折）寻优。
    """
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 注意：SVM 初始设为 probability=False，避免内部嵌套校准
    models = {
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 7, 9]}
        ),
        "SVM": (
            SVC(probability=False, random_state=42),
            {"C": [1, 10],
             "gamma": ["scale", 0.01],
             "kernel": ["rbf"]}
        ),
        "RF": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200],
             "max_depth": [5, 10, None]}
        ),
        "XGB": (
            XGBClassifier(eval_metric="logloss",
                          random_state=42,
                          n_jobs=-1,
                          verbosity=0),
            {"max_depth": [3, 5, 7],
             "learning_rate": [0.05, 0.1],
             "n_estimators": [100, 200],
             "subsample": [0.8],
             "colsample_bytree": [0.8],
             "scale_pos_weight": [1]}   # 若标签不平衡可改为 [1, 1.5]，但会增大搜索量
        ),
    }
    return models, inner_cv


# ============================================================
# 特征缓存（避免重复计算，节省时间）
# ============================================================
def get_cached_features(subject_id: int):
    """若 cache 存在则直接读取，否则重新提取并保存。"""
    cache_path = os.path.join(CACHE_DIR, f"s{subject_id:02d}_features.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 重新提取
    eeg, raw_labels = load_subject(DATA_DIR, subject_id)
    features        = extract_features(eeg, verbose=False)
    y_val, y_aro    = build_labels(raw_labels)

    data = {"features": features, "y_val": y_val, "y_aro": y_aro}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data


# ============================================================
# 单次 LOSO 轮训练 & 评估（优化版）
# ============================================================
def run_one_fold(X_train, y_train, X_test, y_test,
                 model_name, estimator, param_grid, inner_cv):
    """
    一次 LOSO fold：
    1. StandardScaler（fit on train）
    2. PCA（保留 95% 方差，fit on train）
    3. GridSearchCV（在训练集上 3 折寻优）
    4. 评估（on test）
    返回指标字典
    """
    # --- 1. 标准化 ---
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # --- 2. PCA 降维（保留 95% 方差） ---
    pca = PCA(n_components=0.95)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    # --- 3. GridSearchCV（使用 PCA 后的特征） ---
    grid = GridSearchCV(estimator, param_grid,
                        cv=inner_cv, scoring="f1_macro",
                        n_jobs=-1, refit=True)
    grid.fit(X_tr_pca, y_train)
    best_params = grid.best_params_

    # --- 4. 最终模型（处理 SVM 概率） ---
    if model_name == "SVM":
        # SVM 需要用 probability=True 训练以获取概率
        best_model = SVC(probability=True, **best_params)
        best_model.fit(X_tr_pca, y_train)
    else:
        # KNN / RF / XGB 均可直接从 GridSearchCV 获取
        best_model = grid.best_estimator_

    # --- 5. 预测与评估 ---
    y_pred = best_model.predict(X_te_pca)
    try:
        y_prob = best_model.predict_proba(X_te_pca)[:, 1]
    except Exception:
        y_prob = None

    metrics = {
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":     recall_score(y_test, y_pred,    average="macro", zero_division=0),
        "f1":         f1_score(y_test, y_pred,        average="macro", zero_division=0),
        "auc":        roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5,
        "best_params": best_params,
    }
    return metrics, best_model, (scaler, pca)


# ============================================================
# LOSO 主循环（串行，每个 fold 内部并行）
# ============================================================
def run_loso(task: str, all_features: list, all_labels: list):
    """
    task    : "valence" 或 "arousal"
    返回：{model_name: [32个指标字典, ...]}
    """
    models, inner_cv = get_models()
    results = {m: [] for m in models}

    print(f"\n{'='*60}")
    print(f"  任务：{task.upper()}   （LOSO, {N_SUBJECTS} folds）")
    print(f"{'='*60}")

    for test_subj in range(N_SUBJECTS):
        print(f"\n  Fold {test_subj+1:02d}/{N_SUBJECTS}  "
              f"（测试被试 s{test_subj+1:02d}）", end="  ")

        # 构建训练集 & 测试集
        X_test  = all_features[test_subj]
        y_test  = all_labels[test_subj]
        X_train = np.vstack([all_features[i]
                             for i in range(N_SUBJECTS) if i != test_subj])
        y_train = np.hstack([all_labels[i]
                             for i in range(N_SUBJECTS) if i != test_subj])

        fold_line = []
        for model_name, (estimator, param_grid) in models.items():
            t0 = time.time()
            metrics, _, _ = run_one_fold(
                X_train, y_train, X_test, y_test,
                model_name, estimator, param_grid, inner_cv
            )
            elapsed = time.time() - t0
            results[model_name].append(metrics)
            fold_line.append(f"{model_name}={metrics['accuracy']:.3f}")

        print("  |  " + "   ".join(fold_line))

    return results


# ============================================================
# 统计汇总
# ============================================================
def summarize(results: dict, task: str) -> dict:
    summary = {}
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]

    print(f"\n{'='*60}")
    print(f"  {task.upper()} 最终结果（mean ± std, N={N_SUBJECTS} folds）")
    print(f"{'='*60}")
    print(f"  {'Model':<8}  {'Accuracy':>12}  {'F1':>12}  {'AUC':>12}")
    print(f"  {'-'*52}")

    for model_name, fold_list in results.items():
        stats = {}
        for k in metric_keys:
            vals = [f[k] for f in fold_list]
            stats[k] = {"mean": float(np.mean(vals)),
                        "std":  float(np.std(vals))}
        summary[model_name] = stats

        acc = stats["accuracy"]
        f1  = stats["f1"]
        auc = stats["auc"]
        print(f"  {model_name:<8}  "
              f"{acc['mean']*100:>6.2f}±{acc['std']*100:.2f}%  "
              f"{f1['mean']*100:>6.2f}±{f1['std']*100:.2f}%  "
              f"{auc['mean']:>6.4f}±{auc['std']:.4f}")

    print(f"{'='*60}")
    return summary


# ============================================================
# SHAP 分析（在 PCA 降维前的原始 352 维特征上分析）
# ============================================================
def run_shap_analysis(all_features: list, all_labels: list,
                      task: str, top_n: int = 10):
    """
    使用 SHAP 分析特征重要性（基于 XGBoost 全局模型）
    在全部数据上训练 XGBoost，计算 SHAP 值，输出 Top-N 重要特征
    """
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 未安装 shap，跳过。运行: pip install shap")
        return None

    print(f"\n[SHAP] 计算 {task.upper()} 任务的特征重要性...")

    X_all = np.vstack(all_features)
    y_all = np.hstack(all_labels)

    # 标准化（与 LOSO 内部一致）
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)

    # 训练一个 XGBoost（使用一组合理的超参数，无需网格搜索）
    xgb = XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=200,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_sc, y_all)

    # 创建 TreeExplainer
    explainer = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_sc)

    # 计算平均绝对 SHAP 值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    # 获取特征名称
    feat_names = get_feature_names()  # 已在 feature_extract.py 中定义

    print(f"\n  Top-{top_n} 重要特征（{task.upper()}）：")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:>2}. {feat_names[idx]:<35}  mean|SHAP| = {mean_abs_shap[idx]:.5f}")

    return {
        "top_features": [feat_names[i] for i in top_idx],
        "top_shap_mean": mean_abs_shap[top_idx].tolist(),
    }


# ============================================================
# 保存结果
# ============================================================
def save_results(valence_summary, arousal_summary,
                 valence_shap, arousal_shap,
                 valence_folds, arousal_folds):
    out = {
        "valence": {"summary": valence_summary, "folds": valence_folds},
        "arousal": {"summary": arousal_summary, "folds": arousal_folds},
        "shap":    {"valence": valence_shap, "arousal": arousal_shap},
    }

    def convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    path = os.path.join(RESULTS_DIR, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\n[INFO] 完整结果已保存到：{path}")


# ============================================================
# 主程序
# ============================================================
def main():
    total_start = time.time()

    print("=" * 60)
    print("  DEAP EEG 情绪识别 — LOSO 论文实验（v3.2，加速版）")
    print("  NexSymp 2026")
    print("=" * 60)

    # ---- Step 1：预提取全部被试特征 ----
    print(f"\n[Step 1] 加载 / 提取 {N_SUBJECTS} 个被试的特征（352 维）...")
    all_features   = []
    all_val_labels = []
    all_aro_labels = []

    for sid in range(1, N_SUBJECTS + 1):
        data = get_cached_features(sid)
        all_features.append(data["features"])
        all_val_labels.append(data["y_val"])
        all_aro_labels.append(data["y_aro"])
        if sid % 8 == 0 or sid == N_SUBJECTS:
            print(f"  s{sid:02d}/{N_SUBJECTS} 完成")

    print(f"\n  特征维度确认：{all_features[0].shape}  （实际 {all_features[0].shape[1]} 维）")
    print(f"  Valence 总样本：{sum(len(y) for y in all_val_labels)}")
    print(f"  Arousal 总样本：{sum(len(y) for y in all_aro_labels)}")

    # ---- Step 2：LOSO 实验 ----
    print(f"\n[Step 2] LOSO 实验（共 {N_SUBJECTS} folds × 4 模型 × 2 任务）")

    val_results  = run_loso("valence", all_features, all_val_labels)
    val_summary  = summarize(val_results, "valence")

    aro_results  = run_loso("arousal", all_features, all_aro_labels)
    aro_summary  = summarize(aro_results, "arousal")

    # ---- Step 3：SHAP 分析 ----
    print(f"\n[Step 3] SHAP 特征重要性分析 ...")
    val_shap = run_shap_analysis(all_features, all_val_labels, "valence")
    aro_shap = run_shap_analysis(all_features, all_aro_labels, "arousal")

    # ---- Step 4：保存结果 ----
    print(f"\n[Step 4] 保存结果 ...")
    def fold_to_serializable(results_dict):
        out = {}
        for model, folds in results_dict.items():
            out[model] = []
            for f in folds:
                row = {k: (float(v) if isinstance(v, (float, np.floating))
                           else v) for k, v in f.items()}
                out[model].append(row)
        return out

    save_results(
        val_summary, aro_summary,
        val_shap,    aro_shap,
        fold_to_serializable(val_results),
        fold_to_serializable(aro_results),
    )

    # ---- 最终汇总打印（论文 Table 1 格式）----
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"论文 Table 1 — 模型性能对比（LOSO，mean±std）")
    print(f"{'='*60}")
    print(f"  {'Model':<8} | {'Valence Acc':>13} {'Valence F1':>12} "
          f"| {'Arousal Acc':>13} {'Arousal F1':>12}")
    print(f"  {'-'*62}")
    for model in ["KNN", "SVM", "RF", "XGB"]:
        va = val_summary[model]["accuracy"]
        vf = val_summary[model]["f1"]
        aa = aro_summary[model]["accuracy"]
        af = aro_summary[model]["f1"]
        print(f"  {model:<8} | "
              f"{va['mean']*100:>6.2f}±{va['std']*100:.2f}%   "
              f"{vf['mean']*100:>6.2f}±{vf['std']*100:.2f}%   | "
              f"{aa['mean']*100:>6.2f}±{aa['std']*100:.2f}%   "
              f"{af['mean']*100:>6.2f}±{af['std']*100:.2f}%")
    print(f"\n  总耗时：{total_elapsed/60:.1f} 分钟")
    print(f"  结果文件：{RESULTS_DIR}\\results.json")
    print(f"{'='*60}")
    print("实验完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()