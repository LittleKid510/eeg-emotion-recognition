import os
import numpy as np
import pickle

CACHE_DIR = r"D:\EEG\cache"
OUTPUT_DIR = r"D:\EEG\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_features = []
all_val_labels = []
all_aro_labels = []
all_subj_ids = []

for subj_id in range(1, 33):
    cache_path = os.path.join(CACHE_DIR, f"s{subj_id:02d}_features.pkl")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    feats = data["features"]          # (40, 352) 或 (40,480)
    y_val = data["y_val"]
    y_aro = data["y_aro"]
    all_features.append(feats)
    all_val_labels.append(y_val)
    all_aro_labels.append(y_aro)
    all_subj_ids.append(np.full(40, subj_id - 1))

X_all = np.vstack(all_features)
y_val_all = np.hstack(all_val_labels)
y_aro_all = np.hstack(all_aro_labels)
subject_ids = np.hstack(all_subj_ids)

np.save(os.path.join(OUTPUT_DIR, "X_all.npy"), X_all)
np.save(os.path.join(OUTPUT_DIR, "y_val_all.npy"), y_val_all)
np.save(os.path.join(OUTPUT_DIR, "y_aro_all.npy"), y_aro_all)
np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), subject_ids)

print(f"Saved to {OUTPUT_DIR}")
print(f"X_all shape: {X_all.shape}")