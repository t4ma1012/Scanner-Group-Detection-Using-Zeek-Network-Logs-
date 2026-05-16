"""
Tìm trọng số ensemble tối ưu bằng Grid Search + Stratified K-Fold CV
Giải quyết yêu cầu của thầy: phải có lý do thực nghiệm cho trọng số
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import json, warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. ĐỌC DATA
# ==========================================
df = pd.read_csv("final_dataset_ML.csv")
print(f"[*] Tổng mẫu: {len(df)}")
print(f"[*] Phân bố:\n{df['tool'].value_counts()}\n")

# ==========================================
# 2. TÁCH FLOW-BASED vs PAYLOAD-BASED FEATURES
# ==========================================
FLOW_FEATURES = [
    'conn_rate', 'packets_per_conn', 'bytes_per_conn',   # Rate-based
    'port_entropy', 'ip_entropy', 'diff_dest_ports',      # Entropy-based
    'inter_arrival_time_std', 'duration_std', 'mean_duration',  # Timing-based
    'syn_ratio', 'rst_ratio', 'success_rate'              # State-based
]

PAYLOAD_FEATURES = [
    'valid_payload_ratio',        # Tỷ lệ có payload hợp lệ (Zeek DPD)
    'suspicious_history_ratio'    # Tỷ lệ history bất thường
]

X_flow    = df[FLOW_FEATURES].fillna(0).values
X_payload = df[PAYLOAD_FEATURES].fillna(0).values
y         = df['label'].values  # binary: 0=benign, 1=scanner

print(f"[*] Flow features   : {len(FLOW_FEATURES)} features")
print(f"[*] Payload features: {len(PAYLOAD_FEATURES)} features")
print(f"[*] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

# ==========================================
# 3. GRID SEARCH TRỌNG SỐ ENSEMBLE
# ==========================================
# Dùng Stratified K-Fold để đảm bảo mỗi fold có đủ cả 2 class
n_splits = min(5, int(np.min(np.bincount(y))))  # Tránh lỗi nếu class quá ít
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"[*] Dùng {n_splits}-Fold Stratified CV")
print(f"[*] Grid search alpha từ 0.0 đến 1.0 (bước 0.05)")
print(f"{'='*60}")

# RF cho flow-based
rf_flow = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    min_samples_split=2, random_state=42, n_jobs=-1
)

# RF cho payload-based (đơn giản hơn vì chỉ có 2 features)
rf_payload = RandomForestClassifier(
    n_estimators=50, max_depth=5,
    min_samples_split=2, random_state=42, n_jobs=-1
)

results = []
alphas = np.arange(0.0, 1.05, 0.05)

for alpha in alphas:
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_flow, y)):
        # Train
        rf_flow.fit(X_flow[train_idx], y[train_idx])
        rf_payload.fit(X_payload[train_idx], y[train_idx])

        # Predict proba
        flow_prob    = rf_flow.predict_proba(X_flow[val_idx])[:, 1]
        payload_prob = rf_payload.predict_proba(X_payload[val_idx])[:, 1]

        # Ensemble: Final = alpha * Flow + (1-alpha) * Payload
        final_prob = alpha * flow_prob + (1 - alpha) * payload_prob
        final_pred = (final_prob >= 0.5).astype(int)

        f1 = f1_score(y[val_idx], final_pred, average='weighted', zero_division=0)
        fold_f1s.append(f1)

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    results.append({
        'alpha': round(alpha, 2),
        'f1_mean': round(mean_f1, 4),
        'f1_std': round(std_f1, 4)
    })

# ==========================================
# 4. IN KẾT QUẢ
# ==========================================
print(f"\n{'Alpha (Flow)':>12} | {'1-Alpha (Payload)':>17} | {'F1 Mean':>8} | {'F1 Std':>8}")
print(f"{'-'*60}")
for r in results:
    marker = " ← BEST" if r == max(results, key=lambda x: x['f1_mean']) else ""
    print(f"{r['alpha']:>12.2f} | {1-r['alpha']:>17.2f} | {r['f1_mean']:>8.4f} | {r['f1_std']:>8.4f}{marker}")

# ==========================================
# 5. CHỌN TRỌNG SỐ TỐT NHẤT
# ==========================================
best = max(results, key=lambda x: x['f1_mean'])
best_alpha = best['alpha']

print(f"\n{'='*60}")
print(f"[✅] TRỌNG SỐ TỐT NHẤT:")
print(f"     alpha (flow)    = {best_alpha:.2f}")
print(f"     1-alpha (payload) = {1-best_alpha:.2f}")
print(f"     F1 Score = {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
print(f"\n[✅] CÔNG THỨC ENSEMBLE:")
print(f"     Final = {best_alpha:.2f} × Flow_Prob + {1-best_alpha:.2f} × Payload_Prob")
print(f"{'='*60}")

# ==========================================
# 6. TRAIN MODEL CUỐI CÙNG VỚI TOÀN BỘ DATA
# ==========================================
print(f"\n[*] Training final model với toàn bộ {len(df)} mẫu...")
rf_flow.fit(X_flow, y)
rf_payload.fit(X_payload, y)

# Predict trên toàn bộ data để xem confusion matrix
flow_prob_all    = rf_flow.predict_proba(X_flow)[:, 1]
payload_prob_all = rf_payload.predict_proba(X_payload)[:, 1]
final_prob_all   = best_alpha * flow_prob_all + (1 - best_alpha) * payload_prob_all
final_pred_all   = (final_prob_all >= 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print(f"\n[*] Classification Report (train set - chỉ để tham khảo):")
print(classification_report(y, final_pred_all,
      target_names=['Benign', 'Scanner'], digits=4))

print(f"[*] Feature Importance (Flow-based):")
for name, imp in sorted(zip(FLOW_FEATURES, rf_flow.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"    {name:30s}: {imp:.4f}")

print(f"\n[*] Feature Importance (Payload-based):")
for name, imp in sorted(zip(PAYLOAD_FEATURES, rf_payload.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"    {name:30s}: {imp:.4f}")

# ==========================================
# 7. LƯU KẾT QUẢ
# ==========================================
import pickle
with open('rf_flow.pkl', 'wb') as f:
    pickle.dump(rf_flow, f)
with open('rf_payload.pkl', 'wb') as f:
    pickle.dump(rf_payload, f)

output = {
    'best_alpha': best_alpha,
    'formula': f"Final = {best_alpha:.2f} × Flow_Prob + {1-best_alpha:.2f} × Payload_Prob",
    'cv_results': results,
    'n_folds': n_splits,
    'n_samples': len(df),
    'flow_features': FLOW_FEATURES,
    'payload_features': PAYLOAD_FEATURES,
    'best_f1': best['f1_mean'],
    'best_f1_std': best['f1_std']
}
with open('ensemble_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[✅] Đã lưu: rf_flow.pkl, rf_payload.pkl, ensemble_results.json")
print(f"[✅] Trọng số có cơ sở thực nghiệm: alpha={best_alpha:.2f} cho F1={best['f1_mean']:.4f}")