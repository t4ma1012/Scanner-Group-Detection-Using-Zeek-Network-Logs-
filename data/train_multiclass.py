"""
Train Multiclass Random Forest: nmap / masscan / zmap / benign
Có cross-validation và grid search ensemble weight
Đã fix lỗi JSON serializable
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle, json, warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. ĐỌC DATA
# ==========================================
df = pd.read_csv("final_dataset_ML.csv")

# Gom nhóm tool thành 4 class
def map_tool(tool):
    if tool == 'benign':       return 'benign'
    if 'masscan' in tool:      return 'masscan'
    if 'nmap' in tool:         return 'nmap'
    if 'zmap' in tool:         return 'zmap'
    return 'unknown'

df['class'] = df['tool'].apply(map_tool)
df = df[df['class'] != 'unknown']

print("=== MULTICLASS CLASSIFICATION ===")
print(f"[*] Tổng mẫu: {len(df)}")
print(f"[*] Phân bố class:\n{df['class'].value_counts()}\n")

# ==========================================
# 2. FEATURES
# ==========================================
FLOW_FEATURES = [
    'conn_rate', 'packets_per_conn', 'bytes_per_conn',
    'port_entropy', 'ip_entropy', 'diff_dest_ports',
    'inter_arrival_time_std', 'duration_std', 'mean_duration',
    'syn_ratio', 'rst_ratio', 'success_rate'
]
PAYLOAD_FEATURES = [
    'valid_payload_ratio',
    'suspicious_history_ratio'
]
ALL_FEATURES = FLOW_FEATURES + PAYLOAD_FEATURES

# Encode label
le = LabelEncoder()
y = le.fit_transform(df['class'])
classes = le.classes_
print(f"[*] Classes: {list(classes)}")
print(f"[*] Encoded: {dict(zip(classes, range(len(classes))))}\n")

X_flow    = df[FLOW_FEATURES].fillna(0).values
X_payload = df[PAYLOAD_FEATURES].fillna(0).values

# ==========================================
# 3. CROSS-VALIDATION TỪNG MODEL
# ==========================================
n_splits = min(5, df['class'].value_counts().min())
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"[*] {n_splits}-Fold Stratified CV")
print("="*60)

# RF Flow-based
rf_flow = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    min_samples_split=2, random_state=42, n_jobs=-1
)
cv_flow = cross_validate(rf_flow, X_flow, y, cv=skf,
                         scoring=['f1_weighted', 'f1_macro'])
print(f"[Flow-based RF]")
print(f"  F1 Weighted: {cv_flow['test_f1_weighted'].mean():.4f} ± {cv_flow['test_f1_weighted'].std():.4f}")
print(f"  F1 Macro   : {cv_flow['test_f1_macro'].mean():.4f} ± {cv_flow['test_f1_macro'].std():.4f}")

# RF Payload-based
rf_payload = RandomForestClassifier(
    n_estimators=50, max_depth=5,
    min_samples_split=2, random_state=42, n_jobs=-1
)
cv_payload = cross_validate(rf_payload, X_payload, y, cv=skf,
                            scoring=['f1_weighted', 'f1_macro'])
print(f"\n[Payload-based RF]")
print(f"  F1 Weighted: {cv_payload['test_f1_weighted'].mean():.4f} ± {cv_payload['test_f1_weighted'].std():.4f}")
print(f"  F1 Macro   : {cv_payload['test_f1_macro'].mean():.4f} ± {cv_payload['test_f1_macro'].std():.4f}")

# ==========================================
# 4. GRID SEARCH ENSEMBLE WEIGHT
# ==========================================
print(f"\n[*] Grid Search Ensemble Weight...")
print(f"{'Alpha':>8} | {'F1 Weighted':>12} | {'F1 Macro':>10} | {'Std':>8}")
print(f"{'-'*50}")

results = []
alphas = np.arange(0.0, 1.05, 0.05)

for alpha in alphas:
    fold_f1w, fold_f1m = [], []

    for train_idx, val_idx in skf.split(X_flow, y):
        rf_flow.fit(X_flow[train_idx], y[train_idx])
        rf_payload.fit(X_payload[train_idx], y[train_idx])

        flow_prob    = rf_flow.predict_proba(X_flow[val_idx])
        payload_prob = rf_payload.predict_proba(X_payload[val_idx])

        final_prob = alpha * flow_prob + (1 - alpha) * payload_prob
        final_pred = np.argmax(final_prob, axis=1)

        fold_f1w.append(f1_score(y[val_idx], final_pred, average='weighted', zero_division=0))
        fold_f1m.append(f1_score(y[val_idx], final_pred, average='macro', zero_division=0))

    results.append({
        'alpha': round(alpha, 2),
        'f1_weighted_mean': round(np.mean(fold_f1w), 4),
        'f1_macro_mean': round(np.mean(fold_f1m), 4),
        'f1_weighted_std': round(np.std(fold_f1w), 4),
    })

best = max(results, key=lambda x: x['f1_weighted_mean'])
for r in results:
    marker = " ← BEST" if r['alpha'] == best['alpha'] else ""
    print(f"{r['alpha']:>8.2f} | {r['f1_weighted_mean']:>12.4f} | {r['f1_macro_mean']:>10.4f} | {r['f1_weighted_std']:>8.4f}{marker}")

best_alpha = best['alpha']
print(f"\n{'='*60}")
print(f"[✅] BEST: alpha={best_alpha:.2f}")
print(f"     Final = {best_alpha:.2f} × Flow_Prob + {1-best_alpha:.2f} × Payload_Prob")
print(f"     F1 Weighted = {best['f1_weighted_mean']:.4f} ± {best['f1_weighted_std']:.4f}")
print(f"     F1 Macro    = {best['f1_macro_mean']:.4f}")

# ==========================================
# 5. TRAIN FINAL MODEL
# ==========================================
print(f"\n[*] Training final model...")
rf_flow.fit(X_flow, y)
rf_payload.fit(X_payload, y)

flow_prob_all    = rf_flow.predict_proba(X_flow)
payload_prob_all = rf_payload.predict_proba(X_payload)
final_prob_all   = best_alpha * flow_prob_all + (1 - best_alpha) * payload_prob_all
final_pred_all   = np.argmax(final_prob_all, axis=1)

print(f"\n[*] Classification Report:")
print(classification_report(y, final_pred_all,
      target_names=classes, digits=4))

print(f"[*] Confusion Matrix:")
cm = confusion_matrix(y, final_pred_all)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df)

print(f"\n[*] Feature Importance (Flow-based Top 5):")
for name, imp in sorted(zip(FLOW_FEATURES, rf_flow.feature_importances_),
                         key=lambda x: -x[1])[:5]:
    print(f"    {name:30s}: {imp:.4f}")

# ==========================================
# 6. LƯU MODEL VÀ KẾT QUẢ JSON (ĐÃ FIX LỖI ÉP KIỂU)
# ==========================================
with open('rf_flow_multiclass.pkl', 'wb') as f:
    pickle.dump(rf_flow, f)
with open('rf_payload_multiclass.pkl', 'wb') as f:
    pickle.dump(rf_payload, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

output = {
    'task': 'multiclass',
    'classes': [str(c) for c in classes],
    'best_alpha': float(best_alpha),
    'formula': f"Final = {best_alpha:.2f} x Flow + {1-best_alpha:.2f} x Payload",
    'cv_f1_weighted': float(best['f1_weighted_mean']),
    'cv_f1_weighted_std': float(best['f1_weighted_std']),
    'cv_f1_macro': float(best['f1_macro_mean']),
    'n_samples': int(len(df)),
    'n_folds': int(n_splits),
    'class_distribution': {str(k): int(v) for k, v in df['class'].value_counts().to_dict().items()},
    'grid_search_results': [
        {k: float(v) if isinstance(v, (np.float64, np.float32, float)) else v for k, v in res.items()} 
        for res in results
    ]
}

with open('multiclass_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[✅] Đã lưu: rf_flow_multiclass.pkl, rf_payload_multiclass.pkl")
print(f"[✅] Đã lưu: label_encoder.pkl, multiclass_results.json")
print(f"[✅] Công thức: Final = {best_alpha:.2f} × Flow + {1-best_alpha:.2f} × Payload")