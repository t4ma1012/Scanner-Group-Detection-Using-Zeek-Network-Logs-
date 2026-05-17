"""
Train Multiclass với bộ Dữ liệu Khổng lồ
Tự động gộp data, chia 2 tập (Train 80% - Test 20%)
Đã fix lỗi Imbalanced Data (class_weight='balanced') & tối ưu theo F1 Macro
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle, json, warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. ĐỌC DATA VÀ CHIA TẬP TRAIN / TEST
# ==========================================
df = pd.read_csv("final_dataset_ML.csv")

def map_tool(tool):
    if tool == 'benign':       return 'benign'
    if 'masscan' in tool:      return 'masscan'
    if 'nmap' in tool:         return 'nmap'
    if 'zmap' in tool:         return 'zmap'
    return 'unknown'

df['class'] = df['tool'].apply(map_tool)
df = df[df['class'] != 'unknown']

print("=== CHUẨN BỊ DỮ LIỆU ĐÀO TẠO ===")
print(f"[*] Tổng mẫu Data: {len(df)}")
print(f"[*] Phân bố toàn bộ data:\n{df['class'].value_counts()}\n")

FLOW_FEATURES = ['conn_rate', 'packets_per_conn', 'bytes_per_conn', 'port_entropy', 'ip_entropy', 'diff_dest_ports', 'inter_arrival_time_std', 'duration_std', 'mean_duration', 'syn_ratio', 'rst_ratio', 'success_rate']
PAYLOAD_FEATURES = ['valid_payload_ratio', 'suspicious_history_ratio']

# Encode label
le = LabelEncoder()
y = le.fit_transform(df['class'])
classes = le.classes_

X_flow    = df[FLOW_FEATURES].fillna(0).values
X_payload = df[PAYLOAD_FEATURES].fillna(0).values

# CHIA 2 TẬP: TRAIN (80%) VÀ TEST (20%) - Dùng Stratify để chia đều các class
X_flow_train, X_flow_test, X_payload_train, X_payload_test, y_train, y_test = train_test_split(
    X_flow, X_payload, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[*] Đã chia 2 tập dữ liệu (Tỷ lệ 80/20):")
print(f"    - Tập TRAIN (Để AI học và tìm Alpha): {len(y_train)} mẫu")
print(f"    - Tập TEST  (Chưa từng nhìn thấy, để thi): {len(y_test)} mẫu\n")

# ==========================================
# 2. GRID SEARCH TRÊN TẬP TRAIN (Đã fix Imbalanced)
# ==========================================
n_splits = min(5, pd.Series(y_train).value_counts().min())
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# THÊM class_weight='balanced' ĐỂ ÉP AI PHẢI CHÚ Ý ĐẾN NMAP/ZMAP
rf_flow = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
rf_payload = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1)

alphas = np.arange(0.0, 1.05, 0.05)
results = []

print(f"[*] Bắt đầu Grid Search tìm tỷ lệ Ensemble trên tập TRAIN...")
print(f"{'Alpha':>8} | {'F1 Weighted':>12} | {'F1 Macro':>10} | {'Std':>8}")
print(f"{'-'*50}")

for alpha in alphas:
    fold_f1w, fold_f1m = [], []
    for train_idx, val_idx in skf.split(X_flow_train, y_train):
        rf_flow.fit(X_flow_train[train_idx], y_train[train_idx])
        rf_payload.fit(X_payload_train[train_idx], y_train[train_idx])
        
        prob_f = rf_flow.predict_proba(X_flow_train[val_idx])
        prob_p = rf_payload.predict_proba(X_payload_train[val_idx])
        
        final_prob = alpha * prob_f + (1 - alpha) * prob_p
        pred = np.argmax(final_prob, axis=1)
        
        fold_f1w.append(f1_score(y_train[val_idx], pred, average='weighted', zero_division=0))
        fold_f1m.append(f1_score(y_train[val_idx], pred, average='macro', zero_division=0))
    
    results.append({
        'alpha': round(alpha, 2),
        'f1_weighted_mean': round(np.mean(fold_f1w), 4),
        'f1_macro_mean': round(np.mean(fold_f1m), 4),
        'f1_weighted_std': round(np.std(fold_f1w), 4),
    })

# CHỌN BEST ALPHA DỰA TRÊN F1 MACRO (Bảo vệ class thiểu số)
best = max(results, key=lambda x: x['f1_macro_mean'])
best_alpha = best['alpha']

for r in results:
    marker = " ← BEST" if r['alpha'] == best['alpha'] else ""
    print(f"{r['alpha']:>8.2f} | {r['f1_weighted_mean']:>12.4f} | {r['f1_macro_mean']:>10.4f} | {r['f1_weighted_std']:>8.4f}{marker}")

print(f"\n[✅] Đã tìm ra công thức vàng: Final = {best_alpha:.2f} * Flow + {1-best_alpha:.2f} * Payload\n")

# ==========================================
# 3. KIỂM TRA THỰC TẾ TRÊN TẬP TEST (UNSEEN DATA)
# ==========================================
print("="*60)
print("=== KẾT QUẢ KIỂM TRA TRÊN TẬP TEST (BÀI THI CUỐI KỲ) ===")
print("="*60)

# Train lại mô hình bằng TOÀN BỘ tập Train
rf_flow.fit(X_flow_train, y_train)
rf_payload.fit(X_payload_train, y_train)

# Đem AI đi thi trên tập Test
test_prob_flow = rf_flow.predict_proba(X_flow_test)
test_prob_payload = rf_payload.predict_proba(X_payload_test)
test_final_prob = best_alpha * test_prob_flow + (1 - best_alpha) * test_prob_payload
y_test_pred = np.argmax(test_final_prob, axis=1)

print(classification_report(y_test, y_test_pred, target_names=classes, digits=4))

print("\n[*] Ma trận nhầm lẫn (Confusion Matrix) trên tập Test:")
cm = confusion_matrix(y_test, y_test_pred)
print(pd.DataFrame(cm, index=classes, columns=classes))

# ==========================================
# 4. LƯU MÔ HÌNH VÀ KẾT QUẢ JSON
# ==========================================
with open('rf_flow_multiclass.pkl', 'wb') as f: pickle.dump(rf_flow, f)
with open('rf_payload_multiclass.pkl', 'wb') as f: pickle.dump(rf_payload, f)
with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)

output = {
    'task': 'multiclass',
    'classes': [str(c) for c in classes],
    'best_alpha': float(best_alpha),
    'formula': f"Final = {best_alpha:.2f} x Flow + {1-best_alpha:.2f} x Payload",
    'cv_f1_macro': float(best['f1_macro_mean']),
    'n_samples': int(len(df)),
    'n_folds': int(n_splits),
    'class_distribution': {str(k): int(v) for k, v in df['class'].value_counts().to_dict().items()},
    'grid_search_results': [
        {k: float(v) if isinstance(v, (np.float64, np.float32, float)) else v for k, v in res.items()} 
        for res in results
    ]
}
with open('multiclass_results.json', 'w') as f: json.dump(output, f, indent=2)

print(f"\n[✅] Đã lưu mô hình thành công. Sẵn sàng cho bài Test Evasion!")