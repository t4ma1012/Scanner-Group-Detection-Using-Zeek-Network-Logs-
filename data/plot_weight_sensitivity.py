import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DỮ LIỆU & ĐẶC TRƯNG
# ==========================================
df = pd.read_csv("final_dataset_ML.csv")

flow_features = [
    'conn_rate', 'packets_per_conn', 'bytes_per_conn', 'port_entropy',
    'ip_entropy', 'diff_dest_ports', 'inter_arrival_time_std', 'duration_std',
    'mean_duration', 'syn_ratio', 'rst_ratio', 'success_rate'
]
payload_features = ['valid_payload_ratio', 'suspicious_history_ratio']

X_flow = df[flow_features]
X_payload = df[payload_features]
y = df['label']

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ CV
# ==========================================
rf_flow = RandomForestClassifier(n_estimators=50, random_state=42)
lr_payload = LogisticRegression(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Chạy với bước nhảy siêu nhỏ 0.01 (101 điểm ảnh) để thấy rõ hiệu ứng bậc thang
weights = np.linspace(0.0, 1.0, 101) 
f1_scores = []

print(f"Đang chạy thực nghiệm với 101 mốc trọng số (bước nhảy 0.01) trên {len(df)} mẫu...")

# ==========================================
# 3. QUÉT TRỌNG SỐ SIÊU CHI TIẾT
# ==========================================
for w_flow in weights:
    w_payload = 1.0 - w_flow
    fold_f1 = []

    for train_idx, val_idx in skf.split(X_flow, y):
        X_f_train, X_f_val = X_flow.iloc[train_idx], X_flow.iloc[val_idx]
        X_p_train, X_p_val = X_payload.iloc[train_idx], X_payload.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        rf_flow.fit(X_f_train, y_train)
        lr_payload.fit(X_p_train, y_train)

        prob_flow = rf_flow.predict_proba(X_f_val)
        prob_payload = lr_payload.predict_proba(X_p_val)

        final_prob = (w_flow * prob_flow) + (w_payload * prob_payload)
        y_pred = np.argmax(final_prob, axis=1)

        fold_f1.append(f1_score(y_val, y_pred, average='weighted'))

    f1_scores.append(np.mean(fold_f1))

# ==========================================
# 4. VẼ BIỂU ĐỒ CHỨNG MINH
# ==========================================
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Vẽ đường F1-Score
plt.plot(weights, f1_scores, marker='', color='b', linewidth=2.5, label='F1-Score Model')

# Đánh dấu các mốc 0.05 bằng các đường dọc mờ (Grid)
for i in np.linspace(0.0, 1.0, 21):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)

plt.title('Biểu đồ Phân tích Độ nhạy Trọng số (Step = 0.01)', fontsize=14, fontweight='bold')
plt.xlabel('Trọng số của Flow-based Model (w_flow)', fontsize=12)
plt.ylabel('F1-Score Trung bình (5-Fold CV)', fontsize=12)
plt.xticks(np.linspace(0.0, 1.0, 11)) # Hiển thị nhãn trục X mỗi 0.1
plt.legend()
plt.tight_layout()

# Lưu biểu đồ
output_file = 'weight_sensitivity_curve.png'
plt.savefig(output_file, dpi=300)
print(f"\n[V] Đã lưu biểu đồ làm bằng chứng tại: {output_file}")