"""
Evasion Testing: Test model với slow scan và fake UA
So sánh detection rate giữa normal scan vs evasion techniques
"""
import pandas as pd
import numpy as np
import pickle, json, os
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD MODEL
# ==========================================
with open('../rf_flow_multiclass.pkl', 'rb') as f:
    rf_flow = pickle.load(f)
with open('../rf_payload_multiclass.pkl', 'rb') as f:
    rf_payload = pickle.load(f)
with open('../label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('../multiclass_results.json', 'r') as f:
    results = json.load(f)

BEST_ALPHA = results['best_alpha']
print(f"[*] Loaded models, alpha={BEST_ALPHA}")
print(f"[*] Classes: {list(le.classes_)}\n")

# ==========================================
# 2. FEATURE EXTRACTION (giống train)
# ==========================================
ZEEK_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
]

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

def calc_entropy(series):
    vc = series.value_counts()
    return entropy(vc) if len(vc) > 0 else 0

def extract_features(conn_log_gz):
    df = pd.read_csv(conn_log_gz, sep='\t', comment='#',
                     header=None, names=ZEEK_COLUMNS, compression='gzip')
    if df.empty:
        return pd.DataFrame()

    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df = df.replace('-', 0)
    for col in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    grouped = df.groupby(['id.orig_h', pd.Grouper(key='ts', freq='60s')])
    features = []

    for (src_ip, time_window), group in grouped:
        if len(group) < 2: continue
        total_conn = len(group)

        conn_rate             = total_conn / 60.0
        packets_per_conn      = (group['orig_pkts'] + group['resp_pkts']).mean()
        bytes_per_conn        = (group['orig_bytes'] + group['resp_bytes']).mean()
        port_entropy          = calc_entropy(group['id.resp_p'])
        ip_entropy            = calc_entropy(group['id.resp_h'])
        diff_dest_ports       = group['id.resp_p'].nunique()
        iat                   = group['ts'].diff().dt.total_seconds().dropna()
        inter_arrival_time_std = iat.std() if len(iat) > 1 else 0
        duration_std          = group['duration'].std()
        mean_duration         = group['duration'].mean()
        syn_ratio             = len(group[group['conn_state'] == 'S0']) / total_conn
        rst_ratio             = len(group[group['conn_state'].str.contains('REJ|RST', na=False)]) / total_conn
        success_rate          = len(group[group['conn_state'] == 'SF']) / total_conn
        valid_payload_ratio   = len(group[group['service'] != 0]) / total_conn
        suspicious_history_ratio = len(group[group['history'] == 'S']) / total_conn

        features.append({
            'conn_rate': conn_rate, 'packets_per_conn': packets_per_conn,
            'bytes_per_conn': bytes_per_conn, 'port_entropy': port_entropy,
            'ip_entropy': ip_entropy, 'diff_dest_ports': diff_dest_ports,
            'inter_arrival_time_std': inter_arrival_time_std,
            'duration_std': duration_std, 'mean_duration': mean_duration,
            'syn_ratio': syn_ratio, 'rst_ratio': rst_ratio,
            'success_rate': success_rate,
            'valid_payload_ratio': valid_payload_ratio,
            'suspicious_history_ratio': suspicious_history_ratio
        })

    return pd.DataFrame(features)

# ==========================================
# 3. PREDICT
# ==========================================
def predict(feat_df):
    if feat_df.empty:
        return [], []

    X_flow    = feat_df[FLOW_FEATURES].fillna(0).values
    X_payload = feat_df[PAYLOAD_FEATURES].fillna(0).values

    flow_prob    = rf_flow.predict_proba(X_flow)
    payload_prob = rf_payload.predict_proba(X_payload)
    final_prob   = BEST_ALPHA * flow_prob + (1 - BEST_ALPHA) * payload_prob
    final_pred   = np.argmax(final_prob, axis=1)

    labels     = le.inverse_transform(final_pred)
    max_probs  = np.max(final_prob, axis=1)
    return labels, max_probs

# ==========================================
# 4. CHẠY TEST
# ==========================================
print("=" * 60)
print("EVASION TESTING RESULTS")
print("=" * 60)

test_cases = {
    'E1 - Slow Scan (nmap --scan-delay 1s)': 'e1_slow_scan/conn.log.gz',
    'E2 - Fake User-Agent (nmap -sV --script)': 'e2_fake_ua/conn.log.gz',
}

evasion_results = {}

for name, path in test_cases.items():
    print(f"\n[*] Testing: {name}")
    print(f"    File: {path}")

    if not os.path.exists(path):
        print(f"    [!] File not found, skipping...")
        continue

    feat_df = extract_features(path)
    if feat_df.empty:
        print(f"    [!] Không extract được features (quá ít data)")
        continue

    print(f"    Windows extracted: {len(feat_df)}")
    labels, probs = predict(feat_df)

    # Đếm kết quả
    from collections import Counter
    counts = Counter(labels)
    scanner_detected = sum(v for k, v in counts.items() if k != 'benign')
    total = len(labels)
    detection_rate = scanner_detected / total * 100

    print(f"    Predictions: {dict(counts)}")
    print(f"    Detection Rate: {scanner_detected}/{total} = {detection_rate:.1f}%")
    print(f"    Confidence (mean): {np.mean(probs):.4f}")

    # Chi tiết từng window
    print(f"\n    Chi tiết từng window:")
    print(f"    {'Window':>8} | {'Prediction':>10} | {'Confidence':>10} | {'Detected?':>10}")
    print(f"    {'-'*50}")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        detected = "✅ YES" if label != 'benign' else "❌ NO (evasion!)"
        print(f"    {i+1:>8} | {label:>10} | {prob:>10.4f} | {detected}")

    evasion_results[name] = {
        'windows': total,
        'detection_rate': detection_rate,
        'predictions': dict(counts),
        'mean_confidence': float(np.mean(probs))
    }

# ==========================================
# 5. TỔNG KẾT
# ==========================================
print(f"\n{'='*60}")
print(f"TỔNG KẾT EVASION TESTING")
print(f"{'='*60}")
print(f"{'Kịch bản':^35} | {'Detection Rate':^15} | {'Windows':^8}")
print(f"{'-'*65}")
for name, res in evasion_results.items():
    print(f"{name:^35} | {res['detection_rate']:>13.1f}% | {res['windows']:^8}")

print(f"\n[*] Nhận xét:")
for name, res in evasion_results.items():
    dr = res['detection_rate']
    if dr >= 80:
        print(f"    {name}: Model phát hiện tốt ({dr:.1f}%) → Evasion THẤT BẠI")
    elif dr >= 50:
        print(f"    {name}: Model phát hiện được một phần ({dr:.1f}%) → Evasion HIỆU QUẢ MỘT PHẦN")
    else:
        print(f"    {name}: Model bị qua mặt ({dr:.1f}%) → Evasion THÀNH CÔNG")

# Lưu kết quả
with open('evasion_results.json', 'w') as f:
    json.dump(evasion_results, f, indent=2)
print(f"\n[✅] Đã lưu: evasion_results.json")