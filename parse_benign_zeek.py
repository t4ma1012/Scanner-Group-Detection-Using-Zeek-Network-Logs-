import pandas as pd, numpy as np, json, pickle, gzip
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

def parse_log(path, gz=False):
    import gzip as gzmod
    rows, cols = [], None
    opener = gzmod.open(path, 'rt') if gz else open(path, 'r')
    with opener as f:
        for line in f:
            line = line.strip()
            if line.startswith('#fields'):
                cols = line.split('\t')[1:]
            elif not line.startswith('#') and cols:
                parts = line.split('\t')
                if len(parts) == len(cols):
                    rows.append(parts)
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

print("[*] Loading scanner data (zeek_logs_archive/conn.log.gz)...")
scan_df = parse_log('zeek_logs_archive/conn.log.gz', gz=True)
print(f"    Rows: {len(scan_df)}")

print("[*] Loading benign data (conn.log)...")
benign_df = parse_log('conn.log', gz=False)
print(f"    Rows: {len(benign_df)}")

# Combine both
print("[*] Combining datasets...")
combined = pd.concat([scan_df, benign_df], ignore_index=True)
print(f"    Total: {len(combined)} rows")

# Rename columns
combined.rename(columns={
    'id.orig_h':'orig_h','id.orig_p':'orig_p',
    'id.resp_h':'resp_h','id.resp_p':'resp_p'
}, inplace=True)

# Convert types
for col in ['ts','orig_p','resp_p']:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')
for col in ['duration','orig_bytes','resp_bytes']:
    combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0)
combined = combined.dropna(subset=['ts'])

print(f"[✓] After cleaning: {len(combined)} rows")

# Extract 30s windows
SCAN_START = 1775878845
SCAN_END   = 1775889595

features_list = []
window_start  = int(combined['ts'].min())
wc = 0

print("[*] Extracting 30s windows...")
while window_start < combined['ts'].max():
    w = combined[(combined['ts'] >= window_start) &
                 (combined['ts'] <  window_start + 30)]
    if len(w) > 0:
        wc += 1
        features_list.append({
            'packet_count'    : len(w),
            'unique_src_ips'  : w['orig_h'].nunique(),
            'unique_dst_ips'  : w['resp_h'].nunique(),
            'unique_dst_ports': w['resp_p'].nunique(),
            'avg_duration'    : round(w['duration'].mean(), 4),
            'total_orig_bytes': int(w['orig_bytes'].sum()),
            'total_resp_bytes': int(w['resp_bytes'].sum()),
            'proto_tcp'       : (w['proto'] == 'tcp').sum(),
            'proto_udp'       : (w['proto'] == 'udp').sum(),
            'proto_icmp'      : (w['proto'] == 'icmp').sum(),
            'low_port_ratio'  : round((w['resp_p'] < 1024).sum() / len(w), 4),
            'window_start'    : window_start,
        })
    window_start += 30

feat_df = pd.DataFrame(features_list)

# Label: scanner = trong khoảng scan time
feat_df['label'] = (
    (feat_df['window_start'] >= SCAN_START) &
    (feat_df['window_start'] <= SCAN_END)
).astype(int)

scanner_count = feat_df['label'].sum()
bg_count      = len(feat_df) - scanner_count
print(f"\n[✓] Windows: {len(feat_df)}")
print(f"    Scanner   : {scanner_count}")
print(f"    Background: {bg_count}")

# Train
feature_cols = ['packet_count','unique_src_ips','unique_dst_ips',
                'unique_dst_ports','avg_duration','total_orig_bytes',
                'total_resp_bytes','proto_tcp','proto_udp','proto_icmp',
                'low_port_ratio']

X = feat_df[feature_cols].fillna(0).values
y = feat_df['label'].values

print("\n[*] Training RF with 5-fold CV...")
rf = RandomForestClassifier(
    n_estimators=150, max_depth=12,
    min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', random_state=42, n_jobs=-1,
    class_weight='balanced'
)

n_splits = min(5, min(scanner_count, bg_count))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv  = cross_validate(rf, X, y, cv=skf,
                     scoring=['f1_weighted','f1_macro','roc_auc'])

print(f"\n[✓] {n_splits}-fold CV Results:")
for m in ['f1_weighted','f1_macro','roc_auc']:
    s = cv[f'test_{m}']
    print(f"    {m:15s}: {s.mean():.4f} ± {s.std():.4f}")

rf.fit(X, y)
y_pred = rf.predict(X)
cm     = confusion_matrix(y, y_pred)
print(f"\n[✓] Confusion Matrix:")
print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
print(f"\n{classification_report(y, y_pred, target_names=['Background','Scanner'], digits=4)}")

# Save
with open('classifier_rf_week234.pkl','wb') as f:
    pickle.dump(rf, f)

results = {
    'phase': 'WEEK_2_3_COMBINED',
    'scanner_data': 'zeek_logs_archive/conn.log.gz (original)',
    'benign_data': 'conn.log (2h live generation)',
    'total_windows': int(len(feat_df)),
    'scanner_windows': int(scanner_count),
    'background_windows': int(bg_count),
    'cv_f1_weighted': float(cv['test_f1_weighted'].mean()),
    'cv_f1_weighted_std': float(cv['test_f1_weighted'].std()),
    'cv_f1_macro': float(cv['test_f1_macro'].mean()),
    'cv_f1_macro_std': float(cv['test_f1_macro'].std()),
    'cv_roc_auc': float(cv['test_roc_auc'].mean()),
    'cv_roc_auc_std': float(cv['test_roc_auc'].std()),
    'cm': {
        'tn': int(cm[0,0]),
        'fp': int(cm[0,1]),
        'fn': int(cm[1,0]),
        'tp': int(cm[1,1])
    }
}
with open('week_2_3_final_results.json','w') as f:
    json.dump(results, f, indent=2)

print(f"\n[✅] classifier_rf_week234.pkl saved")
print(f"[✅] week_2_3_final_results.json saved")
print(f"\n{json.dumps(results, indent=2)}")