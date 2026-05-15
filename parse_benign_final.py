#!/usr/bin/env python3
import pandas as pd, numpy as np, json, pickle, gzip
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

ARCHIVE = 'zeek_logs_archive'
SCAN_START, SCAN_END = 1775878845, 1775889595

def parse_log(path):
    rows, cols = [], None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#fields'): cols = line.split('\t')[1:]
                elif not line.startswith('#') and cols and len(line.split('\t')) == len(cols):
                    rows.append(line.split('\t'))
    except: pass
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

print("[*] Parsing Zeek logs...")
http = parse_log(f'{ARCHIVE}/http.log')
ssl = parse_log(f'{ARCHIVE}/ssl.log')
dns = parse_log(f'{ARCHIVE}/dns.log')

if len(http) > 0: http['ts'] = pd.to_numeric(http['ts'], errors='coerce'); http = http.dropna(subset=['ts'])
if len(ssl) > 0: ssl['ts'] = pd.to_numeric(ssl['ts'], errors='coerce'); ssl = ssl.dropna(subset=['ts'])
if len(dns) > 0: dns['ts'] = pd.to_numeric(dns['ts'], errors='coerce'); dns = dns.dropna(subset=['ts'])

print(f"[✓] HTTP: {len(http)}, SSL: {len(ssl)}, DNS: {len(dns)}")

rows, cols = [], None
with gzip.open(f'{ARCHIVE}/conn.log.gz', 'rt') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#fields'): cols = line.split('\t')[1:]
        elif not line.startswith('#') and cols and len(line.split('\t')) == len(cols):
            rows.append(line.split('\t'))

conn = pd.DataFrame(rows, columns=cols)
for c in ['ts','orig_p','resp_p']: conn[c] = pd.to_numeric(conn[c], errors='coerce')
for c in ['duration','orig_bytes','resp_bytes']: conn[c] = pd.to_numeric(conn[c], errors='coerce').fillna(0)
conn = conn.dropna(subset=['ts'])
print(f"[✓] conn.log: {len(conn)} rows")

print("[*] Extracting features...")
feat, window_start = [], int(conn['ts'].min())
while window_start < conn['ts'].max():
    w = conn[(conn['ts'] >= window_start) & (conn['ts'] < window_start + 30)]
    if len(w) > 0:
        h = len(http[(http['ts'] >= window_start) & (http['ts'] < window_start + 30)]) if len(http) > 0 else 0
        s = len(ssl[(ssl['ts'] >= window_start) & (ssl['ts'] < window_start + 30)]) if len(ssl) > 0 else 0
        d = len(dns[(dns['ts'] >= window_start) & (dns['ts'] < window_start + 30)]) if len(dns) > 0 else 0
        feat.append({
            'packet_count': len(w), 'unique_src_ips': w['id.orig_h'].nunique(),
            'unique_dst_ips': w['id.resp_h'].nunique(), 'unique_dst_ports': w['id.resp_p'].nunique(),
            'avg_duration': round(w['duration'].mean(), 4), 'total_orig_bytes': int(w['orig_bytes'].sum()),
            'total_resp_bytes': int(w['resp_bytes'].sum()), 'proto_tcp': (w['proto'] == 'tcp').sum(),
            'proto_udp': (w['proto'] == 'udp').sum(), 'proto_icmp': (w['proto'] == 'icmp').sum(),
            'low_port_ratio': round((w['id.resp_p'] < 1024).sum() / len(w), 4),
            'http_activity': h, 'ssl_activity': s, 'dns_activity': d,
            'window_start': window_start
        })
    window_start += 30

feat_df = pd.DataFrame(feat)
feat_df['label'] = ((feat_df['window_start'] >= SCAN_START) & (feat_df['window_start'] <= SCAN_END)).astype(int)
scanner = feat_df['label'].sum()
bg = len(feat_df) - scanner
print(f"[✓] Features: {feat_df.shape[0]} windows | Scanner: {scanner} | Background: {bg}")

X = feat_df[['packet_count','unique_src_ips','unique_dst_ips','unique_dst_ports','avg_duration','total_orig_bytes','total_resp_bytes','proto_tcp','proto_udp','proto_icmp','low_port_ratio','http_activity','ssl_activity','dns_activity']].fillna(0).values
y = feat_df['label'].values

print("[*] Training RF with balanced weights...")
rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_validate(rf, X, y, cv=skf, scoring=['f1_weighted','f1_macro','roc_auc'])

print("\n[✓] CV Results:")
print(f"  F1 Weighted: {cv['test_f1_weighted'].mean():.4f} ± {cv['test_f1_weighted'].std():.4f}")
print(f"  F1 Macro:    {cv['test_f1_macro'].mean():.4f} ± {cv['test_f1_macro'].std():.4f}")
print(f"  ROC AUC:     {cv['test_roc_auc'].mean():.4f} ± {cv['test_roc_auc'].std():.4f}")

rf.fit(X, y)
y_pred = rf.predict(X)
cm = confusion_matrix(y, y_pred)
print(f"\n[✓] Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"\n{classification_report(y, y_pred, target_names=['Background','Scanner'], digits=4)}")

with open('classifier_rf_v3.pkl','wb') as f: pickle.dump(rf, f)
results = {
    'phase': 'WEEK_2_3_HYBRID',
    'benign_generation': '2_hours_real_traffic',
    'windows_total': int(len(feat_df)),
    'scanner_windows': int(scanner),
    'background_windows': int(bg),
    'f1_weighted': float(cv['test_f1_weighted'].mean()),
    'f1_weighted_std': float(cv['test_f1_weighted'].std()),
    'f1_macro': float(cv['test_f1_macro'].mean()),
    'f1_macro_std': float(cv['test_f1_macro'].std()),
    'roc_auc': float(cv['test_roc_auc'].mean()),
    'roc_auc_std': float(cv['test_roc_auc'].std()),
    'cm': {'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])},
    'benign_signals': {'http': int(len(http)), 'ssl': int(len(ssl)), 'dns': int(len(dns))}
}
with open('week_2_3_results_final.json','w') as f: json.dump(results, f, indent=2)
print(f"\n[✅] Saved: classifier_rf_v3.pkl + week_2_3_results_final.json")
print(json.dumps(results, indent=2))
