#!/usr/bin/env python3
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
import pickle, json, datetime

SCAN_START = 1775878845
SCAN_END   = 1775889595

print(f"[*] Scan window: {datetime.datetime.utcfromtimestamp(SCAN_START)} → {datetime.datetime.utcfromtimestamp(SCAN_END)} UTC")

feat_df = pd.read_csv('features.csv')
print(f"[✓] Loaded: {feat_df.shape}")

feat_df['label'] = (
    (feat_df['window_start'] >= SCAN_START) &
    (feat_df['window_start'] <= SCAN_END)
).astype(int)

scanner_count = feat_df['label'].sum()
bg_count      = len(feat_df) - scanner_count
print(f"[*] Scanner   : {scanner_count} windows")
print(f"[*] Background: {bg_count} windows")

feature_cols = ['packet_count','unique_src_ips','unique_dst_ips',
                'unique_dst_ports','avg_duration','total_orig_bytes',
                'total_resp_bytes','proto_tcp','proto_udp','proto_icmp',
                'low_port_ratio']

X = feat_df[feature_cols].fillna(0).values
y = feat_df['label'].values

rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=8, min_samples_leaf=3,
    max_features='sqrt', random_state=42, n_jobs=-1
)

n_splits = min(5, min(scanner_count, bg_count))
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv  = cross_validate(rf, X, y, cv=skf,
                     scoring=['f1_weighted','f1_macro','roc_auc'])

print(f"\n[✓] {n_splits}-fold CV:")
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

top5 = sorted(zip(rf.feature_importances_, feature_cols), reverse=True)[:5]
print("[*] Top-5 Features:")
for imp, name in top5:
    print(f"    {name:20s}: {imp:.4f}")

with open('classifier_rf.pkl','wb') as f:
    pickle.dump(rf, f)

results = {
    'labeling'  : 'timestamp-based (no leakage)',
    'scan_start': SCAN_START,
    'scan_end'  : SCAN_END,
    'n_windows' : int(len(feat_df)),
    'scanner'   : int(scanner_count),
    'background': int(bg_count),
    'cv_f1'     : float(cv['test_f1_weighted'].mean()),
    'cv_f1_std' : float(cv['test_f1_weighted'].std()),
    'cv_roc_auc': float(cv['test_roc_auc'].mean()),
    'cm'        : {'TN':int(cm[0,0]),'FP':int(cm[0,1]),
                   'FN':int(cm[1,0]),'TP':int(cm[1,1])},
    'top5'      : [(n, round(float(i),4)) for i,n in top5]
}
with open('week_1_results.json','w') as f:
    json.dump(results, f, indent=2)

print(f"\n[✅] classifier_rf.pkl saved")
print(f"[✅] week_1_results.json saved")
print(f"\n{json.dumps(results, indent=2)}")
