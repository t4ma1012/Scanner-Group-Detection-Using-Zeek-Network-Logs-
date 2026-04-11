=== WEEK 1 BACKUP ===
Date: Apr 11 2026
Project: Scanner Grouping from Zeek Logs

ARTIFACTS:
- classifier_rf.pkl       : Trained Random Forest (F1=0.8734)
- features.csv            : 226 windows × 12 features
- week_1_results.json     : Full metrics + confusion matrix

SCRIPTS:
- train_model_week1.py    : Training with timestamp-based labeling
- extract_features_week1.py : Feature extraction from conn.log

STATISTICS:
- Scanner windows    : 206 (04:40:45 → 06:19:55 UTC)
- Background windows : 20
- CV F1              : 0.8734 ± 0.0197
- ROC AUC            : 0.8780 ± 0.0554
- Recall (Scanner)   : 1.0000 (0 miss)
- Precision (Scanner): 0.9406

NEXT: WEEK 2 (Payload Fingerprinting)
