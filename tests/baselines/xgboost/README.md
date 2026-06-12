# XGBoost Baseline

This directory contains the XGBoost supervised baseline for the Graphagate dataset.

## Execution

You can run this baseline using:

```bash
python -m tests.baselines.xgboost.xgboost_baseline
```

## Details

This baseline treats every event as an independent static feature vector, ignoring relational and temporal graph structures. It trains a supervised `XGBClassifier` on both benign and anomalous instances from the training split. Hyperparameters are tuned via `RandomizedSearchCV` maximizing `roc_auc`. It is constrained to not use GPU (`tree_method='hist'`) and limits multithreading to 18 cores (`n_jobs=18`). 

Metrics such as Precision, Recall, ROC-AUC, Average Precision, and per-anomaly type breakdown are calculated, using the same evaluation techniques as other baselines.
