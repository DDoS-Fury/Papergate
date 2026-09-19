import sys
import os
sys.path.insert(0, 'tests')
sys.path.insert(0, 'src')

import numpy as np
import torch
from datasets.uwf_zeekdata import load_uwf_stream
from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn

# Quick 1-epoch test on a small subset to inspect the exact scores
data = load_uwf_stream('data/uwf_zeekdata24', max_benign_events=3000, max_attack_events_per_cat=300)

n = len(data.dst)
# Compute exact train_frac and val_frac
n_train_b = 2100  # 70% of 3000
n_val_b = 300     # 10% of 300
train_frac = n_train_b / n
val_frac = n_val_b / n

print(f"n={n}, train_frac={train_frac:.4f}, val_frac={val_frac:.4f}")
print(f"train_end = {int(n * train_frac)}, val_end = {int(n * train_frac) + int(n * val_frac)}")

cfg = TGNConfig(epochs=1, batch_size=128, eval_batch_size=256, train_frac=train_frac, val_frac=val_frac)

metrics = train_tgn(cfg, dataset=data, save=False)
print("\nMetrics with exact split:")
print(f"Aggregate AUC: {metrics['agg_auc']:.4f}")
print(f"Aggregate AP:  {metrics['agg_ap']:.4f}")
for k, v in metrics["per_type"].items():
    print(f"  {k}: AUC={v['auc']:.4f}, AP={v['ap']:.4f}, n={v['n']}")
