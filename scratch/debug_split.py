import sys
import os
sys.path.insert(0, 'tests')
sys.path.insert(0, 'src')

import numpy as np
from datasets.uwf_zeekdata import load_uwf_stream

data, actual_train_frac, actual_val_frac = load_uwf_stream(
    'data/uwf_zeekdata24', max_benign_events=30000, max_attack_events_per_cat=3000
)

n = len(data.dst)
train_end = int(n * actual_train_frac)
val_end = train_end + int(n * actual_val_frac)

print(f"\nExact Verified Split on {n} events:")
print(f"Train: [0, {train_end}) - attacks: {data.y[:train_end].sum().item()} (should be 0)")
print(f"Val:   [{train_end}, {val_end}) - attacks: {data.y[train_end:val_end].sum().item()} (should be 0)")
print(f"Test:  [{val_end}, {n}) - attacks: {data.y[val_end:].sum().item()} (should be 7471)")

# Check types in Val
val_types = data.types[train_end:val_end].numpy()
unique, counts = np.unique(val_types, return_counts=True)
print("Types in Val (should be {0: 3000}):", dict(zip(unique, counts)))

# Check types in Test
test_types = data.types[val_end:].numpy()
unique, counts = np.unique(test_types, return_counts=True)
print("Types in Test:", dict(zip(unique, counts)))
