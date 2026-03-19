#!/usr/bin/env python3
"""Verify that preprocessed data loads correctly."""

import sys
from src.dataset import LatentDataset

data_root = sys.argv[1]
ds = LatentDataset(data_root)
print(f"Dataset: {len(ds)} samples")
sample = ds[0]
for k, v in sample.items():
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
print("Preprocess verification: OK")
