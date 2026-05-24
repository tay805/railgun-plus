"""Stage 3b: a PyTorch Dataset over converted (F_in, F_out, mask) samples.

Two modes:
  - in-memory: pass a list of Instance objects; samples are materialized lazily.
  - on-disk:   point at a directory of .npz shards produced by
               scripts/generate_data.py (recommended for Colab — generate once,
               save to Drive, reload across sessions).
"""
from __future__ import annotations

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import instance_to_samples, normalize_features
from .types import Instance


class InMemoryMapfDataset(Dataset):
    """Materializes all (F_in, F_out, mask) samples from given instances.

    Fine for small/medium data. For large datasets use the on-disk shards.
    """
    def __init__(self, instances: list[Instance], normalize: bool = True):
        self.samples = []
        for inst in instances:
            for F_in, F_out, mask in instance_to_samples(inst):
                if normalize:
                    F_in = normalize_features(F_in)
                self.samples.append((F_in, F_out, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        F_in, F_out, mask = self.samples[i]
        return (torch.from_numpy(F_in),
                torch.from_numpy(F_out),
                torch.from_numpy(mask))


class ShardedMapfDataset(Dataset):
    """Loads samples from .npz shards on disk (e.g. Google Drive).

    Each shard is an .npz with arrays: F_in (N,k,H,W), F_out (N,H,W),
    mask (N,H,W). All shards must share H,W (group by map size, or pad to a
    common size before saving).
    """
    def __init__(self, shard_dir: str):
        self.files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz shards in {shard_dir}")
        # Build an index: (file_idx, within_idx)
        self.index = []
        self._cache = {}
        for fi, f in enumerate(self.files):
            with np.load(f) as d:
                n = d["F_out"].shape[0]
            self.index.extend((fi, j) for j in range(n))

    def _load(self, fi):
        if fi not in self._cache:
            # keep at most one shard cached to bound memory
            self._cache.clear()
            self._cache[fi] = dict(np.load(self.files[fi]))
        return self._cache[fi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, j = self.index[i]
        d = self._load(fi)
        return (torch.from_numpy(d["F_in"][j]),
                torch.from_numpy(d["F_out"][j]),
                torch.from_numpy(d["mask"][j]))
