"""Generate PIBT expert data and save as .npz shards.

Usage (locally or in Colab):
    python scripts/generate_data.py --out data/shards --instances 200 \
        --map-size 32 --density 0.2 --agents 32 --seed 0

On Colab, point --out at a Google Drive path so data survives session resets,
e.g. --out /content/drive/MyDrive/railgun-plus/data/shards
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from railgun_plus.data.generate import generate_with_pogema
from railgun_plus.data.features import instance_to_samples, normalize_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for shards")
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--agents", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="samples per .npz shard")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Generating {args.instances} instances "
          f"({args.map_size}x{args.map_size}, density={args.density}, "
          f"{args.agents} agents)...")
    instances = generate_with_pogema(
        num_instances=args.instances, map_size=args.map_size,
        obstacle_density=args.density, num_agents=args.agents, seed=args.seed)
    print(f"  solved {len(instances)} instances")

    buf_in, buf_out, buf_mask = [], [], []
    shard_idx = 0

    def flush():
        nonlocal shard_idx, buf_in, buf_out, buf_mask
        if not buf_in:
            return
        path = os.path.join(args.out, f"shard_{args.seed:04d}_{shard_idx:04d}.npz")
        np.savez_compressed(path,
                            F_in=np.stack(buf_in),
                            F_out=np.stack(buf_out),
                            mask=np.stack(buf_mask))
        print(f"  wrote {path} ({len(buf_in)} samples)")
        shard_idx += 1
        buf_in, buf_out, buf_mask = [], [], []

    total = 0
    for inst in instances:
        for F_in, F_out, mask in instance_to_samples(inst):
            buf_in.append(normalize_features(F_in))
            buf_out.append(F_out)
            buf_mask.append(mask)
            total += 1
            if len(buf_in) >= args.shard_size:
                flush()
    flush()
    print(f"Done. {total} samples across {shard_idx} shards in {args.out}")


if __name__ == "__main__":
    main()
