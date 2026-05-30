"""Generate PIBT expert data and save as .npz shards — RESUMABLE.

Usage (locally or in Colab):
    python scripts/generate_data.py --out data/shards --instances 200 \
        --map-size 32 --density 0.2 --agents 32 --seed 0

On Colab, point --out at a Google Drive path so data + progress survive
session resets, e.g.
    --out /content/drive/MyDrive/railgun-plus/data/shards

RESUMABILITY
------------
If Colab disconnects mid-run, just re-run the SAME command. The script:
  1. Writes each shard to disk AS SOON as it fills (not at the end), so
     completed shards are never lost.
  2. Keeps a manifest (`progress_<config>.json`) recording how many instances
     have already been generated for this exact config. On restart it reads the
     manifest and only generates the REMAINING instances.
A "config" is identified by (map_size, density, agents, seed), so different
agent counts / seeds have independent progress and never collide.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from railgun_plus.data.generate import generate_with_pogema
from railgun_plus.data.features import instance_to_samples, normalize_features
from railgun_plus.data.features_v2 import (instance_to_samples_v2,
                                           normalize_features_v2)


def config_tag(args) -> str:
    """Stable identifier for this generation config (used in filenames).
    Includes expert and feature-version so different configs never collide."""
    return (f"{args.expert}_{args.features}_m{args.map_size}_d{int(args.density*100)}"
            f"_a{args.agents}_s{args.seed}")


def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"instances_done": 0, "shards_done": 0, "samples_done": 0}


def save_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, path)  # atomic; survives a crash mid-write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--agents", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="samples per .npz shard")
    ap.add_argument("--batch", type=int, default=10,
                    help="instances to generate per chunk before saving "
                         "progress (smaller = more frequent checkpoints)")
    ap.add_argument("--expert", choices=["pibt", "lacam"], default="pibt",
                    help="expert solver for trajectories")
    ap.add_argument("--lacam-bin", default=None,
                    help="path to compiled LaCAM binary (required if "
                         "--expert lacam)")
    ap.add_argument("--lacam-time-ms", type=int, default=10000,
                    help="LaCAM anytime budget per instance in ms")
    ap.add_argument("--features", choices=["v1", "v2"], default="v1",
                    help="v1 = 6-channel RAILGUN features; v2 = 9-channel with "
                         "Step-2 coordination signals (density, predicted "
                         "occupancy, per-cell remaining cost)")
    args = ap.parse_args()

    if args.expert == "lacam" and not args.lacam_bin:
        ap.error("--expert lacam requires --lacam-bin <path-to-binary>")

    os.makedirs(args.out, exist_ok=True)
    tag = config_tag(args)
    manifest_path = os.path.join(args.out, f"progress_{tag}.json")
    manifest = load_manifest(manifest_path)

    done = manifest["instances_done"]
    if done >= args.instances:
        print(f"[{tag}] already complete: {done}/{args.instances} instances. "
              f"Nothing to do.")
        return
    if done > 0:
        print(f"[{tag}] RESUMING: {done}/{args.instances} instances already "
              f"generated. Continuing from there.")
    else:
        print(f"[{tag}] starting fresh: target {args.instances} instances "
              f"({args.map_size}x{args.map_size}, density={args.density}, "
              f"{args.agents} agents)")

    shard_idx = manifest["shards_done"]
    total_samples = manifest["samples_done"]
    buf_in, buf_out, buf_mask = [], [], []

    def flush():
        nonlocal shard_idx, buf_in, buf_out, buf_mask
        if not buf_in:
            return
        # filename includes config tag so different configs never overwrite
        path = os.path.join(args.out, f"shard_{tag}_{shard_idx:04d}.npz")
        np.savez_compressed(path, F_in=np.stack(buf_in),
                            F_out=np.stack(buf_out), mask=np.stack(buf_mask))
        print(f"  wrote {os.path.basename(path)} ({len(buf_in)} samples)")
        shard_idx += 1
        buf_in, buf_out, buf_mask = [], [], []

    # Generate in batches; checkpoint the manifest after each batch so a
    # disconnect loses at most one batch of work.
    remaining = args.instances - done
    generated_this_run = 0
    while remaining > 0:
        n = min(args.batch, remaining)
        # IMPORTANT: vary the seed by how many we've already done, so resumed
        # runs produce NEW instances rather than repeating the first ones.
        chunk_seed = args.seed * 100000 + done + generated_this_run
        if args.expert == "lacam":
            from railgun_plus.data.lacam_expert import generate_with_lacam
            insts = generate_with_lacam(
                num_instances=n, lacam_bin=args.lacam_bin,
                map_size=args.map_size, obstacle_density=args.density,
                num_agents=args.agents, seed=chunk_seed,
                time_limit_ms=args.lacam_time_ms)
        else:
            insts = generate_with_pogema(
                num_instances=n, map_size=args.map_size,
                obstacle_density=args.density, num_agents=args.agents,
                seed=chunk_seed)

        # pick v1 or v2 feature conversion
        if args.features == "v2":
            sampler, normalizer = instance_to_samples_v2, normalize_features_v2
        else:
            sampler, normalizer = instance_to_samples, normalize_features

        for inst in insts:
            for F_in, F_out, mask in sampler(inst):
                buf_in.append(normalizer(F_in))
                buf_out.append(F_out)
                buf_mask.append(mask)
                total_samples += 1
                if len(buf_in) >= args.shard_size:
                    flush()

        generated_this_run += len(insts)
        remaining -= n  # advance by requested n (some may fail to solve)

        # checkpoint progress
        manifest = {"instances_done": done + generated_this_run,
                    "shards_done": shard_idx,
                    "samples_done": total_samples}
        # flush partial buffer to a shard too, so nothing is only in RAM
        flush()
        manifest["shards_done"] = shard_idx
        save_manifest(manifest_path, manifest)
        print(f"  progress: {manifest['instances_done']}/{args.instances} "
              f"instances, {total_samples} samples (checkpointed)")

    flush()
    save_manifest(manifest_path, {"instances_done": args.instances,
                                  "shards_done": shard_idx,
                                  "samples_done": total_samples})
    print(f"[{tag}] DONE. {total_samples} samples across {shard_idx} shards "
          f"in {args.out}")


if __name__ == "__main__":
    main()
