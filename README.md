# RAILGUN+ : Map-Based MAPF Policy + PIBT Corrector

A centralized, **map-based** learned policy for Multi-Agent Path Finding (a
RAILGUN-style U-Net) composed with a **PIBT corrector** at inference to
eliminate the deadlock/congestion collapse that pure greedy rollouts suffer at
high agent density.

**Contribution.** The same trained U-Net is run two ways — a greedy rollout
(baseline RAILGUN behaviour) and a PIBT-corrected rollout (ours). PIBT turns
the network's per-cell action probabilities into a guaranteed collision-free
joint move. The CSR gap between the two curves, on the *same* model, isolates
the corrector's effect from expert-data quality.

## Workflow: VS Code (Mac) + Colab only

- **VS Code on your Mac** is the source of truth. You edit code here and
  `git push`.
- **Colab** is the GPU runner. It clones your repo, installs deps, generates
  data to Google Drive, trains, and evaluates. **Never edit code in Colab** —
  edit in VS Code, push, re-run the clone cell.
- **Google Drive** stores data shards and checkpoints so they survive Colab
  session resets.

```
  VS Code (edit) --push--> GitHub --clone--> Colab (GPU) --save--> Google Drive
```

## Project layout

```
src/railgun_plus/
  data/
    types.py        # Instance: the uniform format ALL data routes emit
    grid_utils.py   # BFS distance field, neighbors (no deps -> no import cycles)
    features.py     # Stage 3: Instance -> (n,m,k) tensors  [route-agnostic]
    generate.py     # Stage 2 Route C: PIBT on POGEMA maps  (+ Route A stub)
    dataset.py      # PyTorch Datasets (in-memory + on-disk shards)
  models/
    unet.py         # RAILGUN U-Net (~30M params). Swap-seam for other backbones.
  solvers/
    pibt.py         # PIBT: data generator AND inference corrector
    corrector.py    # greedy_rollout vs corrected_rollout (the experiment)
  eval/
    metrics.py      # CSR, SoC, makespan
  train.py          # masked cross-entropy training loop
scripts/
  generate_data.py  # CLI: make .npz shards (save to Drive on Colab)
notebooks/
  01_setup_generate_train.ipynb
  02_evaluate.ipynb
tests/
  test_pipeline.py  # run `pytest` before pushing — catches shape/transpose bugs
```

## Local setup (VS Code on Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # editable install + pytest
pytest                       # all tests should pass
```

You do NOT need a GPU locally. Local is for editing, running `pytest`, and
small CPU sanity checks. Training happens on Colab.

## First run (the week-1 path)

1. **Push this repo to your GitHub** (see below).
2. Open `notebooks/01_setup_generate_train.ipynb` in Colab
   (File -> Open notebook -> GitHub, or upload it).
3. Edit the `REPO_URL` cell to your repo.
4. Run cells top to bottom: it mounts Drive, clones, installs, runs the unit
   tests, generates 50 PIBT instances, and trains 10 epochs. This verifies the
   **entire pipeline end to end** on a small scale.
5. Open `notebooks/02_evaluate.ipynb`, point it at the same checkpoint, and run
   it to get the **greedy-vs-corrected CSR plot** — your first result.
6. Once verified, scale up data generation (uncomment the multi-agent-count
   cell in Notebook 1) and retrain.

## Push to GitHub (one time)

```bash
cd railgun-plus
git init
git add .
git commit -m "RAILGUN+ scaffold: data pipeline, U-Net, PIBT corrector, eval"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/railgun-plus.git
git push -u origin main
```

## Data routes

- **Route C (PIBT) — implemented, default.** `generate_with_pogema()` solves
  POGEMA maps with PIBT. Zero compilation, runs anywhere. PIBT is also the
  corrector, so the same code is reused. Lower expert SoC, but the cleanest
  ablation: a weak expert + corrector still fixes deadlocks.
- **Route A (LaCAM) — stub, later.** `load_lacam_instances()` in
  `data/generate.py` is a stub. Implement it to load LaCAM-quality data into
  the same `Instance` format; everything downstream is unchanged. Use this to
  chase lower SoC after Route C results are in hand.

## Swapping the backbone

The pipeline depends only on a module mapping `(B,k,H,W) -> (B,5,H,W)`. To try
a non-U-Net **map-based** network (e.g. CNN + transformer bottleneck), match
that signature in `models/` and nothing else changes. Switching to an
*agent-based* representation would change the data conversion and is out of
scope (and would overlap with MAPF-GPT).

## Notes

- Maps are padded to multiples of 16 (5-level U-Net divisibility); a validity
  mask excludes padding from the loss.
- Coordinate convention is `(row, col)`, `grid==1` is obstacle. Documented in
  `data/types.py`.
- Run `pytest` before every push. The tests catch the silent
  feature-construction bugs that otherwise waste GPU hours.
