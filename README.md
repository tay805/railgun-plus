# RAILGUN+ : Map-Based MAPF Policy + PIBT Corrector

A centralized, **map-based** learned policy for Multi-Agent Path Finding (a
RAILGUN-style U-Net) composed with a **PIBT corrector** at inference to
eliminate the deadlock/congestion collapse that pure greedy rollouts suffer at
high agent density.

## Novelty (the one-sentence claim)

> RAILGUN+ is the first to show that a centralized, map-based CNN policy —
> which alone collapses into deadlock at high agent density — becomes a
> competitive MAPF solver when its per-cell action distribution is used as the
> priority ordering for a PIBT corrector, distilling an expensive search-based
> expert (LaCAM) into a fast, deadlock-free reactive policy.

Three concrete contributions over the RAILGUN paper:
1. **Deadlock fix.** RAILGUN named congestion/deadlock as its key unsolved
   weakness. Feeding the network's softmax into PIBT as its preference ordering
   is a specific, novel mechanism that resolves it (greedy CSR collapses to ~0
   at high density; corrected stays high).
2. **Pure-PIBT ablation = honest rigor.** We compare against PIBT-with-no-network,
   answering the question most learned-MAPF papers dodge: *does the network
   actually beat the trivial reactive baseline?* A LaCAM teacher (not PIBT) is
   what makes "yes" achievable — distilling a near-optimal expert the corrector
   can't reach on its own.
3. **Distillation framing.** A fast CNN that distills slow near-optimal search,
   made deadlock-free by PIBT, inheriting RAILGUN's map-size-independent
   scalability.

**Why upDdate to LaCAM and not keep PIBT as the teacher:** PIBT is free at inference (it's the
corrector), so imitating it can't beat it. LaCAM is near-SoC-optimal but too
slow to run per-step, so distilling it adds real value.

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

## Evaluation (Notebook 2)

`notebooks/02_evaluate.ipynb` runs the full comparison via
`railgun_plus.eval.harness` and produces paper-style outputs.

**Methods compared (all runnable in Colab):**
- `expert` — PIBT run to completion (oracle / upper reference).
- `pibt_only` — pure PIBT at inference, network ignored (shows what the net adds).
- `greedy` — trained net + naive conflict resolution = baseline RAILGUN (deadlocks).
- `corrected` — trained net + PIBT corrector = **the contribution**.

**Metrics:** CSR, normalized SoC (`SoC / lower-bound`, where 1.0 = optimal and
the lower bound is the sum of per-agent shortest paths), and makespan (paper
Table I). SoC/makespan are averaged over *solved* instances only and reported
next to `n_solved` so the "solved nothing -> SoC looks like 0" artifact is
visible, not misleading.

**Outputs** (saved to Drive `results/`): `results_table.csv` plus
`csr_all_methods.png`, `soc_ratio_all_methods.png`, `makespan_all_methods.png`.

### Comparing against SCRIMP / DCC / MAPF-GPT (later)

Those are separate trained systems; reproducing them in Colab is its own
multi-week effort, so they are deliberately NOT run here. The honest path is to
wrap this model as a POGEMA agent and run it through `pogema-toolbox`'s official
benchmark, which already tabulates those baselines on the same six POGEMA
metrics as the RAILGUN paper. `railgun_plus.eval.harness.pogema_benchmark_stub`
documents the intended approach. Do this once the scaled-up Route-C model is
solid.

## Scaling up for real results

The first end-to-end run uses ~50 instances (a plumbing test). For numbers you
can trust, generate more data across agent counts and train longer (in
Notebook 1's multi-agent-count block, ~200 instances each across
{16,32,64,96} agents, then train 30-50 epochs).

### Resumable data generation

`scripts/generate_data.py` is resumable. It writes each shard to disk as soon
as it fills (never only in RAM) and keeps a `progress_<config>.json` manifest
in the output dir. If Colab disconnects, **re-run the exact same command** — it
reads the manifest and generates only the remaining instances. Each
(map_size, density, agents, seed) config has its own manifest, so the
multi-agent-count loop resumes per agent count: completed counts are skipped
instantly, the interrupted one continues, the rest proceed. `--batch` controls
checkpoint frequency (smaller = less lost work on disconnect). To regenerate
from scratch, delete that config's `progress_*.json` and its shards.

## Data routes

- **Route A (LaCAM) — the expert for real results.** Compile LaCAM once on
  Kaggle (`scripts/build_lacam.sh`), save the binary as a Kaggle dataset, then
  generate with `--expert lacam --lacam-bin <path>`. Near-SoC-optimal teacher;
  the network distilling it can beat pure PIBT. See
  `notebooks/00_generate_lacam.ipynb`. The only fragile spot is
  `parse_lacam_solution` in `data/lacam_expert.py` — if your LaCAM fork's output
  format differs, that one function is the fix (docstring explains).
- **Route C (PIBT) — pure Python, no compilation.** `generate_with_pogema`
  solves with PIBT. Good for the deadlock ablation, but the net can't beat PIBT
  when PIBT is also the teacher. Use `--expert pibt` (default).

### Kaggle workflow (three notebooks)

1. `00_generate_lacam.ipynb` — build/attach LaCAM, generate data, save as a
   dataset.
2. `01_setup_generate_train.ipynb` — train the U-Net on that data (resumable,
   early stopping, saves `best.pt`).
3. `02_evaluate.ipynb` — four-method comparison (expert / pibt_only / greedy /
   corrected) with CSR, deadlock-rate, SoC-ratio, makespan tables + plots.

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
