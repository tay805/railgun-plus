# RAILGUN+ — Project Journal

This is the single source of truth for "where are we" across the multi-step
improvement plan. **Update this file after running each step.**

## The plan (Steps 1 → 4, cheapest first)

We're climbing a ladder of progressively more expensive improvements. After
each step we evaluate and decide whether to stop (good enough) or escalate.

| Step | What | Cost | Retrain? | Notebook |
|------|------|------|----------|----------|
| 1 | Corrector variants (different ways to combine softmax + PIBT) | hours | no | `03_step1_variants.ipynb` |
| 2 | Feature engineering (density, predicted occupancy, etc.) | days | yes | TBD |
| 3 | Multi-step / temporal input | days | yes | TBD |
| 4 | RL fine-tuning (PPO) | weeks | yes | TBD |

**Stopping rule:** if at any step the corrected method beats `pibt_only`
across all densities by a clear margin, stop and write up. Don't escalate
further unless you're confident the additional cost is justified.

## Current status

**Step 0 (baseline, complete).** Model trained on LaCAM data for 9 epochs,
early-stopped, val acc 88.6%. Best.pt saved.

Eval results (from `02_evaluate.ipynb`, after the eval-harness fix):

| agents | greedy | corrected (v1) | pibt_only | expert (LaCAM) |
|--------|--------|----------------|-----------|----------------|
| 16     | 0.30   | **0.98**       | 0.94      | 1.00           |
| 32     | 0.08   | 0.86           | **0.88**  | 1.00           |
| 64     | 0.00   | 0.58           | **0.66**  | 1.00           |
| 96     | 0.00   | 0.25           | **0.39**  | 1.00           |
| 128    | 0.00   | 0.17           | **0.42**  | 1.00           |

**Headline findings so far:**
- Deadlock fix works: greedy CSR collapses to 0; corrected stays positive.
- Corrected beats pibt_only only at 16 agents.
- Large gap between pibt_only and LaCAM expert -> room for the net to add value.
- The 96/128 columns are noisy (test sets shrank to 28/12 instances).

**Next action:** run Step 1 (`03_step1_variants.ipynb`).

## Step 1 results — corrector variants

**Status: COMPLETE.** None of the variants helped. v1 (current `corrected`) is
the best of the four.

```
pibt_only CSRs: {16: 0.94, 32: 0.88, 64: 0.66, 96: 0.39, 128: 0.42}

Variants ranked by sum of CSR:
  corrected (v1)              CSR={16:0.98,32:0.86,64:0.58,96:0.25,128:0.167}  beats_pibt_at_1/5
  variant:v2_softmax_tiebreak CSR={16:0.96,32:0.76,64:0.44,96:0.179,128:0.083} beats_pibt_at_1/5
  variant:v4_prio_by_conf     CSR={16:0.84,32:0.64,64:0.24,96:0.036,128:0.0}   beats_pibt_at_0/5
  variant:v3_conf_gated       CSR={16:0.88,32:0.5, 64:0.1, 96:0.0,  128:0.0}   beats_pibt_at_0/5
```

**Interpretation:** corrector formulation is NOT the bottleneck. Notably, v3
(confidence-gated) was the WORST -- gating the network out when uncertain
removes signal exactly when it's most needed (congestion situations where PIBT
also lacks an obvious choice). The full softmax distribution from v1 carries
useful coordination signal, even when noisy.

**Decision: proceed to Step 2 (feature engineering).** The network's INPUTS
lack coordination signal; how we use its outputs is already near-optimal.

## Step 2 — feature engineering with coordination signals

**Status: BUILT, ready to run.** Notebooks:
- `04_step2_generate_train_v2.ipynb` -- generate v2 features and train new model
- `05_step2_evaluate.ipynb` -- evaluate v2 model with adaptive-density test sets

**What changed:**
1. New `features_v2.py` adds three coordination channels (channels 6, 7, 8):
   - **local agent-density** (5x5 sum) — "this corridor is crowded"
   - **predicted next-step occupancy** under greedy go-toward-goal — "this cell will be contested"
   - **per-cell remaining cost-to-goal** of occupying agent — distinguishes urgent vs near-goal agents
2. Corrector auto-detects 6- vs 9-channel models so the SAME inference code works for both.
3. New `data/test_sets.py` (`build_test_sets`) uses an adaptive-density schedule
   so test sets at 96/128 agents are 50 instances each (vs 28/12 before).
4. Caught and fixed a bug in the density integral-image during testing -- the
   first naive-baseline test mismatched by up to 5; rewritten with a proper
   zero-prefixed SAT and verified against 60 random trials.

**Run order:**
1. Run `04_..._v2.ipynb` -- generates v2 shards (~1 hour), retrains from scratch (~few hours, early-stops). Save best.pt as a Kaggle dataset.
2. Run `05_step2_evaluate.ipynb` -- builds adaptive test sets, runs all four methods, prints `corrected vs pibt_only` deltas.

**Decision rule:**
- corrected beats pibt_only at >= 3/7 densities (and meaningfully at 32-64): **STOP**, write up.
- Marginal: continue to Step 3 (multi-step temporal input).
- No improvement over Step 0: corrector + features both ruled out — Step 3 or Step 4.

## Step 3 — multi-step temporal input

*(not yet run; build only if Step 2 doesn't suffice)*

## Step 4 — RL fine-tuning

*(not yet run)*

## Fixes resolved

- ~~Test sets shrink at 96/128~~ -- fixed in Step 2 via `build_test_sets` adaptive density.
- ~~Eval bug: expert always ran PIBT~~ -- fixed earlier (`_expert_paths` now invokes LaCAM).
