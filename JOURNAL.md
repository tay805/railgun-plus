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

**Run date:** [30-05-26]

Verdict from the automated check at the end of the notebook:
pibt_only CSRs: {16: 0.94, 32: 0.88, 64: 0.66, 96: 0.393, 128: 0.417}

Variants ranked by sum of CSR across all agent counts:
  corrected                                CSR={16: 0.98, 32: 0.86, 64: 0.58, 96: 0.25, 128: 0.167}  delta_vs_pibt={16: 0.04, 32: -0.02, 64: -0.08, 96: -0.143, 128: -0.25}  beats_pibt_at_1/5_densities
  variant:v2_softmax_tiebreak              CSR={16: 0.96, 32: 0.76, 64: 0.44, 96: 0.179, 128: 0.083}  delta_vs_pibt={16: 0.02, 32: -0.12, 64: -0.22, 96: -0.214, 128: -0.333}  beats_pibt_at_1/5_densities
  variant:v4_prio_by_conf                  CSR={16: 0.84, 32: 0.64, 64: 0.24, 96: 0.036, 128: 0.0}  delta_vs_pibt={16: -0.1, 32: -0.24, 64: -0.42, 96: -0.357, 128: -0.417}  beats_pibt_at_0/5_densities
  variant:v3_conf_gated                    CSR={16: 0.88, 32: 0.5, 64: 0.1, 96: 0.0, 128: 0.0}  delta_vs_pibt={16: -0.06, 32: -0.38, 64: -0.56, 96: -0.393, 128: -0.417}  beats_pibt_at_0/5_densities

**Winning variant:** none. v1 (current `corrected`) remains the best.

**Interpretation:** Corrector formulation is NOT the bottleneck. Gating the
network out when uncertain (v3) was the worst, indicating that the noisy
softmax carries useful coordination signal even at low confidence. The network
has nothing better to give us than what v1 already extracts.

**Decision:** proceed to Step 2 (feature engineering). The network's INPUTS
lack coordination signal, not the way we use its outputs.

**Plot files:**
- `results_step1/step1_csr.png`
- `results_step1/step1_deadlock.png`
- `results_step1/step1_table.csv`



**Decision:**
- [ ] A variant clearly beats pibt_only at 32-64 agents -> **STOP**, write up.
- [x ] Marginal improvement only -> **proceed to Step 2** (feature engineering).
- [x ] No improvement -> **proceed to Step 2** (corrector formulation isn't the bottleneck).

## Step 2 results — feature engineering

*(not yet run)*

Planned new feature channels:
- local agent-density heatmap (5x5 neighbourhood count)
- predicted next-step occupancy under independent shortest-path
- per-cell remaining cost-to-goal of the occupying agent

## Step 3 results — multi-step temporal input

*(not yet run)*

## Step 4 results — RL fine-tuning

*(not yet run)*

## Fixes still pending regardless of step

- Test sets at 96/128 agents shrink to 28/12 instances because POGEMA can't
  generate solvable random-density-0.2 instances at that scale. Fix: either
  drop those agent counts, or lower density to 0.1, or switch agent counts to
  {16, 32, 48, 64, 80}. The 16/32/64 numbers are publication-stable; the
  96/128 numbers are too noisy.
