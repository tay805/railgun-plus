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

*(fill in after running the notebook)*

Verdict from the automated check at the end of the notebook:

```
[paste the "Variants ranked by sum of CSR" output here]
```

**Winning variant:** _(name or "none — proceed to Step 2")_

**Plot files:**
- `results_step1/step1_csr.png`
- `results_step1/step1_deadlock.png`
- `results_step1/step1_table.csv`

**Decision:**
- [ ] A variant clearly beats pibt_only at 32-64 agents -> **STOP**, write up.
- [ ] Marginal improvement only -> **proceed to Step 2** (feature engineering).
- [ ] No improvement -> **proceed to Step 2** (corrector formulation isn't the bottleneck).

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
