# Amazon ML Challenge 2026 — Business Entity Resolution

Match business records across 3 noisy sources with no shared identifiers. Pure ML, no
blockchain. Scored with **F₀.₅** (precision-weighted 2×), macro-averaged per Source-1 entity.

| | |
|---|---|
| Public LB (archived baseline) | 0.945193 |
| Expected full-run (shipped defaults) | **~0.98** (dev-slice 0.991732 @ 3%) |
| Top-10 cut | 0.986 |
| Measured ceiling | ~0.999 |

Shipped defaults (already on): dense e5 ∪ BM25-k5 ∪ empty-addr BM25-k40 ∪ script neighbors
∪ housenum p-adjust. See `EXPERIMENTS.md` for every measured keep/revert.

## Start here

| Doc | Contents |
|---|---|
| **`AGENTS.md`** | **Read first.** Working rules, promotion gate, known dead ends, measurement loop. |
| **`EXPERIMENTS.md`** | **The log.** Every change, delta, verdict. Append every time. |
| **`FINDINGS.md`** | Ground-truth evidence + error taxonomy. Supersedes other docs on conflicts. |
| **`code/business_entity_resolution/README.md`** | **How to install, run, and tune batches for RAM** (crash avoidance). |
| `LITERATURE.md` · `HARDWARE.md` | What transferred; GPU/disk budgets |
| `student_resource/README.md` | Official problem statement |

## Two non-negotiable process rules

1. **Log every change in `EXPERIMENTS.md`** — date, change, command, before → after, delta,
   verdict. A change with no logged number did not happen; revert it.
2. **Never run the full pipeline (4–9 h) on a hunch.** Promote only when: positive at
   `--dev-frac 0.03`, still positive at `--dev-frac 0.3`, pending gains total **≥ +0.005**, and
   you can state the mechanism in one sentence.

## Quick start (Mac / Linux)

```bash
cd code/business_entity_resolution

# 1. Env (skip if .venv already works)
# uv venv .venv -p 3.11
# uv pip install --python .venv/bin/python -r requirements.txt
# uv pip install --python .venv/bin/python -r requirements-dense.txt   # needed: dense is ON by default

.venv/bin/python tests/smoke_test.py          # must print SMOKE TEST PASS

# 2. Dev slice (~3–8 min with dense on Apple Silicon)
DS=../../student_resource/dataset
.venv/bin/python -u src/run.py train --data-dir $DS --work-dir work_dev \
    --dev-frac 0.03 --keep-intermediates --dense-batch 256

cat work_dev/models/report.json               # trust deltas, not absolutes
```

**Full train+predict** (only after the `AGENTS.md` promotion gate):

```bash
.venv/bin/python -u src/run.py all --data-dir $DS --work-dir work_full \
    --out-dir ../../output --dense-batch 256
```

On a CUDA box with ≥24 GB VRAM use `--dense-batch 1024`. Details and OOM recovery live in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)
§ "RAM / batch sizes".

## Caches — do not commit

All `work/`, `work_*`, `work_exp_*`, `*.npy`, `*.parquet` under the pipeline are gitignored.
They are multi-GB memmaps. Delete unused ones anytime:

```bash
rm -rf code/business_entity_resolution/work_*
# keep regenerating with a fresh --work-dir name per experiment
```

## Before every submission

```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Must print `PASS`. A rejected upload costs a submission slot.

## Where the score leaks (measured)

| Category | Share |
|---|---|
| **blocking_miss** | **63.1%** |
| **missed_in_candidates** | **29.1%** |
| FP categories combined | 7.8% |

**92.2% of errors are false negatives.** Stage-2 AUC is 0.99996 — precision is not the work.
US 0.9896 / India 0.9749 (India = blocking recall). France has **no labels** anywhere.

## Known dead ends (do not repeat)

| Idea | Result |
|---|---|
| Better global / per-country τ | flat or worse |
| BM25 as blocking *replacement* | much worse; **union only** |
| Domain/acronym near-copy feature | −0.000617 → reverted |
| Group-size / reranker / name force-include | no keep on slice |
| Phone feature | **no phone field** |

## Hard constraints

- **No external data lookup** — no ER APIs, registries, geocoding, internet augmentation.
- Final model **MIT/Apache-2.0, ≤ 8B** (LightGBM MIT; e5-small MIT, 118M).
- Decide on the holdout, not the public LB.

## Layout

```
├── AGENTS.md  EXPERIMENTS.md  FINDINGS.md
├── README.md  research.md  strategy.md  PLAN.md  HARDWARE.md
├── code/business_entity_resolution/   the pipeline (submission layout)
├── student_resource/                  given data + official validator
├── baseline/output_0.945193/          archived scored submission
├── output/                            live submission artifacts
└── work/                              small analysis scratch (not pipeline cache)
```
