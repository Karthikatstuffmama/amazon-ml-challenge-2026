# Amazon ML Challenge 2026 — Business Entity Resolution

Match business records across 3 noisy sources with no shared identifiers. Pure ML, no
blockchain. Scored with **F₀.₅** (precision-weighted 2×), macro-averaged per Source-1 entity.

| | |
|---|---|
| Leaderboard score | **0.945193** |
| Top-10 cut | **0.986** |
| Measured ceiling | **~0.999** |
| Dev-slice baseline | 0.983787 (`--dev-frac 0.03`, optimistic) |

## Start here

| Doc | Contents |
|---|---|
| **`AGENTS.md`** | **Read first.** Working rules, the promotion gate for full runs, known dead ends, setup, and the measurement loop. |
| **`EXPERIMENTS.md`** | **The log.** Every change, its measured delta, and the verdict. Append to it every time. |
| **`FINDINGS.md`** | Ground-truth evidence: verified transformations, the real error taxonomy, per-country scores. Supersedes other docs where they conflict. |
| **`LITERATURE.md`** | Published ER techniques **tested on our data** — which transferred, which did not, and why. |
| `code/business_entity_resolution/README.md` | How to run and reproduce the pipeline |
| `research.md` · `strategy.md` · `PLAN.md` | Dataset facts, approach, phased plan |
| `HARDWARE.md` | RTX 5080 laptop runbook (CUDA 12.8, disk/RAM/VRAM budgets) |
| `student_resource/README.md` | The official problem statement |

## Two non-negotiable process rules

1. **Log every change in `EXPERIMENTS.md`** — date, change, command, before → after, delta,
   verdict. A change with no logged number did not happen; revert it.
2. **Never run the full pipeline (4–9 h) on a hunch.** Promote only when: positive at
   `--dev-frac 0.03`, still positive at `--dev-frac 0.3`, pending gains total **≥ +0.005**, and
   you can state the mechanism in one sentence.

## Where the score actually leaks

Measured on the real pipeline against real ground truth — 13,378 holdout entities, 1,882 errors:

| Category | Share |
|---|---|
| **blocking_miss** — true match never became a candidate | **63.1%** |
| **missed_in_candidates** — was a candidate, not selected | **29.1%** |
| fp_distractor / fp_stolen / fp_on_singleton | 7.8% combined |

**92.2% of errors are false negatives.** Despite F₀.₅ being precision-weighted, precision is not
where the work is: stage-2 AUC is 0.99996, so the decision rule is already conservative enough
that almost nothing over-merges. And the misses are **recoverable** — 99.92% of missed pairs
share at least one token with their S1 record; top-K and the `max_df` caps discarded them.

Per-country (both fully labelled): **US 0.9896, India 0.9749** — India's gap is blocking recall
(95.70% vs 98.57%), concentrated in Devanagari names. France has **no labels anywhere** (it
exists only in the unlabelled test set), so it can never be scored directly; India is the proxy.

**Act in this order:** (1) `run.py all --dense` — it was off and targets the 37% non-Latin
misses; (2) mined Devanagari→English token table, adaptive `max_df`, split K budgets;
(3) per-entity group-size prediction; (4) precision last.

## Proven structural facts

Exact, across all 2,083,574 entities and 7,638,365 matched records:

- **One-owner holds** — every S2/S3 record belongs to at most one S1. Zero violations.
- **Caps hold** — n_S2 ≤ 5, n_S3 ≤ 6. Zero violations.
- Every S1 entity is unique by normalized name + address (zero collisions), so nothing is
  fundamentally unresolvable.

Supporting: singletons 5.585%, mean 3.461 matches, ~27% distractor records, and the dataset is a
**synthetic generator with a finite rule set** — so alias tables should be mined from ground
truth, never hand-written.

## Known dead ends (measured — don't repeat)

| Idea | Result |
|---|---|
| Better global threshold | best single τ is **worse** than the shipped rule (−0.000274) |
| Per-country thresholds | −0.000083, no gain |
| Segment-specific cuts | model already well calibrated there |
| Hand-tuned similarity rules instead of the GBM | AUC is 0.99996; rules lose |
| Phone/domain feature | **there is no phone field** — only name, address, country |
| **BM25 top-k as a blocking replacement** (Sparkly, PVLDB'23) | **much worse: 94.29% @k=40 vs 97.42% @21.** Union it instead — see `LITERATURE.md` |

`oracle_topk` 0.990587 vs `oracle_subset` 0.990727 ⇒ ranking is effectively solved. The
remaining ≈0.0065 is entirely **per-entity k selection**.

## Quick start

```bash
cd code/business_entity_resolution
.venv/bin/python tests/smoke_test.py          # must print SMOKE TEST PASS

DS=../../student_resource/dataset
.venv/bin/python src/run.py train --data-dir $DS --work-dir work_dev \
    --dev-frac 0.03 --keep-intermediates      # ~3 min
cat work_dev/models/report.json
.venv/bin/python src/error_analysis.py --work-dir work_dev
```

## Before every submission

```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Must print `PASS`. A rejected upload costs a submission slot.

## Hard constraints

- **No external data lookup** — no ER APIs, registries, geocoding, or internet augmentation.
  Only the provided files. Top teams are audited; violation means disqualification.
- Final model **MIT/Apache-2.0, ≤ 8B parameters** (LightGBM MIT; e5-small MIT, 118M).
- Decide on the holdout, not the public leaderboard — the private split sets the rank.

## Layout

```
├── AGENTS.md  EXPERIMENTS.md  FINDINGS.md        process + evidence
├── README.md  research.md  strategy.md  PLAN.md  HARDWARE.md
├── code/business_entity_resolution/   the pipeline (required submission layout)
│   ├── src/  tests/  .venv/  work_dev/  requirements*.txt  package_submission.py
├── student_resource/     given data, official validator, Documentation_template.md
├── baseline/output_0.945193/   the scored 0.945 submission, archived
├── output/               live submission artifacts
└── work/                 analysis scratch
```
