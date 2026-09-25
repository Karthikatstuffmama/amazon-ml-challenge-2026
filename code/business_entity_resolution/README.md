# Business Entity Resolution — reproduction guide

Pipeline: **normalise → IDF-weighted multi-key blocking (plus an optional GPU multilingual kNN
pass) → 69 country-agnostic features → 2-stage LightGBM → F₀.₅-optimal decision with the
one-owner constraint.**

> **New here?** Read `../../AGENTS.md` first (working rules and the promotion gate), then
> `../../FINDINGS.md` (what is measured and where the score actually leaks). Log every change in
> `../../EXPERIMENTS.md`.

## 1. Environment

Python **3.11** (pinned wheels exist for it; 3.14 lags on LightGBM/pyarrow). A prebuilt
`.venv/` may already be present — check before rebuilding.

```bash
# from this directory
uv venv .venv -p 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# optional GPU dense pass. RTX 50-series (Blackwell, sm_120) REQUIRES CUDA 12.8 wheels;
# cu121 will fail with "no kernel image is available for execution on the device".
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-dense.txt
```

## 2. Smoke test (~1–2 min, synthetic data, proves the install)

```bash
.venv/bin/python tests/smoke_test.py          # must print: SMOKE TEST PASS
```

It generates a dataset mimicking the real generator's transformations, runs every stage, and
asserts the submission contract (one row per test S1 entity, no duplicate ids, S2/S3 ids only,
matches ⊆ candidates).

## 3. Dev-scale loop — this is where you should live

Full runs take 4–9 h. Iterate on an entity-consistent slice instead (~3 min).

```bash
DS=../../student_resource/dataset

.venv/bin/python src/run.py train --data-dir $DS --work-dir work_dev \
    --dev-frac 0.03 --keep-intermediates

cat work_dev/models/report.json               # the headline numbers
```

**`--keep-intermediates` is required** if you intend to run the analysis scripts: the pipeline
deletes stage-2 feature matrices by default (`src/pipeline.py:263`) to bound disk use.

Dev-slice scores are **optimistic** — the slice reports ~0.984 where full scale scored 0.945,
because a smaller world has fewer confusable records. **Trust deltas, not absolutes**, and
confirm at `--dev-frac 0.3` before promoting anything to a full run.

## 4. Full run (only after the promotion gate in `AGENTS.md`)

```bash
.venv/bin/python src/run.py all --data-dir $DS --work-dir work --out-dir ../../output
# with the GPU multilingual pass (currently the top-priority experiment):
.venv/bin/python src/run.py all --data-dir $DS --work-dir work --out-dir ../../output --dense
```

Produces `../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`, then runs a
self-check plus the official `utils/validate_submission.py`.

Stages are cached under the work dir; re-running skips finished stages. `--force` recomputes.

| command | what it does |
|---|---|
| `run.py prepare --split train` | read, normalise, partition by country (parquet cache) |
| `run.py block --split train` | candidate generation (also runs prepare) |
| `run.py blocking-report` | recall ceiling, pairs per S1, reduction ratio |
| `run.py features --split train` | feature memmaps |
| `run.py train` | the above for train, then stage-1/2 LightGBM, calibration, decision tuning, `work/models/report.json` |
| `run.py predict` | test: prepare, block, features, inference, outputs, validation |
| `run.py all` | train then predict |

Useful flags: `--dev-frac 0.1`, `--train-countries US` (leave-one-country-out),
`--k-key 60`, `--decision threshold|expected_f`, `--max-train-rows`, `--no-monotone`,
`--dense`, `--keep-intermediates`.

## 5. Analysis tools

| script | purpose |
|---|---|
| `src/diagnose.py` | official metric on any submission: per-country, blocking ceiling, FP-vs-FN headroom, cap violations, deterministic holdout. Stdlib+numpy only, runs anywhere. Verified: ground truth vs itself = exactly 1.000000 |
| `src/error_analysis.py` | labels the trained model's holdout errors: `blocking_miss` / `missed_in_candidates` / `fp_stolen` / `fp_distractor` / `fp_on_singleton` |
| `src/profile_misses.py` | why blocking missed — trait lift among missed vs found pairs |
| `src/decision_analysis.py` | decision-layer bounds: best global τ, per-country τ, oracle top-k, oracle subset |
| `src/verify_gt.py` | measures the generator's transformation rates; `--one-owner` proves one-owner and the count caps |

```bash
.venv/bin/python src/error_analysis.py    --work-dir work_dev
.venv/bin/python src/profile_misses.py    --work-dir work_dev
.venv/bin/python src/decision_analysis.py --work-dir work_dev
```

## 6. Which metric to trust

| metric | meaning |
|---|---|
| `holdout_blocking_recall` | fraction of true pairs that became candidates |
| `holdout_oracle_f05` | **hard ceiling** given the candidate set — if it is below target, model work cannot help; fix blocking |
| `holdout_stage2_f05` | what the model actually scores |
| `pairs_per_s1` | candidate-set size — **explicitly graded**, keep it small |
| `per_country_holdout_f05` | per-country split (US vs India; France has no labels) |

## 7. Layout

```
src/
  run.py        CLI / orchestration
  config.py     every knob + the train/test feature contract
  io_utils.py   robust TSV reader, streaming writers, self-check, official validator hook
  normalize.py  anyascii transliteration, name/address canonicalisation, skeletons
  prepare.py    stage 0: normalise → Arrow parquet, country partitions, GT arrays
  blocking.py   stage 1: uint64 key families + vectorised sort-merge join, top-K
  dense.py      optional: multilingual-e5-small embeddings + exact GPU kNN
  features.py   stage 2: rapidfuzz cpdist + hashed TF-IDF + group context → fp16 memmap
  stage2.py     collective features on stage-1 probabilities
  pipeline.py   fold protocol, LightGBM CV, calibration, decision tuning, inference
  decide.py     one-owner constraint, expected-F₀.₅ selection, exact metric
  diagnose.py error_analysis.py profile_misses.py decision_analysis.py verify_gt.py
tests/smoke_test.py
package_submission.py   builds <team>_submission.zip in the required layout
```

## 8. Packaging the submission

```bash
.venv/bin/python package_submission.py --team MyTeam
```

Defaults: outputs from `../../output/`, doc from `../../student_resource/Documentation_template.md`,
zip written to the project root. Caches (`work/`, `work_dev/`), `.venv/` and `output/` are excluded
from the code copy.

Validate first — a rejected upload wastes a submission slot:

```bash
python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

## 9. Licences and fair play

LightGBM (MIT) is the final model. The optional encoder, multilingual-e5-small, is MIT with 118M
parameters — well inside the MIT/Apache ≤8B rule.

**No external data lookup.** Every dictionary in `normalize.py` is static domain knowledge
(abbreviations, state codes) and all IDF statistics come from the provided files. Do not add
geocoding, registry lookups, or any internet-sourced augmentation — it is an instant
disqualification and the top teams' code is audited.
