# Experiment log

**Append one row per change, the moment you have the number. A change with no logged number
did not happen — revert it.**

This file is the project's memory. The baseline plateaued at 0.945 *blind* because its run kept
no metrics; nobody knew US and India differed until it was measured. Do not recreate that hole.

---

## How to log

Copy this template, fill it, append under **Log** (newest last). Keep it to a screenful.

```markdown
### YYYY-MM-DD — short title
- **Hypothesis:** the mechanism, in one sentence. Why should this help, given FINDINGS.md?
- **Change:** files touched / config knobs changed.
- **Command:** exact command, including --dev-frac.
- **Before → After:** holdout_stage2_f05 X → Y (**delta**)
  - blocking_recall A → B | oracle_f05 C → D | pairs_per_s1 E → F
  - per-country: US u1 → u2, India i1 → i2
- **Verdict:** KEEP / REVERT / INCONCLUSIVE — and why.
- **Notes:** surprises, side effects, follow-ups.
```

Always report `pairs_per_s1` alongside recall: candidate-set size is **explicitly graded**, so a
recall gain bought with a much larger candidate set may be a net loss in the final ranking.

## The promotion gate (before any full-scale run)

A full run costs 4–9 h. All four must hold:

1. Positive delta at `--dev-frac 0.03`.
2. Delta **survives** at `--dev-frac 0.3` (does not shrink toward zero).
3. Combined pending gain **≥ +0.005** on the dev slice.
4. You can state the **mechanism** in one sentence, consistent with the error taxonomy.

Dev-slice scores are optimistic (~0.984 where full scale gave 0.945). **Trust deltas, not
absolutes.**

## Metric reference

| Metric | Source | Baseline (dev 0.03) |
|---|---|---|
| `holdout_stage2_f05` | `work_dev/models/report.json` | 0.983787 |
| `holdout_blocking_recall` | same | 0.974226 |
| `holdout_oracle_f05` | same — **ceiling given candidates** | 0.990727 |
| `pairs_per_s1` | same — **graded, keep small** | 20.94 |
| `per_country_holdout_f05` | same | US 0.989606 / India 0.974928 |
| **Leaderboard** | Unstop portal | **0.945193** |

Targets: **0.986** = top 10. **~0.999** = measured ceiling.

---

# Log

### 2026-09-26 — BASELINE (reference row, do not overwrite)
- **Hypothesis:** n/a — establishing the reference.
- **Change:** none. Pipeline as inherited: normalize → IDF multi-key blocking (`k_key=40`) → 69
  features → 2-stage LightGBM (10 folds) → `expected_f` decision + greedy one-owner mask.
  **`--dense` was OFF.**
- **Command:** `src/run.py train --data-dir ../../student_resource/dataset --work-dir work_dev --dev-frac 0.03 --keep-intermediates`
- **Results:** `holdout_stage2_f05` **0.983787** | blocking_recall 0.974226 | oracle_f05 0.990727
  | pairs_per_s1 20.94 | US 0.989606 / India 0.974928 | stage-1 AUC 0.99994, stage-2 AUC 0.99996
  | decision `expected_f τ=0.02 γ=0.85 miss=0.0`
- **Full-scale leaderboard:** **0.945193** (the dev slice is optimistic by ~0.039)
- **Verdict:** KEEP as reference.
- **Notes:** Loss decomposes as **blocking 0.0093** + **matcher/decision 0.0069**.

### 2026-09-26 — Verified the generator's transformations against ground truth
- **Hypothesis:** the noise is a finite synthetic rule set, so alias tables should be mined, not
  hand-written.
- **Change:** added `src/verify_gt.py`. No pipeline change.
- **Command:** `src/verify_gt.py --sample work/sample.json`; `--one-owner .../train_ground_truth.tsv`
- **Results (139,271 true pairs):** street abbrev 29.1%, partial reorder 28.0%,
  **address UPPERCASED 53.35% of S2 vs 0.01% of S3**, number perturbed 15.7%, name truncated
  15.6%, diacritics 13.6%, legal-suffix dropped 13.5%, non-Latin name 6.7%, domain form 5.7%,
  transposition 5.6%, typo 5.5%, empty address 4.5%, **random name only 2.08%**.
  **Proven exactly on the full 2.08M entities / 7.64M records: one-owner holds, caps hold
  (n_S2 ≤ 5, n_S3 ≤ 6), zero violations.**
- **Verdict:** KEEP (evidence).
- **Notes:** Two corrections. "Appended descriptor words (Service/Center/Enterprises)" is only
  **0.94%** — hand-coding that list would have been useless. `address_UPPERCASED` is a strong,
  previously unnoticed **S2-vs-S3 discriminator** and is not yet exploited as a feature.

### 2026-09-26 — Error taxonomy of the trained model
- **Hypothesis:** find out where the 0.945 actually leaks instead of guessing.
- **Change:** added `src/error_analysis.py`, `src/profile_misses.py`.
- **Command:** `src/error_analysis.py --work-dir work_dev`; `src/profile_misses.py --work-dir work_dev`
- **Results:** 13,378 holdout entities, 1,627 (12.16%) with ≥1 error, 1,882 error items:
  **blocking_miss 63.1%**, missed_in_candidates 29.1%, fp_distractor 6.1%, fp_stolen 1.0%,
  fp_on_singleton 0.7% → **92.2% false negatives**.
  Miss traits (lift vs found): no shared name token 72.1% (5.5×), **non-Latin name 37.1%
  (6.1×)**, **empty address 23.0% (6.1×)**, **no shared token at all 0.08%**.
  Per-country miss mix: India blocking_miss 71.6%, US 51.0%.
- **Verdict:** KEEP (evidence). **Reorders the whole plan: blocking first, precision last.**
- **Notes:** **99.92% of missed pairs are reachable** — blocking had the signal and the
  top-K/`max_df` caps discarded it. `anyascii` transliterates phonetically (`पावर`→`pavara`),
  which never token-matches `power`; a mined token table is needed.

### 2026-09-26 — Decision-layer bounds (threshold work is exhausted)
- **Hypothesis:** better thresholds / per-country cuts / segment rules would gain score.
- **Change:** added `src/decision_analysis.py`. No pipeline change.
- **Command:** `src/decision_analysis.py --work-dir work_dev`
- **Results:** shipped rule 0.984071 | **best single global τ=0.69 → 0.983797 (−0.000274)** |
  **per-country τ → −0.000083** | oracle_topk 0.990587 (+0.006516) | oracle_subset 0.990727
  (+0.006655). Calibration by segment is tight (non-Latin 0.3080 actual vs 0.3064 predicted;
  empty-address 0.5609 vs 0.5670; plain 0.2968 vs 0.2969).
  Counts: mean predicted 3.344, mean oracle-k 3.369, mean true 3.462.
- **Verdict:** **REVERT the idea.** Threshold, per-country and segment tuning are all dead ends.
- **Notes:** `oracle_topk` ≈ `oracle_subset` (gap 0.00014) ⇒ **ranking is effectively solved**.
  The remaining ≈0.0065 is entirely **per-entity k selection** — a group-size prediction problem.
  True matches with empty addresses score `mean_p|y=1` 0.9106 vs 0.9943 for plain records; that
  is where marginal k decisions go wrong.

### 2026-09-26 — Repo restructure + smoke test (no score impact)
- **Change:** pipeline moved `README/` → `code/business_entity_resolution/` (the required
  submission layout); analysis scripts consolidated into its `src/`; 0.945 submission archived to
  `baseline/output_0.945193/`; added `tests/smoke_test.py` (was referenced but missing); fixed
  `package_submission.py` paths and added `work_dev` to its skip list.
- **Command:** `.venv/bin/python tests/smoke_test.py`
- **Results:** `SMOKE TEST PASS` — full pipeline on synthetic data, submission contract asserted.
- **Verdict:** KEEP. No model change; scores unaffected.

### 2026-09-26 — BM25 top-k blocking as a REPLACEMENT (literature-driven) — NEGATIVE
- **Hypothesis:** Sparkly (PVLDB 2023) reports top-k TF/IDF blocking beating 8 SOTA blockers.
  Our blocker *discards* keys above `max_df_pair=500`/`max_df_name=200` (0.16% of a 310k index),
  so pairs sharing only common tokens get no candidate; BM25 ranks instead of discarding, and
  99.92% of our missed pairs do share a token.
- **Change:** added `src/blocking_probe.py` (measurement only — no pipeline change).
- **Command:** `src/blocking_probe.py --work-dir work_dev`
- **Results:** BM25 alone, same slice/holdout: recall 85.03% @k=5, 91.30% @k=10, 93.08% @k=20,
  **94.29% @k=40** — versus incumbent **97.42% @ 20.94 cands/entity**.
- **Verdict:** **REVERT the idea. Do not replace blocking with BM25.**
- **Notes:** Mechanism for the failure: our signal lives in field *combinations* (house number ×
  rare street token); pooling all tokens into one bag lets common city/state tokens dominate.
  Published blocker rankings are benchmark averages — this dataset is not the average. Testing
  cost ~1 h and saved a costly rebuild.

### 2026-09-26 — UNION of incumbent + BM25 top-k — POSITIVE, accumulate
- **Hypothesis:** (Papadakis survey) union complementary key families. BM25 losing outright does
  not mean it fails on the *same* pairs as the incumbent.
- **Change:** `src/blocking_probe.py` extended to report union recall, candidate cost, rescue
  rate and the exact oracle F₀.₅ ceiling. Still measurement only.
- **Command:** `src/blocking_probe.py --work-dir work_dev`
- **Results (oracle ceiling, incumbent = 0.990727):**
  | k | union recall | cands/ent | rescued | oracle F₀.₅ | Δ |
  |---|---|---|---|---|---|
  | **5** | **98.18%** | **22.32** | **29.5%** | **0.993753** | **+0.003026** |
  | 10 | 98.50% | 26.01 | 41.6% | 0.994617 | +0.003890 |
  | 20 | 98.60% | 34.74 | 45.8% | 0.994932 | +0.004205 |
  | 40 | 98.72% | 53.43 | 50.4% | 0.995304 | +0.004577 |
- **Verdict:** **KEEP the direction — k=5. Do NOT promote to a full run alone.**
- **Notes:** k=5 is the knee: +0.003026 ceiling for +1.24 candidates/entity (+5.9%). Beyond k=10
  the trade collapses (k=40 = +154% candidates for +0.0015 more ceiling) and candidate size is
  graded. **This is a CEILING gain, not realized score**: the rescued pairs are the hard ones the
  incumbent could not reach, and extra candidates add false-positive risk, so expect **+0.002 to
  +0.003 realized** — below the +0.005 gate on its own. Next step is implementing the union in
  `blocking.py` and measuring *realized* `holdout_stage2_f05`.

---

## Next up (highest measured value first)

| # | Action | Why | Expected |
|---|---|---|---|
| 1 | **Union BM25 top-5 into blocking** (`blocking.py`) | **measured**: +0.003026 oracle ceiling for +1.24 cands/entity; rescues 29.5% of misses | **+0.002–0.003 realized** |
| 2 | **`--dense`** (already implemented, was OFF) | e5-small embeds Devanagari + Latin in one space → hits the 37% non-Latin misses | recall ↑, **cheap, test next** |
| 3 | Mined Devanagari→English token table | phonetic transliteration cannot bridge `पावर`/`power`; note `n_translit` already exists in the pipeline | recall ↑ |
| 4 | Adaptive `max_df` caps + split K budgets for address vs name keys | 99.92% of misses were reachable; the two failure modes are disjoint | recall ↑ |
| 5 | Per-entity group-size head (predict n_S2, n_S3, clip to caps) | the only decision lever with headroom (0.0065) | up to +0.006 |
| 6 | Meta-blocking edge pruning (CEP/CNP) to shrink the candidate set | candidate size is graded; 21 cands for 3.46 true matches | size ↓ |
| 7 | `address_UPPERCASED` as a source-identity feature | 53.35% of S2 vs 0.01% of S3, currently unused | small |
| 8 | Precision work | only 7.8% of errors | last |

**Accumulation note:** items 1–4 are all recall plays and should be stacked, then measured
together against the +0.005 gate before any full run.
