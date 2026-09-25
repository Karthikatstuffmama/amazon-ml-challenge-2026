# Literature review — what transfers to this problem, and what does not

> Every claim below was **tested on our own data** before being recommended. Where a
> published finding failed to transfer, that is recorded too — those are the expensive
> mistakes this file exists to prevent.

## Sources

| Paper | Venue | Relevance |
|---|---|---|
| Paulsen, Govind & Doan — [*Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching*](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) | PVLDB 16(7):1507–1519, 2023 | top-k TF/IDF blocking beats 8 SOTA blockers; authors recommend top-k because it improves recall |
| Papadakis, Skoutas, Thanos & Palpanas — [*Blocking and Filtering Techniques for Entity Resolution: A Survey*](https://arxiv.org/abs/1905.06167) | ACM CSUR, 2020 | schema-agnostic blocking reaches higher recall than schema-based; unioning key families is standard practice |
| Zeakis, Papadakis, Skoutas & Koubarakis — [*Pre-trained Embeddings for Entity Resolution: An Experimental Analysis*](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf) | PVLDB 16(9):2225–2238, 2023 | 12 language models × 17 benchmarks for blocking and matching |
| Brinkmann et al. — [*SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines*](https://arxiv.org/abs/2303.03132) | 2023 | learned blocking producing candidate sets ~half the size of alternatives |
| Papadakis et al. — [*Meta-Blocking: Taking Entity Resolution to the Next Level*](http://dit.unitn.it/~themis/publications/tkde13-metablocking.pdf) | TKDE, 2014 | prune a block collection via block-filtering and edge-pruning (CEP/CNP/WEP) |
| Thirumuruganathan et al. — *DeepBlocker* / Wu et al. — [*Blocker and Matcher Can Mutually Benefit*](https://vldb.org/pvldb/vol17/p292-wu.pdf) | PVLDB 2021 / 17(2), 2023 | deep and co-learned blocking |

*Content rephrased and summarised for licensing compliance.*

---

## Tested claim 1 — "top-k TF/IDF blocking is a strong baseline" (Sparkly)

**Why it looked compelling here.** Our blocker matches exact keys, then **discards** any key
whose document frequency exceeds `max_df_pair=500` / `max_df_name=200` — on a 310k-record index
that is 0.16% / 0.06%. A true pair whose only shared tokens are common ones therefore receives
no candidate at all. BM25 top-k never discards: a common term simply earns little score, so the
pair still surfaces when nothing better competes. `FINDINGS.md` measured that **99.92% of missed
pairs do share a token** — precisely the population this difference should address.

**Result: the claim did NOT transfer.** Schema-agnostic BM25 top-k, alone, is clearly worse
(`src/blocking_probe.py`, dev slice, same holdout):

| k | BM25 recall | cands/entity |
|---|---|---|
| 5 | 85.03% | 5.00 |
| 10 | 91.30% | 10.00 |
| 20 | 93.08% | 20.00 |
| 40 | 94.29% | 40.00 |
| **incumbent** | **97.42%** | **20.94** |

BM25 needs 40 candidates to reach 94.3%, where the incumbent gets 97.4% with 21. **Do not
replace blocking with BM25.** The reason is intelligible: our discriminative signal is
concentrated in field *combinations* (house number × rare street token), and pooling every token
into one bag dilutes it — common city/state tokens dominate the bag. The incumbent's compound
keys are simply more precise for this data. Published blocker rankings are averages over
benchmarks; this dataset is not the average.

## Tested claim 2 — "union complementary key families" (Papadakis survey) ✅ **TRANSFERS**

The right question was never "is BM25 better" but "does BM25 find pairs the incumbent *misses*".
It does, and cheaply:

| k | union recall | cands/entity | incumbent misses rescued | **oracle F₀.₅** | vs incumbent |
|---|---|---|---|---|---|
| — | 97.42% | 21.08 | — | 0.990727 | — |
| **5** | **98.18%** | **22.32** | **29.5%** | **0.993753** | **+0.003026** |
| 10 | 98.50% | 26.01 | 41.6% | 0.994617 | +0.003890 |
| 15 | 98.57% | 30.27 | 44.4% | 0.994847 | +0.004120 |
| 20 | 98.60% | 34.74 | 45.8% | 0.994932 | +0.004205 |
| 40 | 98.72% | 53.43 | 50.4% | 0.995304 | +0.004577 |

**k=5 is the knee: +0.003026 on the oracle ceiling for +1.24 candidates/entity (+5.9%).** Past
k=10 the trade collapses — k=40 costs +154% candidates for only +0.0015 more ceiling, and
candidate-set size is **explicitly graded**.

The two blockers fail on *different* pairs, which is exactly the survey's argument for unioning
schema-agnostic with schema-based blocking.

### What this is and is not

It is a **ceiling** gain, not a realized score gain. The model currently realizes 0.9838 of a
0.9907 ceiling. The rescued pairs are by construction the *hard* ones the incumbent could not
reach, so realization will be lower than on average pairs, and extra candidates also create
extra false-positive opportunity. **Honest expectation: +0.002 to +0.003 realized at k=5.**

Per the promotion gate in `AGENTS.md`, that is **below +0.005 on its own** — accumulate it with
other wins rather than triggering a full run for it alone.

## Untested but literature-supported (ranked)

1. **Embedding choice for the `--dense` pass.** Zeakis et al. benchmark 12 models for blocking
   specifically. The pipeline defaults to `multilingual-e5-small` (MIT, 118M). Worth checking
   their results before assuming it is the right pick — but note `--dense` has *never been run
   here*, so turning it on at all is the higher-value test.
2. **Meta-blocking edge pruning** (CEP/CNP/WEP) to *shrink* the candidate set at fixed recall.
   Directly targets the graded criterion; our 21 candidates/entity for a mean of 3.46 true
   matches suggests room.
3. **SC-Block**'s claim of roughly half-size candidate sets. Same graded criterion. Requires
   training a learned blocker — larger effort, weigh against the blocking work already queued.

## Notes on our data that the literature does not cover

- The generator is **synthetic with a finite rule set**, so alias tables should be mined from
  ground truth rather than hand-written or imported. Published blockers assume natural noise.
- **One-owner and the count caps (n_S2 ≤ 5, n_S3 ≤ 6) hold exactly** across 7.64M records. This
  is a much stronger structural constraint than standard ER assumes, and it is under-exploited:
  the pipeline uses one-owner only as a greedy tie-break.
- `address_UPPERCASED` separates S2 from S3 almost perfectly (53.35% vs 0.01%) and is currently
  unused as a feature.
- The **matching** literature (Ditto, cross-encoders) is low priority here: stage-2 AUC is
  already 0.99996 and `oracle_topk` ≈ `oracle_subset`, so ranking is effectively solved.
  Recall, not matching, is the bottleneck.

## Reproduce

```bash
cd code/business_entity_resolution
.venv/bin/python src/blocking_probe.py --work-dir work_dev      # ~5 min
```
