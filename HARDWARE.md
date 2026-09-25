# Hardware Runbook — RTX 5080 Laptop

Target machine for full-scale runs.

| | | Implication |
|---|---|---|
| CPU | Intel Core Ultra 9 275HX @ 2.70 GHz (Arrow Lake-HX, 8 P-cores + 16 E-cores, 24 threads) | strong parallel blocking/features; **E-cores can slow LightGBM — benchmark thread count** |
| RAM | 32 GB (31.4 usable) | **the binding constraint** — everything large must be memmapped, never loaded |
| GPU | RTX 5080 Laptop, 16 GB GDDR7 (Blackwell, **sm_120**) | **requires CUDA 12.8+**; fits full dense index per country partition |
| Disk | 823 / 954 GB used → **~131 GB free** | workable, but keep intermediates off; free to ~200 GB for comfort |

---

## 1. Environment (Windows 11)

**Pin Python 3.11.** A stale cache showed a 3.14 run had been attempted; LightGBM /
pyarrow wheels lag on 3.14 and will either fail to build or fall back to slow paths.

```powershell
winget install astral-sh.uv
cd code\business_entity_resolution
uv venv .venv -p 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### GPU wheels — Blackwell needs CUDA 12.8

The RTX 5080 is `sm_120`. **cu121 wheels will not run on it** (you get "no kernel image is
available for execution on the device"). Install the cu128 build:

```powershell
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-dense.txt
```

Verify before committing to a long run:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expect `... True NVIDIA GeForce RTX 5080 Laptop GPU (12, 0)`. If capability prints `(12, 0)`
but `is_available()` is False, the driver is older than the toolkit — update the NVIDIA driver.

---

## 2. Disk budget

~24.2 M source records (12.53 M train + 11.70 M test). Estimated peak footprint:

| Artifact | Size |
|---|---|
| dataset (given) | 2.3 GB |
| normalized parquet caches | 4–5 GB |
| candidate pairs (≈157 M pairs, both splits) | ~2 GB |
| stage-1 features, fp16 × 69 cols | ~21 GB |
| stage-2 features | ~6 GB |
| dense embeddings, fp16 × 384-d | ~19 GB |
| models / folds / outputs | ~4 GB |
| **peak, `keep_intermediates=False`** | **~40 GB** |
| **peak, everything retained + dense** | **~60 GB** |

131 GB free is enough — with two conditions:

1. Leave `keep_intermediates: bool = False` (the default in
   `code/business_entity_resolution/src/config.py`). Its
   docstring already warns the retained matrices can exceed 40 GB on their own.
2. Put `work/` on the internal NVMe, not an external drive. Blocking and feature stages are
   IO-bound on random reads.

**Add a Defender exclusion for `work/`.** Real-time scanning on millions of memmap writes is
a large, silent slowdown:

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\amazon-ml-challenge-2026\work"
```

---

## 3. RAM guards (32 GB)

The full stage-1 feature matrix is 88 M rows × 69 cols. As fp32 in RAM that is 24 GB and will
OOM; as an fp16 memmap read in chunks it is fine. Settings to adjust in `src/config.py`:

| Knob | Default | Set to | Why |
|---|---|---|---|
| `join_budget_rows` | 40,000,000 | **20,000,000** | raw join rows per chunk; the main OOM risk |
| `feat_chunk` | 2,000,000 | 1,000,000 | smaller feature chunks |
| `max_train_rows` | 30,000,000 | 20,000,000 if LightGBM OOMs | 30 M × 69 fp32 ≈ 8.3 GB plus histograms |

Watch peak RSS on the first full stage; if it crosses ~26 GB, halve `join_budget_rows` again.

---

## 4. Thread tuning — measure, don't assume

Hybrid P/E-core CPUs frequently run LightGBM **faster on P-cores only**, because the scheduler
lets slow E-core threads stall each histogram sync barrier.

```powershell
# benchmark both on a small slice before the long run
$env:OMP_NUM_THREADS=8;  python src\run.py train --dev-frac 0.05
$env:OMP_NUM_THREADS=24; python src\run.py train --dev-frac 0.05
```

Use all 24 threads for normalization, blocking and rapidfuzz features (embarrassingly
parallel); use whichever won for LightGBM.

---

## 5. Expected runtimes

Rough, for full scale on this machine:

| Stage | Time |
|---|---|
| prepare / normalize (24.2 M records) | 10–20 min |
| blocking, both splits | 30–60 min |
| features (≈157 M pairs, rapidfuzz) | 60–120 min |
| LightGBM 2-stage CV | 60–120 min |
| dense embeddings, e5-small fp16 @ len 64 | 45–90 min |
| cross-encoder rerank, ~8 M uncertain pairs | 45–90 min |
| inference + decision + output | 20–40 min |
| **full run, no dense** | **~4–6 h** |
| **full run, with dense + rerank** | **~6–9 h** |

Iterate with `--dev-frac 0.1` for 10–20 minute cycles. Reserve full runs for overnight.

---

## 6. VRAM plan (16 GB)

| Use | Footprint | Fits |
|---|---|---|
| e5-small (118 M) fp16 inference, batch 512 × len 64 | <2 GB | easily |
| exact GPU kNN index, 384-d fp16, 11.7 M records | ~9 GB | yes, per country partition |
| bge-m3 (568 M) cross-encoder fp16, batch 128 × len 128 | ~3 GB | yes |

Do kNN **per country partition** rather than globally — it keeps the index under 16 GB and is
also semantically correct, since matches never cross countries. Chunk the query side.

---

## 7. Overnight run checklist

- [ ] Plugged in; Windows power plan **Best Performance**
- [ ] Sleep and hibernate disabled (`powercfg /change standby-timeout-ac 0`)
- [ ] Laptop on a cooling stand, rear vents clear — sustained 24-core + GPU load will throttle
      a laptop chassis, and a throttled run is a slow run, not a failed one
- [ ] ≥100 GB free confirmed after `work/` is seeded
- [ ] Defender exclusion added for `work/`
- [ ] `torch.cuda.is_available()` verified True
- [ ] Smoke test passes: `python tests\smoke_test.py` → `SMOKE TEST PASS`
- [ ] Stage caching confirmed (re-running skips finished stages; `--force` recomputes)
- [ ] **`report.json` retained** — losing per-country metrics is why the baseline plateaued

---

## 8. macOS note

This workspace is on macOS, where the dense/GPU path is unavailable (no CUDA). Use the Mac for
data analysis, diagnostics (`src/diagnose.py`) and code edits; run blocking, training and the
dense passes on the 5080 machine. The diagnostics script is stdlib + numpy only and runs
anywhere.
