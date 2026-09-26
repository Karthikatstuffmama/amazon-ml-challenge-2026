"""Optional GPU/MPS pass: multilingual sentence embeddings + exact chunked kNN.

Model: intfloat/multilingual-e5-small (MIT, 118M params) - reads Devanagari, Kannada,
Tamil, Telugu, Bengali, Gujarati and French natively, so it recovers cross-script
matches that transliteration + keys miss. Enabled with --dense. Embeddings are
L2-normalised fp16 memmaps (N x 384 x 2 bytes) cached on disk.

kNN is exact brute force (Q @ I^T + topk), chunked on query and index axes.
Optimised for CUDA and Apple MPS (MacBook); falls back to threaded CPU.
"""
from __future__ import annotations

import os
import time

# Must be set before the first `import torch` in this process: on Windows, PyTorch's
# cu12x wheels ship multi-GB CUDA runtime DLLs (curand/cublas/nvJitLink) whose eager
# load can fail with "[WinError 1455] The paging file is too small" on machines with a
# system-managed pagefile, even with ample RAM. Lazy module loading defers per-kernel
# loading and avoids the up-front commit spike. Harmless on Linux/macOS (ignored) and
# a no-op if the caller already set it.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Prefer higher MPS memory use before reclaim (helps large encode/knn batches on unified RAM).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np

from utils import LOG, timed


def _pick_device():
    """Prefer CUDA, then Apple MPS, else CPU. Returns (device_str, use_fp16)."""
    import torch
    if torch.cuda.is_available():
        return "cuda", True
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def _texts(df):
    n = df["business_name"].astype(str).tolist()
    a = df["business_address"].astype(str).tolist()
    return [f"query: {x} | {y}" for x, y in zip(n, a)]


def _best_batch(model, texts_probe, candidates, dev: str) -> int:
    """Pick the fastest encode batch that does not OOM (short probe)."""
    import torch
    best_b, best_r = candidates[0], -1.0
    probe = texts_probe[: max(candidates)]
    # warmup
    with torch.inference_mode():
        model.encode(probe[: min(128, len(probe))], batch_size=min(128, candidates[0]),
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    for b in candidates:
        if b > len(probe):
            continue
        try:
            if dev == "mps":
                torch.mps.empty_cache()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.encode(probe, batch_size=b, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            rate = len(probe) / max(1e-6, time.perf_counter() - t0)
            LOG.info("    batch-tune %d -> %.0f rows/s", b, rate)
            if rate > best_r:
                best_r, best_b = rate, b
        except Exception as e:  # noqa: BLE001 — OOM / MPS abort
            LOG.info("    batch-tune %d failed (%s); stopping ramp", b, type(e).__name__)
            break
    return best_b


def build_embeddings(cfg, split: str, s1, idx):
    """Returns (emb_s1, emb_idx) as read-only fp16 memmaps."""
    from prepare import split_dir
    d = split_dir(cfg, split)
    out = []
    for tag, df in (("s1", s1), ("idx", idx)):
        path = os.path.join(d, f"emb_{tag}.npy")
        if not (os.path.exists(path) and not cfg.force):
            _encode_to(path, _texts(df), cfg)
        out.append(np.load(path, mmap_mode="r"))
    return tuple(out)


def _encode_to(path, texts, cfg):
    """Encode corpus → fp16 memmap with auto-tuned batch + inference_mode.

    Stages stay sequential (S1 then idx; countries later). Inside each chunk the
    accelerator parallelises the matmul — that is the safe parallelism on 16 GB.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    ncpu = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(ncpu)
    os.environ.setdefault("OMP_NUM_THREADS", str(ncpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(ncpu))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(ncpu))

    dev, use_fp16 = _pick_device()
    model = SentenceTransformer(cfg.dense_model, device=dev)
    model.max_seq_length = cfg.dense_max_len
    model.eval()
    if use_fp16 and dev == "cuda":
        model.half()

    base = int(cfg.dense_batch)
    if dev == "cuda":
        cands = sorted({max(base, 512), 512, 768, 1024, 1536})
    elif dev == "mps":
        # Empirically ~256 is the knee on M5 Air; 1024 hung. Probe around the knee.
        cands = [128, 256, 384, 512, 640]
    else:
        cands = [64, 128, 256]

    if getattr(cfg, "dense_batch_fixed", False):
        batch = int(cfg.dense_batch)
        LOG.info("    pinned encode batch=%d on %s (probe skipped)", batch, dev)
    else:
        probe_n = min(len(texts), max(cands) * 2, 2048)
        batch = _best_batch(model, texts[:probe_n], cands, dev) if len(texts) >= 256 else cands[0]
        LOG.info("    selected encode batch=%d on %s", batch, dev)

    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    tmp = path + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, dim))
    # Larger steps = less Python overhead; still sequential chunks.
    step = max(batch * 16, 8192)
    t_start = time.perf_counter()
    with timed(f"dense:encode {os.path.basename(path)} n={n:,} on {dev} batch={batch} step={step}"):
        with torch.inference_mode():
            for i, s in enumerate(range(0, n, step)):
                e = model.encode(
                    texts[s:s + step],
                    batch_size=batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                mm[s:s + len(e)] = e.astype(np.float16)
                done = s + len(e)
                elapsed = max(1e-6, time.perf_counter() - t_start)
                LOG.info("    encoded %d / %d (%.0f%%)  %.0f rows/s",
                         done, n, 100.0 * done / n, done / elapsed)
                # Reclaim rarely — empty_cache every step killed throughput.
                if dev == "mps" and (i + 1) % 8 == 0:
                    torch.mps.empty_cache()
    mm.flush()
    del mm
    os.replace(tmp, path)


def knn_topk(emb_q, emb_i, q_rows, i_rows, k):
    """Exact top-k cosine for rows q_rows of emb_q against rows i_rows of emb_i.

    Returns local indices (into q_rows / i_rows) and similarities, sorted by q.
    Tuned for MPS: keep working tensors on-device, larger chunks, infrequent cache clears.
    """
    import torch
    dev_s, _ = _pick_device()
    dev = torch.device(dev_s)
    dt = torch.float16 if dev_s == "cuda" else torch.float32
    nq, ni = len(q_rows), len(i_rows)
    k = int(min(k, ni))
    if nq == 0 or k == 0:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))

    if dev_s == "cuda":
        free, _ = torch.cuda.mem_get_info()
    elif dev_s == "mps":
        # Unified memory: push harder than the old 2 GB cap (was leaving M5 idle).
        free = 6 * 2**30
    else:
        free = 3 * 2**30
    dim = emb_i.shape[1]
    bpe = 2 if dt == torch.float16 else 4
    ic = int(max(50_000, min(ni, 0.45 * free / (dim * bpe))))
    qc = int(max(128, min(8192, 0.25 * free / (max(ic, 1) * bpe))))
    i_rows = np.asarray(i_rows)
    q_rows = np.asarray(q_rows)

    with timed(f"dense:knn nq={nq:,} ni={ni:,} k={k} ichunk={ic:,} qchunk={qc} on {dev_s}"):
        with torch.inference_mode():
            best_v = torch.full((nq, k), -2.0, dtype=torch.float32, device=dev)
            best_i = torch.zeros((nq, k), dtype=torch.int64, device=dev)
            for i0 in range(0, ni, ic):
                I = torch.as_tensor(
                    np.ascontiguousarray(emb_i[i_rows[i0:i0 + ic]]),
                    device=dev, dtype=dt,
                )
                kk = min(k, I.shape[0])
                for q0 in range(0, nq, qc):
                    Q = torch.as_tensor(
                        np.ascontiguousarray(emb_q[q_rows[q0:q0 + qc]]),
                        device=dev, dtype=dt,
                    )
                    v, ix = torch.topk(Q @ I.T, kk, dim=1)
                    v = v.float()
                    ix = ix + i0
                    cv = torch.cat([best_v[q0:q0 + qc], v], 1)
                    ci = torch.cat([best_i[q0:q0 + qc], ix], 1)
                    tv, tix = torch.topk(cv, k, dim=1)
                    best_v[q0:q0 + qc] = tv
                    best_i[q0:q0 + qc] = torch.gather(ci, 1, tix)
                del I
                if dev_s == "cuda":
                    torch.cuda.empty_cache()
            v = best_v.cpu().numpy()
            ix = best_i.cpu().numpy()
            if dev_s == "mps":
                torch.mps.empty_cache()

    qq = np.repeat(np.arange(nq, dtype=np.int32), k)
    ii = ix.reshape(-1).astype(np.int32)
    ss = v.reshape(-1).astype(np.float32)
    m = ss > -1.5
    return qq[m], ii[m], ss[m]
