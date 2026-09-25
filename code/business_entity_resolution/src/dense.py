"""Optional GPU pass: multilingual sentence embeddings + exact chunked kNN.

Model: intfloat/multilingual-e5-small (MIT, 118M params) - reads Devanagari, Kannada,
Tamil, Telugu, Bengali, Gujarati and French natively, so it recovers cross-script
matches that transliteration + keys miss. Enabled with --dense. Embeddings are
L2-normalised fp16 memmaps (N x 384 x 2 bytes) cached on disk.

kNN is exact brute force (Q @ I^T + topk) in fp16 tensor cores, chunked on both the
query and index axes so it fits any GPU; ~1.5e15 FLOP for the biggest partition,
i.e. a few minutes on an RTX 5080. Falls back to CPU fp32 automatically.
"""
from __future__ import annotations

import os

# Must be set before the first `import torch` in this process: on Windows, PyTorch's
# cu12x wheels ship multi-GB CUDA runtime DLLs (curand/cublas/nvJitLink) whose eager
# load can fail with "[WinError 1455] The paging file is too small" on machines with a
# system-managed pagefile, even with ample RAM. Lazy module loading defers per-kernel
# loading and avoids the up-front commit spike. Harmless on Linux/macOS (ignored) and
# a no-op if the caller already set it.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np

from utils import LOG, timed


def _texts(df):
    n = df["business_name"].astype(str).tolist()
    a = df["business_address"].astype(str).tolist()
    return [f"query: {x} | {y}" for x, y in zip(n, a)]


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
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg.dense_model, device=dev)
    model.max_seq_length = cfg.dense_max_len
    if dev == "cuda":
        model.half()
    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    tmp = path + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, dim))
    step = 200_000
    with timed(f"dense:encode {os.path.basename(path)} n={n:,} on {dev}"):
        for s in range(0, n, step):
            e = model.encode(texts[s:s + step], batch_size=cfg.dense_batch,
                             normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)
            mm[s:s + len(e)] = e.astype(np.float16)
            if (s // step) % 10 == 0:
                LOG.info("    encoded %d / %d", s + len(e), n)
    mm.flush()
    del mm
    os.replace(tmp, path)


def knn_topk(emb_q, emb_i, q_rows, i_rows, k):
    """Exact top-k cosine for rows q_rows of emb_q against rows i_rows of emb_i.

    Returns local indices (into q_rows / i_rows) and similarities, sorted by q.
    """
    import torch
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    dt = torch.float16 if cuda else torch.float32
    nq, ni = len(q_rows), len(i_rows)
    k = int(min(k, ni))
    if nq == 0 or k == 0:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    # memory plan: index chunk <= 40% of free memory, score block <= 20%
    if cuda:
        free, _ = torch.cuda.mem_get_info()
    else:
        free = 4 * 2**30
    dim = emb_i.shape[1]
    bpe = 2 if cuda else 4
    ic = int(max(100_000, min(ni, 0.4 * free / (dim * bpe))))
    qc = int(max(64, min(8192, 0.2 * free / (ic * bpe))))
    out_q, out_i, out_s = [], [], []
    i_rows = np.asarray(i_rows)
    q_rows = np.asarray(q_rows)
    with timed(f"dense:knn nq={nq:,} ni={ni:,} k={k} ichunk={ic:,} qchunk={qc}"):
        best_v = torch.full((nq, k), -2.0, dtype=torch.float32)
        best_i = torch.zeros((nq, k), dtype=torch.int64)
        for i0 in range(0, ni, ic):
            I = torch.from_numpy(np.ascontiguousarray(emb_i[i_rows[i0:i0 + ic]])).to(dev, dt)
            kk = min(k, I.shape[0])
            for q0 in range(0, nq, qc):
                Q = torch.from_numpy(np.ascontiguousarray(emb_q[q_rows[q0:q0 + qc]])).to(dev, dt)
                v, ix = torch.topk(Q @ I.T, kk, dim=1)
                v = v.float().cpu()
                ix = ix.cpu() + i0
                cv = torch.cat([best_v[q0:q0 + qc], v], 1)
                ci = torch.cat([best_i[q0:q0 + qc], ix], 1)
                tv, tix = torch.topk(cv, k, dim=1)
                best_v[q0:q0 + qc] = tv
                best_i[q0:q0 + qc] = torch.gather(ci, 1, tix)
            del I
            if cuda:
                torch.cuda.empty_cache()
    v = best_v.numpy()
    ix = best_i.numpy()
    qq = np.repeat(np.arange(nq, dtype=np.int32), k)
    ii = ix.reshape(-1).astype(np.int32)
    ss = v.reshape(-1).astype(np.float32)
    m = ss > -1.5
    return qq[m], ii[m], ss[m]
