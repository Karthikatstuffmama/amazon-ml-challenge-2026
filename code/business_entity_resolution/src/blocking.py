"""Stage 1: candidate generation (blocking) - IDF-weighted multi-key sort-merge join.

For each country partition, every record emits integer keys from 6 complementary
families. A query (S1) and an index record (S2/S3) become a candidate pair when they
share >= 1 key. Pair score = sum over shared keys of idf(key) = log(1 + N_idx/df).
This is a sparse retrieval engine over hand-designed keys, implemented with numpy
sort + searchsorted (no Python loops over records, no string keys, no hash maps):

  NT  name token x address token      (name + locality)          both rarest-first
  NS  skel(name tok) x skel(addr tok) (typos + transliteration)
  PH  nysiis(name tok) x skel(addr tok) (phonetic; independent error model from NS -
      catches transpositions/confusions the hand-built consonant skeleton misses, e.g.
      the problem statement's own 'imdastris' <-> 'industries' collide under NYSIIS)
  AN  house number x address token    (address anchor; name may be totally different)
  NH  name token x house number
  NP  name token pair                 (name only; handles missing/foreign address)
  NC  concatenated core name          (domains 'sjdcement.com', spacing variants)
  DENSE (optional) multilingual embedding kNN on GPU

Keys are packed as uint64: [type:8][a:28][b:28]. Tokens are factorised per family,
so key construction is pure integer arithmetic. Generic keys are pruned by document
frequency caps, which also bounds join size. The per-query chunking is budgeted by the
exact number of join rows so peak RAM is predictable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils import LOG, popcount16, rank_in_sorted_groups, safe_name, timed

T_NT, T_NS, T_AN, T_NH, T_NP, T_NC, T_PH, T_DENSE = 0, 1, 2, 3, 4, 5, 6, 7
TYPE_NAMES = {T_NT: "NT", T_NS: "NS", T_AN: "AN", T_NH: "NH", T_NP: "NP", T_NC: "NC",
              T_PH: "PH", T_DENSE: "DENSE"}
_S56 = np.uint64(56)
_S28 = np.uint64(28)
_M28 = (1 << 28) - 1


# --------------------------------------------------------------------------- #
# token families                                                              #
# --------------------------------------------------------------------------- #
def _explode(values, offset: int, first_k: int | None = None):
    s = pd.Series(values, dtype=object)
    lst = s.str.split()
    if first_k is not None:
        lst = lst.str[:first_k]
    ex = lst.explode()
    ex = ex[ex.notna()]
    ex = ex[ex.astype(bool)]
    return ex.index.to_numpy(dtype=np.int64) + offset, ex.to_numpy(dtype=object)


def _family(q_vals, i_vals, first_k=None):
    """Factorise a token family over the union of both sides.

    Returns rec (int64, combined space: q in [0,nq), idx in [nq, nq+ni)), code, df.
    (rec, code) pairs are unique.
    """
    nq = len(q_vals)
    rq, tq = _explode(q_vals, 0, first_k)
    ri, ti = _explode(i_vals, nq, first_k)
    rec = np.concatenate([rq, ri])
    tok = np.concatenate([tq, ti])
    if len(tok) == 0:
        return rec, np.zeros(0, np.int64), np.zeros(0, np.int64)
    codes, uniq = pd.factorize(tok, sort=False)
    U = len(uniq)
    if U >= _M28:
        raise OverflowError("token vocabulary exceeds 2^28")
    comb = np.unique(rec * np.int64(U + 1) + codes.astype(np.int64))
    rec = comb // (U + 1)
    codes = comb % (U + 1)
    df = np.bincount(codes, minlength=U)
    return rec, codes, df


def _topk_rare(rec, codes, df, k):
    """Keep the k rarest tokens (df >= 2) per record."""
    if len(rec) == 0:
        return rec, codes
    d = df[codes]
    m = d >= 2
    rec, codes, d = rec[m], codes[m], d[m]
    order = np.lexsort((codes, d, rec))
    rec, codes = rec[order], codes[order]
    keep = rank_in_sorted_groups(rec) < k
    return rec[keep], codes[keep]


def _single_keys(rec, codes, t):
    return rec, (np.uint64(t) << _S56) | codes.astype(np.uint64)


def _cross_keys(recA, a, recB, b, t, symmetric=False):
    if len(recA) == 0 or len(recB) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint64)
    A = pd.DataFrame({"r": recA, "a": a})
    B = pd.DataFrame({"r": recB, "b": b})
    M = A.merge(B, on="r", how="inner", sort=False)
    ra = M["a"].to_numpy(np.int64)
    rb = M["b"].to_numpy(np.int64)
    r = M["r"].to_numpy(np.int64)
    if symmetric:
        m = ra < rb
        r, ra, rb = r[m], ra[m], rb[m]
    key = (np.uint64(t) << _S56) | (ra.astype(np.uint64) << _S28) | rb.astype(np.uint64)
    return r, key


def build_keys(Q: pd.DataFrame, I: pd.DataFrame, cfg):
    """Return {type: (q_rec, q_key, i_rec, i_key)} with local record indices."""
    nq = len(Q)
    out = {}

    def split(rec, key):
        m = rec < nq
        return rec[m], key[m], rec[~m] - nq, key[~m]

    n_rec, n_code, n_df = _family(Q["n_core"].values, I["n_core"].values)
    a_rec, a_code, a_df = _family(Q["a_alpha"].values, I["a_alpha"].values)
    ns_rec, ns_code, ns_df = _family(Q["n_skel"].values, I["n_skel"].values)
    ph_rec, ph_code, ph_df = _family(Q["n_phon"].values, I["n_phon"].values)
    as_rec, as_code, as_df = _family(Q["a_skel"].values, I["a_skel"].values)
    u_rec, u_code, _ = _family(Q["a_nums"].values, I["a_nums"].values, first_k=cfg.num_topk)
    c_rec, c_code, _ = _family(Q["n_concat"].values, I["n_concat"].values)

    nr, nc = _topk_rare(n_rec, n_code, n_df, cfg.name_topk)
    ar, ac = _topk_rare(a_rec, a_code, a_df, cfg.addr_topk)
    nsr, nsc = _topk_rare(ns_rec, ns_code, ns_df, cfg.name_topk)
    phr, phc = _topk_rare(ph_rec, ph_code, ph_df, cfg.name_topk)
    asr, asc = _topk_rare(as_rec, as_code, as_df, max(1, cfg.addr_topk - 1))
    ar3, ac3 = _topk_rare(a_rec, a_code, a_df, max(1, cfg.addr_topk - 1))
    np3r, np3c = _topk_rare(n_rec, n_code, n_df, cfg.name_topk + 1)

    out[T_NT] = split(*_cross_keys(nr, nc, ar, ac, T_NT))
    out[T_NS] = split(*_cross_keys(nsr, nsc, asr, asc, T_NS))
    out[T_PH] = split(*_cross_keys(phr, phc, asr, asc, T_PH))
    out[T_AN] = split(*_cross_keys(u_rec, u_code, ar3, ac3, T_AN))
    out[T_NH] = split(*_cross_keys(nr, nc, u_rec, u_code, T_NH))
    out[T_NP] = split(*_cross_keys(np3r, np3c, np3r, np3c, T_NP, symmetric=True))
    out[T_NC] = split(*_single_keys(c_rec, c_code, T_NC))
    return out


# --------------------------------------------------------------------------- #
# sort-merge join                                                             #
# --------------------------------------------------------------------------- #
class _TypeIndex:
    """Pre-sorted index side + per-query-key match ranges for one key type."""

    def __init__(self, t, qrec, qkey, irec, ikey, ni, max_df, max_pairs):
        self.t = t
        o = np.argsort(ikey, kind="stable")
        self.ik = ikey[o]
        self.ir = irec[o].astype(np.int32)
        oq = np.argsort(qrec, kind="stable")
        self.qrec = qrec[oq].astype(np.int64)
        qkey = qkey[oq]
        lo = np.searchsorted(self.ik, qkey, "left")
        hi = np.searchsorted(self.ik, qkey, "right")
        cnt = (hi - lo).astype(np.int64)
        if len(qkey):
            uq, inv, cq = np.unique(qkey, return_inverse=True, return_counts=True)
            dfq = cq[inv]
        else:
            dfq = np.zeros(0, np.int64)
        ok = (cnt > 0) & (cnt <= max_df) & (cnt * dfq <= max_pairs)
        self.qrec, self.lo, self.cnt = self.qrec[ok], lo[ok], cnt[ok]
        self.w = np.log1p(ni / np.maximum(self.cnt, 1)).astype(np.float32)

    def rows_per_query(self, nq):
        return np.bincount(self.qrec, weights=self.cnt, minlength=nq)

    def join(self, q_lo, q_hi):
        a = np.searchsorted(self.qrec, q_lo, "left")
        b = np.searchsorted(self.qrec, q_hi, "left")
        qr, lo, cnt, w = self.qrec[a:b], self.lo[a:b], self.cnt[a:b], self.w[a:b]
        total = int(cnt.sum())
        if total == 0:
            return np.zeros(0, np.uint64), np.zeros(0, np.float32)
        rep_q = np.repeat(qr, cnt)
        offs = np.cumsum(cnt) - cnt
        pos = np.repeat(lo - offs, cnt) + np.arange(total, dtype=np.int64)
        rep_i = self.ir[pos]
        pid = (rep_q.astype(np.uint64) << np.uint64(32)) | rep_i.astype(np.uint64)
        wv = np.repeat(w, cnt)
        u, inv = np.unique(pid, return_inverse=True)
        return u, np.bincount(inv, weights=wv, minlength=len(u)).astype(np.float32)


def _query_chunks(rows_per_q: np.ndarray, budget: int):
    """Greedy contiguous query ranges whose total join rows stay under budget."""
    nq = len(rows_per_q)
    if nq == 0:
        return []
    cs = np.cumsum(rows_per_q)
    bounds = [0]
    base = 0.0
    while bounds[-1] < nq:
        nxt = int(np.searchsorted(cs, base + budget, "right"))
        nxt = max(nxt, bounds[-1] + 1)
        nxt = min(nxt, nq)
        bounds.append(nxt)
        base = cs[nxt - 1]
    return list(zip(bounds[:-1], bounds[1:]))


def block_partition(Q, I, cfg):
    """Key-based candidates for one partition. Returns dict of aligned arrays."""
    nq, ni = len(Q), len(I)
    empty = dict(q=np.zeros(0, np.int32), i=np.zeros(0, np.int32),
                 bscore=np.zeros(0, np.float32), bits=np.zeros(0, np.uint16),
                 brank=np.zeros(0, np.int16))
    if nq == 0 or ni == 0:
        return empty
    keys = build_keys(Q, I, cfg)
    idxs = []
    for t, (qr, qk, ir, ik) in keys.items():
        name_only = t in (T_NP, T_NC)
        ti = _TypeIndex(t, qr, qk, ir, ik, ni,
                        cfg.max_df_name if name_only else cfg.max_df_pair,
                        cfg.max_pairs_per_key)
        LOG.info("    key %-3s q_keys=%10d idx_keys=%10d usable=%10d join_rows=%12d",
                 TYPE_NAMES[t], len(qk), len(ik), len(ti.cnt), int(ti.cnt.sum()))
        idxs.append(ti)
    del keys
    rows_q = sum(ti.rows_per_query(nq) for ti in idxs)
    chunks = _query_chunks(rows_q, cfg.join_budget_rows)
    out_q, out_i, out_s, out_b, out_r = [], [], [], [], []
    for (a, b) in chunks:
        pids, ws, bits = [], [], []
        for ti in idxs:
            u, w = ti.join(a, b)
            pids.append(u)
            ws.append(w)
            bits.append(np.full(len(u), 1 << ti.t, dtype=np.uint16))
        pid = np.concatenate(pids)
        if len(pid) == 0:
            continue
        w = np.concatenate(ws)
        bt = np.concatenate(bits)
        u, inv = np.unique(pid, return_inverse=True)
        score = np.bincount(inv, weights=w, minlength=len(u)).astype(np.float32)
        # each type contributes a pid at most once, so summing bits == OR-ing them
        bsum = np.bincount(inv, weights=bt.astype(np.float64), minlength=len(u)).astype(np.uint16)
        q = (u >> np.uint64(32)).astype(np.int32)
        i = (u & np.uint64(0xFFFFFFFF)).astype(np.int32)
        # rank by summed idf; ties broken by number of distinct key families
        ntypes = popcount16(bsum)
        order = np.lexsort((-ntypes, -score, q))
        q, i, score, bsum = q[order], i[order], score[order], bsum[order]
        rk = rank_in_sorted_groups(q)
        keep = rk < cfg.k_key
        out_q.append(q[keep])
        out_i.append(i[keep])
        out_s.append(score[keep])
        out_b.append(bsum[keep])
        out_r.append(rk[keep].astype(np.int16))
    if not out_q:
        return empty
    return dict(q=np.concatenate(out_q), i=np.concatenate(out_i),
                bscore=np.concatenate(out_s), bits=np.concatenate(out_b),
                brank=np.concatenate(out_r))


def merge_dense(P: dict, dq, di, dsim, k_key: int):
    """Union key candidates with dense kNN candidates (bit 7). Keeps q-sorted order."""
    kq = P["q"].astype(np.int64)
    ki = P["i"].astype(np.int64)
    kp = (kq << 32) | ki
    dp = (dq.astype(np.int64) << 32) | di.astype(np.int64)
    allp = np.concatenate([kp, dp])
    u, inv = np.unique(allp, return_inverse=True)
    n = len(u)
    bscore = np.zeros(n, np.float32)
    bits = np.zeros(n, np.uint16)
    brank = np.full(n, k_key + 50, np.int16)
    dcos = np.full(n, np.nan, np.float32)
    ik, idn = inv[:len(kp)], inv[len(kp):]
    bscore[ik] = P["bscore"]
    bits[ik] = P["bits"]
    brank[ik] = P["brank"]
    bits[idn] |= np.uint16(1 << T_DENSE)
    dcos[idn] = dsim
    return dict(q=(u >> 32).astype(np.int32), i=(u & 0xFFFFFFFF).astype(np.int32),
                bscore=bscore, bits=bits, brank=brank, dcos=dcos)


BLOCK_COLS = ["n_core", "n_skel", "n_phon", "n_concat", "a_alpha", "a_skel", "a_nums"]


def run_blocking(cfg, split: str, s1, idx, parts, emb=None) -> dict:
    """Blocks every partition; caches work/<split>/pairs_<country>.npz. Returns stats."""
    from prepare import split_dir
    d = split_dir(cfg, split)
    stats = {}
    for ck, (s1_rows, idx_rows) in parts.items():
        path = os.path.join(d, f"pairs_{safe_name(ck)}.npz")
        if os.path.exists(path) and not cfg.force:
            z = np.load(path)
            stats[ck] = dict(n_s1=len(s1_rows), n_idx=len(idx_rows), pairs=len(z["q"]))
            continue
        with timed(f"block:{split}:{ck!r} S1={len(s1_rows):,} IDX={len(idx_rows):,}"):
            Q = s1.iloc[s1_rows][BLOCK_COLS].reset_index(drop=True)
            I = idx.iloc[idx_rows][BLOCK_COLS].reset_index(drop=True)
            P = block_partition(Q, I, cfg)
            if emb is not None and len(s1_rows) and len(idx_rows):
                from dense import knn_topk
                eq, ei = emb
                dq, di, ds = knn_topk(eq, ei, s1_rows, idx_rows, cfg.k_dense)
                P = merge_dense(P, dq, di, ds, cfg.k_key)
            else:
                P["dcos"] = np.full(len(P["q"]), np.nan, np.float32)
            np.savez(path, **P)
            stats[ck] = dict(n_s1=len(s1_rows), n_idx=len(idx_rows), pairs=len(P["q"]))
            LOG.info("  %r: %d pairs (%.1f / S1)", ck, len(P["q"]),
                     len(P["q"]) / max(1, len(s1_rows)))
    return stats


def load_pairs(cfg, split: str, ck: str) -> dict:
    from prepare import split_dir
    z = np.load(os.path.join(split_dir(cfg, split), f"pairs_{safe_name(ck)}.npz"))
    return {k: z[k] for k in z.files}
