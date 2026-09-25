"""Single source of truth for every tunable knob. Serialised next to the models so a
test run can prove it used the same feature/blocking settings as training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # ---------------- paths ----------------
    data_dir: str = ""                # folder containing train/ and test/
    work_dir: str = "work"            # caches, memmaps, models
    out_dir: str = "output"           # matching_results.tsv + candidate_pairs.tsv
    validator: str = ""               # optional path to utils/validate_submission.py

    # ---------------- runtime --------------
    workers: int = 0                  # 0 => os.cpu_count()
    seed: int = 42
    force: bool = False               # recompute cached stages
    keep_intermediates: bool = False  # keep large per-partition feat_*/s2_*.npy after use
                                       # (default False: delete them once consumed within a
                                       # run to bound disk use — at full scale, stage-1 (fp16)
                                       # + stage-2 feature matrices for train AND test can
                                       # exceed 40 GB combined if all kept simultaneously;
                                       # pairs_*.npz/parquet caches, which are cheap in size
                                       # and expensive to recompute, are always kept)

    # ---------------- blocking -------------
    k_key: int = 40                   # max key-based candidates per S1 record
    k_dense: int = 15                 # extra candidates from the (optional) dense pass
    name_topk: int = 2                # rarest name tokens used in keys
    addr_topk: int = 4                # rarest address tokens used in keys
    num_topk: int = 2                 # first N numbers of the address used in keys
    max_df_pair: int = 500            # drop a compound key if > this many index records share it
    max_df_name: int = 200            # stricter cap for name-only keys
    max_pairs_per_key: int = 50_000   # df_query * df_index cap (kills pathological keys)
    join_budget_rows: int = 40_000_000  # raw join rows per query chunk (RAM guard)

    # ---------------- dense (optional GPU pass) ---------------
    dense: bool = False
    dense_model: str = "intfloat/multilingual-e5-small"   # MIT licence, 118M params
    dense_batch: int = 512
    dense_max_len: int = 64

    # ---------------- features ------------
    feat_chunk: int = 2_000_000       # pairs per feature chunk
    hash_features: int = 2 ** 20      # hashing-trick width for TF-IDF

    # ---------------- training ------------
    stage1_folds: list = field(default_factory=lambda: [0, 1, 2])
    stage2_folds: list = field(default_factory=lambda: [3, 4, 5, 6, 7])
    holdout_folds: list = field(default_factory=lambda: [8, 9])
    lgb_rounds: int = 3000
    lgb_lr: float = 0.08
    lgb_leaves: int = 191
    lgb_min_leaf: int = 150
    early_stop: int = 100
    monotone: bool = True
    max_train_rows: int = 30_000_000
    dev_frac: float = 1.0             # <1.0 => consistent entity sub-sample of TRAIN (fast iteration)
    train_countries: list = field(default_factory=list)   # [] => all (LOCO experiments)

    # ---------------- decision ------------
    decision: str = "auto"            # auto | threshold | expected_f

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Settings that change the *meaning* of features; must be identical at train and test.
FEATURE_CONTRACT = ("k_key", "k_dense", "name_topk", "addr_topk", "num_topk",
                    "max_df_pair", "max_df_name", "max_pairs_per_key", "dense",
                    "dense_model", "hash_features")
