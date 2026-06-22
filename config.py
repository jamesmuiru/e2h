"""
config.py — single source of truth for all paths and hyperparameters.
"""
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/scratch/lustre/users/hmwangi/embed2heights")
# Notice the double 'data/data' to match your ls output
DATA_ROOT    = PROJECT_ROOT / "data" / "data"
CONDA_ENV    = "/scratch/lustre/users/hmwangi/conda_envs/sr3_env"

# ── Data paths ─────────────────────────────────────────────────────────────
CATALOG      = DATA_ROOT / "catalog.v1.parquet"

TRAIN_DIR    = DATA_ROOT / "train"
TEST_DIR     = DATA_ROOT / "test"

# Train modality folders
TRAIN_FOLDERS = {
    "alphaearth":   TRAIN_DIR / "alphaearth_emb",
    "tessera":      TRAIN_DIR / "tessera_emb",
    "terramind_s1": TRAIN_DIR / "terramind_s1_emb",
    "terramind_s2": TRAIN_DIR / "terramind_s2_emb",
    "thor_s1":      TRAIN_DIR / "thor_s1_emb",
    "thor_s2":      TRAIN_DIR / "thor_s2_emb",
    "labels":       TRAIN_DIR / "labels",
}

# Test modality folders
TEST_FOLDERS = {
    "alphaearth":   TEST_DIR / "alphaearth_test_emb",
    "tessera":      TEST_DIR / "tessera_test_emb",
    "terramind_s1": TEST_DIR / "terramind_test_s1_emb",
    "terramind_s2": TEST_DIR / "terramind_test_s2_emb",
    "thor_s1":      TEST_DIR / "thor_test_s1_emb",
    "thor_s2":      TEST_DIR / "thor_test_s2_emb",
}

# ── Output paths ───────────────────────────────────────────────────────────
RUNS_DIR     = PROJECT_ROOT / "runs"
EXP_NAME     = "exp01_convnext"
EXP_DIR      = RUNS_DIR / EXP_NAME
CKPT_DIR     = EXP_DIR / "checkpoints"
LOG_DIR      = EXP_DIR / "logs"
PRED_DIR     = EXP_DIR / "predictions"
SUBMIT_DIR   = EXP_DIR / "submission"

# ── Model hyperparameters ──────────────────────────────────────────────────
MODEL = dict(
    D           = 256,
    token_dim   = 512,
    n_heads     = 8,
    n_tx_layers = 4,
)

# ── Training hyperparameters ───────────────────────────────────────────────
TRAIN = dict(
    seed        = 42,
    val_frac    = 0.10,
    batch_size  = 16,       
    num_workers = 16,       
    epochs      = 80,
    lr          = 1e-4,
    weight_decay= 1e-2,
    w_seg       = 0.45,
    w_height    = 0.55,
    grad_clip   = 1.0,
    save_top_k  = 3,
)

# ── Inference ──────────────────────────────────────────────────────────────
INFER = dict(
    tta         = True,
    batch_size  = 32,
)

# ── Evaluation metric weights ──────────────────────────────────────────────
METRIC_WEIGHTS = {
    "mIoU_buildings":         0.25,
    "mIoU_trees":             0.15,
    "mIoU_water":             0.15,
    "RMSE_building_height":   0.25,
    "RMSE_vegetation_height": 0.20,
}