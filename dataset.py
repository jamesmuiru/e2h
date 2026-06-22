"""
dataset.py — GeoFMDataset with defensive loading, shape enforcement, and robust matching.
"""

import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try: import tifffile
except ImportError: tifffile = None
try: import rasterio
except ImportError: rasterio = None

from config import CATALOG, DATA_ROOT

# ── 1. File Loading & Sanitization ────────────────────────────────────────

def load_multiband_array(path: Path) -> np.ndarray:
    """Load, sanitize NaNs, and ensure (H, W, C) format."""
    suffix = path.suffix.lower()
    arr = None
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        if tifffile:
            try: arr = tifffile.imread(str(path))
            except: pass
        if arr is None and rasterio:
            try:
                with rasterio.open(str(path)) as src:
                    arr = src.read()
            except: pass
            
    if arr is None:
        raise RuntimeError(f"Could not load file: {path}")

    # DEFENSIVE: Sanitize NaNs/Infs immediately
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Force HWC: Move channels to last if they are at the front (CHW)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.moveaxis(arr, 0, -1)
        
    return arr.astype(np.float32)

def force_shape(arr: np.ndarray, target_h: int, target_w: int, target_c: int) -> np.ndarray:
    """Pad or Crop to strictly match (target_h, target_w, target_c)."""
    h, w, c = arr.shape
    if h == target_h and w == target_w and c == target_c: return arr
    out = np.zeros((target_h, target_w, target_c), dtype=np.float32)
    h_crop, w_crop, c_crop = min(h, target_h), min(w, target_w), min(c, target_c)
    out[:h_crop, :w_crop, :c_crop] = arr[:h_crop, :w_crop, :c_crop]
    return out

# ── 2. Catalog Processing ─────────────────────────────────────────────────

def _extract_patch_stem(filename: str) -> str:
    """Hunt for the 6-character Patch ID (e.g. 0000_BE) and ignore all prefixes/suffixes."""
    match = re.search(r'(\d{4}_[A-Z]{2})', filename)
    if match:
        return match.group(1)
        
    # Fallback just in case
    stem = Path(filename).stem
    for prefix in ["alphaearth_emb_", "tessera_emb_", "label_", "s1_", "s2_", "terramind_", "thor_", "gee_emb_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = stem.replace("_embeddings", "").replace("_embedding", "")
    stem = re.sub(r'_\d{4}$', '', stem)
    return stem

def build_sample_lists(catalog_path: Path = CATALOG, base_dir: Path = DATA_ROOT):
    df = pd.read_parquet(catalog_path)
    train_by_mod = defaultdict(dict)
    test_by_mod = defaultdict(dict)
    
    TRAIN_MODS = {
        "data/train/alphaearth_emb/": "alphaearth", "data/train/tessera_emb/": "tessera",
        "data/train/terramind_s1_emb/": "terramind_s1", "data/train/terramind_s2_emb/": "terramind_s2",
        "data/train/thor_s1_emb/": "thor_s1", "data/train/thor_s2_emb/": "thor_s2",
        "data/train/labels/": "labels",
    }
    
    TEST_MODS = {
        "data/test/alphaearth_test_emb/": "alphaearth", "data/test/tessera_test_emb/": "tessera",
        "data/test/terramind_test_s1_emb/": "terramind_s1", "data/test/terramind_test_s2_emb/": "terramind_s2",
        "data/test/thor_test_s1_emb/": "thor_s1", "data/test/thor_test_s2_emb/": "thor_s2",
    }
    
    for id_str in df["id"].astype(str):
        clean_rel_path = id_str[5:] if id_str.startswith("data/") else id_str
        abs_path = base_dir / clean_rel_path
        
        for prefix, mod_key in TRAIN_MODS.items():
            if id_str.startswith(prefix):
                stem = _extract_patch_stem(Path(id_str).name)
                train_by_mod[mod_key][stem] = abs_path
                
        for prefix, mod_key in TEST_MODS.items():
            if id_str.startswith(prefix):
                stem = _extract_patch_stem(Path(id_str).name)
                test_by_mod[mod_key][stem] = abs_path
    
    # Build Train
    train_samples = []
    for stem in sorted(train_by_mod["alphaearth"].keys()):
        try:
            paths = {mod: train_by_mod[mod][stem] for mod in TRAIN_MODS.values()}
            if all(p.exists() for p in paths.values()):
                paths["stem"] = stem
                train_samples.append(paths)
        except KeyError: continue
        
    # Build Test
    test_stems = set(test_by_mod["alphaearth"].keys()) if "alphaearth" in test_by_mod else set()
    if not test_stems and "thor_s1" in test_by_mod:
        test_stems = set(test_by_mod["thor_s1"].keys())
        for mod in ["thor_s2", "terramind_s1", "terramind_s2"]:
            if mod in test_by_mod:
                test_stems &= set(test_by_mod[mod].keys())

    test_samples = []
    for stem in sorted(test_stems):
        entry = {"stem": stem}
        valid = True
        for mod in TEST_MODS.values():
            p = test_by_mod[mod].get(stem)
            if p and p.exists():
                entry[mod] = p
            else:
                valid = False
                break
        if valid:
            test_samples.append(entry)
            
    return train_samples, test_samples

def split_samples(samples: list, val_frac: float = 0.10, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_frac))
    return [samples[i] for i in idx[n_val:]], [samples[i] for i in idx[:n_val]]

# ── 3. Dataset Classes ────────────────────────────────────────────────────

def chw(a): return torch.from_numpy(a.transpose(2, 0, 1).copy())
def norm(a): return (a - a.mean()) / (a.std() + 1e-6)

class GeoFMDataset(Dataset):
    def __init__(self, sample_list: list, augment: bool = False):
        self.samples = sample_list
        self.augment = augment
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        alpha   = force_shape(load_multiband_array(s["alphaearth"]), 256, 256, 64)
        tessera = force_shape(load_multiband_array(s["tessera"]), 256, 256, 128)
        tm_s1   = force_shape(load_multiband_array(s["terramind_s1"]), 16, 16, 768)
        tm_s2   = force_shape(load_multiband_array(s["terramind_s2"]), 16, 16, 768)
        th_s1   = force_shape(load_multiband_array(s["thor_s1"]), 16, 16, 768)
        th_s2   = force_shape(load_multiband_array(s["thor_s2"]), 16, 16, 768)
        label   = force_shape(load_multiband_array(s["labels"]), 256, 256, 4)
        
        tokens = np.concatenate([tm_s1, tm_s2], axis=-1)
        thor   = np.concatenate([th_s1, th_s2], axis=-1)
        
        return {
            "alpha": chw(norm(alpha)), "tessera": chw(norm(tessera)),
            "tokens": chw(norm(tokens)), "thor": chw(norm(thor)),
            "label": chw(label), "stem": s["stem"]
        }

class GeoFMTestDataset(Dataset):
    def __init__(self, sample_list: list):
        self.samples = sample_list
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        alpha   = force_shape(load_multiband_array(s["alphaearth"]), 256, 256, 64)
        tessera = force_shape(load_multiband_array(s["tessera"]), 256, 256, 128)
        tm_s1   = force_shape(load_multiband_array(s["terramind_s1"]), 16, 16, 768)
        tm_s2   = force_shape(load_multiband_array(s["terramind_s2"]), 16, 16, 768)
        th_s1   = force_shape(load_multiband_array(s["thor_s1"]), 16, 16, 768)
        th_s2   = force_shape(load_multiband_array(s["thor_s2"]), 16, 16, 768)
        
        tokens = np.concatenate([tm_s1, tm_s2], axis=-1)
        thor   = np.concatenate([th_s1, th_s2], axis=-1)
        
        return {
            "alpha": chw(norm(alpha)), "tessera": chw(norm(tessera)),
            "tokens": chw(norm(tokens)), "thor": chw(norm(thor)),
            "stem": s["stem"]
        }

if __name__ == "__main__":
    train_samples, test_samples = build_sample_lists()
    print(f"Matched Train Samples: {len(train_samples)}")
    print(f"Matched Test Samples: {len(test_samples)}")
