
"""

predict.py — Inference on the test set. Saves one .npy per patch.



Usage:

    cd /scratch/lustre/users/hmwangi/embed2heights

    python predict.py

    python predict.py --ckpt runs/exp01_convnext/checkpoints/ckpt_best.pt

    python predict.py --no_tta   # disable test-time augmentation

"""



import argparse

import sys

import time

from pathlib import Path



import numpy as np

import torch

from torch.amp import autocast



from config import (

    CATALOG, PROJECT_ROOT, CKPT_DIR, PRED_DIR, MODEL, INFER, DATA_ROOT

)

from dataset import (

    GeoFMTestDataset, build_sample_lists

)

from model import build_model





# ══════════════════════════════════════════════════════════════════════════════

# TTA helpers

# ══════════════════════════════════════════════════════════════════════════════



def _flip_h(t: torch.Tensor) -> torch.Tensor: return torch.flip(t, [-1])

def _flip_v(t: torch.Tensor) -> torch.Tensor: return torch.flip(t, [-2])



def _pred_batch(model, alpha, tessera, tokens, thor, device):

    alpha   = alpha.to(device,   non_blocking=True)

    tessera = tessera.to(device, non_blocking=True)

    tokens  = tokens.to(device,  non_blocking=True)

    thor    = thor.to(device,    non_blocking=True)

    with torch.no_grad(), autocast('cuda'):

        seg, ht = model(alpha, tessera, tokens, thor)

        

    # THE FIX: Output raw meters directly! Cap at a realistic physical height (e.g., 100 meters)

    # This prevents the exponential math from blowing up the accurate numbers.

    ht_m = ht.clamp(min=0.0, max=100.0)

    

    return torch.cat([seg, ht_m], dim=1).float().cpu()   # (B, 4, H, W)





def predict_with_tta(model, batch, device) -> torch.Tensor:

    """

    3 forward passes: original + h-flip + v-flip, averaged.

    Returns (B, 4, H, W) float32.

    """

    a  = batch["alpha"];   te = batch["tessera"]

    tk = batch["tokens"];  th = batch["thor"]



    p0 = _pred_batch(model, a, te, tk, th, device)



    # h-flip: flip input, flip output back

    ph = _pred_batch(model, _flip_h(a), _flip_h(te), tk, th, device)

    ph = _flip_h(ph)



    # v-flip

    pv = _pred_batch(model, _flip_v(a), _flip_v(te), tk, th, device)

    pv = _flip_v(pv)



    return torch.stack([p0, ph, pv], dim=0).mean(dim=0)   # (B, 4, H, W)





def predict_no_tta(model, batch, device) -> torch.Tensor:

    a  = batch["alpha"];   te = batch["tessera"]

    tk = batch["tokens"];  th = batch["thor"]

    return _pred_batch(model, a, te, tk, th, device)





# ══════════════════════════════════════════════════════════════════════════════

# Main

# ══════════════════════════════════════════════════════════════════════════════



def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--ckpt",        type=str, default=str(CKPT_DIR / "ckpt_best.pt"),

                   help="Path to model checkpoint")

    p.add_argument("--no_tta",      action="store_true",

                   help="Disable test-time augmentation")

    p.add_argument("--batch_size",  type=int, default=INFER["batch_size"])

    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()





def main():

    args  = parse_args()

    ckpt_path = Path(args.ckpt)

    use_tta   = not args.no_tta



    PRED_DIR.mkdir(parents=True, exist_ok=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)

    print("Embed2Heights — Inference")

    print("=" * 70)

    print(f"Device     : {device}")

    print(f"Checkpoint : {ckpt_path}")

    print(f"TTA        : {use_tta}")

    print(f"Pred dir   : {PRED_DIR}")



    # ── Load model ───────────────────────────────────────────────────────────

    if not ckpt_path.exists():

        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")



    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = build_model(MODEL, pretrained=False).to(device)

    

    # Handle DataParallel if the checkpoint was saved from multiple GPUs

    state_dict = ckpt["model"]

    if list(state_dict.keys())[0].startswith('module.'):

        state_dict = {k[7:]: v for k, v in state_dict.items()}

        

    model.load_state_dict(state_dict)

    model.eval()

    print(f"Loaded epoch {ckpt['epoch']}  score {ckpt['score']:.4f}")



    # ── Test dataset ─────────────────────────────────────────────────────────

    print("\nBuilding test sample list...")

    _, test_samples = build_sample_lists(catalog_path=CATALOG, base_dir=DATA_ROOT)

    print(f"Test patches: {len(test_samples)}")



    if not test_samples:

        print("[ERROR] No test samples found. Check TEST_FOLDERS in config.py match your data layout.")

        return



    # Filter out samples with any missing modality file

    valid = []

    missing_count = 0

    for s in test_samples:

        paths = [s.get(k) for k in ["alphaearth", "tessera", "terramind_s1",

                                     "terramind_s2", "thor_s1", "thor_s2"]]

        if all(p is not None and Path(p).exists() for p in paths):

            valid.append(s)

        else:

            missing_count += 1

    print(f"Valid: {len(valid)}  |  Skipped (missing files): {missing_count}")



    from torch.utils.data import DataLoader

    test_ds     = GeoFMTestDataset(valid)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,

                             num_workers=args.num_workers, pin_memory=True)



    # ── Run inference ────────────────────────────────────────────────────────

    pred_fn = predict_with_tta if use_tta else predict_no_tta



    print(f"\nRunning inference ({len(test_loader)} batches)...")

    t0 = time.time()

    n_saved = 0



    for i, batch in enumerate(test_loader):

        stems = batch["stem"]

        preds = pred_fn(model, batch, device)   # (B, 4, H, W) float32



        for b, stem in enumerate(stems):

            arr = preds[b].numpy().astype(np.float32)   # (4, H, W)

            # Clamp to physically valid ranges

            arr[:3] = np.clip(arr[:3], 0.0, 1.0)        # fractions

            arr[3]  = np.clip(arr[3],  0.0, None)       # height ≥ 0

            out_path = PRED_DIR / f"{stem}.npy"

            np.save(out_path, arr)

            n_saved += 1



        if (i + 1) % 20 == 0:

            elapsed = time.time() - t0

            print(f"  Batch {i+1}/{len(test_loader)}  |  {n_saved} patches saved  |  {elapsed:.0f}s")



    elapsed = time.time() - t0

    print(f"\nDone. {n_saved} predictions saved to {PRED_DIR}  ({elapsed:.0f}s)")



    # ── Quick sanity check ───────────────────────────────────────────────────

    saved = sorted(PRED_DIR.glob("*.npy"))

    if saved:

        sample = np.load(saved[0])

        print(f"\nSample check — {saved[0].name}:")

        print(f"  shape : {sample.shape}  (expected [4, 256, 256])")

        print(f"  dtype : {sample.dtype}")

        print(f"  bldg  : [{sample[0].min():.3f}, {sample[0].max():.3f}]")

        print(f"  veg   : [{sample[1].min():.3f}, {sample[1].max():.3f}]")

        print(f"  water : [{sample[2].min():.3f}, {sample[2].max():.3f}]")

        print(f"  height: [{sample[3].min():.2f}, {sample[3].max():.2f}] metres")





if __name__ == "__main__":

    main()

