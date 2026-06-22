"""
train.py — Main training loop for Embed2Heights.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import (
    TRAIN, CKPT_DIR, LOG_DIR, MODEL, METRIC_WEIGHTS, CATALOG
)
from dataset import build_sample_lists, split_samples, GeoFMDataset
from model import build_model
from losses import compute_loss, compute_metrics, weighted_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN["lr"])
    parser.add_argument("--num_workers", type=int, default=TRAIN["num_workers"])
    parser.add_argument("--no_pretrain", action="store_true")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Embed2Heights — Training")
    print("=" * 70)
    print(f"Device: {device}")
    
    # ── 1. Data ────────────────────────────────────────────────────────
    print("\nLoading catalog and checking files...")
    train_samples, _ = build_sample_lists(CATALOG)
    train_list, val_list = split_samples(train_samples, val_frac=TRAIN["val_frac"], seed=TRAIN["seed"])
    
    train_ds = GeoFMDataset(train_list, augment=True)
    val_ds   = GeoFMDataset(val_list, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)
    
    # ── 2. Model ───────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(MODEL, pretrained=not args.no_pretrain).to(device)
    
    # Wrap for multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRAIN["weight_decay"])
    scaler    = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ── 3. Logging Setup ───────────────────────────────────────────────
    csv_path = LOG_DIR / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_score"])
        
    best_score = -1.0
    
    # ── 4. Training Loop ───────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            
            alpha   = batch["alpha"].to(device, non_blocking=True)
            tessera = batch["tessera"].to(device, non_blocking=True)
            tokens  = batch["tokens"].to(device, non_blocking=True)
            thor    = batch["thor"].to(device, non_blocking=True)
            label   = batch["label"].to(device, non_blocking=True)
            
            # Keep predictions fast in 16-bit
            with autocast('cuda'):
                seg_pred, ht_pred = model(alpha, tessera, tokens, thor)
                
            # Compute loss safely outside autocast in 32-bit
            loss, _, _ = compute_loss(seg_pred.float(), ht_pred.float(), label.float(), TRAIN["w_seg"], TRAIN["w_height"])
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        # ── 5. Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                alpha   = batch["alpha"].to(device, non_blocking=True)
                tessera = batch["tessera"].to(device, non_blocking=True)
                tokens  = batch["tokens"].to(device, non_blocking=True)
                thor    = batch["thor"].to(device, non_blocking=True)
                label   = batch["label"].to(device, non_blocking=True)
                
                with autocast('cuda'):
                    seg_pred, ht_pred = model(alpha, tessera, tokens, thor)
                    
                # Compute validation loss safely in 32-bit
                loss, _, _ = compute_loss(seg_pred.float(), ht_pred.float(), label.float(), TRAIN["w_seg"], TRAIN["w_height"])
                    
                val_loss += loss.item()
                batch_metrics = compute_metrics(seg_pred, ht_pred, batch["label"])
                all_metrics.append(batch_metrics)
                
        val_loss /= len(val_loader)
        
        # Aggregate metrics
        avg_metrics = {}
        for k in METRIC_WEIGHTS.keys():
            avg_metrics[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
            
        score = weighted_score(avg_metrics, METRIC_WEIGHTS)
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:03d}/{args.epochs} [{elapsed:.0f}s] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Score: {score:.4f}")
        
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, score])
            
        # Save Best Checkpoint
        if score > best_score:
            best_score = score
            save_path = CKPT_DIR / "ckpt_best.pt"
            # Handle unwrapping if DataParallel was used
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({"epoch": epoch, "model": state_dict, "score": score}, save_path)
            print(f"  ⭐ Saved new best checkpoint (Score: {score:.4f}) to {save_path.name}")

if __name__ == "__main__":
    main()