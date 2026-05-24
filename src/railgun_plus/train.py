"""Stage 4: supervised training (masked cross-entropy over occupied cells).

Loss is computed ONLY at cells where an agent stands (the mask). Features:
  - per-epoch checkpoints (epoch{N}.pt) + rolling latest.pt + best.pt
  - tqdm progress with live loss/acc
  - automatic train/val split
  - early stopping on validation accuracy (patience-based)
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def masked_ce_loss(logits, target, mask):
    B, C, H, W = logits.shape
    logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
    target = target.reshape(-1)
    mask = mask.reshape(-1).bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def masked_accuracy(logits, target, mask):
    pred = logits.argmax(dim=1)
    m = mask.bool()
    if m.sum() == 0:
        return 0.0
    return (pred[m] == target[m]).float().mean().item()


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        bs = len(x)
        tot += masked_accuracy(logits, y, m) * bs
        n += bs
    return tot / max(1, n)


def train(model, dataset, *, epochs=10, batch_size=8, lr=1e-3,
          weight_decay=1e-3, device="cuda", ckpt_dir="checkpoints",
          val_fraction=0.1, early_stop_patience=5, min_delta=1e-3,
          log_every=20, seed=0):
    """Train with validation-based early stopping and best-checkpoint saving.

    Args:
        val_fraction: fraction of data held out for validation (0 disables
            both validation and early stopping).
        early_stop_patience: stop if val accuracy hasn't improved by at least
            `min_delta` for this many consecutive epochs.
        min_delta: minimum val-accuracy improvement to count as "better".

    Saves to ckpt_dir:
        epoch{N}.pt  -- every epoch
        latest.pt    -- most recent epoch
        best.pt      -- highest val accuracy so far (use THIS for evaluation)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    model = model.to(device)

    # --- train / val split ---
    if val_fraction and val_fraction > 0:
        n_val = max(1, int(len(dataset) * val_fraction))
        n_train = len(dataset) - n_val
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        print(f"split: {n_train} train / {n_val} val samples")
    else:
        train_ds, val_loader = dataset, None
        print(f"no validation split ({len(dataset)} train samples); "
              f"early stopping disabled")

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            weight_decay=weight_decay)

    history = []
    best_val = -1.0
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        for step, (x, y, m) in enumerate(pbar):
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = masked_ce_loss(logits, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if step % log_every == 0:
                acc = masked_accuracy(logits, y, m)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 train_acc=f"{acc:.3f}")

        avg_loss = running / max(1, len(loader))
        entry = {"epoch": epoch + 1, "train_loss": avg_loss}

        # --- validation + early stopping ---
        if val_loader is not None:
            val_acc = evaluate_accuracy(model, val_loader, device)
            entry["val_acc"] = val_acc
            improved = val_acc > best_val + min_delta
            if improved:
                best_val = val_acc
                epochs_since_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "val_acc": val_acc},
                           os.path.join(ckpt_dir, "best.pt"))
                entry["saved_best"] = True
            else:
                epochs_since_improve += 1
                entry["saved_best"] = False

        # per-epoch + latest checkpoints (always)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   os.path.join(ckpt_dir, f"epoch{epoch+1}.pt"))
        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   os.path.join(ckpt_dir, "latest.pt"))

        history.append(entry)
        print(entry)

        if val_loader is not None and epochs_since_improve >= early_stop_patience:
            print(f"early stopping: val acc hasn't improved by >{min_delta} "
                  f"for {early_stop_patience} epochs "
                  f"(best val acc = {best_val:.3f}). Use best.pt.")
            break

    if val_loader is not None:
        print(f"\nTraining done. Best val acc = {best_val:.3f}. "
              f"Evaluate with best.pt (not latest.pt).")
    return history
