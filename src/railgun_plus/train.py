"""Stage 4: supervised training (masked cross-entropy over occupied cells).

Loss is computed ONLY at cells where an agent stands (the mask), since the
network's job is to predict the action of each occupied cell. Empty cells
carry no supervision.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def masked_ce_loss(logits, target, mask):
    """logits (B,5,H,W), target (B,H,W) int, mask (B,H,W) {0,1}."""
    B, C, H, W = logits.shape
    logits = logits.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, 5)
    target = target.reshape(-1)                          # (B*H*W,)
    mask = mask.reshape(-1).bool()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def masked_accuracy(logits, target, mask):
    pred = logits.argmax(dim=1)            # (B,H,W)
    m = mask.bool()
    if m.sum() == 0:
        return 0.0
    return (pred[m] == target[m]).float().mean().item()


def train(model, dataset, *, epochs=10, batch_size=8, lr=1e-3,
          weight_decay=1e-3, device="cuda", ckpt_dir="checkpoints",
          val_dataset=None, log_every=20):
    os.makedirs(ckpt_dir, exist_ok=True)
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            weight_decay=weight_decay)

    history = []
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
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.3f}")

        avg = running / max(1, len(loader))
        entry = {"epoch": epoch + 1, "train_loss": avg}

        if val_dataset is not None:
            entry["val_acc"] = evaluate_accuracy(model, val_dataset, device,
                                                 batch_size)
        history.append(entry)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   os.path.join(ckpt_dir, f"epoch{epoch+1}.pt"))
        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   os.path.join(ckpt_dir, "latest.pt"))
        print(entry)
    return history


@torch.no_grad()
def evaluate_accuracy(model, dataset, device, batch_size=8):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    tot, n = 0.0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        tot += masked_accuracy(logits, y, m) * len(x)
        n += len(x)
    return tot / max(1, n)
