"""
LVCG pretraining on MIMIC-IV ECG (self-supervised reconstruction).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from lvcg.data.pipeline import make_dataloaders
from lvcg.models import build_model
from lvcg.models.lvcg import base_beat_loss, beat_level_loss, temporal_loss
from lvcg.models.utils.loss import masked_reconstruction_loss, random_lead_mask
from lvcg.utils.config import Config, add_cli_overrides, apply_overrides, load_config
from lvcg.utils.run_id import ensure_run_dirs


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_id = f"{cfg.run.get('m', 'm5')}{cfg.run.get('s', 's1')}{cfg.run.get('k', 'k1')}"
    ckpt_dir = ensure_run_dirs(cfg.run.get("checkpoint_root", "./checkpoints"), run_id)
    log_dir = ensure_run_dirs(cfg.run.get("log_root", "./logs"), run_id)
    print(f"Run ID: {run_id}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Log dir: {log_dir}")

    model = build_model(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = make_dataloaders(cfg.raw)
    print(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset)}"
    )

    train_cfg = cfg.train
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    lambda_temporal = float(train_cfg.get("lambda_temporal", 0.1))
    lambda_beat = float(train_cfg.get("lambda_beat", 1.0))
    lambda_base = float(train_cfg.get("lambda_base", 1.0))
    num_visible = int(train_cfg.get("num_visible", 3))

    max_steps = int(train_cfg.get("max_steps", 500000))
    log_interval = int(train_cfg.get("log_interval", 100))
    save_interval = int(train_cfg.get("save_interval", 50000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    model.train()
    global_step = 0
    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break

            ecg = batch["ecg"].to(device)
            B = ecg.shape[0]
            visible_indices, mask = random_lead_mask(
                B, num_visible=num_visible, device=device
            )

            optimizer.zero_grad()
            outputs = model.forward_train(ecg, visible_indices)

            loss_recon = masked_reconstruction_loss(outputs["recon"], ecg, mask)
            loss_temporal = temporal_loss(
                outputs["states_pred"],
                outputs["states_real"],
                outputs["beat_mask"],
            )
            loss_base = base_beat_loss(outputs["V_base_hat"], outputs["V_base"])
            loss_beat = beat_level_loss(
                outputs["V_hat_beats"][:, 1:-1],
                outputs["V_beats"],
                outputs["beat_mask_full"][:, 1:-1],
            )

            loss = (
                loss_recon
                + lambda_temporal * loss_temporal
                + lambda_beat * loss_beat
                + lambda_base * loss_base
            )
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if global_step % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "recon": f"{loss_recon.item():.4f}",
                        "temp": f"{loss_temporal.item():.4f}",
                    }
                )

            if global_step > 0 and global_step % save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"\nSaved checkpoint: {ckpt_path}")

            global_step += 1
            pbar.update(1)

    pbar.close()

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(payload, final_path)
    print(f"Saved checkpoint: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LVCG pretraining")
    parser = add_cli_overrides(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    train(cfg)


if __name__ == "__main__":
    main()
