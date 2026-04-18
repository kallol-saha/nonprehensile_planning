"""Train start/goal-conditioned trajectory diffusion (no polygon, clamped endpoints).

Usage:
    python scripts/train_startgoal_diffusion.py --data_dir assets/push_data \
        --epochs 500 --batch_size 128 --wandb --wandb_run_name v1
"""

import argparse
import json
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from visplan.training.push_dataset import make_push_dataloaders
from visplan.training.models.startgoal_diffusion import StartGoalDiffusionUNet


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default="checkpoints/startgoal_diffusion")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="startgoal-diffusion")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch["trajectory"].shape[0]
        n += batch["trajectory"].shape[0]
    return total_loss / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        loss = model.compute_loss(batch)
        total_loss += loss.item() * batch["trajectory"].shape[0]
        n += batch["trajectory"].shape[0]
    return total_loss / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    train_loader, val_loader = make_push_dataloaders(
        args.data_dir, batch_size=args.batch_size, val_frac=args.val_frac,
        num_workers=args.num_workers)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    traj_len = meta["max_steps"] + 1
    print(f"Max trajectory length: {traj_len}")

    model = StartGoalDiffusionUNet(
        traj_dim=4, traj_len=traj_len,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)
    print(f"Model: StartGoalDiffusionUNet | Params: {count_params(model):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if args.wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "val_loss": val_loss, "lr": lr,
                       "epoch_time_s": dt}, step=epoch)

        if epoch % args.log_every == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | "
                  f"val {val_loss:.6f} | lr {lr:.2e} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best.pt"))
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"epoch{epoch:04d}.pt"))

    if args.wandb:
        wandb.log({"best_val_loss": best_val})
        wandb.finish()

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    main()
