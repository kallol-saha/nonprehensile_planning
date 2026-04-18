"""Train trajectory prediction models on Voronoi reassembly data.

Usage:
  python scripts/train.py --model unet --data_dir assets/data --epochs 200
  python scripts/train.py --model dit --data_dir assets/data --epochs 200
  python scripts/train.py --model regression --data_dir assets/data --epochs 200
"""

import argparse
import os
import time

import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from visplan.training.dataset import make_dataloaders
from visplan.training.models.diffusion_unet import DiffusionUNet
from visplan.training.models.diffusion_transformer import DiffusionTransformer
from visplan.training.models.regression_baseline import RegressionBaseline


MODEL_REGISTRY = {
    "unet": DiffusionUNet,
    "dit": DiffusionTransformer,
    "regression": RegressionBaseline,
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--visible_range", type=float, default=0.6)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--diffusion_steps", type=int, default=100)
    p.add_argument("--traj_len", type=int, default=32)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="voronoi-reassembly")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging (enabled by default).")
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"{args.model}",
            config=vars(args),
            dir=args.output_dir,
        )

    # Data
    train_loader, val_loader = make_dataloaders(
        args.data_dir, batch_size=args.batch_size, val_frac=args.val_frac,
        num_workers=args.num_workers, visible_range=args.visible_range)
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")

    # Model
    model_cls = MODEL_REGISTRY[args.model]
    model_kwargs = dict(traj_dim=4, traj_len=args.traj_len, embed_dim=args.embed_dim)
    if args.model in ("unet", "dit"):
        model_kwargs["num_diffusion_steps"] = args.diffusion_steps
    model = model_cls(**model_kwargs).to(device)
    n_params = count_params(model)
    print(f"Model: {args.model} | Params: {n_params:,}")
    if use_wandb:
        wandb.summary["num_params"] = n_params
        wandb.watch(model, log="gradients", log_freq=100)

    # Optimizer
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

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": lr,
                "epoch_time_s": dt,
            }, step=epoch)

        if epoch % args.log_every == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | train {train_loss:.5f} | "
                  f"val {val_loss:.5f} | lr {lr:.2e} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"{args.model}_best.pt"))
            if use_wandb:
                wandb.summary["best_val_loss"] = best_val
                wandb.summary["best_epoch"] = epoch

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir,
                                    f"{args.model}_epoch{epoch:04d}.pt"))

    print(f"Done. Best val loss: {best_val:.5f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
