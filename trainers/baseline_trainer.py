"""
trainers/baseline_trainer.py
-----------------------------
Training loop for the BiomedCLIP baseline on ROCO.

Features:
  - AMP (bfloat16 on 5090, float16 fallback)
  - Gradient clipping (max_norm=1.0)
  - LR warmup + cosine decay
  - Checkpoint saving (best val R@1 + latest)
  - Optional wandb logging
  - Evaluation every N epochs

Usage:
    from trainers.baseline_trainer import BaselineTrainer
    trainer = BaselineTrainer(model, train_loader, val_loader, config, device)
    trainer.train()
"""

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from evaluators.retrieval_eval import RetrievalEvaluator


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BaselineTrainer:
    """
    Args:
        model:        BiomedCLIPBaseline
        train_loader: DataLoader (ROCO train)
        val_loader:   DataLoader (ROCO val)
        config:       full config dict (from baseline.yaml)
        device:       torch.device
    """

    def __init__(self, model, train_loader, val_loader, config: dict, device: torch.device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        train_cfg = config["training"]
        self.epochs           = train_cfg["epochs"]
        self.lr               = float(train_cfg["lr"])
        self.weight_decay     = float(train_cfg.get("weight_decay", 0.1))
        self.grad_clip        = float(train_cfg.get("grad_clip", 1.0))
        self.warmup_epochs    = int(train_cfg.get("warmup_epochs", 1))
        self.eval_every       = int(train_cfg.get("eval_every_epochs", 1))
        self.checkpoint_dir   = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AMP — bfloat16 is native on RTX 5090 (Blackwell) via torch >= 2.3
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler    = GradScaler(enabled=(self.amp_dtype == torch.float16))

        # Optimizer — AdamW with separate LR for logit_scale
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        # Scheduler
        total_steps  = self.epochs * len(self.train_loader)
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Evaluator
        self.evaluator = RetrievalEvaluator(self.model, device)

        # Wandb (optional)
        self.use_wandb = config.get("wandb", {}).get("enabled", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config["wandb"].get("project", "afrnet-baseline"),
                    config=config,
                )
                self.wandb = wandb
            except ImportError:
                print("⚠  wandb not installed; disabling logging. Run: uv add wandb")
                self.use_wandb = False

        self.best_r1 = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*55}")
        print(f"  Training baseline — {self.epochs} epochs")
        print(f"  Device     : {self.device}")
        print(f"  AMP dtype  : {self.amp_dtype}")
        print(f"  Batch size : {self.config['training']['batch_size']}")
        print(f"  LR         : {self.lr}")
        print(f"  Train batches : {len(self.train_loader):,}")
        print(f"  Val   batches : {len(self.val_loader):,}")
        print(f"{'='*55}\n")

        self.model.param_summary()
        print()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            epoch_time = time.time() - t0

            log_dict = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "lr":         self.scheduler.get_last_lr()[0],
                "epoch_time": epoch_time,
            }

            if epoch % self.eval_every == 0 or epoch == self.epochs:
                metrics = self.evaluator.evaluate(self.val_loader)
                log_dict.update({f"val/{k}": v for k, v in metrics.items()})

                r1 = metrics.get("i2t_R@1", 0.0)
                if r1 > self.best_r1:
                    self.best_r1 = r1
                    self._save_checkpoint("best.pt", epoch, metrics)
                    print(f"  💾 New best i2t_R@1 = {r1:.4f} — saved best.pt")

            self._save_checkpoint("latest.pt", epoch, {})

            print(
                f"Epoch {epoch:>3}/{self.epochs} | "
                f"loss={train_loss:.4f} | "
                f"lr={log_dict['lr']:.2e} | "
                f"time={epoch_time:.0f}s"
            )

            if self.use_wandb:
                self.wandb.log(log_dict)

        print(f"\n✅ Training complete. Best i2t_R@1 = {self.best_r1:.4f}")
        if self.use_wandb:
            self.wandb.finish()

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for images, tokens in pbar:
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward under AMP
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                img_feats = self.model.encode_image(images)
                txt_feats = self.model.encode_text(tokens)
                loss      = self.model.clip_loss(img_feats, txt_feats)

            # Backward
            if self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # bfloat16: no scaler needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "metrics":     metrics,
                "best_r1":     self.best_r1,
            },
            path,
        )

    def load_checkpoint(self, filename: str = "best.pt"):
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.best_r1 = ckpt.get("best_r1", 0.0)
        print(f"✅ Loaded checkpoint '{filename}' (epoch {ckpt['epoch']}, best R@1={self.best_r1:.4f})")
        return ckpt
