"""
trainers/afrnet_trainer.py
--------------------------
Training loop for AFR-Net on ROCO. Backbone is frozen by default; only the
AFR-Net modules (ROAM/GTGA/DDTP/OT), projection heads, and the loss temperature
train. Reuses evaluators/retrieval_eval.py unchanged (AFRNet exposes
encode_image / encode_text).

Key differences from BaselineTrainer:
  - loss comes from losses.AFRLoss and needs a GLOBAL STEP for OT warmup
  - forward returns a dict, not (feats)
  - eval uses AFRNet.encode_image/encode_text (decoupled retrieval globals)
"""

import math
import time
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from evaluators.retrieval_eval import RetrievalEvaluator
from losses.losses import AFRLoss, LossConfig


def cosine_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))
    return optim.lr_scheduler.LambdaLR(optimizer, f)


class AFRNetTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tcfg = config["training"]
        self.epochs = tcfg["epochs"]
        self.lr = float(tcfg["lr"])
        self.weight_decay = float(tcfg.get("weight_decay", 0.1))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", 1))
        self.eval_every = int(tcfg.get("eval_every_epochs", 1))
        self.accum = int(tcfg.get("grad_accum", 1))
        self.ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # freeze backbone (default True for prove-it-works phase)
        if config["model"].get("freeze_backbone", True):
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)
            self.model.backbone.eval()

        lcfg = config.get("loss", {})
        self.loss_fn = AFRLoss(LossConfig(
            ot_weight=float(lcfg.get("ot_weight", 0.1)),
            ot_warmup_steps=int(lcfg.get("ot_warmup_steps", 1000)),
            distill_weight=float(lcfg.get("distill_weight", 0.0)),
        )).to(device)

        params = [p for p in self.model.parameters() if p.requires_grad] + \
                 [p for p in self.loss_fn.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay,
                                     betas=(0.9, 0.98), eps=1e-6)

        total = self.epochs * len(train_loader)
        self.scheduler = cosine_warmup(self.optimizer,
                                       self.warmup_epochs * len(train_loader), total)

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.evaluator = RetrievalEvaluator(self.model, device)
        self.best_r1 = 0.0
        self.global_step = 0

        self.use_wandb = config.get("wandb", {}).get("enabled", False)
        if self.use_wandb:
            import wandb
            wandb.init(project=config["wandb"].get("project", "afrnet"), config=config)
            self.wandb = wandb

    def train(self):
        print(f"\n{'='*55}\n  AFR-Net training — {self.epochs} epochs on {self.device}")
        print(f"  AMP: {self.amp_dtype} | batch {self.config['training']['batch_size']}"
              f" x accum {self.accum} | train batches {len(self.train_loader):,}\n{'='*55}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tl = self._train_epoch(epoch)
            log = {"epoch": epoch, "train_loss": tl,
                   "lr": self.scheduler.get_last_lr()[0], "time": time.time() - t0}

            if epoch % self.eval_every == 0 or epoch == self.epochs:
                metrics = self.evaluator.evaluate(self.val_loader)
                log.update({f"val/{k}": v for k, v in metrics.items()})
                r1 = metrics.get("i2t_R@1", 0.0)
                if r1 > self.best_r1:
                    self.best_r1 = r1
                    self._save("best.pt", epoch, metrics)
                    print(f"  💾 new best i2t_R@1={r1:.4f}")
            self._save("latest.pt", epoch, {})
            print(f"Epoch {epoch}/{self.epochs} | loss={tl:.4f} "
                  f"| lr={log['lr']:.2e} | {log['time']:.0f}s")
            if self.use_wandb:
                self.wandb.log(log)

        print(f"\n✅ done. best i2t_R@1={self.best_r1:.4f}")
        if self.use_wandb:
            self.wandb.finish()

    def _train_epoch(self, epoch):
        self.model.train()
        if self.config["model"].get("freeze_backbone", True):
            self.model.backbone.eval()   # keep frozen BN/dropout in eval mode
        total = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for i, (images, tokens) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                out = self.model(images, tokens)
                loss, parts = self.loss_fn(
                    out["v_img"], out["t_txt"], out["L_OT"], step=self.global_step,
                    v_cls_frozen=out["v_cls_frozen"], t_frozen=out["t_frozen"])
                loss = loss / self.accum

            loss.backward()
            if (i + 1) % self.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            total += loss.item() * self.accum
            pbar.set_postfix(loss=f"{loss.item()*self.accum:.3f}",
                             ot=f"{parts['ot_alpha']:.2f}")
        return total / len(self.train_loader)

    def _save(self, name, epoch, metrics):
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "loss_state": self.loss_fn.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "metrics": metrics, "best_r1": self.best_r1,
                    "global_step": self.global_step}, self.ckpt_dir / name)

    def load_checkpoint(self, name="best.pt"):
        ckpt = torch.load(self.ckpt_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.loss_fn.load_state_dict(ckpt["loss_state"])
        self.best_r1 = ckpt.get("best_r1", 0.0)
        self.global_step = ckpt.get("global_step", 0)
        print(f"✅ loaded {name} (epoch {ckpt['epoch']}, best R@1={self.best_r1:.4f})")
        return ckpt
