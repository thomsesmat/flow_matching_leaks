"""Rectified-Flow Lightning module (any backbone, any latent shape): training and reconstruction error."""

import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from rf_membership.models.registry import build_backbone


class RFModule(pl.LightningModule):
    """Linear Flow Matching, X_lambda = (1 - lambda) X0 + lambda X1, X0 ~ N(0, I), target v = X1 - X0.

    flow_scheduler: 'sigmoid' (lambda = sigmoid(N(0,1)), logit-normal sampling) or 'linear' (uniform).
    The validation loss (early stopping) is (1 - lambda)-weighted, lambda uniform.
    """

    def __init__(self, model, config, flow_scheduler="sigmoid"):
        super().__init__()
        assert flow_scheduler in ("sigmoid", "linear")
        self.flow_scheduler = flow_scheduler
        self.save_hyperparameters(ignore=["model"])
        self.rft = model
        self.config = config
        self.loss_fn_full = nn.MSELoss(reduction="none")

    def forward(self, x, timesteps):
        return self.rft(x, timesteps)

    @staticmethod
    def interpolate(x, noises, times):
        times = torch.as_tensor(times, device=x.device, dtype=x.dtype).view(-1, *([1] * (x.dim() - 1)))
        return times * x + (1.0 - times) * noises

    def _step(self, x, times):
        noises = torch.randn_like(x)
        v_hat = self(self.interpolate(x, noises, times), times)
        return self.loss_fn_full(v_hat, x - noises)

    def sample_times(self, n, device, uniform=False):
        grid = getattr(self.rft, "num_lambda", None)  # linear baseline: lambda on a grid
        if grid:
            return torch.randint(0, grid, (n,), device=device).float() / (grid - 1)
        if self.flow_scheduler == "sigmoid" and not uniform:
            return torch.sigmoid(torch.randn(n, device=device))
        return torch.rand(n, device=device)

    def training_step(self, batch):
        times = self.sample_times(batch.shape[0], batch.device)
        loss = self._step(batch, times).mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        times = self.sample_times(batch.shape[0], batch.device, uniform=True)
        loss = ((1 - times.view(-1, *([1] * (batch.dim() - 1)))) * self._step(batch, times)).mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        cfg = self.config
        opt = torch.optim.AdamW(self.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"], eta_min=cfg["min_lr"])
        return {"optimizer": opt, "lr_scheduler": sched}

    @torch.no_grad()
    def reconstruction_error(self, x1, lam, noises, steps=1):
        """Per-sample MSE between x1 and its reconstruction from x_lambda.

        steps=1: x1_hat = x_lambda + (1 - lambda) v_theta(x_lambda, lambda) (protocol of the paper), so that
        MSE(lambda) = (1 - lambda)^2 ||v_theta - v||^2. steps>1: Euler integration from lambda to 1.
        """
        out = self.interpolate(x1, noises, lam)
        t = float(lam)
        h = (1.0 - t) / steps
        for _ in range(steps):
            v = self(out, torch.full((x1.shape[0],), t, device=x1.device))
            out = out + h * v
            t += h
        return self.loss_fn_full(x1, out).flatten(1).mean(dim=1)


def load_model(ckpt_path, device=None):
    """Rebuild a model from ``<dir>/run_info.json`` (written by ``rf_membership.train``) and the checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "run_info.json")) as f:
        info = json.load(f)
    backbone = build_backbone(info["model"], info["latent_shape"], info["size"])
    model = RFModule.load_from_checkpoint(ckpt_path, model=backbone, config=info["training"],
                                          flow_scheduler=info["flow_scheduler"], map_location=device)
    return model.eval().to(device)
