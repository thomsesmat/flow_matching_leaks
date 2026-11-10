"""Train a Flow Matching model (transformer | unet | linear) on latents, (C, T) sequences or (C, H, W) images.

    python -m rf_membership.train --train train.pt --val val.pt --out-dir outputs/run1 [--model transformer] [--size medium]

``--train`` / ``--val`` are latent files (see ``rf_membership.utils.loaders`` for the format). Writes the best and last
checkpoints and ``run_info.json`` to ``--out-dir``.
"""

import argparse
import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from rf_membership.config import ARCHITECTURES, TRAINING
from rf_membership.models.registry import MODELS, build_backbone
from rf_membership.models.rf_module import RFModule
from rf_membership.utils.loaders import LatentDataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="transformer", choices=MODELS)
    p.add_argument("--size", default="medium", choices=list(ARCHITECTURES), help="Transformer size")
    p.add_argument("--flow-scheduler", default="sigmoid", choices=["sigmoid", "linear"],
                   help="'sigmoid': logit-normal lambda sampling, 'linear': uniform")
    p.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    p.add_argument("--max-epochs", type=int, default=TRAINING["max_epochs"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast-dev-run", type=int, default=0)
    args = p.parse_args()

    pl.seed_everything(args.seed)
    train_ds, val_ds = LatentDataset(args.train), LatentDataset(args.val)
    shape = tuple(train_ds.latents.shape[1:])

    training = {**TRAINING, "batch_size": args.batch_size, "max_epochs": args.max_epochs}
    model = RFModule(build_backbone(args.model, shape, args.size), training, args.flow_scheduler)
    print(f"Parameters: {sum(q.numel() for q in model.parameters()) / 1e6:.0f}M")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump({"model": args.model, "size": args.size, "latent_shape": shape, "training": training,
                   "flow_scheduler": args.flow_scheduler, "seed": args.seed}, f, indent=2)

    trainer = pl.Trainer(
        accelerator="auto", max_epochs=args.max_epochs, precision=training["precision"],
        gradient_clip_val=training["gradient_clip"], fast_dev_run=args.fast_dev_run or False,
        default_root_dir=args.out_dir,
        callbacks=[
            ModelCheckpoint(dirpath=args.out_dir, monitor="val_loss", mode="min", save_last=True, save_top_k=1),
            EarlyStopping(monitor="val_loss", mode="min", patience=training["es_patience"]),
        ],
    )
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=args.num_workers)
    trainer.fit(model, loader(train_ds, True), loader(val_ds, False))


if __name__ == "__main__":
    main()
