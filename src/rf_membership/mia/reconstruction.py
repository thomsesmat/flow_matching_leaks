"""Per-sample reconstruction errors along the interpolation path (detection protocol of the paper, Sec. 5.1).

For every sample x1 and lambda in {0, 0.1, ..., 1}, K noises x0 ~ N(0, I) give x_lambda = (1 - lambda) x0 + lambda x1,
which is reconstructed in one step; the MSE is averaged over the K noises. Run on members (training set) and
non-members (held-out set); each writes a CSV with one row per sample and one column per lambda.

    python -m rf_membership.mia.reconstruction --ckpt outputs/run1/last.ckpt \\
        --members train.pt --nonmembers test.pt --out-dir outputs/run1/recon
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from rf_membership.models.rf_module import load_model
from rf_membership.utils.loaders import load_latents

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]


@torch.no_grad()
def reconstruction_profile(model, latents, num_samples, num_noise, batch_size, seed):
    """(n, 11) array: mean over ``num_noise`` noises of the reconstruction MSE, per sample and lambda."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(latents), generator=g)[:num_samples]
    latents = latents[idx]
    out = []
    for start in range(0, len(latents), batch_size):
        x1 = latents[start:start + batch_size].to(device)
        errs = torch.zeros(x1.shape[0], len(LAMBDAS), device=device)
        for _ in range(num_noise):
            noises = torch.randn_like(x1)
            for j, lam in enumerate(LAMBDAS):
                errs[:, j] += model.reconstruction_error(x1, lam, noises)
        out.append((errs / num_noise).cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--members", required=True, help="latent file of training samples")
    p.add_argument("--nonmembers", required=True, help="latent file of held-out samples")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--num-noise", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = load_model(args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, path in [("members", args.members), ("nonmembers", args.nonmembers)]:
        prof = reconstruction_profile(model, load_latents(path), args.num_samples, args.num_noise,
                                      args.batch_size, args.seed)
        pd.DataFrame(prof, columns=[str(lam) for lam in LAMBDAS]).to_csv(os.path.join(args.out_dir, f"{name}.csv"),
                                                                          index=False)
        print(f"{name}: {prof.shape} -> {args.out_dir}/{name}.csv")

    m = pd.read_csv(os.path.join(args.out_dir, "members.csv")).values.mean(0)
    n = pd.read_csv(os.path.join(args.out_dir, "nonmembers.csv")).values.mean(0)
    gap = np.divide(n - m, n + m, out=np.zeros_like(m), where=(n + m) != 0)
    print("Delta_norm(lambda):", dict(zip(LAMBDAS, gap.round(4))), "\npeak at lambda =", LAMBDAS[int(gap.argmax())])


if __name__ == "__main__":
    main()
