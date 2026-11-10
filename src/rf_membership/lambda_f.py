"""Predicted peak location lambda*_F of the membership signal (Proposition 4.1) for a latent space.

    lambda*_F = [tr(S0^2) + tr(S0 S1)] / tr((S0 + S1)^2),   S0 = I (noise N(0, I)),  S1 = covariance of the latents

S1 is the C x C channel covariance of the latents, every time frame / pixel being one observation.

    python -m rf_membership.lambda_f --latents train.pt
"""

import argparse

import torch

from rf_membership.utils.loaders import load_latents


def channel_covariance(latents):
    """Covariance over channels of latents (N, C, ...)."""
    x = latents.double().transpose(0, 1).reshape(latents.shape[1], -1)  # (C, N*T)
    x = x - x.mean(dim=1, keepdim=True)
    return x @ x.T / (x.shape[1] - 1)


def compute_lambda_F(sigma_0, sigma_1):
    s = sigma_0 + sigma_1
    return ((torch.trace(sigma_0 @ sigma_0) + torch.trace(sigma_0 @ sigma_1)) / torch.trace(s @ s)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", required=True, help="latent file of the training set")
    args = p.parse_args()
    s1 = channel_covariance(load_latents(args.latents))
    lam = compute_lambda_F(torch.eye(s1.shape[0], dtype=s1.dtype), s1)
    print(f"lambda*_F = {lam:.4f}")


if __name__ == "__main__":
    main()
