"""Latent input format.

A latent file is a ``.pt`` file containing either a tensor of shape ``(N, C, T)`` or a list of dicts, each with a
``"latent"`` entry of shape ``(C, T)``. Every sample is one latent sequence ``x1``.
"""

import torch
from torch.utils.data import Dataset


def load_latents(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.float()
    return torch.stack([torch.as_tensor(d["latent"]) for d in data]).float()


class LatentDataset(Dataset):
    def __init__(self, path):
        self.latents = load_latents(path)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]
