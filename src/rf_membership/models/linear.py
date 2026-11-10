import torch
import torch.nn as nn


class Linear(nn.Module):
    """Linear baseline: one independent affine map over channels per lambda of the grid {0, 1/(n-1), ..., 1}.

    Works on latents (B, C, ...) of any number of trailing dims (audio: time, image: H x W); lambda is rounded to the grid.
    """

    num_lambda_default = 11

    def __init__(self, input_size, num_lambda=11):
        super().__init__()
        self.num_lambda = num_lambda
        self.lin_layers = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_lambda)])
        for layer in self.lin_layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x, t):
        shape = x.shape
        flat = x.flatten(2).transpose(1, 2)  # (B, positions, C)
        idx = (t.to(x.device) * (self.num_lambda - 1)).round().long().expand(shape[0])
        out = torch.zeros_like(flat)
        for i in range(self.num_lambda):
            mask = idx == i
            if mask.any():
                out[mask] = self.lin_layers[i](flat[mask]).to(out.dtype)
        return out.transpose(1, 2).reshape(shape)
