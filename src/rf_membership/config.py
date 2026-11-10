"""Model sizes and training hyper-parameters (Appendix B.3)."""

ARCHITECTURES = {
    # hidden_size, depth, num_heads, mlp_ratio
    "small": {"hidden_size": 576, "depth": 24, "num_heads": 12, "mlp_ratio": 4.0},
    "medium": {"hidden_size": 1032, "depth": 24, "num_heads": 12, "mlp_ratio": 4.0},
    "large": {"hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0},
}

# UNet configurations of Appendix B.3.2 (1D for (C, T) latents, 2D for (C, H, W) latents)
UNETS = {
    "1d": {"model_channels": 192, "channel_mult": (1, 2, 4, 4), "num_res_blocks": 2,
           "attention_levels": (2, 3), "num_heads": 8, "dropout": 0.1},
    "2d": {"model_channels": 192, "channel_mult": (1, 2, 4, 4), "num_res_blocks": 2,
           "attention_levels": (1, 2, 3), "num_heads": 8, "dropout": 0.0},
}

TRAINING = {
    "lr": 1e-4,
    "min_lr": 1e-6,
    "betas": (0.9, 0.999),
    "max_epochs": 2000,
    "gradient_clip": 1.0,
    "es_patience": 25,
    "batch_size": 128,
    "precision": "bf16-mixed",
}
