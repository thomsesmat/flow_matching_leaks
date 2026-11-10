# Where Flow Matching Leaks: Membership Signals Along the Interpolation Path

Code for *Where Flow Matching Leaks: Characterising Membership Signals Along the Interpolation Path*
(Sesmat, Meseguer-Brocal, Peeters; ICML 2026, arXiv:2606.07271). Three things: train a Flow Matching model on latents,
run the membership inference attack (MIA) on it, and predict the optimal λ*_F of a latent space.

## Installation

```bash
pip install -e .            # python >= 3.10, GPU needed
```

## Data format

Bring your latents, produced with any autoencoder. A latent file is a `.pt` file holding either a tensor
`(N, C, T)` (sequences, e.g. audio) / `(N, C, H, W)` (images), or a list of dicts with a `"latent"` entry of shape
`(C, T)` / `(C, H, W)`. You need:

* `train.pt` (members) and `val.pt` (validation, for early stopping) for training;
* `train.pt` (members) and `test.pt` (non-members, same distribution, never seen in training) for the attack.

## Commands

```bash
# 1. Train (--model transformer | unet | linear ; --size small|medium|large for the transformer ;
python -m rf_membership.train --train train.pt --val val.pt --out-dir outputs/run1 --model transformer

# 2. MIA: reconstruction profile along λ in {0, 0.1, ..., 1} (K=100 noises, 5000 samples per set), then MLP attack
python -m rf_membership.mia.reconstruction --ckpt outputs/run1/last.ckpt \
    --members train.pt --nonmembers test.pt --out-dir outputs/run1/recon
python -m rf_membership.mia.mlp_attack --recon-dir outputs/run1/recon --out-dir outputs/run1/mlp

# 3. Predicted peak λ*_F of the latent space
python -m rf_membership.lambda_f --latents train.pt
```

`mia.reconstruction` also prints the train-test gap Δ_norm(λ) and its peak; `mia.mlp_attack` reports AUC and
TPR@1%/5% FPR and saves the MLP (`mlp_attack.pt`). Use `--help` for all options (e.g. `--num-samples 64 --num-noise 2`
for a quick test). Training hyper-parameters (AdamW, lr 1e-4, clipping 1.0, early stopping patience 25, batch 128) are
in `src/rf_membership/config.py`. Heavy steps should be submitted to your scheduler (`scripts/example_train.slurm`).

## Tests

```bash
pytest tests
```

## Citation

```bibtex
@inproceedings{sesmat2026flowleaks,
  title     = {Where Flow Matching Leaks: Characterising Membership Signals Along the Interpolation Path},
  author    = {Sesmat, Thomas and Meseguer-Brocal, Gabriel and Peeters, Geoffroy},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
