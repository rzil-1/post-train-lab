# 35-Day Post-Training & Reasoning RL Learning Log

## Day 1 — PyTorch, Autograd & Mixed Precision (AMP)
- Built a 3-layer FashionMNIST MLP (`week1_transformers/day1_mlp_amp.py`) from scratch on Colab T4 using `torch.autocast("cuda", dtype=torch.float16)` and `torch.amp.GradScaler`, reaching 87.47% test accuracy in 5 epochs with zero NaNs.
- Verified mixed-precision casting directly in runtime logs: master weights stayed `torch.float32`, linear logits cast to `torch.float16` inside `autocast`, and `CrossEntropyLoss` stayed `torch.float32` for numerical stability.
- Implemented manual global L2 gradient norm computation (`compute_grad_norm`) and asserted `< 1e-4` agreement with PyTorch's `clip_grad_norm_` after calling `scaler.unscale_(optimizer)`.
- Verified 4-step gradient accumulation (`effective batch=512`, `Opt Steps` dropped from `469` to `118` per epoch, 87.20% test accuracy), observing tighter gradient norms (`0.6965` vs. `0.9502`) from lower batch variance.
- Observed `GradScaler`'s dynamic `growth_interval=2000` behavior in action: scale doubled (`65536` → `131072`) in Run 1 Epoch 5 after passing 2,000 clean steps, while Run 2 stayed at `65536` (590 steps).

