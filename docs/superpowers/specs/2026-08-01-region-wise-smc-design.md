# Region-Wise Self-Mutual Confidence Design

## Objective

Replace SynFoC's sample-level scalar pseudo-label mixing ratio with a pixel-wise confidence map. The mechanism chooses locally between EMA U-Net and EMA MedSAM probabilities while keeping pseudo-label generation gradient-free.

## Existing Behavior

The current loop averages Dice-based U-Net teacher/student confidence and U-Net/MedSAM mutual confidence, then broadcasts their product across the image. This discards spatial information and reduces U-Net influence whenever the models disagree, even when U-Net is locally more reliable.

## Scope

The first implementation uses pure pixel-wise confidence without patch pooling or connected-component post-processing. It supports every dataset and class count already handled by `code/train.py`.

It does not add morphology, shape priors, another MedSAM student forward, or new command-line hyperparameters.

## Inputs and Alignment

Pseudo-label generation uses `sam_teacher_prob`, `unet_teacher_prob`, and `unet_student_prob`. All tensors are aligned to U-Net resolution `(B, C, H, W)`. MedSAM probabilities use bilinear interpolation because they are continuous; hard labels and masks continue to use nearest-neighbor interpolation.

## Pixel-Wise Confidence

```python
unet_local_conf = unet_teacher_prob.max(dim=1, keepdim=True).values
sam_local_conf = sam_teacher_prob.max(dim=1, keepdim=True).values

self_conf = 1.0 - 0.5 * (
    unet_teacher_prob - unet_student_prob
).abs().sum(dim=1, keepdim=True)

mutual_conf = 1.0 - 0.5 * (
    unet_teacher_prob - sam_teacher_prob
).abs().sum(dim=1, keepdim=True)
```

Both similarity maps are clamped to `[0, 1]` for numerical safety. Each confidence map has shape `(B, 1, H, W)`; no mean participates in pseudo-label calculation.

## Relative Model Selection

```python
unet_score = unet_local_conf * self_conf
sam_score = sam_local_conf
relative_alpha = unet_score / (unet_score + sam_score + eps)

alpha_map = mutual_conf * 0.5 + (1.0 - mutual_conf) * relative_alpha
alpha_map = alpha_map.clamp(0.0, 1.0)
```

When distributions agree, `alpha_map` approaches `0.5`. When they disagree, it approaches the relative confidence ratio, allowing a stable U-Net prediction to override a locally weaker MedSAM prediction.

## Pseudo-Label Blending

```python
blended_prob = (
    alpha_map * unet_teacher_prob
    + (1.0 - alpha_map) * sam_teacher_prob
)
blended_conf, pseudo_label = blended_prob.max(dim=1)
confidence_mask = (blended_conf > threshold).unsqueeze(1).float()
```

The blended probability map is resized to MedSAM's low-resolution output before generating the corresponding MedSAM pseudo-label and mask.

## Gradient Safety

EMA MedSAM inference, EMA U-Net inference, weak-image U-Net student inference for stability, probability conversion, interpolation, confidence evaluation, blending, hard pseudo-label generation, and mask generation all run inside `torch.no_grad()`.

Strong-image student forwards and losses remain outside that block and retain gradients.

## Logging

Scalar monitoring uses only detached reductions: `self_conf.mean().item()`, `mutual_conf.mean().item()`, and `alpha_map.mean().item()`. These means never affect pseudo-label generation.

## Code Organization

A pure tensor helper in `code/utils/region_smc.py` returns blended probabilities, `self_conf`, `mutual_conf`, and `alpha_map`. `code/train.py` aligns the probability tensors and calls this helper. Keeping the helper outside the training entry point makes it testable on CPU without triggering argument parsing, dataset setup, or checkpoint loading.

## Validation

CPU tests verify shapes, `[0, 1]` bounds, U-Net preference when stable and confident, MedSAM preference when U-Net is weak or unstable, alpha near `0.5` under agreement, gradient-free outputs under `torch.no_grad()`, and multi-class compatibility.

The final syntax check is `python -m compileall train.py test.py dataloaders networks utils` from `code/`.

## Acceptance Criteria

- No sample-level scalar participates in probability blending.
- `self_conf`, `mutual_conf`, and `alpha_map` preserve `(B, 1, H, W)` structure.
- U-Net and MedSAM probabilities are mixed pixel by pixel.
- Pseudo-label confidence evaluation and blending are gradient-free.
- Existing CutMix, masking, losses, and dataset behavior remain compatible.
