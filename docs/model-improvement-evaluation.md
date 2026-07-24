# Evaluation of Possible SynFoC Improvements

This document summarizes practical research directions for improving the SynFoC-style mixed-domain semi-supervised medical image segmentation model in this repository. The analysis is based on `2503.16997v1.pdf`, the current `code/train.py` implementation, and the summarized experiment logs in `code/summary.json`.

## 1. Context

SynFoC combines a foundation model, MedSAM/SAM, with a conventional model, U-Net, for Mixed Domain Semi-Supervised Medical Image Segmentation (MiDSS). In this setting, labeled data come from one domain, while unlabeled data come from multiple mixed domains.

The paper identifies two complementary failure modes:

- U-Net can overfit the labeled source domain when the domain gap is large.
- MedSAM can produce high-confidence but incorrect predictions, causing pseudo-label error accumulation.

SynFoC addresses this by synergistically training both models. MedSAM provides stronger early pseudo-labels, while U-Net can later correct some of MedSAM's high-confidence mistakes.

## 2. Current Implementation Observations

The main training logic is implemented in `code/train.py`.

The current pseudo-label ensemble uses a sample-level ratio:

```python
ratio = torch.tensor(self_conf * mutual_conf).view(len(ulb_x_w), 1, 1, 1).cuda()
unet_size_prob_ulb_x_w = (1 - ratio) * unet_size_sam_prob_ulb_x_w + ratio * unet_prob_ulb_x_w
```

This means the same U-Net/MedSAM mixing weight is applied to every pixel in an unlabeled image. However, segmentation errors are usually spatially uneven. MedSAM may be reliable in the interior but poor near boundaries; U-Net may correct some local regions but fail elsewhere.

The implementation also uses a fixed confidence threshold:

```python
unet_size_mask = (unet_size_prob > threshold).unsqueeze(1).float()
```

The summarized logs show very high final mask ratios, often around `0.98-0.99`. This suggests that almost all pseudo-label pixels are used during training, which may allow incorrect high-confidence predictions to continue supervising the models.

## 3. Most Promising Improvement Directions

### 3.1 Region-Wise Self-Mutual Confidence

The most promising improvement is to replace the current image-level ensemble ratio with a pixel-wise, patch-wise, or connected-component-wise confidence map.

Instead of assigning one ratio to the whole image, compute local reliability from:

- prediction agreement between U-Net and MedSAM,
- entropy or max-probability confidence,
- teacher-student stability,
- boundary uncertainty,
- connected-component size and shape plausibility.

Expected benefit:

- MedSAM can dominate regions where it is locally stable and confident.
- U-Net can dominate regions where it disagrees with MedSAM but is stable across teacher/student predictions.
- Boundary and small-object regions can receive more cautious supervision.

This directly targets the key weakness shown in Figure 1 of the paper: high-confidence wrong pseudo-labels.

### 3.2 Dynamic Confidence Threshold

The fixed threshold `0.95` can be replaced with a schedule or adaptive rule.

Possible variants:

- epoch-based threshold: stricter early, relaxed later;
- percentile-based threshold per batch;
- class-specific threshold for multi-class datasets;
- domain-aware threshold estimated from unlabeled sample statistics;
- lower threshold for stable agreement regions and higher threshold for disagreement regions.

Expected benefit:

- reduce early error accumulation;
- avoid using nearly every pixel as pseudo-supervision;
- improve robustness on difficult domains.

This is a low-risk improvement and should be tested before larger architecture changes.

### 3.3 Boundary and Small-Object Aware Training

The paper's limitations mention failure cases on extremely small targets and low-contrast boundaries. This can be addressed with losses or sampling strategies that emphasize boundary quality.

Possible additions:

- boundary loss;
- Hausdorff-distance-aware loss;
- clDice or topology-aware loss;
- edge-weighted Dice/CE loss;
- crop sampling that increases the frequency of small foreground targets;
- stronger local augmentations near annotated objects.

Expected benefit:

- better segmentation of small anatomical structures or lesions;
- fewer over-segmentation and under-segmentation errors;
- likely stronger gains on BUSI and M&Ms than on Prostate.

### 3.4 Domain-Aware Unlabeled Sampling

MiDSS assumes unlabeled data come from mixed domains, often without known domain labels. The current code samples from the mixed unlabeled pool directly.

Potential improvement:

- cluster unlabeled images using image-level features;
- balance batches across pseudo-domains;
- track pseudo-label quality per pseudo-domain;
- apply domain-specific thresholds or adapters.

Expected benefit:

- reduce bias toward dominant unlabeled domains;
- improve transfer to harder domains;
- make training more stable when one domain has much noisier pseudo-labels.

This is promising but more complex than confidence-map improvements.

### 3.5 Stronger Adaptation of the Foundation Model

The current method adapts SAM/MedSAM using LoRA and mask-decoder training. Improvements could explore:

- different LoRA ranks;
- layer-wise LoRA placement;
- adapter modules in selected encoder blocks;
- prompt encoder freezing versus partial tuning;
- separate learning rates for LoRA, prompt encoder, and mask decoder.

Expected benefit:

- better downstream adaptation;
- possible improvement when MedSAM plateaus early.

Risk:

- higher compute cost;
- easier overfitting with very few labeled samples;
- harder ablation space.

## 4. Recommended Research Plan

The recommended path is incremental:

1. Establish reliable baselines from the current code.
2. Add adaptive confidence thresholding.
3. Add region-wise confidence ensemble.
4. Add boundary-aware loss only after pseudo-label quality improves.
5. Test domain-aware sampling if domain imbalance remains visible.

The best first target is:

> Region-wise Self-Mutual Confidence with adaptive thresholding.

This direction is close to the original SynFoC idea, keeps the same two-model architecture, and directly addresses the known error mode of high-confidence incorrect pseudo-labels.

## 5. Suggested Ablation Experiments

Use the current SynFoC implementation as the baseline.

Recommended ablations:

| Experiment | Change | Purpose |
| --- | --- | --- |
| Baseline | current SynFoC | reference |
| A1 | adaptive threshold only | test pseudo-label filtering |
| A2 | pixel-wise confidence map | test local ensemble |
| A3 | patch-wise confidence map | compare smoother local weighting |
| A4 | connected-component confidence | test object-level reliability |
| A5 | region-wise confidence + adaptive threshold | main proposed variant |
| A6 | A5 + boundary-aware loss | test boundary/small-object gains |

Run at least on:

- Prostate, because it is the main dataset in the paper;
- BUSI, because current summarized results are relatively weak;
- M&Ms, because the paper reports small-target limitations.

## 6. Metrics to Report

Report the same metrics as the paper where available:

- Dice coefficient;
- Jaccard;
- 95HD;
- ASD;
- per-domain Dice;
- average Dice across domains.

Also report training diagnostics:

- mask ratio;
- self-confidence;
- mutual-confidence;
- ensemble pseudo-label Dice if labels are available for analysis;
- percentage of ignored pseudo-label pixels;
- boundary Dice or surface Dice if boundary loss is added.

## 7. Expected Outcome

The most realistic improvement is not a dramatic architecture replacement, but a better pseudo-label selection and fusion mechanism.

Expected gains:

- modest but consistent average Dice improvement on Prostate and Fundus;
- larger potential improvement on BUSI and M&Ms;
- reduced late-training degradation when pseudo-labels become overconfident;
- better boundary behavior if boundary-aware loss is added.

## 8. Main Risks

- Pixel-wise confidence maps may be noisy and produce unstable supervision.
- Adaptive thresholding can reduce the number of supervised pixels too much.
- Boundary losses may improve contours but hurt region Dice if overweighted.
- Domain-aware clustering may introduce unnecessary complexity if the current domain gap is already handled by SynFoC.

## 9. Practical Recommendation

Start with the smallest meaningful change:

1. Log confidence histograms and mask ratios per dataset/domain.
2. Replace the fixed threshold with an adaptive threshold.
3. Replace scalar `ratio` with a region-wise confidence map.
4. Compare against baseline using identical seeds and labeled data settings.

If this improves pseudo-label quality and Dice, then add boundary-aware loss as a second-stage improvement.

