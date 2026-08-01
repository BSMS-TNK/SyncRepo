# Region-Wise SMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SynFoC's sample-level pseudo-label mixing ratio with a gradient-free pixel-wise Region-Wise Self-Mutual Confidence map.

**Architecture:** Add a small tensor-only utility that computes U-Net self-confidence, U-Net/MedSAM mutual confidence, relative model scores, and the spatial U-Net weight. The training loop aligns the three probability maps, calls the utility inside `torch.no_grad()`, generates hard pseudo-labels at U-Net and MedSAM resolutions, and reduces maps only for logging.

**Tech Stack:** Python, PyTorch, `unittest`

---

## File Structure

- Create `code/utils/region_smc.py`: pure pixel-wise confidence and blending function.
- Create `code/tests/test_region_smc.py`: CPU behavior tests for shapes, bounds, model selection, multiclass operation, and gradient safety.
- Modify `code/train.py`: replace scalar Dice confidence generation with spatial Region-Wise SMC and update monitoring reductions.

### Task 1: Define Pixel-Wise Confidence Behavior

**Files:**
- Create: `code/tests/test_region_smc.py`
- Create: `code/utils/region_smc.py`

- [ ] **Step 1: Write the failing shape, bounds, and gradient test**

```python
import unittest

import torch

from utils.region_smc import blend_region_wise_probabilities


class RegionWiseSMCTest(unittest.TestCase):
    def test_returns_spatial_maps_without_gradients(self):
        sam_prob = torch.tensor(
            [[[[0.8, 0.3]], [[0.2, 0.7]]]], requires_grad=True
        )
        unet_teacher_prob = torch.tensor(
            [[[[0.7, 0.4]], [[0.3, 0.6]]]], requires_grad=True
        )
        unet_student_prob = torch.tensor(
            [[[[0.6, 0.5]], [[0.4, 0.5]]]], requires_grad=True
        )

        blended, self_conf, mutual_conf, alpha_map = (
            blend_region_wise_probabilities(
                sam_prob,
                unet_teacher_prob,
                unet_student_prob,
            )
        )

        self.assertEqual(blended.shape, (1, 2, 1, 2))
        self.assertEqual(self_conf.shape, (1, 1, 1, 2))
        self.assertEqual(mutual_conf.shape, (1, 1, 1, 2))
        self.assertEqual(alpha_map.shape, (1, 1, 1, 2))
        for tensor in (blended, self_conf, mutual_conf, alpha_map):
            self.assertFalse(tensor.requires_grad)
        for confidence_map in (self_conf, mutual_conf, alpha_map):
            self.assertTrue(torch.all(confidence_map >= 0.0))
            self.assertTrue(torch.all(confidence_map <= 1.0))
        torch.testing.assert_close(
            blended.sum(dim=1), torch.ones_like(blended[:, 0])
        )
```

- [ ] **Step 2: Run the test and verify RED**

Run from `code/`:

```bash
python -m unittest tests.test_region_smc.RegionWiseSMCTest.test_returns_spatial_maps_without_gradients -v
```

Expected: FAIL because `utils.region_smc` does not exist.

- [ ] **Step 3: Implement the minimal tensor helper**

```python
import torch


@torch.no_grad()
def blend_region_wise_probabilities(
    sam_teacher_prob,
    unet_teacher_prob,
    unet_student_prob,
    eps=1e-6,
):
    unet_local_conf = unet_teacher_prob.max(dim=1, keepdim=True).values
    sam_local_conf = sam_teacher_prob.max(dim=1, keepdim=True).values

    self_conf = 1.0 - 0.5 * (
        unet_teacher_prob - unet_student_prob
    ).abs().sum(dim=1, keepdim=True)
    mutual_conf = 1.0 - 0.5 * (
        unet_teacher_prob - sam_teacher_prob
    ).abs().sum(dim=1, keepdim=True)
    self_conf = self_conf.clamp(0.0, 1.0)
    mutual_conf = mutual_conf.clamp(0.0, 1.0)

    unet_score = unet_local_conf * self_conf
    sam_score = sam_local_conf
    relative_alpha = unet_score / (unet_score + sam_score + eps)
    alpha_map = (
        mutual_conf * 0.5
        + (1.0 - mutual_conf) * relative_alpha
    ).clamp(0.0, 1.0)

    blended_prob = (
        alpha_map * unet_teacher_prob
        + (1.0 - alpha_map) * sam_teacher_prob
    )
    return blended_prob, self_conf, mutual_conf, alpha_map
```

- [ ] **Step 4: Run the test and verify GREEN**

Run:

```bash
python -m unittest tests.test_region_smc.RegionWiseSMCTest.test_returns_spatial_maps_without_gradients -v
```

Expected: PASS.

### Task 2: Verify Local Model Selection

**Files:**
- Modify: `code/tests/test_region_smc.py`
- Modify: `code/utils/region_smc.py`

- [ ] **Step 1: Add failing tests for agreement and disagreement**

```python
    def test_agreement_moves_alpha_to_half(self):
        shared_prob = torch.tensor([[[[0.9]], [[0.1]]]])

        _, _, mutual_conf, alpha_map = blend_region_wise_probabilities(
            shared_prob,
            shared_prob,
            shared_prob,
        )

        torch.testing.assert_close(mutual_conf, torch.ones_like(mutual_conf))
        torch.testing.assert_close(alpha_map, torch.full_like(alpha_map, 0.5))

    def test_stable_confident_unet_is_preferred_during_disagreement(self):
        sam_prob = torch.tensor([[[[0.55]], [[0.45]]]])
        unet_prob = torch.tensor([[[[0.01]], [[0.99]]]])

        blended, _, _, alpha_map = blend_region_wise_probabilities(
            sam_prob,
            unet_prob,
            unet_prob,
        )

        self.assertGreater(alpha_map.item(), 0.5)
        self.assertEqual(blended.argmax(dim=1).item(), 1)

    def test_unstable_unet_is_deprioritized(self):
        sam_prob = torch.tensor([[[[0.99]], [[0.01]]]])
        unet_teacher_prob = torch.tensor([[[[0.45]], [[0.55]]]])
        unet_student_prob = sam_prob.clone()

        blended, self_conf, _, alpha_map = blend_region_wise_probabilities(
            sam_prob,
            unet_teacher_prob,
            unet_student_prob,
        )

        self.assertLess(self_conf.item(), 0.5)
        self.assertLess(alpha_map.item(), 0.5)
        self.assertEqual(blended.argmax(dim=1).item(), 0)
```

- [ ] **Step 2: Run the selection tests and verify RED or behavior gaps**

Run:

```bash
python -m unittest tests.test_region_smc -v
```

Expected before final helper completion: at least one selection assertion fails if the score or agreement formula is incomplete.

- [ ] **Step 3: Complete the score and alpha calculation**

Ensure `code/utils/region_smc.py` exactly implements:

```python
unet_score = unet_local_conf * self_conf
sam_score = sam_local_conf
relative_alpha = unet_score / (unet_score + sam_score + eps)
alpha_map = (
    mutual_conf * 0.5
    + (1.0 - mutual_conf) * relative_alpha
).clamp(0.0, 1.0)
```

- [ ] **Step 4: Run all Region-Wise SMC tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_region_smc -v
```

Expected: all tests PASS.

### Task 3: Cover Multi-Class Inputs

**Files:**
- Modify: `code/tests/test_region_smc.py`

- [ ] **Step 1: Add the failing multi-class test**

```python
    def test_supports_multiclass_probability_maps(self):
        sam_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
        unet_teacher_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)
        unet_student_prob = torch.softmax(torch.randn(2, 3, 4, 5), dim=1)

        blended, self_conf, mutual_conf, alpha_map = (
            blend_region_wise_probabilities(
                sam_prob,
                unet_teacher_prob,
                unet_student_prob,
            )
        )

        self.assertEqual(blended.shape, (2, 3, 4, 5))
        self.assertEqual(self_conf.shape, (2, 1, 4, 5))
        self.assertEqual(mutual_conf.shape, (2, 1, 4, 5))
        self.assertEqual(alpha_map.shape, (2, 1, 4, 5))
```

- [ ] **Step 2: Run the multi-class test**

Run:

```bash
python -m unittest tests.test_region_smc.RegionWiseSMCTest.test_supports_multiclass_probability_maps -v
```

Expected: PASS if the helper remains class-count agnostic; otherwise FAIL and expose a hard-coded channel assumption.

- [ ] **Step 3: Remove any class-count assumptions**

Keep all reductions expressed as `dim=1, keepdim=True`; do not branch on dataset names or class counts.

- [ ] **Step 4: Re-run the complete helper tests**

Run:

```bash
python -m unittest tests.test_region_smc -v
```

Expected: all tests PASS.

### Task 4: Integrate Region-Wise SMC Into Training

**Files:**
- Modify: `code/train.py:31`
- Modify: `code/train.py:527`
- Modify: `code/train.py:654`

- [ ] **Step 1: Import the helper**

Add beside the existing utility imports:

```python
from utils.region_smc import blend_region_wise_probabilities
```

- [ ] **Step 2: Replace scalar confidence generation inside `torch.no_grad()`**

Replace the current weak-image confidence block with:

```python
with torch.no_grad():
    sam_output_ulb_x_w = ema_SAM_model(
        ulb_x_w, multimask_output, args.img_size
    )
    sam_logits_ulb_x_w = sam_output_ulb_x_w['low_res_logits']
    sam_prob_ulb_x_w = torch.softmax(sam_logits_ulb_x_w, dim=1)
    _, sam_pseudo_label = torch.max(sam_prob_ulb_x_w, dim=1)

    # Align continuous MedSAM probabilities with U-Net's spatial resolution.
    unet_size_sam_prob_ulb_x_w = F.interpolate(
        sam_prob_ulb_x_w,
        size=(patch_size, patch_size),
        mode='bilinear',
        align_corners=False,
    )

    unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
    unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
    _, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)

    # This student prediction is used only to estimate local stability.
    unet_stu_output_ulb_x_w = unet_model(ulb_unet_size_x_w)
    unet_stu_prob_ulb_x_w = torch.softmax(
        unet_stu_output_ulb_x_w, dim=1
    )

    (
        unet_size_prob_ulb_x_w,
        self_conf,
        mutual_conf,
        alpha_map,
    ) = blend_region_wise_probabilities(
        unet_size_sam_prob_ulb_x_w,
        unet_prob_ulb_x_w,
        unet_stu_prob_ulb_x_w,
    )

    unet_size_prob, unet_size_pseudo_label = torch.max(
        unet_size_prob_ulb_x_w, dim=1
    )
    unet_size_mask = (unet_size_prob > threshold).unsqueeze(1).float()

    low_res_prob_ulb_x_w = F.interpolate(
        unet_size_prob_ulb_x_w,
        size=(low_res, low_res),
        mode='bilinear',
        align_corners=False,
    )
    low_res_prob, low_res_pseudo_label = torch.max(
        low_res_prob_ulb_x_w, dim=1
    )
    low_res_mask = (low_res_prob > threshold).unsqueeze(1).float()

    self_conf_mean = self_conf.mean().item()
    mutual_conf_mean = mutual_conf.mean().item()
    alpha_mean = alpha_map.mean().item()
```

Delete the old hard-label Dice confidence branches, `np.mean` confidence reductions, `low_res_unet_pseudo_label`, and scalar `ratio` construction.

- [ ] **Step 3: Keep running statistics scalar-only**

Use monitoring values after the no-gradient block:

```python
self_conf_sta.update(self_conf_mean)
mutual_conf_sta.update(mutual_conf_mean)
ratio_sta.update(alpha_mean)
```

- [ ] **Step 4: Update TensorBoard and progress logging**

Replace every confidence logging reduction with:

```python
writer.add_scalar('train/self_conf', self_conf_mean, iter_num)
writer.add_scalar('train/mutual_conf', mutual_conf_mean, iter_num)
writer.add_scalar('train/ratio', alpha_mean, iter_num)
```

Pass `self_conf_mean`, `mutual_conf_mean`, and `alpha_mean` to each dataset-specific progress description instead of `np.mean(...)` expressions.

- [ ] **Step 5: Confirm no obsolete scalar blending remains**

Run from the repository root:

```powershell
Select-String -Path code\train.py -Pattern 'np.mean\(self_conf|self_conf \* mutual_conf|view\(len\(ulb_x_w\),1,1,1\)'
```

Expected: no matches.

### Task 5: Verify Integration

**Files:**
- Verify: `code/utils/region_smc.py`
- Verify: `code/tests/test_region_smc.py`
- Verify: `code/train.py`

- [ ] **Step 1: Run focused tests**

Run from `code/`:

```bash
python -m unittest tests.test_region_smc -v
```

Expected: all Region-Wise SMC tests PASS.

- [ ] **Step 2: Run existing unit tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run the syntax check**

Run:

```bash
python -m compileall train.py test.py dataloaders networks utils
```

Expected: compilation succeeds without syntax errors.

- [ ] **Step 4: Inspect the focused diff**

Run from the repository root:

```bash
git diff --check
git diff -- code/train.py code/utils/region_smc.py code/tests/test_region_smc.py docs/superpowers/specs/2026-08-01-region-wise-smc-design.md
```

Expected: no whitespace errors; changes are limited to Region-Wise SMC, its tests, and documentation.

No commit is included because repository changes must not be committed unless the user explicitly requests it.
