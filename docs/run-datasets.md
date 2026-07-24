# How to Run SynFoC on Each Dataset

This guide explains how to prepare paths, train, evaluate, and summarize results for the SynFoC implementation in this repository.

Run commands from the `code/` directory unless stated otherwise.

```bash
cd code
```

## 1. Environment Requirements

This repository does not include a dependency manifest. Prepare a Python environment with CUDA-capable PyTorch and the packages imported by `train.py` and `test.py`.

Common required packages include:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `opencv-python`
- `scikit-image`
- `Pillow`
- `tqdm`
- `tensorboardX`
- `matplotlib`
- `medpy`
- `h5py`
- `SimpleITK`

Use a quick syntax check after setting up the environment:

```bash
python -m compileall train.py test.py dataloaders networks utils
```

## 2. Checkpoints

`train.py` expects pretrained SAM/MedSAM checkpoints under `../checkpoints/` relative to `code/`.

Expected files:

```text
checkpoints/
  sam_vit_b_01ec64.pth
  medsam_vit_b.pth
```

The `--model` argument selects which checkpoint is used:

| `--model` | Checkpoint |
| --- | --- |
| `SAM` | `../checkpoints/sam_vit_b_01ec64.pth` |
| `MedSAM` | `../checkpoints/medsam_vit_b.pth` |

Recommended default:

```bash
--model MedSAM --AdamW --warmup
```

## 3. Dataset Paths

The dataset paths are hard-coded near the bottom of `code/train.py` and `code/test.py`.

Current expected paths, relative to `code/`:

| Dataset argument | Expected path | Domains |
| --- | --- | --- |
| `fundus` | `../data/Fundus` | 4 |
| `prostate` | `../data/ProstateSlice` | 6 |
| `MNMS` | `../data/mnms` | 4 |
| `BUSI` | `../data/Dataset_BUSI_with_GT` | 2 |

If your datasets are stored elsewhere, update `train_data_path` in both `train.py` and `test.py`.

## 4. Output Layout

Training saves logs and checkpoints under:

```text
model/<dataset>/train/<save_name>/
```

Important outputs:

```text
log.txt
log/
train.py
unet_avg_dice_best_model.pth
SAM_avg_dice_best_model.pth
```

The checkpoint files are written only when `--save_model` is enabled.

Important note: `test.py` loads from:

```text
model/<dataset>/<save_name>/
```

Therefore, for checkpoints produced by `train.py`, pass:

```bash
--save_name train/<experiment_name>
```

For example, if training used `--save_name prostate_dm1_lb40_full`, evaluate with:

```bash
python test.py --dataset prostate --save_name train/prostate_dm1_lb40_full --model unet --gpu 0 --overwrite
```

## 5. Key Arguments

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | One of `fundus`, `prostate`, `MNMS`, `BUSI` |
| `--lb_domain` | Domain used as the labeled source domain |
| `--lb_num` | Number of labeled samples selected from `lb_domain` |
| `--save_name` | Experiment name |
| `--gpu` | CUDA device id |
| `--model` | `MedSAM` or `SAM` for training; `unet` or `SAM` for testing |
| `--AdamW` | Use AdamW for SAM/MedSAM adaptation |
| `--warmup` | Warm up the SAM/MedSAM learning rate |
| `--save_model` | Save best U-Net and SAM/MedSAM checkpoints |
| `--overwrite` | Allow writing into an existing experiment directory |
| `--rank` | LoRA rank, default `4` |
| `--threshold` | Confidence threshold for pseudo-label supervision, default `0.95` |

## 6. Fundus

### 6.1 Expected Structure

`FundusSegmentation` expects:

```text
data/Fundus/
  Domain1/
    train/ROIs/image/*.png
    train/ROIs/mask/*.png
    test/ROIs/image/*.png
    test/ROIs/mask/*.png
  Domain2/
    train/ROIs/image/*.png
    train/ROIs/mask/*.png
    test/ROIs/image/*.png
    test/ROIs/mask/*.png
  Domain3/
    ...
  Domain4/
    ...
```

Domain mapping in code:

| Domain id | Name |
| --- | --- |
| 1 | DGS |
| 2 | RIM |
| 3 | REF |
| 4 | REF_val |

Fundus has 2 foreground targets:

- cup
- disc

### 6.2 Train

Example with Domain 1 as labeled domain and 40 labeled samples:

```bash
python train.py \
  --dataset fundus \
  --lb_domain 1 \
  --lb_num 40 \
  --save_name fundus_dm1_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Default effective settings in `train.py`:

- `max_iterations = 30000`
- `patch_size = 256`
- `num_classes = 2`
- `domain_num = 4`

### 6.3 Test

Evaluate the U-Net checkpoint:

```bash
python test.py \
  --dataset fundus \
  --save_name train/fundus_dm1_lb40_full \
  --model unet \
  --gpu 0 \
  --overwrite
```

Evaluate the SAM/MedSAM LoRA checkpoint:

```bash
python test.py \
  --dataset fundus \
  --save_name train/fundus_dm1_lb40_full \
  --model SAM \
  --gpu 0 \
  --overwrite
```

## 7. Prostate

### 7.1 Expected Structure

`ProstateSegmentation` expects:

```text
data/ProstateSlice/
  BIDMC/
    train/image/*.png
    train/mask/*.png
    test/image/*.png
    test/mask/*.png
  BMC/
    train/image/*.png
    train/mask/*.png
    test/image/*.png
    test/mask/*.png
  HK/
    ...
  I2CVB/
    ...
  RUNMC/
    ...
  UCL/
    ...
```

Domain mapping in code:

| Domain id | Name |
| --- | --- |
| 1 | BIDMC |
| 2 | BMC |
| 3 | HK |
| 4 | I2CVB |
| 5 | RUNMC |
| 6 | UCL |

Prostate has 1 foreground target:

- base

### 7.2 Train

Example with BIDMC as labeled domain:

```bash
python train.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 40 \
  --save_name prostate_dm1_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Example with HK as labeled domain:

```bash
python train.py \
  --dataset prostate \
  --lb_domain 3 \
  --lb_num 40 \
  --save_name prostate_dm3_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Default effective settings in `train.py`:

- `max_iterations = 60000`
- `patch_size = 384`
- `num_classes = 1`
- `domain_num = 6`

### 7.3 Test

Evaluate U-Net:

```bash
python test.py \
  --dataset prostate \
  --save_name train/prostate_dm1_lb40_full \
  --model unet \
  --gpu 0 \
  --overwrite
```

Evaluate SAM/MedSAM:

```bash
python test.py \
  --dataset prostate \
  --save_name train/prostate_dm1_lb40_full \
  --model SAM \
  --gpu 0 \
  --overwrite
```

## 8. M&Ms / MNMS

### 8.1 Expected Structure

Use `--dataset MNMS`.

`MNMSSegmentation` expects:

```text
data/mnms/
  vendorA/
    train/image/*.png
    train/mask/*.png
    test/image/*.png
    test/mask/*.png
  vendorB/
    train/image/*.png
    train/mask/*.png
    test/image/*.png
    test/mask/*.png
  vendorC/
    ...
  vendorD/
    ...
```

Domain mapping in code:

| Domain id | Name |
| --- | --- |
| 1 | vendorA |
| 2 | vendorB |
| 3 | vendorC |
| 4 | vendorD |

M&Ms has 3 foreground targets:

- lv
- myo
- rv

The mask loader expects RGB masks where each foreground class is encoded through a channel with value `255`.

### 8.2 Train

```bash
python train.py \
  --dataset MNMS \
  --lb_domain 1 \
  --lb_num 40 \
  --save_name MNMS_dm1_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Default effective settings in `train.py`:

- `max_iterations = 60000`
- `patch_size = 288`
- `num_classes = 3`
- `domain_num = 4`

### 8.3 Test

Evaluate U-Net:

```bash
python test.py \
  --dataset MNMS \
  --save_name train/MNMS_dm1_lb40_full \
  --model unet \
  --gpu 0 \
  --overwrite
```

Evaluate SAM/MedSAM:

```bash
python test.py \
  --dataset MNMS \
  --save_name train/MNMS_dm1_lb40_full \
  --model SAM \
  --gpu 0 \
  --overwrite
```

## 9. BUSI

### 9.1 Expected Structure

Use `--dataset BUSI`.

`BUSISegmentation` expects the original BUSI-style folder layout:

```text
data/Dataset_BUSI_with_GT/
  benign/
    benign (1).png
    benign (1)_mask.png
    benign (2).png
    benign (2)_mask.png
    ...
  malignant/
    malignant (1).png
    malignant (1)_mask.png
    malignant (2).png
    malignant (2)_mask.png
    ...
```

The code maps domains as:

| Domain id | Name |
| --- | --- |
| 1 | benign |
| 2 | malignant |

The `normal/` folder may exist in the original BUSI dataset, but this implementation uses only benign and malignant domains.

BUSI has 1 foreground target:

- base

The dataset class splits each domain internally:

- first 80%: train
- last 20%: test

### 9.2 Train

Example with benign as labeled domain:

```bash
python train.py \
  --dataset BUSI \
  --lb_domain 1 \
  --lb_num 40 \
  --save_name BUSI_dm1_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Example with malignant as labeled domain:

```bash
python train.py \
  --dataset BUSI \
  --lb_domain 2 \
  --lb_num 40 \
  --save_name BUSI_dm2_lb40_full \
  --gpu 0 \
  --AdamW \
  --warmup \
  --model MedSAM \
  --save_model
```

Default effective settings in `train.py`:

- `max_iterations = 30000`
- `patch_size = 256`
- `num_classes = 1`
- `domain_num = 2`

### 9.3 Test

Evaluate U-Net:

```bash
python test.py \
  --dataset BUSI \
  --save_name train/BUSI_dm1_lb40_full \
  --model unet \
  --gpu 0 \
  --overwrite
```

Evaluate SAM/MedSAM:

```bash
python test.py \
  --dataset BUSI \
  --save_name train/BUSI_dm1_lb40_full \
  --model SAM \
  --gpu 0 \
  --overwrite
```

## 10. Running Multiple Labeled Domains

To reproduce MiDSS-style experiments, repeat training with different `--lb_domain` values.

Example for Prostate:

```bash
python train.py --dataset prostate --lb_domain 1 --lb_num 40 --save_name prostate_dm1_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python train.py --dataset prostate --lb_domain 2 --lb_num 40 --save_name prostate_dm2_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python train.py --dataset prostate --lb_domain 3 --lb_num 40 --save_name prostate_dm3_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python train.py --dataset prostate --lb_domain 4 --lb_num 40 --save_name prostate_dm4_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python train.py --dataset prostate --lb_domain 5 --lb_num 40 --save_name prostate_dm5_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python train.py --dataset prostate --lb_domain 6 --lb_num 40 --save_name prostate_dm6_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
```

For existing experiment directories, add `--overwrite`.

## 11. Summarize Logs

After experiments finish, summarize all logs:

```bash
python summarize_logs.py \
  --model-dir ../model \
  --format json \
  --output summary.json
```

Markdown output:

```bash
python summarize_logs.py \
  --model-dir ../model \
  --format markdown \
  --output summary.md
```

CSV output:

```bash
python summarize_logs.py \
  --model-dir ../model \
  --format csv \
  --output summary.csv
```

## 12. Common Problems

### Experiment Directory Already Exists

If the output folder already exists, training or testing raises an exception unless `--overwrite` is passed.

Use:

```bash
--overwrite
```

### Test Cannot Find Checkpoint

Check whether `--save_name` points to the training subdirectory.

If training saved to:

```text
model/prostate/train/prostate_dm1_lb40_full/
```

then test with:

```bash
--save_name train/prostate_dm1_lb40_full
```

Also confirm training used `--save_model`; otherwise the best model checkpoint may not exist.

### Wrong Dataset Path

Update `train_data_path` in both `train.py` and `test.py`, then rerun.

### CUDA Out of Memory

Try:

- reduce `--label_bs` and `--unlabel_bs`;
- use a smaller `--img_size`;
- keep `--amp 1`;
- train one experiment at a time per GPU.

Note that `train.py` overrides `label_bs` and `unlabel_bs` based on `lb_num` near the bottom of the file. If you need manual batch sizes, edit that block.

### Missing MedSAM/SAM Checkpoint

Place checkpoints in:

```text
checkpoints/
  sam_vit_b_01ec64.pth
  medsam_vit_b.pth
```

or update the checkpoint dictionary in `train.py`.

## 13. Recommended Minimal Run Order

For a first smoke test:

1. Verify paths and checkpoints.
2. Run `compileall`.
3. Train one small setting with `--save_model`.
4. Test U-Net.
5. Test SAM/MedSAM.
6. Summarize logs.

Example:

```bash
cd code
python -m compileall train.py test.py dataloaders networks utils
python train.py --dataset prostate --lb_domain 1 --lb_num 40 --save_name prostate_dm1_lb40_full --gpu 0 --AdamW --warmup --model MedSAM --save_model
python test.py --dataset prostate --save_name train/prostate_dm1_lb40_full --model unet --gpu 0 --overwrite
python test.py --dataset prostate --save_name train/prostate_dm1_lb40_full --model SAM --gpu 0 --overwrite
python summarize_logs.py --model-dir ../model --format json --output summary.json
```

