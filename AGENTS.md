# Repository Guidelines

## Project Structure & Module Organization

- `code/train.py` and `code/test.py` are the main training and evaluation entry points.
- `code/networks/` contains conventional segmentation models; `code/segment_anything/` contains the bundled SAM implementation and model components.
- `code/dataloaders/` handles dataset loading and augmentation, while `code/utils/` provides losses, metrics, schedules, and shared helpers.
- `data_format/` documents the expected directory layout for Fundus, Prostate, M&Ms, and BUSI datasets. Treat its images as examples or local research data, not general source files.
- `model/<dataset>/train/<experiment>/` stores experiment scripts, logs, and generated outputs. Large weights belong in `checkpoints/` or `model/` and must remain untracked.

## Build, Test, and Development Commands

This project is script-driven and has no build system or dependency manifest. Create a compatible Python/PyTorch environment before running commands from `code/`.

```bash
cd code
python train.py --dataset prostate --lb_domain 1 --lb_num 40 \
  --save_name trial --gpu 0 --AdamW --warmup --model MedSAM
python test.py --dataset prostate --save_name trial --gpu 0
python -m compileall train.py test.py dataloaders networks utils
```

Update dataset and checkpoint paths in `train.py` and `test.py` for your machine. The compile command is a lightweight syntax check; training and evaluation require CUDA-capable dependencies and prepared data.

## Coding Style & Naming Conventions

Use Python with four-space indentation and follow existing PEP 8-style patterns. Name functions and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Keep dataset-specific branching explicit and preserve existing CLI argument names for reproducibility. No formatter or linter is configured, so avoid unrelated reformatting and keep imports grouped logically.

## Testing Guidelines

There is currently no automated unit-test suite; `code/test.py` is an evaluation program, not a test runner. Validate changes with `compileall`, then run the smallest relevant dataset/experiment configuration. For model or metric changes, report the dataset, labeled domain/count, checkpoint, seed, and key Dice results in the pull request.

## Commit & Pull Request Guidelines

Git history uses short, imperative summaries such as `Update code (ignored large model weights)`. Keep commits focused and describe the observable change. Pull requests should include purpose, affected datasets/modules, exact commands used, and evaluation results. Link related issues or papers, and attach sample masks or metric plots when outputs change. Never commit raw datasets, `.pth` checkpoints, caches, TensorBoard events, or machine-specific paths.
