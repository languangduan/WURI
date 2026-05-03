# WURI: Watching Unfolding Risk in Agent Interactions

This repository contains code for preprocessing multi-source trajectory data, training a representation-based prefix monitor, and running prefix-level evaluation.


---

## Overview

The workflow has three steps:

1. Preprocess raw datasets into a unified JSONL format.
2. Train the model on the processed training split.
3. Run prefix-level evaluation on validation and held-out splits.

---

## Environment

Create a Python environment and install dependencies:

```bash
conda create -n anonymous_env python=3.10
conda activate anonymous_env
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install the packages required by your local setup, e.g.:

```bash
pip install torch numpy scikit-learn tqdm transformers sentence-transformers
```

---

## Data Preprocessing

Before training, preprocess the raw datasets:

```bash
python [PREPROCESS_SCRIPT_PATH] \
  --agenthazard      [RAW_AGENTHAZARD_PATH] \
  --toolbench        [RAW_TOOLBENCH_PATH] \
  --atbench          [RAW_ATBENCH_PATH] \
  --safedialbench_en [RAW_SAFEDIALBENCH_EN_PATH] \
  --safedialbench_zh [RAW_SAFEDIALBENCH_ZH_PATH] \
  --output_dir       data/processed_multisource \
  --zeroshot_hazard_types 6 7 \
  --val_ratio        0.10 \
  --sdb_test_ratio   0.15 \
  --min_steps        2 \
  --max_steps        20 \
  --seed             42
```

This creates processed files under:

```text
data/processed_multisource/
```

Expected outputs include:

```text
train.jsonl
val.jsonl
at_zeroshot.jsonl
sdb_test_en.jsonl
sdb_test_zh.jsonl
ah_test.jsonl
tb_test.jsonl
stats.json
label_map.json
```

---

## Training

Run:

```bash
python training/train_stage1_v7.py \
  --train_jsonl data/processed_multisource/train.jsonl \
  --val_jsonl   data/processed_multisource/val.jsonl   \
  --exclude_train_hazards "6,7"                        \
  --enable_unseen_analysis                             \
  --neg_recon_weight 7.7                               \
  --max_traj_length  22                                \
  --num_workers 4                                      \
  --save_dir results/trace_stage1_v7
```

The best checkpoint is saved as:

```text
results/trace_stage1_v7/stage1_best.pth
```

---

## Prefix-Level Evaluation

Run:

```bash
python analysis/prefix_eval.py \
  --ckpt_path    results/trace_stage1_v7/stage1_best.pth \
  --center_jsonl data/processed_multisource/train.jsonl  \
  --eval_jsonl \
      data/processed_multisource/val.jsonl          \
      data/processed_multisource/sdb_test_en.jsonl  \
      data/processed_multisource/sdb_test_zh.jsonl  \
      data/processed_multisource/at_zeroshot.jsonl  \
      data/processed_multisource/ah_test.jsonl      \
      data/processed_multisource/tb_test.jsonl      \
  --eval_names val sdb_en sdb_zh at_zeroshot ah_test tb_test \
  --benign_jsonl data/processed_multisource/tb_test.jsonl \
  --use_full_for_short \
  --max_prefix      20  \
  --max_traj_length 22  \
  --target_tpr      0.8 \
  --batch_size      32  \
  --output_dir      analysis/prefix_eval_v7_full_v2
```

Evaluation results are written to:

```text
analysis/prefix_eval_v7_full_v2/
```

---

## Expected Directory Structure

```text
.
├── training/
│   └── train_stage1_v7.py
├── analysis/
│   └── prefix_eval.py
├── data/
│   └── processed_multisource/
├── results/
│   └── trace_stage1_v7/
└── README.md
```

---

## Notes

- Run preprocessing before training or evaluation.
- Make sure all raw dataset paths are correctly specified.
- The order of `--eval_jsonl` must match the order of `--eval_names`.
- The evaluation checkpoint path should match the training output directory.
- Results may vary slightly depending on random seeds, hardware, and dependency versions.

---

## License

The license will be specified in the final public release.
