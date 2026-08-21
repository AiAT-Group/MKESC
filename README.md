# MKESC

Code for **Human-Centered Emotional Support Conversation via Cognitive-Behavioral Knowledge Reasoning**.

This repository contains the model, data-loading, preprocessing, training, inference, and evaluation code. Conversation datasets, pretrained model weights, checkpoints, generated outputs, and dataset-specific intermediate files are intentionally not included.

## Repository layout

```text
CONFIG/       Model configuration files
RUN/          Reproducible command templates for preprocessing, training, and inference
tools/        Offline CB-CoT and hierarchical DPR preprocessing tools
inputters/    Conversation input and feature construction
models/       MKESC, no-persona ablation, and BlenderBot baseline implementations
utils/        Training and evaluation utilities
metric/       Automatic evaluation metrics
apex/         Vendored NVIDIA Apex Python sources used by the mixed-precision path
DPR-reader/   Placeholder for the external DPR reader model
```

## Requirements

The original environment targets Python 3.8 and CUDA-enabled PyTorch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default command templates use `--fp16 false`. Mixed-precision training additionally requires a compatible CUDA/Apex environment.

## External resources

### Pretrained model

Download [facebook/blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M) and place the model files in `Blenderbot_small-90M/`. The model weights are excluded from this repository.

### Dataset

Place the approved, locally available JSONL conversation files in `DATA/` using these filenames:

```text
DATA/6_train.txt
DATA/6_valid.txt
DATA/6_test.txt
```

The paper reports experiments on ESConv and cross-domain Empathetic Dialogue data. The datasets and their derived CB-CoT/DPR annotations are intentionally excluded from this repository.

Do not commit the dataset or generated preprocessing/training outputs. See `DATA/README.md` for the expected role of this directory.

### Optional GloVe evaluation resource

Embedding-based metrics require the converted GloVe file in `glove6B/`. The directory is ignored by Git. To use a different location, set `MKESC_GLOVE_DIR` before running the conversion or evaluation code.

### Method preprocessing resources

The paper's offline CB-CoT and hierarchical DPR retrieval stages are retained in `tools/`. CB-CoT uses an OpenAI-compatible LLM endpoint; retrieval uses the external DPR reader described in [DPR-reader/README.md](DPR-reader/README.md). See [tools/README.md](tools/README.md) for the pipeline. API responses and retrieval annotations are generated artifacts and are not committed.

### Baselines reported in the paper

The paper compares MKESC with BlenderBot-Joint, MISC, TransESC, KEMI, PAL, DKPE, and MultiAgentESC. Their full implementations were not present in the source snapshot. The local vanilla BlenderBot branch is retained as the available baseline implementation; the other baseline names refer to external systems and are not claimed to be reproduced here.

## Usage

Run the commands from the repository root.

1. Preprocess the training data:

   ```bash
   bash RUN/prepare_strat.sh
   ```

2. Train the model:

   ```bash
   bash RUN/train_strat.sh
   ```

3. Run inference with a checkpoint:

   ```bash
   CHECKPOINT=DATA/path/to/epoch-4.bin \
   INPUT_FILE=DATA/6_test.txt \
   bash RUN/infer_strat.sh
   ```

   Add `ADD_NLG_EVAL=true` when the optional GloVe and METEOR evaluation resources are available. The large METEOR paraphrase table is intentionally omitted; see `metric/pycocoevalcap/meteor/data/README.md`.

Set `CUDA_VISIBLE_DEVICES` when a specific GPU should be used, for example `CUDA_VISIBLE_DEVICES=0 bash RUN/train_strat.sh`. Set `SEED` to repeat the paper's multi-seed experiments.

The public copy keeps the supported prepare/train/inference path. An older interactive shell script was omitted because its `interact.py` entrypoint was not present in the source directory.

The paper's local BlenderBot baseline is retained as `inputters/vanilla.py`, `models/vanilla_blenderbot_small.py`, and `CONFIG/vanilla.json`. The DialogPT variants and interactive-only branch were not part of the paper's reported method or baselines and were omitted.

The same command templates can run the local baseline by setting `CONFIG_NAME=vanilla INPUTTER_NAME=vanilla`.

## Evaluation

For a JSONL result file containing `response` and `generation` fields:

```bash
python eval_cider.py DATA/result.jsonl
python eval_meteor.py DATA/result.jsonl
```

## License

The project license is currently unspecified. Third-party components retain their upstream attribution and license files where available.

## Citation

The final paper citation and publication link are currently unspecified and will be added once the paper metadata is finalized.
