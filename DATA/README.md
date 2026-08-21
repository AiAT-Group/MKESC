# Local data and generated artifacts

This directory is intentionally kept free of dataset files in the public repository.

Place the locally obtained JSONL conversation splits here:

```text
6_train.txt
6_valid.txt
6_test.txt
```

`prepare.py` also writes tokenized data and metadata here. Training checkpoints and inference outputs are generated under this directory as well; all of these files are ignored by Git.

The offline CB-CoT and DPR retrieval tools also write augmented JSONL files here. These annotations are intermediate outputs and must be regenerated locally when reproducing the paper.
