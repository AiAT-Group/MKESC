# Method preprocessing tools

These scripts preserve the offline method stages described in the paper without bundling datasets, API outputs, or retrieval caches.

The R1 manuscript names the CB-CoT model inconsistently: Fig. 2 says `DeepSeek-V3.2`, while Table II says `DeepSeek-R1-V3.2`. Confirm the exact SiliconFlow model ID with the authors before running the tool; set it with `--model` or `MKESC_CB_COT_MODEL`. The script intentionally does not request API JSON mode because provider support for reasoning-model JSON mode is not universal.

## Pipeline

1. `generate_cb_cot.py` implements the offline Cognitive-Behavioral Chain-of-Thought stage shown in **Fig. 4** and corresponding to **Equations (1)-(2)**. It adds `persona_situation`, `persona_thought`, `persona_emotion`, and `persona_behavior` to system turns.
2. `retrieve_exemplars.py` implements the hierarchical DPR retrieval stage shown in **Fig. 5** and corresponding to **Equations (3)-(5)**. It first retrieves globally similar situations and then selects persona-aware exemplars. The default `--num-exemplars 3` follows the setting selected in **Fig. 7**.
3. `inputters/strat.py` converts the annotated JSONL data into model features. `models/strat_blenderbot_small.py` implements the bidirectional cross-attention and adaptive weighted fusion described in **Equations (6)-(11)**.
4. `train.py` and `infer.py` implement response generation with the BlenderBot-small backbone and the NLL training objective in **Equations (12)-(13)**.

## Example commands

Run these commands from the repository root after obtaining the approved dataset and external models:

```bash
export SILICONFLOW_API_KEY="<your-key>"
export MKESC_CB_COT_MODEL="<author-confirmed-model-id>"
python tools/generate_cb_cot.py \
  --input DATA/6_train.txt \
  --output DATA/6_train_cb_cot.txt

python tools/retrieve_exemplars.py \
  --query-file DATA/6_train_cb_cot.txt \
  --retrieval-corpus DATA/6_train_cb_cot.txt \
  --dpr-model-dir ./DPR-reader \
  --output DATA/6_train_annotated.txt
```

The same two preprocessing stages should be applied to validation and test inputs as required by the experiment. Generated JSONL files must remain local and are ignored by Git.

The scripts expect the canonical JSONL structure consumed by `inputters/strat.py`: top-level `situation`, `persona_list`, and `dialog` fields; user turns use `speaker: "usr"`, and system turns use `speaker: "sys"`, `text`, and `strategy`.
