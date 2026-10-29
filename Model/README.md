# Model experiments

The original `model.py` is now split into:

- `models.py`: the original tapering head and a fresh DeBERTa backbone per instance.
- `data.py`: dataset loading and batch tokenization.
- `train.py`: a shared training entry point for independent experiments.
- `evaluate.py`: loss, accuracy, macro F1, per-class precision/recall, and a confusion matrix.
- `configs/`: head-only and full-fine-tuning experiment settings.

Install dependencies in your Python environment:

```sh
python -m pip install torch transformers datasets sentencepiece
```

Run from the repository root:

```sh
python -m Model.train --config Model/configs/tapering_head_only.json
python -m Model.train --config Model/configs/tapering_full.json
```

These commands train models; importing the modules does not load datasets,
download weights, or start training. The first run needs access to Hugging Face.
The old single-example prediction used an untrained head; the entry point now
trains and validates that head instead.

The configs retain the original dataset's `test` split for validation. Once a
dedicated validation split is available, update `validation_split` and reserve
`test` for final evaluation. Labels must be integers from 0 to `num_labels - 1`;
the dataset's existing label mapping is preserved.

The learning rates and epoch counts are starting settings to tune, not measured
best settings. `device` defaults to `cpu`; set it to `cuda` or `mps` when supported
by your machine. Reduce `batch_size` if memory is limited.

Each run saves its config, tokenizer, per-epoch `metrics.json`, and the checkpoint
with the highest validation macro F1 (`best.pt`). A checkpoint contains
`model_state_dict`, `config`, and `epoch`. To restore one, pass its `config` to
`build_model`, then load `model_state_dict` with `model.load_state_dict(...)`.
`evaluate(model, loader, device, num_labels)` evaluates a loaded model; confusion
matrix rows are true classes and columns are predicted classes. Missing-class
precision, recall, and F1 are zero.

Use a new `output_dir` for each run; existing directories are rejected to protect
previous results. The same seed and data splits are used in both configs, though
hardware-dependent nondeterminism may remain. Add future heads in `models.py`
and register them in `build_model`; the training and evaluation code can be reused.

The model returns raw logits for cross-entropy loss. For inference probabilities,
use `logits.softmax(dim=-1)`. W&B integration is not included; metrics are currently
printed and saved locally.

## Common Pile dataset

The published [Common Pile release](https://huggingface.co/datasets/Rodrigo-Martin/productivity_enhancer_common_pile)
contains 6,000 examples, with 2,000 per class. It uses the same `video_title`,
`video_description`, and `productivity_label` columns. Labels are based on titles
only, matching the current model input:

- `0`: science and technical education, including mathematics, computing,
  engineering, natural sciences, and medical science.
- `1`: other instruction or substantive explanation, such as history, languages,
  cooking, arts, fitness, and finance.
- `2`: non-educational content, including entertainment, performances, promotion,
  personal updates, and news without substantive teaching or explanation.

The splits contain 4,800 training, 600 validation, and 600 test examples, balanced
within each split, with no shared channels between splits. Use the matching
experiment configs:

```sh
python -m Model.train --config Model/configs/common_pile_head_only.json
python -m Model.train --config Model/configs/common_pile_full.json
```

The current training loop uses unweighted cross-entropy. The release's labels are
AI-generated; use an independently human-labeled sample to assess real-world
blocking quality. Source attribution, the full label policy, and preparation
details are preserved in the published dataset card and accompanying files. The
one-time labeling pipeline and local downloads have been removed from this project;
training loads the published dataset directly from Hugging Face.
