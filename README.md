# ECG Classification

Software-first starter repository for binary ECG classification. This repo turns the proposal guide into a runnable baseline: load data, preprocess signals, train a small 1D CNN, evaluate results, and keep the final classifier boundary explicit for later hardware mapping.

## What is in the repo

- `src/data_loader.py`: dataset loading, demo-data generation, and PyTorch dataloaders
- `src/preprocess.py`: signal shaping, normalization, binary label conversion, and dataset splits
- `src/model.py`: small feature extractor plus a separate final classifier layer
- `src/train.py`: end-to-end training loop and checkpoint saving
- `src/evaluate.py`: held-out evaluation with confusion matrix, precision, recall, and F1
- `src/inference.py`: single-sample prediction and feature-vector inspection
- `src/export_parts.py`: split a trained checkpoint into separate feature extractor and classifier files
- `data/README.md`: expected dataset format for the first software prototype
- `EE533_software_first_implementation_guide.docx`: original planning document

## Recommended first milestone

Build a binary classifier for `normal` vs `abnormal` ECG segments on a regular computer before moving any logic to custom hardware. The model is intentionally small so we can validate the pipeline first and cleanly expose the boundary between:

1. `feature_extractor`
2. `classifier`

That interface is the first candidate for later GPU mapping.

## Current model architecture

The baseline model is a small 1D CNN for binary ECG classification. By default, training resizes each ECG segment to length `256`, and single-channel inputs are shaped as `(batch, 1, 256)`.

The current default network in `src/model.py` is:

```text
Input: (batch, 1, 256)

Feature extractor:
- Conv1d(1 -> 16, kernel_size=7, padding=3)
- BatchNorm1d(16)
- ReLU
- MaxPool1d(2)

- Conv1d(16 -> 32, kernel_size=7, padding=3)
- BatchNorm1d(32)
- ReLU
- MaxPool1d(2)

- Conv1d(32 -> 64, kernel_size=7, padding=3)
- BatchNorm1d(64)
- ReLU
- MaxPool1d(2)

- AdaptiveAvgPool1d(1)
- Flatten

Classifier:
- Dropout(0.2)
- Linear(64 -> 2)

Output:
- logits with shape (batch, 2)
```

With the default input length of `256`, the tensor shape changes like this:

```text
(batch, 1, 256)
-> (batch, 16, 256)
-> (batch, 16, 128)
-> (batch, 32, 128)
-> (batch, 32, 64)
-> (batch, 64, 64)
-> (batch, 64, 32)
-> (batch, 64, 1)
-> (batch, 64)
-> (batch, 2)
```

Intuition:

- `Conv1d` increases the number of learned feature channels.
- `MaxPool1d(2)` halves the signal length while keeping the strongest local response in each window.
- `AdaptiveAvgPool1d(1)` compresses each channel into one value, producing a final `64`-dimensional feature vector for the classifier.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a smoke test without a real dataset:

```bash
python -m src.train --demo --run-name demo_baseline --epochs 5
python -m src.evaluate --checkpoint checkpoints/demo_baseline.pt --demo
python -m src.inference --checkpoint checkpoints/demo_baseline.pt --demo --index 0
```

Prepare a real MIT-BIH beat dataset in `.npz` format:

```bash
python -m src.prepare_mitbih --download --output data/processed/mitbih_binary.npz
```

This script downloads MIT-BIH Arrhythmia Database records into `data/raw/mitdb/`, extracts fixed-length beat-centered ECG segments, and writes a training-ready dataset with:

- `signals`: `(num_samples, signal_length)`
- `labels`: `(num_samples,)`, where `0` means a symbol listed in `--normal-symbols` and `1` means any other included beat symbol

Useful variants:

```bash
python -m src.prepare_mitbih --download --records 100 101 102
python -m src.prepare_mitbih --stream --records 100 --output data/processed/mitbih_100_binary.npz
python -m src.prepare_mitbih --download --normal-symbols N L R e j
```

Run with a real processed dataset:

```bash
python -m src.train --dataset data/processed/mitbih_binary.npz --run-name mitbih_baseline
python -m src.evaluate --checkpoint checkpoints/mitbih_baseline.pt --dataset data/processed/mitbih_binary.npz
python -m src.inference --checkpoint checkpoints/mitbih_baseline.pt --dataset data/processed/mitbih_binary.npz --split test --index 0
```

Export the trained model as two separate parts for hardware mapping:

```bash
python -m src.export_parts --checkpoint checkpoints/mitbih_baseline.pt
```

This saves `checkpoints/mitbih_baseline_feature_extractor.pt` (the 1D CNN) and `checkpoints/mitbih_baseline_classifier.pt` (the linear head) as independent state dicts.

Export the classifier into FPGA-friendly files:

```bash
python -m src.export_classifier_hw \
  --classifier-checkpoint checkpoints/mitbih_baseline_classifier.pt \
  --out-dir hardware_export \
  --prefix netfpga_classifier \
  --layout row-major \
  --word-bits 16 \
  --frac-bits 8
```

This writes:

- `hardware_export/netfpga_classifier_meta.json`
- `hardware_export/netfpga_classifier_weight_matrix.txt`
- `hardware_export/netfpga_classifier_bias_float.txt`
- `hardware_export/netfpga_classifier_weight_q16_8.mem`
- `hardware_export/netfpga_classifier_bias_q16_8.mem`

For the current model, the hardware classifier contract is:

- input feature vector: length `64`
- weight matrix: shape `(2, 64)`
- bias vector: shape `(2,)`
- math: `logits = W * features + b`

Export the classifier as BF16 files for a BF16-capable hardware path:

```bash
python -m src.export_classifier_bf16 \
  --classifier-checkpoint checkpoints/mitbih_baseline_classifier.pt \
  --out-dir hardware_export \
  --prefix netfpga_classifier \
  --layout row-major
```

This writes BF16 memory files such as:

- `hardware_export/netfpga_classifier_weight_bf16.mem`
- `hardware_export/netfpga_classifier_bias_bf16.mem`

Compare the original FP32 classifier against a BF16-emulated classifier before changing training:

```bash
python -m src.emulate_classifier_bf16 \
  --checkpoint checkpoints/mitbih_baseline.pt \
  --dataset data/processed/mitbih_binary.npz \
  --split test \
  --accumulation fp32
```

This reports how much BF16 storage changes the logits and whether the final predictions still agree.

Export CNN output features for FPGA input together with software reference logits and predictions:

```bash
python -m src.export_fpga_reference \
  --checkpoint checkpoints/mitbih_baseline.pt \
  --dataset data/processed/mitbih_binary.npz \
  --split test \
  --max-samples 128 \
  --accumulation fp32 \
  --out-dir hardware_export
```

This writes:

- `<prefix>_features_fp32.txt`: one feature vector per sample, for inspection
- `<prefix>_features_bf16.mem`: flattened BF16 feature stream for FPGA input
- `<prefix>_labels.txt`: ground-truth labels
- `<prefix>_fp32_logits.txt`: software FP32 classifier outputs
- `<prefix>_bf16_logits.txt`: BF16-emulated classifier outputs
- `<prefix>_fp32_pred.txt`: FP32 predicted classes
- `<prefix>_bf16_pred.txt`: BF16-emulated predicted classes
- `<prefix>_reference_bundle.npz`: all exported arrays in one bundle
- `<prefix>_meta.json`: export summary and accuracy/agreement metrics

## Expected dataset format

The baseline code expects a processed `.npz` file with:

- `signals`: shape `(num_samples, signal_length)` or `(num_samples, channels, signal_length)`
- `labels`: shape `(num_samples,)`

If labels are not already binary, the preprocessing step maps the configured `normal_label` to `0` and every other label to `1`.

Example:

```python
import numpy as np

signals = np.random.randn(100, 256).astype("float32")
labels = np.random.randint(0, 2, size=100).astype("int64")
np.savez("data/processed/example_binary_ecg.npz", signals=signals, labels=labels)
```

## Project layout

```text
ECG_classification/
├── checkpoints/
├── data/
│   ├── processed/
│   ├── raw/
│   └── README.md
├── notebooks/
├── results/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── bf16_utils.py
│   ├── evaluate.py
│   ├── emulate_classifier_bf16.py
│   ├── export_classifier_bf16.py
│   ├── export_classifier_hw.py
│   ├── export_parts.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   └── train.py
├── EE533_software_first_implementation_guide.docx
├── README.md
└── requirements.txt
```

## Outputs

- Checkpoints are saved to `checkpoints/<run_name>.pt`
- Training summaries are saved to `results/<run_name>_train.json`
- Evaluation summaries are saved to `results/<checkpoint_stem>_<split>_metrics.json`
- Exported parts are saved to `checkpoints/<run_name>_feature_extractor.pt` and `checkpoints/<run_name>_classifier.pt`

## Next steps

- Plot sample waveforms and class distributions in `notebooks/`
- Replace the classifier call with a hardware wrapper once the software boundary is stable
