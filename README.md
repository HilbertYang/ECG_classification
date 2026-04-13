# ECG Classification

Software-first starter repository for binary ECG classification. This repo turns the proposal guide into a runnable baseline: load data, preprocess signals, train a small 1D CNN, evaluate results, and keep the final classifier boundary explicit for later hardware mapping.

## Code paths

This repository now keeps two usable software paths side by side:

- `src_original/`: the original baseline implementation, including the earlier multi-layer CNN design and the original command paths from the first version of the project
- `src/`: the current lightweight implementation, simplified to better match a hardware-friendly deployment path

If you are following older notes, older screenshots, or earlier commands from this project, replace `src/...` with `src_original/...` conceptually and use `python3 -m src_original.<module>` at the command line. New work should usually use `src/`.

## What is in the repo

- `src_original/`: original baseline pipeline and original model implementation
- `src/`: current lightweight pipeline and current model implementation
- `data/README.md`: expected dataset format for the first software prototype
- `EE533_software_first_implementation_guide.docx`: original planning document

## Recommended first milestone

Build a binary classifier for `normal` vs `abnormal` ECG segments on a regular computer before moving any logic to custom hardware. The model is intentionally small so we can validate the pipeline first and cleanly expose the boundary between:

1. `feature_extractor`
2. `classifier`

That interface is the first candidate for later GPU mapping.

## Model variants

Both versions default to ECG segments of length `256`, with single-channel inputs shaped as `(batch, 1, 256)`.

### Current lightweight model in `src/model.py`

The current implementation keeps the same high-level split between `feature_extractor` and `classifier`, but simplifies the network so it is easier to map to hardware.

```text
Input: (batch, 1, 256)

Feature extractor:
- Conv1d(1 -> 1, kernel_size=7, padding=3)
- BatchNorm1d(1)
- ReLU
- MaxPool1d(2)
- Flatten

Classifier:
- Linear(128 -> 2)

Output:
- logits with shape (batch, 2)
```

With the default input length of `256`, the tensor shape changes like this:

```text
(batch, 1, 256)
-> (batch, 1, 256)
-> (batch, 1, 128)
-> (batch, 128)
-> (batch, 2)
```

Intuition:

- `Conv1d` learns a single filtered ECG channel.
- `MaxPool1d(2)` halves the signal length while keeping the strongest local response in each window.
- `Flatten` turns the pooled waveform into a `128`-dimensional feature vector for the classifier.

### Original baseline model in `src_original/model.py`

The original baseline is still preserved in `src_original/`. That version uses a deeper 3-layer CNN:

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
- Linear(64 -> 2)
```

Use `src_original/` if you want the earlier software baseline behavior. Use `src/` if you want the current hardware-oriented lightweight path.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a smoke test with the current lightweight implementation:

```bash
python3 -m src.train --demo --run-name demo_lightweight --epochs 5
python3 -m src.evaluate --checkpoint checkpoints/demo_lightweight.pt --demo
python3 -m src.inference --checkpoint checkpoints/demo_lightweight.pt --demo --index 0
```

Run the same smoke test with the original baseline implementation:

```bash
python3 -m src_original.train --demo --run-name demo_original --epochs 5
python3 -m src_original.evaluate --checkpoint checkpoints/demo_original.pt --demo
python3 -m src_original.inference --checkpoint checkpoints/demo_original.pt --demo --index 0
```

Prepare a real MIT-BIH beat dataset in `.npz` format:

```bash
python3 -m src.prepare_mitbih --download --output data/processed/mitbih_binary.npz
```

This script downloads MIT-BIH Arrhythmia Database records into `data/raw/mitdb/`, extracts fixed-length beat-centered ECG segments, and writes a training-ready dataset with:

- `signals`: `(num_samples, signal_length)`
- `labels`: `(num_samples,)`, where `0` means a symbol listed in `--normal-symbols` and `1` means any other included beat symbol

Useful variants:

```bash
python3 -m src.prepare_mitbih --download --records 100 101 102
python3 -m src.prepare_mitbih --stream --records 100 --output data/processed/mitbih_100_binary.npz
python3 -m src.prepare_mitbih --download --normal-symbols N L R e j
```

Run the current lightweight model with a real processed dataset:

```bash
python3 -m src.train --dataset data/processed/mitbih_binary.npz --run-name mitbih_lightweight
python3 -m src.evaluate --checkpoint checkpoints/mitbih_lightweight.pt --dataset data/processed/mitbih_binary.npz
python3 -m src.inference --checkpoint checkpoints/mitbih_lightweight.pt --dataset data/processed/mitbih_binary.npz --split test --index 0
```

Run the original baseline model with the same dataset:

```bash
python3 -m src_original.train --dataset data/processed/mitbih_binary.npz --run-name mitbih_original
python3 -m src_original.evaluate --checkpoint checkpoints/mitbih_original.pt --dataset data/processed/mitbih_binary.npz
python3 -m src_original.inference --checkpoint checkpoints/mitbih_original.pt --dataset data/processed/mitbih_binary.npz --split test --index 0
```

Export the trained model as two separate parts for hardware mapping:

```bash
python3 -m src.export_parts --checkpoint checkpoints/mitbih_lightweight.pt
```

This saves `checkpoints/mitbih_lightweight_feature_extractor.pt` (the lightweight CNN feature extractor) and `checkpoints/mitbih_lightweight_classifier.pt` (the linear head) as independent state dicts.

Export the classifier into FPGA-friendly files:

```bash
python3 -m src.export_classifier_hw \
  --classifier-checkpoint checkpoints/mitbih_lightweight_classifier.pt \
  --out-dir hardware_export/param_fp32 \
  --prefix netfpga_classifier \
  --layout row-major \
  --word-bits 16 \
  --frac-bits 8
```

This writes:

- `hardware_export/param_fp32/netfpga_classifier_meta.json`
- `hardware_export/param_fp32/netfpga_classifier_weight_matrix.txt`
- `hardware_export/param_fp32/netfpga_classifier_bias_float.txt`
- `hardware_export/param_fp32/netfpga_classifier_weight_q16_8.mem`
- `hardware_export/param_fp32/netfpga_classifier_bias_q16_8.mem`

For the current model, the hardware classifier contract is:

- input feature vector: length `128`
- weight matrix: shape `(2, 128)`
- bias vector: shape `(2,)`
- math: `logits = W * features + b`

Export the classifier as BF16 files for a BF16-capable hardware path:

```bash
python3 -m src.export_classifier_bf16 \
  --classifier-checkpoint checkpoints/mitbih_lightweight_classifier.pt \
  --out-dir hardware_export/param_bf16 \
  --prefix netfpga_classifier \
  --layout row-major
```

This writes BF16 memory files such as:

- `hardware_export/param_bf16/netfpga_classifier_weight_bf16.mem`
- `hardware_export/param_bf16/netfpga_classifier_bias_bf16.mem`

Compare the original FP32 classifier against a BF16-emulated classifier before changing training:

```bash
python3 -m src.emulate_classifier_bf16 \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --dataset data/processed/mitbih_binary.npz \
  --split test \
  --accumulation fp32
```

This reports how much BF16 storage changes the logits and whether the final predictions still agree.

Export CNN output features for FPGA input together with software reference logits and predictions:

```bash
python3 -m src.export_fpga_reference \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --dataset data/processed/mitbih_binary.npz \
  --split test \
  --max-samples 128 \
  --accumulation fp32 \
  --out-dir hardware_export/extracted_feature
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

For a tiny hardware-debug subset, you can export a specific MIT-BIH record directly from dataset order:

```bash
python3 -m src.export_fpga_reference \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --dataset data/processed/mitbih_binary.npz \
  --record-id 100 \
  --first-n-beats 16 \
  --accumulation fp32 \
  --out-dir hardware_export/extracted_feature \
  --prefix mini_mitbih_record_100_16beats
```

This also writes:

- `<prefix>_record_ids.txt`
- `<prefix>_beat_symbols.txt`
- `<prefix>_beat_samples.txt`

## Current `hardware_export/` layout

The current repository organizes hardware-facing artifacts into three subdirectories:

```text
hardware_export/
├── extracted_feature/
│   ├── mitbih_baseline_test_bf16_logits.txt
│   ├── mitbih_baseline_test_bf16_pred.txt
│   ├── mitbih_baseline_test_features_bf16.mem
│   ├── mitbih_baseline_test_features_fp32.txt
│   ├── mitbih_baseline_test_fp32_logits.txt
│   ├── mitbih_baseline_test_fp32_pred.txt
│   ├── mitbih_baseline_test_labels.txt
│   ├── mitbih_baseline_test_meta.json
│   └── mitbih_baseline_test_reference_bundle.npz
├── param_bf16/
│   ├── netfpga_classifier_bf16_meta.json
│   ├── netfpga_classifier_bias_bf16.mem
│   ├── netfpga_classifier_bias_bf16_float.txt
│   ├── netfpga_classifier_weight_bf16.mem
│   └── netfpga_classifier_weight_bf16_float.txt
└── param_fp32/
    ├── netfpga_classifier_bias_float.txt
    ├── netfpga_classifier_bias_q16_8.mem
    ├── netfpga_classifier_meta.json
    ├── netfpga_classifier_weight_matrix.txt
    └── netfpga_classifier_weight_q16_8.mem
```

What each folder is for:

- `hardware_export/param_fp32/`: classifier parameters exported for a fixed-point FPGA path
- `hardware_export/param_bf16/`: classifier parameters exported for a BF16-capable FPGA or accelerator path
- `hardware_export/extracted_feature/`: CNN output features plus software golden results for hardware comparison

What the files are for:

- `*_weight_matrix.txt`: human-readable floating-point classifier weights, useful for inspection and debugging
- `*_bias_float.txt`: human-readable floating-point bias values
- `*_weight_q16_8.mem`: fixed-point weight memory image for FPGA initialization
- `*_bias_q16_8.mem`: fixed-point bias memory image for FPGA initialization
- `*_weight_bf16.mem`: BF16 weight memory image for BF16 hardware input
- `*_bias_bf16.mem`: BF16 bias memory image for BF16 hardware input
- `*_weight_bf16_float.txt`: rounded BF16 weights converted back to float text for checking quantization effects
- `*_bias_bf16_float.txt`: rounded BF16 bias converted back to float text for checking quantization effects
- `*_meta.json`: metadata about tensor shapes, quantization format, source checkpoint, and export settings
- `*_features_fp32.txt`: one CNN feature vector per sample in readable float format
- `*_features_bf16.mem`: flattened BF16 feature stream to feed into FPGA logic
- `*_labels.txt`: ground-truth labels for each exported sample
- `*_fp32_logits.txt`: software FP32 classifier outputs, used as a golden reference
- `*_bf16_logits.txt`: BF16-emulated classifier outputs, used when hardware is expected to behave like BF16
- `*_fp32_pred.txt`: final FP32 predicted class per sample
- `*_bf16_pred.txt`: final BF16-emulated predicted class per sample
- `*_reference_bundle.npz`: all exported arrays bundled together for quick loading in Python

Recommended comparison flow:

1. Use `param_fp32/` or `param_bf16/` to initialize the hardware classifier weights and bias.
2. Feed the FPGA with the feature vectors from `extracted_feature/*_features_bf16.mem`.
3. Compare FPGA logits or final classes against `*_fp32_logits.txt` / `*_fp32_pred.txt` or against the BF16 versions if your hardware math is BF16-like.
4. Use `*_meta.json` and `*_reference_bundle.npz` when you need shapes, counts, or a compact software-side debug bundle.

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
├── hardware_export/
│   ├── extracted_feature/
│   ├── param_bf16/
│   └── param_fp32/
├── logs/
├── notebooks/
├── results/
├── slurm/
│   ├── current/
│   │   ├── 01_setup_env.slurm
│   │   ├── 02_demo_train.slurm
│   │   ├── 03_gpu_train.slurm
│   │   ├── 04_export_parts.slurm
│   │   ├── 05_export_classifier_hw.slurm
│   │   ├── 06_export_classifier_bf16.slurm
│   │   ├── 07_emulate_classifier_bf16.slurm
│   │   ├── 08_export_fpga_reference.slurm
│   │   └── 09_export_fpga_reference_mini.slurm
│   └── original/
│       ├── 01_setup_env.slurm
│       ├── 02_demo_train.slurm
│       ├── 03_gpu_train.slurm
│       ├── 04_export_parts.slurm
│       ├── 05_export_classifier_hw.slurm
│       ├── 06_export_classifier_bf16.slurm
│       ├── 07_emulate_classifier_bf16.slurm
│       ├── 08_export_fpga_reference.slurm
│       └── 09_export_fpga_reference_mini.slurm
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── bf16_utils.py
│   ├── evaluate.py
│   ├── emulate_classifier_bf16.py
│   ├── export_classifier_bf16.py
│   ├── export_classifier_hw.py
│   ├── export_fpga_reference.py
│   ├── export_parts.py
│   ├── inference.py
│   ├── model.py
│   ├── prepare_mitbih.py
│   ├── preprocess.py
│   └── train.py
├── src_original/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── bf16_utils.py
│   ├── evaluate.py
│   ├── emulate_classifier_bf16.py
│   ├── export_classifier_bf16.py
│   ├── export_classifier_hw.py
│   ├── export_fpga_reference.py
│   ├── export_parts.py
│   ├── inference.py
│   ├── model.py
│   ├── prepare_mitbih.py
│   ├── preprocess.py
│   └── train.py
├── CLAUDE.md
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
