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

## Current workflow

If you are using the current lightweight BF16-oriented path, the recommended order is:

1. Set up the environment:

```bash
sbatch slurm/current/01_setup_env.slurm
```

2. Run a quick smoke test if needed:

```bash
sbatch slurm/current/02_demo_train.slurm
```

3. Train the real model:

```bash
sbatch slurm/current/03_gpu_train.slurm
```

4. Optional software-side BF16 verification:

```bash
sbatch slurm/current/04_emulate_classifier_bf16.slurm
```

5. Export software reference data for hardware comparison:

```bash
sbatch slurm/current/05_export_fpga_reference.slurm
```

6. Export a tiny debug subset from record 200:

```bash
sbatch slurm/current/06_export_fpga_reference_mini_record200_4.slurm
```

7. Export fused frontend parameters as BF16:

```bash
sbatch slurm/current/07_export_fused_frontend_bf16.slurm
```

8. Export the corresponding raw BF16 inputs for record 200 first 4 beats:

```bash
sbatch slurm/current/08_export_input_bf16_record200_4.slurm
```

For the current hardware debug flow, the most important outputs are:

- `hardware_export/fused_frontend_bf16/`: BF16 fused CNN and classifier parameters
- `hardware_export/input_bf16/`: BF16 raw input windows, including record 200 first 4 beats
- `hardware_export/mini_feature/`: software golden outputs for the same tiny debug subset

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

For the current model, the hardware classifier contract is:

- input feature vector: length `128`
- weight matrix: shape `(2, 128)`
- bias vector: shape `(2,)`
- math: `logits = W * features + b`

Export fused frontend parameters as BF16 files:

```bash
python3 -m src.export_fused_frontend_bf16 \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --out-dir hardware_export/fused_frontend_bf16 \
  --prefix ecg_fused_frontend
```

This writes BF16 memory files such as:

- `hardware_export/fused_frontend_bf16/ecg_fused_frontend_conv_weight_bf16.mem`
- `hardware_export/fused_frontend_bf16/ecg_fused_frontend_conv_bias_bf16.mem`
- `hardware_export/fused_frontend_bf16/ecg_fused_frontend_classifier_weight_bf16.mem`
- `hardware_export/fused_frontend_bf16/ecg_fused_frontend_classifier_bias_bf16.mem`

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

For a tiny hardware-debug subset, you can export record 200 first 4 beats directly from dataset order:

```bash
python3 -m src.export_fpga_reference \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --dataset data/processed/mitbih_binary.npz \
  --record-id 200 \
  --first-n-beats 4 \
  --accumulation fp32 \
  --out-dir hardware_export/mini_feature \
  --prefix mini_mitbih_record_200_4beats
```

This also writes:

- `<prefix>_record_ids.txt`
- `<prefix>_beat_symbols.txt`
- `<prefix>_beat_samples.txt`

Export raw model inputs as BF16 for the same tiny hardware-debug subset:

```bash
python3 -m src.export_input_bf16 \
  --checkpoint checkpoints/mitbih_lightweight.pt \
  --dataset data/processed/mitbih_binary.npz \
  --record-id 200 \
  --first-n-beats 4 \
  --out-dir hardware_export/input_bf16 \
  --prefix mini_mitbih_record_200_4beats_input
```

This writes BF16 model inputs with shape `(4, 1, 256)` together with labels and beat metadata.

## Current `hardware_export/` layout

The current repository organizes hardware-facing artifacts into three subdirectories:

```text
hardware_export/
├── fused_frontend_bf16/
│   ├── ecg_fused_frontend_bf16_meta.json
│   ├── ecg_fused_frontend_conv_weight_bf16.mem
│   ├── ecg_fused_frontend_conv_bias_bf16.mem
│   ├── ecg_fused_frontend_classifier_weight_bf16.mem
│   └── ecg_fused_frontend_classifier_bias_bf16.mem
├── input_bf16/
│   ├── mini_mitbih_record_200_4beats_input_bf16.mem
│   ├── mini_mitbih_record_200_4beats_input_fp32.txt
│   ├── mini_mitbih_record_200_4beats_input_labels.txt
│   └── mini_mitbih_record_200_4beats_input_meta.json
└── mini_feature/
    ├── mini_mitbih_record_200_4beats_features_bf16.mem
    ├── mini_mitbih_record_200_4beats_fp32_logits.txt
    ├── mini_mitbih_record_200_4beats_fp32_pred.txt
    ├── mini_mitbih_record_200_4beats_labels.txt
    └── mini_mitbih_record_200_4beats_meta.json
```

What each folder is for:

- `hardware_export/fused_frontend_bf16/`: BF16 fused CNN parameters plus BF16 classifier parameters
- `hardware_export/input_bf16/`: BF16 raw model inputs for hardware input testing
- `hardware_export/mini_feature/`: software golden outputs for the record 200 first-4-beats debug subset

What the files are for:

- `*_conv_weight_bf16.mem`: BF16 fused convolution weights
- `*_conv_bias_bf16.mem`: BF16 fused convolution bias
- `*_classifier_weight_bf16.mem`: BF16 classifier weights
- `*_classifier_bias_bf16.mem`: BF16 classifier bias
- `*_input_bf16.mem`: BF16 raw model inputs
- `*_features_bf16.mem`: flattened BF16 feature vectors after the CNN frontend
- `*_labels.txt`: ground-truth labels for each exported sample
- `*_fp32_logits.txt`: software FP32 classifier outputs, used as a golden reference
- `*_fp32_pred.txt`: final FP32 predicted class per sample
- `*_meta.json`: metadata about tensor shapes, source checkpoint, and export settings
- `*_reference_bundle.npz`: bundled software-side arrays for quick debugging in Python

Recommended comparison flow:

1. Use `fused_frontend_bf16/` to initialize the fused CNN weights and classifier weights.
2. Feed the FPGA with raw BF16 inputs from `input_bf16/mini_mitbih_record_200_4beats_input_bf16.mem`.
3. Compare the hardware frontend output or final classes against the software golden files in `mini_feature/`.
4. Use `*_meta.json` when you need exact tensor shapes, kernel sizes, or sample counts.

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
│   │   ├── 04_emulate_classifier_bf16.slurm
│   │   ├── 05_export_fpga_reference.slurm
│   │   ├── 06_export_fpga_reference_mini_record200_4.slurm
│   │   ├── 07_export_fused_frontend_bf16.slurm
│   │   └── 08_export_input_bf16_record200_4.slurm
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
