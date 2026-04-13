"""
Split a trained ECGNet checkpoint into two separate parts:
  - feature_extractor: the 1D CNN (ECGFeatureExtractor)
  - classifier:        the linear head (nn.Linear)

Usage:
    python -m src.export_parts --checkpoint checkpoints/mitbih_baseline.pt
"""

import argparse
from pathlib import Path

import torch

from src.model import build_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ECGNet parts separately")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--out-dir",
        default="checkpoints",
        help="Directory to save the two part files (default: checkpoints/)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    stem = ckpt_path.stem  # e.g. "mitbih_baseline"

    # --- Load full model ---
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.eval()

    # --- Extract parts ---
    feature_extractor = model.feature_extractor
    classifier = model.classifier

    print("\nFeature extractor:")
    print(feature_extractor)
    print(f"\nClassifier:")
    print(classifier)

    # --- Save ---
    fe_path = out_dir / f"{stem}_feature_extractor.pt"
    cl_path = out_dir / f"{stem}_classifier.pt"

    torch.save(feature_extractor.state_dict(), fe_path)
    torch.save(classifier.state_dict(), cl_path)

    print(f"\nSaved:")
    print(f"  Feature extractor → {fe_path}")
    print(f"  Classifier        → {cl_path}")

    # --- Quick sanity check: forward pass through both parts ---
    print("\nRunning sanity check ...")
    signal_length = ckpt.get("signal_length", 256)
    x = torch.randn(1, 1, signal_length)
    with torch.no_grad():
        features = feature_extractor(x)   # (1, 64)
        logits = classifier(features)      # (1, 2)
    pred = logits.argmax(dim=1).item()
    label = "normal" if pred == 0 else "abnormal"
    print(f"  Input shape   : {list(x.shape)}")
    print(f"  Feature shape : {list(features.shape)}")
    print(f"  Logits        : {logits.squeeze().tolist()}")
    print(f"  Prediction    : {pred} ({label})")
    print("\nDone.")


if __name__ == "__main__":
    main()
