from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb


DEFAULT_BEAT_SYMBOLS = [
    "N",
    "L",
    "R",
    "B",
    "A",
    "a",
    "J",
    "S",
    "V",
    "r",
    "F",
    "e",
    "j",
    "n",
    "E",
    "/",
    "f",
    "Q",
    "?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MIT-BIH WFDB records into a binary ECG segment dataset."
    )
    parser.add_argument(
        "--database",
        type=str,
        default="mitdb",
        help="PhysioNet database name used for record listing, downloads, or streaming.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/mitdb"),
        help="Directory containing local WFDB files, or where downloads will be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mitbih_binary.npz"),
        help="Output .npz dataset path.",
    )
    parser.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=None,
        help="Optional record names to include. Defaults to local records or the full database list.",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="atr",
        help="WFDB annotation extension to read.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the requested records and annotations into raw-dir before processing.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Read records directly from PhysioNet instead of local WFDB files.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Signal channel index to extract from each MIT-BIH record.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=256,
        help="Number of samples per extracted ECG segment.",
    )
    parser.add_argument(
        "--normal-symbols",
        type=str,
        nargs="+",
        default=["N"],
        help="Annotation symbols that should become class 0. All other included beat symbols become class 1.",
    )
    parser.add_argument(
        "--include-symbols",
        type=str,
        nargs="+",
        default=DEFAULT_BEAT_SYMBOLS,
        help="Beat symbols to keep when building the dataset.",
    )
    return parser.parse_args()


def resolve_records(raw_dir: Path, database: str, requested_records: list[str] | None) -> list[str]:
    if requested_records:
        return requested_records

    local_records = sorted(path.stem for path in raw_dir.glob("*.hea"))
    if local_records:
        return local_records

    return list(wfdb.get_record_list(database))


def maybe_download_records(
    database: str,
    raw_dir: Path,
    records: list[str],
    annotation: str,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        database,
        str(raw_dir),
        records=records,
        annotators=[annotation],
        keep_subdirs=False,
        overwrite=False,
    )


def build_record_source(record_name: str, raw_dir: Path, stream: bool) -> str:
    if stream:
        return record_name

    record_path = raw_dir / record_name
    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(
            f"Local WFDB record not found: {record_path.with_suffix('.hea')}. "
            "Use --download to fetch the records or --stream to read directly from PhysioNet."
        )
    return str(record_path)


def extract_centered_window(signal: np.ndarray, center: int, length: int) -> np.ndarray:
    start = center - (length // 2)
    end = start + length
    window = np.zeros((length,), dtype=np.float32)

    src_start = max(start, 0)
    src_end = min(end, signal.shape[0])
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    window[dst_start:dst_end] = signal[src_start:src_end].astype(np.float32, copy=False)
    return window


def collect_segments(
    *,
    database: str,
    raw_dir: Path,
    record_name: str,
    annotation: str,
    channel: int,
    segment_length: int,
    include_symbols: set[str],
    normal_symbols: set[str],
    stream: bool,
) -> tuple[list[np.ndarray], list[int], list[str], list[str], list[int], float]:
    record_source = build_record_source(record_name, raw_dir, stream)
    pn_dir = database if stream else None

    record = wfdb.rdrecord(record_source, channels=[channel], pn_dir=pn_dir)
    annotation_record = wfdb.rdann(record_source, annotation, pn_dir=pn_dir)

    signal = record.p_signal[:, 0]
    channel_name = record.sig_name[0]
    fs = float(record.fs)

    segments: list[np.ndarray] = []
    labels: list[int] = []
    symbols: list[str] = []
    channel_names: list[str] = []
    samples: list[int] = []

    for sample, symbol in zip(annotation_record.sample, annotation_record.symbol):
        if symbol not in include_symbols:
            continue

        segments.append(extract_centered_window(signal, int(sample), segment_length))
        labels.append(0 if symbol in normal_symbols else 1)
        symbols.append(symbol)
        channel_names.append(channel_name)
        samples.append(int(sample))

    return segments, labels, symbols, channel_names, samples, fs


def main() -> None:
    args = parse_args()
    include_symbols = set(args.include_symbols)
    normal_symbols = set(args.normal_symbols)

    if not normal_symbols.issubset(include_symbols):
        missing = sorted(normal_symbols.difference(include_symbols))
        raise ValueError(
            "All normal symbols must also be included. Missing from --include-symbols: "
            f"{missing}"
        )

    records = resolve_records(args.raw_dir, args.database, args.records)
    if args.download:
        maybe_download_records(args.database, args.raw_dir, records, args.annotation)
    elif not args.stream and not args.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {args.raw_dir}. "
            "Use --download to fetch MIT-BIH data first, or --stream to read directly from PhysioNet."
        )

    all_segments: list[np.ndarray] = []
    all_labels: list[int] = []
    all_record_ids: list[str] = []
    all_symbols: list[str] = []
    all_channel_names: list[str] = []
    all_samples: list[int] = []
    fs_values: set[float] = set()
    symbol_counter: Counter[str] = Counter()

    for record_name in records:
        segments, labels, symbols, channel_names, samples, fs = collect_segments(
            database=args.database,
            raw_dir=args.raw_dir,
            record_name=record_name,
            annotation=args.annotation,
            channel=args.channel,
            segment_length=args.segment_length,
            include_symbols=include_symbols,
            normal_symbols=normal_symbols,
            stream=args.stream,
        )

        if not segments:
            print(f"Skipping record {record_name}: no matching beat symbols were found.")
            continue

        all_segments.extend(segments)
        all_labels.extend(labels)
        all_record_ids.extend([record_name] * len(segments))
        all_symbols.extend(symbols)
        all_channel_names.extend(channel_names)
        all_samples.extend(samples)
        fs_values.add(fs)
        symbol_counter.update(symbols)

        label_counter = Counter(labels)
        print(
            f"Processed record {record_name}: "
            f"{len(segments)} segments, normal={label_counter.get(0, 0)}, abnormal={label_counter.get(1, 0)}"
        )

    if not all_segments:
        raise RuntimeError("No ECG segments were collected. Check the record list and beat symbol filters.")

    if len(fs_values) != 1:
        raise ValueError(f"Expected a single sampling rate across records, but found {sorted(fs_values)}")

    signals = np.stack(all_segments).astype(np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    record_ids = np.asarray(all_record_ids)
    beat_symbols = np.asarray(all_symbols)
    channel_names = np.asarray(all_channel_names)
    beat_samples = np.asarray(all_samples, dtype=np.int64)
    fs = np.asarray(sorted(fs_values), dtype=np.float32)[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        signals=signals,
        labels=labels,
        record_ids=record_ids,
        beat_symbols=beat_symbols,
        beat_samples=beat_samples,
        channel_names=channel_names,
        fs=fs,
        segment_length=np.int64(args.segment_length),
        source_database=np.asarray(args.database),
        annotation_extension=np.asarray(args.annotation),
        normal_symbols=np.asarray(sorted(normal_symbols)),
        included_symbols=np.asarray(sorted(include_symbols)),
    )

    label_counter = Counter(labels.tolist())
    print(f"Saved processed dataset to {args.output}")
    print(f"Signals shape: {signals.shape}")
    print(f"Label counts: normal={label_counter.get(0, 0)}, abnormal={label_counter.get(1, 0)}")
    print(f"Sampling rate: {float(fs):.1f} Hz")
    print(f"Included beat symbols: {dict(symbol_counter)}")


if __name__ == "__main__":
    main()
