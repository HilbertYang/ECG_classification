# Data Notes

Place original ECG files under `data/raw/` and processed training-ready arrays under `data/processed/`.

## MIT-BIH raw files

For MIT-BIH, one record is usually represented by three files with the same stem, for example:

- `100.dat`
- `100.hea`
- `100.atr`

These three files work together:

- `.dat`: the ECG waveform itself, meaning the sampled signal values.
- `.hea`: the header file, meaning metadata that explains how to read the waveform, such as sampling rate, number of channels, and channel names.
- `.atr`: the annotation file, meaning beat locations and beat symbols such as `N`, `A`, and `V`.

So a record like `100` is not a single file. It is the combination of `100.dat`, `100.hea`, and `100.atr`.

In this repo:

- `wfdb.rdrecord(...)` reads the signal using the `.hea` and `.dat` files.
- `wfdb.rdann(...)` reads the beat annotations from the `.atr` file.

## How this repo turns MIT-BIH into `.npz`

Suggested first target:

- `data/processed/mitbih_binary.npz`

You can generate that file with:

```bash
python -m src.prepare_mitbih --download --output data/processed/mitbih_binary.npz
```

That command does the following:

1. Downloads MIT-BIH record files into `data/raw/mitdb/`.
2. Reads each record's ECG waveform from `.dat` + `.hea`.
3. Reads each beat annotation from `.atr`.
4. Uses each annotated beat position as the center of a fixed-length ECG window.
5. Cuts out one ECG segment per beat, so one record becomes many samples.
6. Converts beat symbols into binary labels:
   - symbols listed in `--normal-symbols` become label `0`
   - other included beat symbols become label `1`
7. Saves all extracted samples into a single training-ready `.npz` file.

In other words:

- one `record` contains many `beats`
- each `beat` becomes one `sample`
- each `sample` gets one `label`

## Output `.npz` format

The training code mainly needs:

- `signals`: `(num_samples, signal_length)` or `(num_samples, channels, signal_length)`
- `labels`: `(num_samples,)`

The MIT-BIH preparation script also stores extra metadata for inspection:

- `record_ids`: which record each sample came from
- `beat_symbols`: the original MIT-BIH annotation symbol for each sample
- `beat_samples`: the original beat location in the record
- `channel_names`: which ECG channel was used
- `fs`: sampling rate
- `segment_length`: extracted window length

So the `.npz` file is the bridge between:

- raw MIT-BIH files in `data/raw/mitdb/`
- model training in `src/train.py`

## Notes

- The default preprocessing uses beat-centered segments of length `256`.
- The default binary mapping treats `N` as normal and other included beat symbols as abnormal.
- Keep large datasets out of Git. The repository tracks only placeholder files so the folder structure stays intact.
