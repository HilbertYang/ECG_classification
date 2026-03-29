# Data Notes

Place original ECG files under `data/raw/` and processed training-ready arrays under `data/processed/`.

For the current baseline code, create a single `.npz` file with:

- `signals`: `(num_samples, signal_length)` or `(num_samples, channels, signal_length)`
- `labels`: `(num_samples,)`

Suggested first target:

- `data/processed/mitbih_binary.npz`

You can generate that file from MIT-BIH WFDB records with:

```bash
python -m src.prepare_mitbih --download --output data/processed/mitbih_binary.npz
```

The script downloads records into `data/raw/mitdb/` and extracts beat-centered fixed-length segments from the requested annotation symbols.

Keep large datasets out of Git. The repository tracks only placeholder files so the folder structure stays intact.
