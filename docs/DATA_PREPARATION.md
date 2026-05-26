# Data preparation

Step-by-step setup after downloading datasets. Official download links are in the [README](../README.md#data).

PhysioNet corpora require [credentialed access](https://physionet.org/settings/credentialing/).

## Quick reference

| Key | Config file | Meaning |
|-----|-------------|---------|
| `data.meta_root` | `configs/train/lvcg_v5_gru.yaml` | MIMIC JSONL manifest path |
| `data.raw_root` | `configs/eval/probing.yaml` | Root of downloaded PTB / ICBEB / CSN extracts |
| `data.splits_root` | `configs/eval/probing.yaml` | Train/val/test CSVs (default: `probing/data_splits`) |

```text
raw_root/                         # probing only (user machine)
  ptbxl/
  icbeb/TrainingSet{1,2,3}/
  csn/

probing/data_splits/              # shipped in repo
  ptbxl/{super_class,sub_class,form,rhythm}/
  icbeb/
  chapman/
```

**No extra `.npy` conversion** — training and probing read WFDB / MAT on the fly. You only need a manifest (pretraining) or the folder layout above (probing).

## What you do not need

| Stage | You need | You do *not* need |
|-------|----------|-------------------|
| Pretraining | JSONL manifest with WFDB paths | Converting MIMIC to `.npy` / LMDB |
| PTB / ICBEB / Chapman | Correct `raw_root/` layout | Regenerating split CSVs (in `probing/data_splits/`) |
| AI-READI | `build_labels.py` + `make_split.py` | MELP or any external probing toolkit |

---

## Pretraining: MIMIC-IV ECG

1. Download and unzip [MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) so WFDB records sit under a tree like `.../files/p####/p########/s########/########` (each record: `.hea` + `.dat`).

2. Build a JSONL manifest (one JSON object per line):

```bash
python scripts/build_mimic_manifest.py \
  --mimic-root /path/to/mimic-iv-ecg/files \
  --out /path/to/mimic_manifest.jsonl
```

Each line looks like:

```json
{"id": "mimic_0", "ecg_path": "/abs/path/to/wfdb/record/stem", "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
```

`ecg_path` is the WFDB stem **without** `.hea` / `.dat`.

3. Set `data.meta_root` in `configs/train/lvcg_v5_gru.yaml`, or pass on the CLI:

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml \
  --data.meta_root /path/to/mimic_manifest.jsonl
```

Training resamples to 100 Hz and normalizes per batch; no offline preprocessing step.

---

## Probing: PTB-XL, ICBEB, Chapman

### 1. Choose `raw_root`

Example: `/data/ecg/raw/`

### 2. Download and extract

| Corpus | Link | Place under |
|--------|------|-------------|
| PTB-XL | [ptb-xl 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | `raw_root/ptbxl/` (must contain `records500/`) |
| ICBEB | [CPSC 2018 training](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/) | `raw_root/icbeb/TrainingSet{1,2,3}/*.mat` |
| Chapman (CSN) | [ecg-arrhythmia 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/) | `raw_root/csn/` |

Example layout:

```text
/data/ecg/raw/
  ptbxl/
  icbeb/
    TrainingSet1/
    TrainingSet2/
    TrainingSet3/
  csn/
```

### 3. Splits (already in repo)

Train/val/test CSVs live under `probing/data_splits/` — no label prep for these six tasks.

| Config name | Raw subdir | Split CSV dir |
|-------------|------------|---------------|
| `ptbxl_super_class` | `ptbxl/` | `ptbxl/super_class/` |
| `ptbxl_sub_class` | `ptbxl/` | `ptbxl/sub_class/` |
| `ptbxl_form` | `ptbxl/` | `ptbxl/form/` |
| `ptbxl_rhythm` | `ptbxl/` | `ptbxl/rhythm/` |
| `icbeb` | `icbeb/` | `icbeb/` |
| `chapman` | `csn/` | `chapman/` |

Chapman uses four classes by default: `SR`, `AFIB`, `1AVB`, `RBBB` (`chapman_classes` in config).

Split CSVs follow the [MELP](https://github.com/HKU-MedAI/MELP) benchmark partitions for reproducibility; only the on-disk layout is LVCG-native.

### 4. Configure probing

Edit `configs/eval/probing.yaml`:

```yaml
data:
  raw_root: /data/ecg/raw
  splits_root: probing/data_splits
  norm_method: zscore
```

### 5. Verify paths

```bash
python scripts/verify_data_layout.py --raw-root /data/ecg/raw
```

### 6. Run probing

Requires a trained checkpoint (`final.pt`). See [README](../README.md#evaluation).

---

## Probing: AI-READI

1. Request access and download from [aireadi.org](https://aireadi.org/dataset) / [FAIRhub](https://fairhub.io/datasets/1).

2. Build labels and splits (once per release):

```bash
python probing/datasets/aireadi/build_labels.py \
  --condition-csv /path/to/aireadi/clinical_data/condition_occurrence.csv

python probing/datasets/aireadi/make_split.py \
  --aireadi-root /path/to/aireadi
```

3. Set paths under `aireadi_condition_cardio` in `configs/eval/probing.yaml` (`ecg_root`, `manifest_tsv`, `labels_parquet`, `split_csv`).
