# AI for Medic ECG

This project processes 12-lead ECG records in WFDB `.hea + .mat` format and prepares a multi-label ECG diagnosis dataset for deep learning experiments.

The project includes scripts for:

- Inspecting a single ECG record
- Batch-reading WFDB ECG records
- Generating metadata and label statistics
- Preparing a multi-label dataset
- Splitting train / validation / test sets
- Training a baseline 1D CNN model

> Raw ECG data and generated `.npy` files are not included in this repository.

---

## Dataset Format

Each ECG record consists of a pair of files:

```text
JSxxxxx.hea
JSxxxxx.mat
```

Each ECG sample has:

```text
12 leads
500 Hz sampling rate
5000 time points
10 seconds duration
multi-label diagnosis codes from Dx
```

The 12 ECG leads are:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Each ECG signal is loaded as:

```text
shape = (5000, 12)
```

where:

```text
5000 = time samples
12   = ECG leads
```

---

## Project Structure

```text
AI for Medic ECG/
├── data/                              # ignored by Git
│   ├── JS00001.hea
│   ├── JS00001.mat
│   └── ...
├── output/                            # ignored by Git
├── output_multilabel/                 # ignored by Git
├── output_split/                      # ignored by Git
├── model_output/                      # ignored by Git
├── inspect_one_record.py
├── prepare_dataset.py
├── prepare_multilabel_dataset.py
├── check_multilabel_dataset.py
├── split_dataset.py
├── train_1d_cnn.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
matplotlib
wfdb
scikit-learn
tensorflow
```

---

## Pipeline

### 1. Inspect One ECG Record

This script reads one `.hea + .mat` ECG record and prints basic information such as sampling rate, signal shape, lead names, diagnosis codes, and data statistics.

Run:

```bash
python inspect_one_record.py
```

Example output:

```text
Sampling frequency fs: 500
Number of signals: 12
Signal length: 5000
Signal shape: (5000, 12)
Lead names: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
```

---

### 2. Prepare Full Dataset

This script batch-reads all WFDB records from the `data/` folder and generates:

```text
output/X.npy
output/metadata.csv
output/dx_counts.csv
```

Run:

```bash
python prepare_dataset.py
```

Expected full dataset shape:

```text
X shape: (10247, 5000, 12)
```

---

### 3. Prepare Multi-label Dataset

This script converts raw `Dx` diagnosis codes into a multi-hot label matrix.

By default, it keeps diagnosis codes with frequency greater than or equal to `200`.

Run:

```bash
python prepare_multilabel_dataset.py
```

Generated files:

```text
output_multilabel/X_filtered.npy
output_multilabel/Y_filtered.npy
output_multilabel/metadata_filtered.csv
output_multilabel/label_map.csv
output_multilabel/selected_dx_counts.csv
output_multilabel/dataset_summary.txt
output_multilabel/prepare_config.json
```

Final multi-label dataset:

```text
X_filtered: (10199, 5000, 12)
Y_filtered: (10199, 20)
```

This means:

```text
10199 ECG samples
5000 time points
12 ECG leads
20 selected diagnosis labels
```

---

### 4. Check Multi-label Dataset

This script verifies whether `X`, `Y`, `metadata`, and `label_map` are consistent.

Run:

```bash
python check_multilabel_dataset.py
```

Checks include:

```text
X shape
Y shape
Y unique values
per-label sample counts
per-sample label count distribution
metadata consistency
```

Expected result:

```text
Y unique values: [0. 1.]
Each sample has at least one label: True
Each label has at least one sample: True
```

---

### 5. Split Train / Validation / Test Sets

This script splits the multi-label dataset into train, validation, and test sets.

Default split ratio:

```text
train: 70%
val:   15%
test:  15%
```

Run:

```bash
python split_dataset.py
```

Generated files:

```text
output_split/X_train.npy
output_split/Y_train.npy
output_split/X_val.npy
output_split/Y_val.npy
output_split/X_test.npy
output_split/Y_test.npy
output_split/metadata_train.csv
output_split/metadata_val.csv
output_split/metadata_test.csv
output_split/label_map.csv
output_split/split_indices.npz
output_split/split_config.json
output_split/split_summary.txt
```

Final split:

