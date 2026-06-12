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

## Dataset Download

The ECG records used in this project can be downloaded from PhysioNet:

```text
https://physionet.org/content/ecg-arrhythmia/1.0.0/
```

Dataset name:

```text
A large scale 12-lead electrocardiogram database for arrhythmia study v1.0.0
```

After downloading the dataset, place the WFDB records under the local `data/` folder:

```text
AI for Medic ECG/
├── data/
│   ├── JS00001.hea
│   ├── JS00001.mat
│   ├── JS00002.hea
│   ├── JS00002.mat
│   └── ...
```

Raw ECG files are not included in this GitHub repository because the dataset files are large.

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
## Workflow Overview
<img width="849" height="2268" alt="c1a7fed301d3b6c28bdbab427c28b58e" src="https://github.com/user-attachments/assets/aea56f53-54b2-4221-bd94-e14cc1e508d3" />




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

```text
X_train: (7139, 5000, 12)
X_val:   (1530, 5000, 12)
X_test:  (1530, 5000, 12)

Y_train: (7139, 20)
Y_val:   (1530, 20)
Y_test:  (1530, 20)
```

---

### 6. Train Baseline 1D CNN

This script trains a baseline 1D CNN model for multi-label ECG classification.

Run:

```bash
python train_1d_cnn.py
```

Model setting:

```text
Input shape:  (5000, 12)
Output shape: (20,)
Activation:   sigmoid
Loss:         binary_crossentropy
Task:         multi-label classification
```

Generated files:

```text
model_output/ecg_1d_cnn_best.keras
model_output/ecg_1d_cnn_final.keras
model_output/training_history.csv
model_output/test_predictions.npy
model_output/test_metrics.txt
```

---

## Multi-label Classification

This project is a multi-label classification task.

One ECG sample may have more than one diagnosis code, for example:

```text
164889003,59118001,164934002
```

Therefore, the label vector is multi-hot encoded:

```text
[0, 1, 0, 1, 0, ...]
```

The model should use:

```text
sigmoid activation
binary cross-entropy loss
```

Do not use:

```text
softmax
categorical cross-entropy
```

because those are for single-label multi-class classification.

---

## Data Summary

After preprocessing and label filtering:

```text
Total samples: 10199
Signal shape:  (5000, 12)
Number of labels: 20
```

Per-sample label count distribution:

```text
1 label : 5481 samples
2 labels: 2921 samples
3 labels: 1153 samples
4 labels: 458 samples
5 labels: 140 samples
6 labels: 38 samples
7 labels: 6 samples
8 labels: 2 samples
```

---

## Selected Diagnosis Labels

The final dataset keeps 20 high-frequency diagnosis labels.

Examples of selected Dx codes:

```text
426177001
164934002
426783006
164889003
427084000
55827005
428750005
426761007
59118001
164890007
```

The complete mapping is stored in:

```text
output_multilabel/label_map.csv
output_split/label_map.csv
```

---

## Git Ignore Policy

The following folders and files are ignored by Git:

```text
data/
output/
output_multilabel/
output_split/
model_output/
.venv/
*.npy
*.npz
*.mat
*.hea
*.keras
*.h5
```

This is intentional because raw ECG data, generated NumPy arrays, and trained models can be very large.

Only source code and configuration files are tracked in this repository.

---

## Reproducibility

The split script uses a fixed random seed by default:

```text
random_state = 42
```

The train / validation / test indices are saved in:

```text
output_split/split_indices.npz
```

This makes it possible to reproduce the same dataset split.

---

## Notes

This repository contains the preprocessing and baseline training pipeline only.

The raw ECG dataset must be placed locally under:

```text
data/
```

before running the scripts.
