# AI for Medic ECG

This project processes 12-lead ECG records in WFDB `.hea + .mat` format and prepares a multi-label ECG diagnosis dataset.

## Dataset Format

Each ECG record consists of:

- `JSxxxxx.hea`
- `JSxxxxx.mat`

Each ECG sample has:

- 12 leads
- 500 Hz sampling rate
- 5000 time points
- 10 seconds duration
- multi-label diagnosis codes from `Dx`

Raw ECG data and generated `.npy` files are not included in this repository.

## Pipeline

### 1. Inspect one ECG record

```bash
python inspect_one_record.py
python prepare_dataset.py
``
python prepare_multilabel_dataset.py
python split_dataset.py
python train_1d_cnn.py
X_filtered: (10199, 5000, 12)
Y_filtered: (10199, 20)

X_train: (7139, 5000, 12)
X_val:   (1530, 5000, 12)
X_test:  (1530, 5000, 12)
Notes
The following folders and files are ignored by Git:

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
