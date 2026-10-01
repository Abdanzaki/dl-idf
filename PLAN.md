# DL-IDF — Deep Learning-Based Intrusion Detection Framework for Industrial IoT

College major project (B.E CSE). All application code is written by the
Antigravity CLI (`agy`); the coordinator only orchestrates, verifies, and
pushes to GitHub. Keep everything explainable for a viva.

## What it does
Takes Industrial IoT network-traffic records, runs them through a trained
neural network, classifies each record as normal or attack, and shows results
on a web dashboard (traffic status, detected attacks, alerts, statistics).

## Architecture (4 layers)
1. **Data layer** — public IIoT dataset (no real hardware).
2. **Processing layer** — cleaning, scaling, train/test split; the scaler is
   saved so inference uses identical transformations.
3. **Detection engine** — trained deep-learning model behind a Flask API:
   traffic features in → normal/attack + confidence out.
4. **Dashboard layer** — Streamlit app: CSV upload + "simulate live traffic",
   predictions table, alerts panel, charts, stat cards.

## Dataset
Primary: **WUSTL-IIoT-2018** (network flow records from an industrial testbed,
labeled attacks). Fallback: TON_IoT network data, then Bot-IoT.

## Models (viva-explainable order)
1. Random Forest baseline (scikit-learn) — the comparison benchmark.
2. MLP: input → 2–3 hidden layers (ReLU, dropout) → sigmoid output,
   binary cross-entropy, early stopping. Primary deep model.
3. 1D-CNN as the advanced comparison.
( Skip LSTM/autoencoder unless time permits. )
Metrics: accuracy, precision, recall, F1, confusion matrix, false-positive
rate. Recall matters most — missing an attack is worse than a false alarm.

## Tech stack
Python, pandas, numpy, scikit-learn, TensorFlow/Keras, Flask, Streamlit,
Jupyter for exploration.

## Folder structure
```
dl-idf/
├── data/                 # raw + processed datasets (gitignored if large)
├── notebooks/            # EDA + experiments
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/           # mlp.py, cnn.py
├── api/app.py            # Flask prediction API
├── dashboard/app.py      # Streamlit dashboard
├── saved/                # trained model (.h5/.keras) + scaler (.pkl)
├── requirements.txt
├── .gitignore
└── README.md             # architecture diagram (ASCII), how to run, results
```

## Build phases (agy executes, coordinator verifies each)
1. **Data** — download WUSTL-IIoT-2018, EDA notebook (class balance, missing
   values, feature types).
2. **Preprocessing** — `src/preprocess.py`: clean, encode, StandardScaler,
   stratified split; persist scaler + processed splits.
3. **Baseline** — Random Forest via sklearn; record metrics.
4. **Deep models** — MLP then 1D-CNN with dropout + early stopping; record
   metrics; comparison table (baseline vs MLP vs CNN).
5. **Selection** — pick winner on recall/F1; save model + scaler to `saved/`.
6. **API** — Flask `api/app.py`: POST /predict (features in, label +
   confidence out), GET /health.
7. **Dashboard** — Streamlit: CSV upload, simulate-live button, predictions
   table, alerts, charts (attacks over time, normal-vs-attack split), stat
   cards. Must call the Flask API (not import the model directly).
8. **Verify** — `python -m py_compile` on all sources, API boot + sample
   prediction, dashboard smoke test, requirements install check.
9. **GitHub** — push to `Abdanzaki/dl-idf` (public).
10. **Docs** — README with architecture, dataset story, model explanation,
    results table, how-to-run, limitations + future work (viva prep).

## Constraints
- No secrets in code or git. Large data files gitignored.
- Keep model code readable and commented — the student must explain it.
- Do not over-engineer: MLP-first, simple dashboard, working demo > features.
