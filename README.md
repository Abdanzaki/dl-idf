# DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **B.E. Computer Science & Engineering — Major Technical Project**  
> An end-to-end cyber-physical intrusion detection system engineered for Supervisory Control and Data Acquisition (SCADA) and Industrial Internet of Things (IIoT) telemetry networks.

---

## 📌 Project Overview

Modern Industrial IoT (IIoT) environments—such as power distribution grids, smart water treatment facilities, and automated manufacturing pipelines—are increasingly targeted by sophisticated cyber-physical attacks. Traditional perimeter firewalls and rule-based Intrusion Detection Systems (IDS) struggle to detect zero-day exploits, evasive packet modifications, and high-frequency network anomalies.

**DL-IDF** provides a modular, reproducible, and production-ready machine learning framework for IIoT threat detection:
1. **Data Engineering**: Cleans, encodes, and normalizes high-volume network telemetry without data leakage.
2. **Multi-Model Benchmarking**: Empirically compares an ensemble baseline (**Random Forest**) against deep neural architectures (**Multi-Layer Perceptron** and **1D-Convolutional Neural Network**).
3. **Decoupled Microservice Inference**: Exposes the trained detection engine via a high-performance **Flask REST API**.
4. **Interactive SOC Dashboard**: Delivers a **Streamlit** Security Operations Center (SOC) dashboard featuring batch PCAP/CSV upload analysis, real-time live packet streaming simulation, instant threat alerts, and executive KPI analytics.

---

## 🏗️ System Architecture

DL-IDF adopts a **Three-Tier Enterprise Microservice Architecture**, strictly separating the presentation layer from the deep learning compute engine:

```
+-------------------------------------------------------------------------------------------------+
|                                     DL-IDF SYSTEM PIPELINE                                      |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [1. Ingestion Layer]                                                                          |
|       TON_IoT Network Flow Telemetry (461,043 records, 45 raw features)                          |
|                                |                                                                |
|                                v                                                                |
|   [2. Data Engineering & Preprocessing] (src/preprocess.py)                                     |
|       - Data cleaning, missing value handling ('-' -> 'none')                                   |
|       - Categorical One-Hot Encoding (Protocol, Service, Connection State -> 50 features)       |
|       - Stratified Train/Test Split (80% Train: 368,834 | 20% Test: 92,209)                    |
|       - StandardScaler (fitted strictly on X_train to prevent Data Leakage)                    |
|                                |                                                                |
|                                v                                                                |
|   [3. Model Training & Comparative Benchmarking]                                                |
|       +----------------------------+--------------------------+-----------------------------+   |
|       |  Random Forest Baseline    |    Deep MLP (128-64-32)  |   1D-CNN (Temporal Motif)   |   |
|       |    (src/train_baseline.py) |     (src/models/mlp.py)  |     (src/models/cnn.py)     |   |
|       |    Accuracy: 99.8417%      |     Accuracy: 94.0461%   |     Accuracy: 94.4929%      |   |
|       |    F1-Score: 99.7734%      |     F1-Score: 91.7814%   |     F1-Score: 92.1604%      |   |
|       +----------------------------+--------------------------+-----------------------------+   |
|                                                                              |                  |
|                                                                              v                  |
|   [4. Model Selection & Persistence]                                  [SELECTED ENGINE]         |
|       saved/selected_model.keras (1D-CNN, 138 KB) + data/processed/scaler.pkl                   |
|                                |                                                                |
|                                v                                                                |
|   [5. Application / Inference Tier] (api/app.py)                                                |
|       Flask REST API (Port 5000)                                                                |
|       - GET  /health   -> Microservice heartbeat & model metadata                               |
|       - POST /predict  -> JSON payload validation, dynamic scaling, sub-millisecond inference   |
|                                |                                                                |
|                                v                                                                |
|   [6. Presentation Tier — SOC Dashboard] (dashboard/app.py)                                     |
|       Streamlit Web Interface (Port 8501)                                                       |
|       - Mode 1: Batch CSV Upload Analysis (with interactive progress bar)                       |
|       - Mode 2: Live IoT Telemetry Simulation (streaming random genuine test-pool flows)        |
|       - Threat Intelligence: Color-coded telemetry, Real-time Alerts Panel, Threat Visuals     |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 📊 Dataset Story

* **Dataset Name**: TON_IoT Network Flow Dataset (`data/raw/Train_Test_Network.csv`)
* **Scale**: 461,043 network flow records across 45 raw features (Bro/Zeek network telemetry).
* **Target Distribution**:
  * **Normal (Benign)**: 300,000 flows (65.07%)
  * **Attack (Malicious)**: 161,043 flows (34.93%)
* **Stratified Splits**:
  * **Train Set (80%)**: 368,834 flows (240,000 normal, 128,834 attacks)
  * **Test Set (20%)**: 92,209 flows (60,000 normal, 32,209 attacks)

### Note on Dataset Selection & Fallback Justification
The project was originally planned around the **WUSTL-IIoT-2018** industrial dataset. However, during initial project provisioning, upstream download access to the Washington University in St. Louis repository was restricted/blocked. As per the documented fallback plan, the benchmark **TON_IoT** dataset (created by the Cyber Range Lab at UNSW Canberra) was adopted. TON_IoT represents a modern, heterogeneous industrial testbed combining Industrial IoT protocols (Modbus, MQTT, HTTP, DNS, SSL) with realistic multi-vector cyber attacks (Backdoors, DoS, DDoS, Port Scans, Ransomware, XSS, and Injection).

---

## 🧠 Model Architectures & Evaluation

### 1. Classical Machine Learning Baseline: Random Forest
* **Algorithm**: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`
* **Role**: Serves as the gold-standard classical tabular ML benchmark against which deep architectures are evaluated.

### 2. Deep Multi-Layer Perceptron (MLP)
* **Architecture**: Fully connected feedforward network with funnel dimensionality compression:
  $$\text{Input}(50) \to \text{Dense}(128, \text{ReLU}) + \text{Dropout}(0.3) \to \text{Dense}(64, \text{ReLU}) + \text{Dropout}(0.3) \to \text{Dense}(32, \text{ReLU}) + \text{Dropout}(0.3) \to \text{Dense}(1, \text{Sigmoid})$$
* **Parameters**: 16,897 trainable parameters.
* **Optimization**: Binary Cross-Entropy loss + Adam optimizer ($\eta = 0.001$).

### 3. 1D Convolutional Neural Network (1D-CNN) — *Selected & Deployed*
* **Architecture**: Localized temporal/spatial feature extractor:
  $$\text{Input}(50, 1) \to \text{Conv1D}(64, 3, \text{ReLU}, \text{same}) \to \text{MaxPool1D}(2) \to \text{Conv1D}(32, 3, \text{ReLU}, \text{same}) \to \text{GlobalMaxPool1D}() \to \text{Dense}(64, \text{ReLU}) + \text{Dropout}(0.3) \to \text{Dense}(1, \text{Sigmoid})$$
* **Parameters**: 8,609 trainable parameters (50% fewer weights than the MLP).
* **Optimization**: Binary Cross-Entropy loss + Adam optimizer ($\eta = 0.001$).

---

### 📈 Honest Benchmark Comparison Table

All models were evaluated on the exact same unseen holdout test split (**92,209 network flows**):

| Model | Architecture Category | Accuracy | Precision | Recall (Sensitivity) | F1-Score | False Positive Rate (FPR) | Train Time | Eval Latency | Model Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RANDOM_FOREST`** | Classical Tree Ensemble | **99.8417%** | **99.7579%** | **99.7889%** | **99.7734%** | **0.1300%** | **21.0s** | 0.66s | Reference Baseline |
| **`MLP`** | Deep Feedforward Net | 94.0461% | 88.6213% | **95.1753%** | 91.7814% | 6.5600% | 86.3s | **0.35s** | Evaluated |
| **`CNN` (1D-CNN)** | 1D Convolutional Net | **94.4929%** | **91.6567%** | 92.6698% | **92.1604%** | **4.5283%** | 312.0s | 0.64s | **SELECTED & DEPLOYED** |

#### Detailed Confusion Matrix Breakdown (Test Set: $N = 92,209$)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **`RANDOM_FOREST`** | 59,922 | 78 | 68 | 32,141 |
| **`MLP`** | 56,064 | 3,936 | 1,554 | 30,655 |
| **`CNN` (1D-CNN)** | 57,283 | 2,717 | 2,361 | 29,848 |

---

### 🎓 Viva Voce: Why Deploy 1D-CNN When Random Forest Scored Higher?

In academic examinations, examiners often ask: *"If Random Forest scored higher (F1 99.77% vs 92.16%), why did you deploy the 1D-CNN?"*

**The 4-Point Technical Defense:**
1. **Inductive Bias of Tabular Features**: Decision tree ensembles excel at orthogonal axis-aligned partitioning on pre-engineered tabular data. This is a well-established empirical finding in literature (Grinsztajn et al., NeurIPS 2022). We report this result transparently and honestly.
2. **Representation Learning & Feature Extraction**: The 1D-CNN operates via weight sharing across sliding kernels, capturing localized cross-feature correlations (e.g. byte volume / duration / protocol flags) without manual feature engineering.
3. **Continuous Probability Calibration & Latency**: The 1D-CNN outputs continuous posterior probabilities ($P \in [0, 1]$) with sub-millisecond inference (over 144,000 flows/sec), making it ideal for streaming edge deployment on resource-constrained microcontrollers.
4. **Architectural Scope**: As a Deep Learning project (DL-IDF), the primary objective is investigating neural representation learning. The framework retains Random Forest as a modular baseline, demonstrating that the system architecture supports hybrid deployment.

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
Clone the repository and install all pinned dependencies:
```bash
git clone https://github.com/Abdanzaki/dl-idf.git
cd dl-idf
pip install -r requirements.txt
```

### 2. Preprocess Dataset
Cleans raw telemetry, applies One-Hot Encoding, performs stratified splitting, and persists standard scalers:
```bash
python3 -m src.preprocess
```
*Outputs: `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `scaler.pkl`, `feature_names.json`.*

### 3. Train Classical Baseline (Random Forest)
Trains and benchmarks the Random Forest classifier:
```bash
python3 src/train_baseline.py
```
*Outputs: `saved/rf_baseline.pkl` and records baseline metrics in `results/metrics.json`.*

### 4. Train Deep Learning Models (MLP & 1D-CNN)
Trains both deep neural networks with Early Stopping and logs comparison metrics:
```bash
python3 src/train_deep.py
```
*Outputs: `saved/mlp.keras`, `saved/cnn.keras`, and updates `results/metrics.json`.*

### 5. Launch Flask Inference API
Starts the REST API inference microservice on port 5000:
```bash
python3 api/app.py
```

### 6. Launch Streamlit SOC Dashboard
In a separate terminal, launch the interactive SOC dashboard:
```bash
streamlit run dashboard/app.py
```
Access the dashboard in your web browser at **`http://localhost:8501`**.

---

## 🔌 REST API Specification

The Flask service runs by default on `http://localhost:5000`:

### 1. Health Probe (`GET /health`)
Verifies service availability and model status.
* **Request**:
  ```bash
  curl -i http://localhost:5000/health
  ```
* **Response (HTTP 200 OK)**:
  ```json
  {
    "model": "1d-cnn",
    "status": "ok"
  }
  ```

### 2. Flow Prediction (`POST /predict`)
Performs real-time classification on an unscaled network flow record.
* **Request**:
  ```bash
  curl -X POST http://localhost:5000/predict \
       -H "Content-Type: application/json" \
       -d '{"features": {"dst_port": 53.0, "duration": 0.17, "src_bytes": 48.0, ...}}'
  ```
* **Response (HTTP 200 OK)**:
  ```json
  {
    "prediction": "normal",
    "confidence": 1.0,
    "attack_probability": 0.0
  }
  ```
* **Error Response (HTTP 400 Bad Request)**:
  Triggered if any required feature is missing, extra features are supplied, or values are non-numeric.

---

## 🔮 Limitations & Future Work (College Viva Prep)

| Area | Current Implementation | Proposed Future Enhancement |
| :--- | :--- | :--- |
| **Attack Granularity** | Binary classification (`normal` vs `attack`). | Multi-class hierarchical classification distinguishing DoS, Scanning, Ransomware, and MITM. |
| **Temporal Dynamics** | 1D-CNN sliding over tabular feature order. | Recurrent / Transformer architectures (LSTM, GRU, Temporal Convolutional Networks) on raw packet inter-arrival times ($I_t$). |
| **Edge Hardware Optimization** | TensorFlow CPU runtime. | Quantization (INT8 post-training quantization) and ONNX/TensorRT compilation for deployment on NVIDIA Jetson / ARM Cortex. |
| **Adversarial Robustness** | Standard dropout regularization. | Adversarial training (Fast Gradient Sign Method) to defend against adversarial packet perturbation attacks. |

---

## 👥 Contributors

* **Abdanzaki** — *Lead Developer & Researcher*
* Department of Computer Science & Engineering

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
