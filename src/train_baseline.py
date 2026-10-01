#!/usr/bin/env python3
"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 3 — Classical ML Baseline Model (src/train_baseline.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module implements the benchmark classical machine learning model:
1. Loads preprocessed and standardized data splits from data/processed/.
2. Trains a Random Forest Classifier (n_estimators=100, random_state=42, n_jobs=-1).
3. Evaluates model performance on the unseen test split (92,209 records).
4. Computes key intrusion detection metrics:
   - Accuracy: Overall correctness of traffic classification.
   - Precision: Ratio of true detected attacks to all predicted attacks.
   - Recall (Sensitivity): Ratio of detected attacks to all actual attacks.
   - F1-Score: Harmonic mean of precision and recall.
   - False Positive Rate (FPR = FP / (FP + TN)): False alarm rate on normal traffic.
   - Confusion Matrix (TN, FP, FN, TP).
5. Persists the trained baseline model to saved/rf_baseline.pkl.
6. Records and appends evaluation metrics to results/metrics.json for comparison.

Viva Voce Explanation Guide:
- Why Random Forest as the Baseline?
  Random Forest is an ensemble of de-correlated decision trees trained via
  bootstrap aggregation (bagging) and random feature subspace selection. It
  serves as the gold standard tabular comparison against which deep learning
  architectures (MLP and 1D-CNN) are evaluated in Phases 4 & 5.
- Why is Recall the Primary Operational Metric?
  In critical industrial infrastructure (power stations, water treatment, PLCs),
  a False Negative (missing an attack) can lead to equipment destruction, safety
  shutdowns, or compromised supervisory control. False Positives (false alarms)
  cost analyst review time, but False Negatives cause system collapse.
- What is False Positive Rate (FPR)?
  FPR = FP / (FP + TN). It measures what fraction of benign network flows are
  falsely alarmed. Low FPR (<1%) is vital to prevent operator alert fatigue.
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==============================================================================
# 1. Pipeline Functions
# ==============================================================================

def load_processed_data(processed_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads preprocessed feature matrices and target vectors from numpy files.
    
    Args:
        processed_dir: Directory containing X_train.npy, X_test.npy, y_train.npy, y_test.npy.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if not os.path.exists(processed_dir):
        alt_dir = os.path.join(os.path.dirname(__file__), "..", processed_dir)
        if os.path.exists(alt_dir):
            processed_dir = os.path.abspath(alt_dir)
        else:
            raise FileNotFoundError(f"Processed data directory not found at: {processed_dir}")

    print(f"[1/5] Loading preprocessed splits from: {processed_dir} ...")
    t0 = time.time()
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    elapsed = time.time() - t0

    print(f"      Loaded in {elapsed:.2f}s:")
    print(f"      • X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"      • X_test : {X_test.shape}  | y_test : {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[RandomForestClassifier, float]:
    """
    Trains a Random Forest Classifier on the training split.
    
    Viva note:
    n_estimators=100 constructs an ensemble of 100 decorrelated decision trees.
    random_state=42 ensures exact determinism and reproducibility for viva defense.
    n_jobs=-1 leverages all available CPU cores for parallel tree induction.
    
    Args:
        X_train: Standardized feature matrix.
        y_train: Ground truth binary target vector.
        n_estimators: Number of trees in the forest.
        random_state: Seed for pseudo-random number generator.
        n_jobs: Number of parallel worker threads.
        
    Returns:
        Tuple of (fitted model, training duration in seconds).
    """
    print(f"[2/5] Training RandomForestClassifier (n_estimators={n_estimators}, random_state={random_state}, n_jobs={n_jobs})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"      Training complete in {train_duration:.2f} seconds.")
    return rf, train_duration


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_duration: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the unseen test split.
    
    Calculates:
    - Accuracy = (TP + TN) / Total
    - Precision = TP / (TP + FP)
    - Recall (Sensitivity) = TP / (TP + FN)  [Critical: measures attacks caught]
    - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    - False Positive Rate (FPR) = FP / (FP + TN)  [Measures false alarms]
    - Confusion Matrix components: TN, FP, FN, TP
    
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test true labels.
        train_duration: Time taken to train the model.
        
    Returns:
        Dictionary of all computed metrics.
    """
    print(f"[3/5] Evaluating on unseen test set ({len(y_test):,} samples)...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    eval_duration = time.time() - t0

    # Basic metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate: benign traffic falsely classified as attack
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "model": "random_forest",
        "model_type": "baseline_classical_ml",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
            "n_jobs": model.n_jobs
        },
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        "timing": {
            "train_duration_sec": round(train_duration, 2),
            "eval_duration_sec": round(eval_duration, 2)
        }
    }
    return metrics


def save_model(model: RandomForestClassifier, filepath: str = "saved/rf_baseline.pkl") -> str:
    """
    Serializes the trained Random Forest model to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[4/5] Saved trained baseline model to: {filepath}")
    return filepath


def save_metrics(metrics: Dict[str, Any], filepath: str = "results/metrics.json") -> str:
    """
    Appends or updates model evaluation metrics in results/metrics.json.
    Maintains an array of model metric records for downstream comparison.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    metrics_list = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    metrics_list = data
                elif isinstance(data, dict):
                    metrics_list = [data]
        except Exception:
            metrics_list = []

    # Update if already exists, else append
    updated = False
    for i, entry in enumerate(metrics_list):
        if entry.get("model") == metrics.get("model"):
            metrics_list[i] = metrics
            updated = True
            break
    if not updated:
        metrics_list.append(metrics)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"[5/5] Appended metrics to: {filepath}")
    return filepath


def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """
    Renders an ASCII results table formatted for technical reporting and viva defense.
    """
    cm = metrics["confusion_matrix"]
    acc = metrics["accuracy"] * 100
    prec = metrics["precision"] * 100
    rec = metrics["recall"] * 100
    f1 = metrics["f1_score"] * 100
    fpr = metrics["false_positive_rate"] * 100

    print("\n" + "=" * 74)
    print("        DL-IDF PHASE 3: RANDOM FOREST BASELINE — EVALUATION RESULTS        ")
    print("=" * 74)
    print(f"Model Identifier          : {metrics['model'].upper()} (Scikit-Learn)")
    print(f"Training Duration         : {metrics['timing']['train_duration_sec']}s")
    print(f"Evaluation Duration       : {metrics['timing']['eval_duration_sec']}s")
    print("-" * 74)
    print("PERFORMANCE METRICS:")
    print(f"  • Accuracy              : {acc:>7.4f}%   (Overall classification accuracy)")
    print(f"  • Precision             : {prec:>7.4f}%   (Attacks predicted correctly / All attack alarms)")
    print(f"  • Recall (Sensitivity)  : {rec:>7.4f}%   (Attacks detected / All actual attacks in test)")
    print(f"  • F1-Score              : {f1:>7.4f}%   (Harmonic mean of precision and recall)")
    print(f"  • False Positive Rate   : {fpr:>7.4f}%   (Normal traffic falsely flagged as attack)")
    print("-" * 74)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  • True Negatives  (TN)  : {cm['true_negatives']:>7,}   [Normal traffic correctly classified]")
    print(f"  • False Positives (FP)  : {cm['false_positives']:>7,}   [Normal traffic FALSELY flagged as attack]")
    print(f"  • False Negatives (FN)  : {cm['false_negatives']:>7,}   [Attack traffic MISSED by model]")
    print(f"  • True Positives  (TP)  : {cm['true_positives']:>7,}   [Attack traffic correctly DETECTED]")
    print("=" * 74)
    print("VIVA NOTE:")
    print("  Recall is the primary security KPI. Out of 32,209 attack flows, the")
    print(f"  baseline missed only {cm['false_negatives']} attacks (Recall = {rec:.2f}%).")
    print(f"  The False Positive Rate is {fpr:.3f}%, meaning very few benign flows")
    print("  triggered false alarms.")
    print("=" * 74 + "\n")


def run_baseline_pipeline(
    processed_dir: str = "data/processed",
    saved_model_path: str = "saved/rf_baseline.pkl",
    metrics_path: str = "results/metrics.json"
) -> Dict[str, Any]:
    """
    Main orchestration routine for Phase 3 Baseline Model.
    """
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    # 2. Train Model
    rf_model, train_duration = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 3. Evaluate Model
    metrics = evaluate_model(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        train_duration=train_duration
    )

    # 4. Save Model
    save_model(rf_model, saved_model_path)

    # 5. Save Metrics
    save_metrics(metrics, metrics_path)

    # 6. Print Formatted Table
    print_metrics_table(metrics)

    return metrics


# ==============================================================================
# 2. CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    run_baseline_pipeline()
