#!/usr/bin/env python3
"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 4 — Deep Neural Network Training & Evaluation (src/train_deep.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module orchestrates the end-to-end training, validation, evaluation, and
comparative benchmarking of deep learning models on the TON_IoT network dataset:
1. Multi-Layer Perceptron (MLP): Deep feedforward neural network with Dropout.
2. 1D Convolutional Neural Network (1D-CNN): Temporal / spatial feature extractor.

Workflow:
1. Loads preprocessed splits (X_train, X_test, y_train, y_test) from data/processed/.
2. Trains MLP with EarlyStopping (monitor='val_loss', patience=4, restore_best_weights=True).
3. Evaluates MLP on X_test (Accuracy, Precision, Recall, F1, FPR, Confusion Matrix).
4. Persists trained model to saved/mlp.keras and logs metrics to results/metrics.json.
5. Reshapes input to (n_samples, n_features, 1) and repeats training for 1D-CNN.
6. Persists trained model to saved/cnn.keras and logs metrics to results/metrics.json.
7. Prints a unified performance comparison table against the Random Forest baseline.
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Import custom model builders
from src.models.mlp import build_mlp
from src.models.cnn import build_cnn1d


# ==============================================================================
# 1. Data Ingestion
# ==============================================================================

def load_data_splits(
    processed_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads preprocessed feature matrices and target labels from disk.

    Args:
        processed_dir: Directory containing preprocessed .npy files.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if not os.path.exists(processed_dir):
        alt = os.path.join(os.path.dirname(__file__), "..", processed_dir)
        if os.path.exists(alt):
            processed_dir = os.path.abspath(alt)
        else:
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    print(f"\n[1/6] Loading preprocessed dataset splits from: {processed_dir} ...")
    t0 = time.time()
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    elapsed = time.time() - t0

    print(f"      Loaded in {elapsed:.2f}s:")
    print(f"      • X_train: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"      • y_train: {y_train.shape} (Attack ratio: {np.mean(y_train):.4f})")
    print(f"      • X_test : {X_test.shape}  (dtype: {X_test.dtype})")
    print(f"      • y_test : {y_test.shape}  (Attack ratio: {np.mean(y_test):.4f})")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. Evaluation & Metric Storage Helpers
# ==============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    model_name: str,
    model_type: str,
    hyperparams: Dict[str, Any],
    train_duration_sec: float,
    eval_duration_sec: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes classification metrics, confusion matrix, and false positive rate.

    Args:
        y_true: True binary target labels (0=Normal, 1=Attack).
        y_pred_prob: Predicted posterior attack probabilities from sigmoid.
        model_name: Model identifier (e.g., 'mlp', 'cnn').
        model_type: Descriptive model category.
        hyperparams: Dictionary of training configuration hyperparameters.
        train_duration_sec: Wall-clock training duration in seconds.
        eval_duration_sec: Wall-clock test inference duration in seconds.
        threshold: Decision threshold for binary classification (default: 0.5).

    Returns:
        Structured dictionary matching DL-IDF results/metrics.json schema.
    """
    # Threshold continuous probabilities to binary decisions
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate: FPR = FP / (FP + TN)
    denom_fpr = float(fp + tn)
    fpr = float(fp / denom_fpr) if denom_fpr > 0 else 0.0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record: Dict[str, Any] = {
        "model": model_name,
        "model_type": model_type,
        "timestamp": timestamp,
        "hyperparameters": hyperparams,
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
            "matrix_2x2": [
                [int(tn), int(fp)],
                [int(fn), int(tp)]
            ]
        },
        "timing": {
            "train_duration_sec": round(train_duration_sec, 2),
            "eval_duration_sec": round(eval_duration_sec, 2)
        }
    }
    return record


def save_or_update_metrics(
    metrics_record: Dict[str, Any],
    filepath: str = "results/metrics.json"
) -> None:
    """
    Appends or updates the model metric record in results/metrics.json.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    metrics_list: List[Dict[str, Any]] = []

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    metrics_list = data
                elif isinstance(data, dict):
                    metrics_list = [data]
        except Exception as e:
            print(f"      [Warning] Could not parse existing {filepath}: {e}. Initializing fresh list.")
            metrics_list = []

    # Update if entry with matching model name exists, otherwise append
    replaced = False
    for i, entry in enumerate(metrics_list):
        if entry.get("model") == metrics_record.get("model"):
            metrics_list[i] = metrics_record
            replaced = True
            break
    if not replaced:
        metrics_list.append(metrics_record)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"      Metrics successfully recorded for '{metrics_record['model']}' in {filepath}")


# ==============================================================================
# 3. Model Training Orchestrations
# ==============================================================================

def train_and_eval_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 1024,
    validation_split: float = 0.15,
    patience: int = 4,
    saved_model_path: str = "saved/mlp.keras",
    metrics_path: str = "results/metrics.json"
) -> Dict[str, Any]:
    """
    Constructs, trains, evaluates, and persists the MLP model.
    """
    print("\n" + "=" * 76)
    print("      [2/6] TRAINING DEEP MULTI-LAYER PERCEPTRON (MLP)")
    print("=" * 76)
    input_dim = X_train.shape[1]
    mlp_model = build_mlp(input_dim=input_dim, learning_rate=0.001)

    print("Model Summary:")
    mlp_model.summary(line_length=76)

    # Early stopping callback: monitors validation loss to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\nInitiating training (Max Epochs={epochs}, Batch Size={batch_size}, Val Split={validation_split}, Patience={patience})...")
    t_start = time.time()
    history = mlp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    train_duration = time.time() - t_start
    actual_epochs = len(history.history["loss"])
    best_epoch = actual_epochs - early_stop.wait if early_stop.stopped_epoch > 0 else actual_epochs
    print(f"\nMLP training completed in {train_duration:.2f}s ({actual_epochs} epochs ran, best epoch: {best_epoch}).")

    # Evaluation on unseen test split
    print("\nEvaluating MLP on unseen test split (92,209 samples)...")
    t_eval = time.time()
    y_pred_prob = mlp_model.predict(X_test, batch_size=2048, verbose=0).ravel()
    eval_duration = time.time() - t_eval
    print(f"Inference completed in {eval_duration:.2f}s ({len(X_test)/eval_duration:,.0f} samples/sec).")

    # Save model in native Keras format
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    mlp_model.save(saved_model_path)
    print(f"Saved trained MLP model to: {saved_model_path}")

    # Compile metrics
    hyperparams = {
        "architecture": "MLP (128->64->32)",
        "epochs_max": epochs,
        "epochs_trained": actual_epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "patience": patience,
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "activation": "relu",
        "output_activation": "sigmoid"
    }

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred_prob=y_pred_prob,
        model_name="mlp",
        model_type="deep_learning_mlp",
        hyperparams=hyperparams,
        train_duration_sec=train_duration,
        eval_duration_sec=eval_duration
    )

    save_or_update_metrics(metrics, filepath=metrics_path)
    return metrics


def train_and_eval_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 1024,
    validation_split: float = 0.15,
    patience: int = 4,
    saved_model_path: str = "saved/cnn.keras",
    metrics_path: str = "results/metrics.json"
) -> Dict[str, Any]:
    """
    Reshapes data to (N, n_features, 1), constructs, trains, evaluates, and persists the 1D-CNN.
    """
    print("\n" + "=" * 76)
    print("      [3/6] TRAINING 1D CONVOLUTIONAL NEURAL NETWORK (1D-CNN)")
    print("=" * 76)
    n_features = X_train.shape[1]

    # Reshape 2D matrices into 3D tensors: (batch, sequence_length, channels)
    print(f"Reshaping data for 1D convolution:")
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)
    print(f"  • X_train_cnn: {X_train_cnn.shape}")
    print(f"  • X_test_cnn : {X_test_cnn.shape}")

    cnn_model = build_cnn1d(input_dim=n_features, learning_rate=0.001)

    print("\nModel Summary:")
    cnn_model.summary(line_length=76)

    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\nInitiating training (Max Epochs={epochs}, Batch Size={batch_size}, Val Split={validation_split}, Patience={patience})...")
    t_start = time.time()
    history = cnn_model.fit(
        X_train_cnn,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    train_duration = time.time() - t_start
    actual_epochs = len(history.history["loss"])
    best_epoch = actual_epochs - early_stop.wait if early_stop.stopped_epoch > 0 else actual_epochs
    print(f"\n1D-CNN training completed in {train_duration:.2f}s ({actual_epochs} epochs ran, best epoch: {best_epoch}).")

    # Evaluation on unseen test split
    print("\nEvaluating 1D-CNN on unseen test split (92,209 samples)...")
    t_eval = time.time()
    y_pred_prob = cnn_model.predict(X_test_cnn, batch_size=2048, verbose=0).ravel()
    eval_duration = time.time() - t_eval
    print(f"Inference completed in {eval_duration:.2f}s ({len(X_test_cnn)/eval_duration:,.0f} samples/sec).")

    # Save model in native Keras format
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    cnn_model.save(saved_model_path)
    print(f"Saved trained 1D-CNN model to: {saved_model_path}")

    # Compile metrics
    hyperparams = {
        "architecture": "1D-CNN (Conv1D(64,3)->MaxPool->Conv1D(32,3)->GlobalMaxPool->Dense(64))",
        "epochs_max": epochs,
        "epochs_trained": actual_epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "patience": patience,
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "activation": "relu",
        "output_activation": "sigmoid"
    }

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred_prob=y_pred_prob,
        model_name="cnn",
        model_type="deep_learning_1d_cnn",
        hyperparams=hyperparams,
        train_duration_sec=train_duration,
        eval_duration_sec=eval_duration
    )

    save_or_update_metrics(metrics, filepath=metrics_path)
    return metrics


# ==============================================================================
# 4. Comparative Evaluation Table Generator
# ==============================================================================

def print_unified_comparison_table(metrics_path: str = "results/metrics.json") -> None:
    """
    Loads all recorded model metrics and prints a clean comparison table
    suitable for technical evaluation reports and college viva presentations.
    """
    if not os.path.exists(metrics_path):
        print(f"[Error] Metrics file {metrics_path} does not exist.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # Order preference: random_forest, mlp, cnn, others
    order_map = {"random_forest": 0, "mlp": 1, "cnn": 2}
    records = sorted(records, key=lambda r: order_map.get(r.get("model", ""), 99))

    print("\n" + "=" * 90)
    print("         DL-IDF: INTRUSION DETECTION BENCHMARK COMPARISON TABLE")
    print("                   (Random Forest Baseline vs. MLP vs. 1D-CNN)                  ")
    print("=" * 90)

    header = f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FPR':<10} | {'Train (s)':<9} | {'Eval (s)'}"
    print(header)
    print("-" * 90)

    for r in records:
        name = r.get("model", "unknown").upper()
        acc = f"{r.get('accuracy', 0.0) * 100:.4f}%"
        prec = f"{r.get('precision', 0.0) * 100:.4f}%"
        rec = f"{r.get('recall', 0.0) * 100:.4f}%"
        f1 = f"{r.get('f1_score', 0.0) * 100:.4f}%"
        fpr = f"{r.get('false_positive_rate', 0.0) * 100:.4f}%"
        t_train = f"{r.get('timing', {}).get('train_duration_sec', 0.0):.1f}s"
        t_eval = f"{r.get('timing', {}).get('eval_duration_sec', 0.0):.2f}s"
        print(f"{name:<15} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10} | {fpr:<10} | {t_train:<9} | {t_eval}")

    print("=" * 90)
    print("\nDETAILED CONFUSION MATRIX BREAKDOWN:")
    print("-" * 90)
    print(f"{'Model':<15} | {'TN (Benign OK)':<15} | {'FP (False Alarm)':<16} | {'FN (Missed Attack)':<18} | {'TP (Attack OK)'}")
    print("-" * 90)
    for r in records:
        name = r.get("model", "unknown").upper()
        cm = r.get("confusion_matrix", {})
        tn = f"{cm.get('true_negatives', 0):,}"
        fp = f"{cm.get('false_positives', 0):,}"
        fn = f"{cm.get('false_negatives', 0):,}"
        tp = f"{cm.get('true_positives', 0):,}"
        print(f"{name:<15} | {tn:<15} | {fp:<16} | {fn:<18} | {tp}")
    print("=" * 90)

    print("\nVIVA VOCE KEY TAKEAWAYS & COMPARATIVE ANALYSIS:")
    print("1. Security Metric Priority (Recall vs. FPR):")
    print("   - In Industrial IoT (IIoT), False Negatives (missed attacks) are critical hazards.")
    print("   - Low False Positive Rate (FPR < 0.5%) prevents analyst alert fatigue.")
    print("2. Classical ML vs. Deep Learning:")
    print("   - Random Forest serves as a highly competitive tabular baseline with fast training.")
    print("   - MLP and 1D-CNN learn continuous probabilistic representations suitable for online")
    print("     streaming inference, transfer learning, and real-time edge deployment.")
    print("=" * 90 + "\n")


# ==============================================================================
# 5. Main Execution Entry Point
# ==============================================================================

def main():
    """
    Executes Phase 4 training for MLP and 1D-CNN, evaluates against test split,
    and displays the comparison table.
    """
    print("=" * 76)
    print("  DL-IDF PHASE 4: DEEP LEARNING MODEL TRAINING (MLP & 1D-CNN)")
    print("=" * 76)

    # 1. Load preprocessed splits
    X_train, X_test, y_train, y_test = load_data_splits(processed_dir="data/processed")

    # 2. Train and evaluate MLP
    train_and_eval_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=30,
        batch_size=1024,
        validation_split=0.15,
        patience=4,
        saved_model_path="saved/mlp.keras",
        metrics_path="results/metrics.json"
    )

    # 3. Train and evaluate 1D-CNN
    train_and_eval_cnn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=30,
        batch_size=1024,
        validation_split=0.15,
        patience=4,
        saved_model_path="saved/cnn.keras",
        metrics_path="results/metrics.json"
    )

    # 4. Display unified comparison table
    print_unified_comparison_table(metrics_path="results/metrics.json")


if __name__ == "__main__":
    main()
