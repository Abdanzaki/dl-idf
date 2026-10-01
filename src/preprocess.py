#!/usr/bin/env python3
"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 2 — Data Preprocessing Pipeline (src/preprocess.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module implements the end-to-end preprocessing pipeline for Industrial IoT
network flow telemetry:
1. Ingestion of raw flow records from data/raw/Train_Test_Network.csv.
2. Dropping non-predictive identifier columns and leakage-prone attributes.
3. Imputation and cleaning of missing values and Zeek/Bro '-' placeholders.
4. One-Hot Encoding of nominal categorical features and binary label encoding.
5. Stratified 80/20 train/test split (random_state=42).
6. StandardScaler normalization fitted STRICTLY on the training split.
7. Serialization of processed arrays, fitted scaler, and feature metadata.

Viva Voce Explanation Guide:
- Data Leakage Prevention: The StandardScaler is fitted ONLY on X_train. Fitting
  on the whole dataset would leak test set statistics (mean & standard deviation)
  into the model.
- One-Hot vs. Label Encoding: Protocols (TCP, UDP, ICMP) and connection states
  have no natural ordinal ranking. Label encoding imposes artificial distance
  metrics (e.g. ICMP=3 > UDP=2 > TCP=1) that bias neural network weights.
  One-hot encoding represents each state as an independent orthogonal vector.
- Dropping IP Addresses & Timestamps: Using raw IP addresses causes shortcut
  learning (the model memorizes that 192.168.1.152 is an attacker instead of
  learning traffic dynamics). Dropping IPs ensures generalization to new subnets.
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. Documented Feature Exclusion Lists (Viva-Explainable)
# ==============================================================================

# Columns dropped to prevent data leakage and topological / temporal shortcut learning:
IDENTIFIER_COLUMNS: List[str] = [
    "ts",        # Timestamp (epoch seconds): Temporal identifier. If included, models
                 # learn to classify attacks by time of day rather than packet flow patterns.
    "src_ip",    # Source IP Address: High-cardinality host identifier. Attackers use
                 # dynamic or spoofed IPs; models must not memorize specific testbed IPs.
    "dst_ip",    # Destination IP Address: Identifies specific testbed servers/PLCs.
                 # Memorizing target IPs fails completely when deployed on a new network.
    "src_port",  # Source Port: Ephemeral client ports (49152-65535) are allocated
                 # dynamically and pseudo-randomly by OS stacks; provides no generalizable signal.
]

# Ground-truth targets that must never be present in the feature matrix X:
TARGET_COLUMNS: List[str] = [
    "label",     # Binary ground truth (0 = Normal, 1 = Attack) -> extracted as target vector y
    "type",      # Multi-class attack taxonomy (e.g. 'dos', 'scanning', 'mitm') -> separated
]

# Free-text strings and ultra-sparse protocol attributes (>99% missing/dashes):
# In Zeek/Bro logs, non-applicable protocol attributes are logged as '-'.
# Dropping features with >99% missing entries eliminates useless zero-variance dimensions.
SPARSE_AND_TEXT_COLUMNS: List[str] = [
    # Free-text strings (cannot be one-hot encoded without NLP tokenization):
    "dns_query",             # 14,149 unique domain query strings (79.4% dashes)
    "http_uri",              # High-cardinality HTTP request URIs (99.9% dashes)
    "http_user_agent",       # Free-text browser user-agent strings (99.9% dashes)
    "ssl_subject",           # X.509 SSL certificate subjects (100% dashes in this dataset)
    "ssl_issuer",            # X.509 SSL certificate issuers (100% dashes in this dataset)

    # Ultra-sparse SSL layer features (>99.9% dashes):
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",

    # Ultra-sparse HTTP layer features (>99.9% dashes):
    "http_trans_depth",
    "http_method",
    "http_version",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
    "http_orig_mime_types",
    "http_resp_mime_types",

    # Ultra-sparse Bro anomaly/weird attributes (>99.7% dashes):
    "weird_name",
    "weird_addl",
    "weird_notice",
]

# Consolidated list of all columns to exclude from the predictive feature set:
ALL_DROPPED_COLUMNS: List[str] = IDENTIFIER_COLUMNS + TARGET_COLUMNS + SPARSE_AND_TEXT_COLUMNS


# ==============================================================================
# 2. Pipeline Functions
# ==============================================================================

def load_raw_data(data_path: str = "data/raw/Train_Test_Network.csv") -> pd.DataFrame:
    """
    Loads raw TON_IoT network flow records from disk.
    
    Args:
        data_path: Relative or absolute path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing the raw ingested network records.
    """
    # Graceful fallback if invoked from different working directories
    if not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", data_path)
        if os.path.exists(alt_path):
            data_path = os.path.abspath(alt_path)
        else:
            raise FileNotFoundError(f"Raw dataset file not found at: {data_path}")

    print(f"[1/6] Ingesting raw network flow dataset from: {data_path}")
    t0 = time.time()
    df = pd.read_csv(data_path, low_memory=False)
    elapsed = time.time() - t0
    print(f"      Loaded {len(df):,} records across {df.shape[1]} attributes in {elapsed:.2f}s.")
    return df


def clean_and_select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extracts binary labels, separates multi-class labels, drops non-predictive/sparse
    columns, and handles protocol sparsity placeholders ('-').
    
    Args:
        df: Raw input DataFrame.
        
    Returns:
        Tuple of:
        - X: Cleaned feature DataFrame
        - y_binary: Binary target vector (0 = normal, 1 = attack)
        - y_type: Multi-class attack string array (for validation and multi-class tracking)
    """
    print("[2/6] Cleaning data, separating ground-truth labels, and dropping non-predictive columns...")

    # 1. Extract target vectors
    if "label" not in df.columns:
        raise KeyError("Target column 'label' not found in raw dataset.")
    y_binary = df["label"].astype(int).values
    y_type = df["type"].astype(str).values if "type" in df.columns else np.array([])

    # 2. Drop non-predictive identifiers, targets, and ultra-sparse columns
    cols_to_drop = [c for c in ALL_DROPPED_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop).copy()

    # 3. Handle missing values & Zeek '-' placeholders
    # Categorical columns: replace '-' with 'none' (represents protocol-level absence)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].replace("-", "none").fillna("none")

    # Numerical columns: fill any residual NaNs with 0
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        X[col] = X[col].fillna(0)

    print(f"      Preserved {X.shape[1]} core predictive features:")
    print(f"      • Numerical Features   ({len(num_cols)}): {num_cols}")
    print(f"      • Categorical Features ({len(cat_cols)}): {cat_cols}")
    return X, y_binary, y_type


def encode_categorical_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encodes nominal categorical features using One-Hot Encoding.
    
    Viva Justification:
    Network protocols ('tcp', 'udp', 'icmp') and connection states ('SF', 'S0', etc.)
    do not possess any inherent hierarchical or numerical order. Using Label
    Encoding (e.g. 1, 2, 3) imposes an artificial mathematical distance that corrupts
    neural network weight updates. One-Hot Encoding maps categories to orthogonal
    binary dimensions, which is mathematically sound for MLPs and CNNs.
    
    Args:
        X: Feature DataFrame with cleaned categorical strings.
        
    Returns:
        Tuple of:
        - X_encoded: DataFrame with all categorical features one-hot encoded.
        - feature_names: List of all 50 final feature names.
    """
    print("[3/6] Applying One-Hot Encoding to nominal categorical features...")
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Apply pd.get_dummies to convert nominal categories into binary columns
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    feature_names = X_encoded.columns.tolist()

    print(f"      Generated {len(feature_names)} numerical input features after One-Hot Encoding.")
    return X_encoded, feature_names


def perform_stratified_split(
    X_matrix: np.ndarray,
    y_vector: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a stratified train/test split.
    
    Viva Justification:
    Stratification ensures both the training set (80%) and testing set (20%)
    reproduce the exact class distribution (65.07% Normal, 34.93% Attack).
    This prevents sampling bias where a split might accidentally underrepresent
    rare attack categories.
    
    Args:
        X_matrix: 2D numpy array of unscaled features.
        y_vector: 1D numpy array of binary labels.
        test_size: Proportion allocated to testing (default 0.20 for 80/20 split).
        random_state: Fixed seed for exact reproducibility across viva runs.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"[4/6] Executing Stratified Train/Test Split (ratio: {int((1-test_size)*100)}/{int(test_size*100)}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix,
        y_vector,
        test_size=test_size,
        random_state=random_state,
        stratify=y_vector
    )
    return X_train, X_test, y_train, y_test


def fit_and_apply_scaler(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Applies StandardScaler fitted STRICTLY on the training split.
    
    Viva Justification (Critical Data Leakage Defense):
    StandardScaler centers features to zero mean and scales to unit variance:
        z = (x - mu_train) / sigma_train
    The mean (mu) and standard deviation (sigma) MUST be computed solely from X_train.
    If scaler.fit() were called on the full dataset or X_test, information about
    the test set distribution would leak into the training process (Data Leakage).
    
    Args:
        X_train: Unscaled training feature matrix.
        X_test: Unscaled testing feature matrix.
        
    Returns:
        Tuple of:
        - X_train_scaled: Normalized training array.
        - X_test_scaled: Normalized test array (scaled using X_train parameters).
        - scaler: The fitted StandardScaler object for persistence and inference.
    """
    print("[5/6] Fitting StandardScaler on X_train ONLY and transforming splits...")
    scaler = StandardScaler()

    # Fit exclusively on training data
    scaler.fit(X_train)

    # Transform both splits
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler


def save_processed_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    feature_names: List[str],
    output_dir: str = "data/processed",
    saved_models_dir: str = "saved"
) -> Dict[str, str]:
    """
    Persists processed numpy arrays, fitted scaler, and feature metadata.
    
    Args:
        X_train, X_test, y_train, y_test: Scaled and split numpy arrays.
        scaler: Fitted StandardScaler instance.
        feature_names: List of final feature column names.
        output_dir: Directory to save processed data splits.
        saved_models_dir: Directory to persist scaler for inference engine.
        
    Returns:
        Dict mapping artifact names to their absolute file paths.
    """
    print(f"[6/6] Saving processed arrays, scaler, and metadata to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    saved_paths = {
        "X_train": os.path.join(output_dir, "X_train.npy"),
        "X_test": os.path.join(output_dir, "X_test.npy"),
        "y_train": os.path.join(output_dir, "y_train.npy"),
        "y_test": os.path.join(output_dir, "y_test.npy"),
        "scaler_processed": os.path.join(output_dir, "scaler.pkl"),
        "scaler_saved": os.path.join(saved_models_dir, "scaler.pkl"),
        "feature_names": os.path.join(output_dir, "feature_names.json"),
        "metadata": os.path.join(output_dir, "metadata.json"),
    }

    # 1. Save processed numpy arrays
    np.save(saved_paths["X_train"], X_train)
    np.save(saved_paths["X_test"], X_test)
    np.save(saved_paths["y_train"], y_train)
    np.save(saved_paths["y_test"], y_test)

    # 2. Save fitted scaler to both data/processed/ and saved/ for the Flask API
    joblib.dump(scaler, saved_paths["scaler_processed"])
    joblib.dump(scaler, saved_paths["scaler_saved"])

    # 3. Save feature schema metadata
    with open(saved_paths["feature_names"], "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    metadata: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset_name": "TON_IoT Network Flow Dataset",
        "total_samples": int(X_train.shape[0] + X_test.shape[0]),
        "num_features": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_class_balance": {
            "normal_count": int(np.sum(y_train == 0)),
            "attack_count": int(np.sum(y_train == 1)),
            "normal_pct": float(np.mean(y_train == 0) * 100),
            "attack_pct": float(np.mean(y_train == 1) * 100),
        },
        "test_class_balance": {
            "normal_count": int(np.sum(y_test == 0)),
            "attack_count": int(np.sum(y_test == 1)),
            "normal_pct": float(np.mean(y_test == 0) * 100),
            "attack_pct": float(np.mean(y_test == 1) * 100),
        },
        "feature_names": feature_names,
    }

    with open(saved_paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return saved_paths


def print_pipeline_summary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> None:
    """
    Prints a clear, formatted summary of shapes, dimensions, and class balances.
    """
    total_samples = len(y_train) + len(y_test)
    train_normal = int(np.sum(y_train == 0))
    train_attack = int(np.sum(y_train == 1))
    test_normal = int(np.sum(y_test == 0))
    test_attack = int(np.sum(y_test == 1))

    print("\n" + "=" * 78)
    print("           DL-IDF PREPROCESSING PIPELINE — EXECUTION SUMMARY           ")
    print("=" * 78)
    print(f"Total Processed Samples    : {total_samples:,}")
    print(f"Total Input Features       : {len(feature_names)}")
    print("-" * 78)
    print("ARRAY SHAPES:")
    print(f"  • X_train shape          : {X_train.shape}  (float32)")
    print(f"  • y_train shape          : {y_train.shape}       (int64)")
    print(f"  • X_test  shape          : {X_test.shape}   (float32)")
    print(f"  • y_test  shape          : {y_test.shape}        (int64)")
    print("-" * 78)
    print("STRATIFIED CLASS BALANCE:")
    print(f"  • Training Split (80%)   : {len(y_train):,} samples")
    print(f"      - Normal (Class 0)   : {train_normal:>8,}  ({train_normal/len(y_train)*100:.2f}%)")
    print(f"      - Attack (Class 1)   : {train_attack:>8,}  ({train_attack/len(y_train)*100:.2f}%)")
    print(f"  • Testing Split  (20%)   : {len(y_test):,} samples")
    print(f"      - Normal (Class 0)   : {test_normal:>8,}  ({test_normal/len(y_test)*100:.2f}%)")
    print(f"      - Attack (Class 1)   : {test_attack:>8,}  ({test_attack/len(y_test)*100:.2f}%)")
    print("=" * 78 + "\n")


def run_preprocessing_pipeline(
    raw_csv_path: str = "data/raw/Train_Test_Network.csv",
    output_dir: str = "data/processed",
    saved_dir: str = "saved",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Main orchestration routine for Phase 2 Preprocessing.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    t_start = time.time()

    # 1. Ingestion
    df = load_raw_data(raw_csv_path)

    # 2. Cleaning & Feature Selection
    X, y_binary, _ = clean_and_select_features(df)

    # 3. Categorical Encoding
    X_encoded, feature_names = encode_categorical_features(X)

    # 4. Stratified Split
    X_train, X_test, y_train, y_test = perform_stratified_split(
        X_encoded.values, y_binary, test_size=test_size, random_state=random_state
    )

    # 5. Scaling (Fitted on train only)
    X_train_scaled, X_test_scaled, scaler = fit_and_apply_scaler(X_train, X_test)

    # 6. Serialization
    save_processed_artifacts(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=feature_names,
        output_dir=output_dir,
        saved_models_dir=saved_dir
    )

    # 7. Print summary
    print_pipeline_summary(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )

    total_time = time.time() - t_start
    print(f"[SUCCESS] Preprocessing completed in {total_time:.2f} seconds.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ==============================================================================
# 3. Entrypoint for CLI execution: python3 -m src.preprocess
# ==============================================================================

if __name__ == "__main__":
    # Allows execution as a standalone module: python3 -m src.preprocess
    run_preprocessing_pipeline()
