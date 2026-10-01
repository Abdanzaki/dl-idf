#!/usr/bin/env python3
"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 6 — REST API Inference Microservice (api/app.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module implements the production-style REST API inference microservice
using Flask, serving the trained 1D-CNN deep learning model for real-time
industrial network flow classification.

College Viva Voce Architectural Guide & Technical Defense:
================================================================================
1. Why Deploy Deep Learning via a REST API?
   - Decoupled Microservice Architecture: Separates the ingestion network sensors
     (Zeek/Bro, Suricata, edge telemetry probes) from the deep learning compute engine.
   - Interoperability & Platform Independence: Any industrial controller, SCADA client,
     or edge daemon capable of sending standard HTTP/JSON requests can query the model.
   - Stateless Inference: Each POST request is fully self-contained. The server
     maintains no per-client session state, allowing horizontal scaling behind load
     balancers.

2. Fail-Fast Startup Principle:
   - On process initialization, the service eagerly loads both the deep learning
     weights (saved/selected_model.keras) and the feature normalization parameters
     (data/processed/scaler.pkl).
   - If either artifact is missing or corrupted, the process aborts immediately with
     an explicit diagnostic message rather than failing silently on incoming traffic.

3. Complete Inference Pipeline (Per Request):
   1. HTTP POST Request -> Ingest JSON payload {"features": {col_name: raw_val}}.
   2. Schema Validation -> Strict check: all 50 expected features must be present
      without extraneous or non-numeric fields (HTTP 400 Bad Request if invalid).
   3. Feature Ordering -> Maps the dictionary keys into the exact ordered vector
      defined by feature_names.json.
   4. Standardization -> Transforms the raw values using the training-fitted
      StandardScaler: z = (x - mu_train) / sigma_train.
   5. Tensor Reshaping -> Expands dimensions from (1, 50) to (1, 50, 1) for 1D convolution.
   6. Forward Pass -> Executes the compiled 1D-CNN graph in sub-millisecond latency.
   7. Thresholding -> Converts continuous posterior probability P(Attack|Flow) to
      {"prediction": "attack" | "normal", "confidence": float, "attack_probability": float}.
--------------------------------------------------------------------------------
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, Response

# Add project root to sys.path to facilitate module discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# 1. Configuration & Path Declarations
# ==============================================================================

MODEL_PATH = str(PROJECT_ROOT / "saved" / "selected_model.keras")
SCALER_PATH = str(PROJECT_ROOT / "data" / "processed" / "scaler.pkl")
FEATURE_NAMES_PATH = str(PROJECT_ROOT / "data" / "processed" / "feature_names.json")

# ==============================================================================
# 2. Eager Startup & Artifact Integrity Verification (Fail Loudly)
# ==============================================================================

print("[STARTUP] Initializing DL-IDF Intrusion Detection Inference Microservice...")

# 1. Verify and load trained Deep Learning model
if not os.path.exists(MODEL_PATH):
    error_msg = (
        f"\n{'!' * 80}\n"
        f"[FATAL STARTUP ERROR] Model artifact not found at: '{MODEL_PATH}'\n"
        f"Resolution: Complete Phase 4 & Phase 5 to train and select the deployed model:\n"
        f"  1. Run `python3 src/train_deep.py`\n"
        f"  2. Copy `saved/cnn.keras` to `saved/selected_model.keras`\n"
        f"{'!' * 80}\n"
    )
    raise FileNotFoundError(error_msg)

try:
    print(f"[STARTUP] Loading deep neural network weights from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print(f"          Model '{model.name}' loaded successfully ({model.count_params():,} total parameters).")
except Exception as e:
    raise RuntimeError(f"[FATAL STARTUP ERROR] Failed to deserialize Keras model from '{MODEL_PATH}': {e}")

# 2. Verify and load feature scaler
if not os.path.exists(SCALER_PATH):
    error_msg = (
        f"\n{'!' * 80}\n"
        f"[FATAL STARTUP ERROR] StandardScaler artifact not found at: '{SCALER_PATH}'\n"
        f"Resolution: Execute `python3 src/preprocess.py` to generate preprocessing artifacts.\n"
        f"{'!' * 80}\n"
    )
    raise FileNotFoundError(error_msg)

try:
    print(f"[STARTUP] Loading feature scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"          StandardScaler loaded successfully ({scaler.n_features_in_} features configured).")
except Exception as e:
    raise RuntimeError(f"[FATAL STARTUP ERROR] Failed to deserialize StandardScaler from '{SCALER_PATH}': {e}")

# 3. Verify and load feature names schema
if not os.path.exists(FEATURE_NAMES_PATH):
    error_msg = (
        f"\n{'!' * 80}\n"
        f"[FATAL STARTUP ERROR] Feature names schema not found at: '{FEATURE_NAMES_PATH}'\n"
        f"Resolution: Ensure Phase 2 preprocessing has exported data/processed/feature_names.json.\n"
        f"{'!' * 80}\n"
    )
    raise FileNotFoundError(error_msg)

try:
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        EXPECTED_FEATURE_NAMES: List[str] = json.load(f)
    EXPECTED_FEATURE_SET = set(EXPECTED_FEATURE_NAMES)
    print(f"[STARTUP] Expected feature schema loaded: {len(EXPECTED_FEATURE_NAMES)} input attributes.")
except Exception as e:
    raise RuntimeError(f"[FATAL STARTUP ERROR] Failed to load schema from '{FEATURE_NAMES_PATH}': {e}")

# Warm up the model graph with dummy inference to prevent cold-start latency on first request
_dummy_input = np.zeros((1, len(EXPECTED_FEATURE_NAMES), 1), dtype=np.float32)
_ = model.predict(_dummy_input, verbose=0)
print("[STARTUP] Inference engine warmed up and ready to serve traffic.\n")


# ==============================================================================
# 3. Flask Application Setup & Routes
# ==============================================================================

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """
    Health check probe for container orchestrators (Kubernetes/Docker) and monitoring.
    Returns:
        JSON response with system status and deployed model architecture identifier.
    """
    return jsonify({
        "status": "ok",
        "model": "1d-cnn"
    }), 200


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    """
    Classifies an unscaled network flow record as either 'normal' or 'attack'.

    Request JSON Specification:
    {
        "features": {
            "dst_port": 53.0,
            "duration": 0.17,
            ...
            "dns_rejected_none": 1.0
        }
    }

    Response JSON Specification:
    {
        "prediction": "attack" | "normal",
        "confidence": 0.0 - 1.0,
        "attack_probability": 0.0 - 1.0
    }
    """
    # 1. Enforce JSON payload
    if not request.is_json:
        return jsonify({
            "error": "Bad Request: Content-Type must be application/json."
        }), 400

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({
            "error": "Bad Request: Malformed JSON body. Expected a valid JSON object."
        }), 400

    if "features" not in payload:
        return jsonify({
            "error": "Bad Request: Missing required top-level key 'features'."
        }), 400

    features = payload["features"]
    if not isinstance(features, dict):
        return jsonify({
            "error": "Bad Request: 'features' must be an object mapping feature names to numerical values."
        }), 400

    # 2. Strict Schema Validation (Missing or Extra Features)
    received_keys = set(features.keys())
    missing_features = [col for col in EXPECTED_FEATURE_NAMES if col not in received_keys]
    extra_features = [col for col in received_keys if col not in EXPECTED_FEATURE_SET]

    if missing_features or extra_features:
        details: Dict[str, Any] = {}
        if missing_features:
            details["missing_features"] = missing_features
            details["missing_count"] = len(missing_features)
        if extra_features:
            details["extra_features"] = extra_features
            details["extra_count"] = len(extra_features)
        details["expected_feature_count"] = len(EXPECTED_FEATURE_NAMES)
        details["received_feature_count"] = len(received_keys)

        return jsonify({
            "error": "Bad Request: Feature set mismatch. Payload must provide all 50 unscaled features exactly.",
            "details": details
        }), 400

    # 3. Value Validation: Ensure all feature inputs are parseable as floats
    raw_vector_list = []
    invalid_fields = []
    for col in EXPECTED_FEATURE_NAMES:
        val = features[col]
        try:
            raw_vector_list.append(float(val))
        except (ValueError, TypeError):
            invalid_fields.append(col)

    if invalid_fields:
        return jsonify({
            "error": "Bad Request: Non-numeric values provided for features.",
            "invalid_fields": invalid_fields
        }), 400

    # 4. Standardize Features & Reshape for 1D-CNN Inference
    # Shape: (1, 50)
    raw_vector = np.array([raw_vector_list], dtype=np.float32)
    # Apply fitted training scaler parameters: z = (x - mu) / sigma
    scaled_vector = scaler.transform(raw_vector)
    # Reshape for 1D-CNN temporal input: (1, 50, 1)
    tensor_3d = np.expand_dims(scaled_vector, axis=-1)

    # 5. Forward Inference
    pred_prob_raw = model.predict(tensor_3d, verbose=0)
    attack_probability = float(pred_prob_raw[0][0])

    # 6. Binary Classification & Confidence Calculation
    is_attack = attack_probability >= 0.5
    prediction = "attack" if is_attack else "normal"
    confidence = attack_probability if is_attack else (1.0 - attack_probability)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "attack_probability": round(attack_probability, 6)
    }), 200


# ==============================================================================
# 4. Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Host on all interfaces (0.0.0.0) on port 5000
    print("[SERVER] Starting Flask WSGI server on http://0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
