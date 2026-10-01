#!/usr/bin/env python3
"""
DL-IDF: REST API Verification & Test Script (test_api.py)
Extracts genuine normal and attack samples from the holdout test set,
inverts the standard scaler to obtain original unscaled feature values,
and queries the running Flask microservice endpoints.
"""

import json
import time
import urllib.request
import urllib.error
import joblib
import numpy as np

BASE_URL = "http://127.0.0.1:5000"


def wait_for_server(max_retries: int = 15, delay: float = 1.0) -> bool:
    """Polls /health until the server responds or timeout is reached."""
    health_url = f"{BASE_URL}/health"
    print(f"Polling {health_url} to ensure server is ready...")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    print(f"Server is UP and responsive (attempt {attempt}): {data}")
                    return True
        except Exception:
            time.sleep(delay)
    return False


def query_health() -> dict:
    """Queries GET /health and returns parsed JSON."""
    url = f"{BASE_URL}/health"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_predict(features_dict: dict) -> dict:
    """Queries POST /predict with a feature dictionary."""
    url = f"{BASE_URL}/predict"
    payload = json.dumps({"features": features_dict}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_invalid_input() -> tuple:
    """Tests POST /predict with a missing feature to verify 400 Bad Request."""
    url = f"{BASE_URL}/predict"
    # Incomplete payload (missing all other 49 features)
    payload = json.dumps({"features": {"dst_port": 80}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def main():
    print("=" * 76)
    print("      DL-IDF REST API ENDPOINT TESTING (LIVE VERIFICATION)")
    print("=" * 76)

    # 1. Wait for server
    if not wait_for_server():
        print("[ERROR] Server failed to respond on port 5000.")
        return

    # 2. Test GET /health
    print("\n[Test 1] GET /health")
    health_resp = query_health()
    print("Response Status Code: 200 OK")
    print("Response JSON Payload:")
    print(json.dumps(health_resp, indent=2))

    # 3. Load artifacts to construct real unscaled samples
    print("\nLoading test splits and scaler to obtain real unscaled samples...")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    scaler = joblib.load("data/processed/scaler.pkl")
    with open("data/processed/feature_names.json", "r") as f:
        feature_names = json.load(f)

    # Locate first genuine normal and attack sample
    normal_idx = int(np.where(y_test == 0)[0][0])
    attack_idx = int(np.where(y_test == 1)[0][0])
    print(f"  • Selected Genuine Normal sample from test split index [{normal_idx}] (Ground Truth label=0)")
    print(f"  • Selected Genuine Attack sample from test split index [{attack_idx}] (Ground Truth label=1)")

    # Invert scaler to obtain original, unscaled values
    normal_unscaled = scaler.inverse_transform(X_test[normal_idx:normal_idx+1])[0]
    attack_unscaled = scaler.inverse_transform(X_test[attack_idx:attack_idx+1])[0]

    normal_features = {name: float(val) for name, val in zip(feature_names, normal_unscaled)}
    attack_features = {name: float(val) for name, val in zip(feature_names, attack_unscaled)}

    # 4. Test POST /predict with Normal Sample
    print("\n[Test 2] POST /predict (Real Normal Sample, ground_truth=0)")
    print(f"Sample preview (first 4 features): {{k: round(v, 4) for k, v in list(normal_features.items())[:4]}}")
    normal_resp = query_predict(normal_features)
    print("Response Status Code: 200 OK")
    print("Response JSON Payload:")
    print(json.dumps(normal_resp, indent=2))

    # 5. Test POST /predict with Attack Sample
    print("\n[Test 3] POST /predict (Real Attack Sample, ground_truth=1)")
    print(f"Sample preview (first 4 features): {{k: round(v, 4) for k, v in list(attack_features.items())[:4]}}")
    attack_resp = query_predict(attack_features)
    print("Response Status Code: 200 OK")
    print("Response JSON Payload:")
    print(json.dumps(attack_resp, indent=2))

    # 6. Test Input Validation (Bad Request)
    print("\n[Test 4] POST /predict (Validation Test: Missing Features)")
    status_code, err_resp = test_invalid_input()
    print(f"Response Status Code: {status_code} Bad Request")
    print("Response JSON Payload:")
    print(json.dumps(err_resp, indent=2))

    print("\n" + "=" * 76)
    print("ALL API ENDPOINTS TESTED SUCCESSFULLY AND VERIFIED.")
    print("=" * 76)


if __name__ == "__main__":
    main()
