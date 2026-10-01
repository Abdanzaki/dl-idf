#!/usr/bin/env python3
"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 7 — Streamlit Security Operations Center (SOC) Dashboard (dashboard/app.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This interactive Streamlit dashboard acts as the visualization and operational
monitoring tier (Presentation Layer) for the DL-IDF system.

Architectural Principles & College Viva Defense:
================================================================================
1. Three-Tier Enterprise Microservice Architecture:
   - Presentation Tier: Streamlit Web UI (dashboard/app.py)
   - Application / Inference Tier: Flask REST API microservice (api/app.py)
   - Persistence & Model Tier: Saved artifacts (saved/selected_model.keras, scaler.pkl)
   * VIVA POINT: Why does the Streamlit app NEVER import the model directly?
     Decoupling ensures that frontend UI re-renders, user sessions, and multi-user
     connections never compete with or corrupt model memory contexts. The inference
     service can be horizontally scaled, containerized, or migrated to edge GPU
     nodes without altering frontend code.

2. Stateless HTTP Telemetry Ingestion:
   - Communication between the dashboard and detection engine occurs strictly via
     RESTful JSON contracts (GET /health and POST /predict at http://localhost:5000).

3. Industrial IoT Operational Modes:
   - Upload Mode: Batch-analyzes recorded network flow captures (PCAP/NetFlow CSVs)
     with automated schema alignment and real-time progress tracking.
   - Simulate Mode: Emulates real-time streaming traffic by sampling authentic holdout
     flows from the TON_IoT test pool with controlled ingestion delays.

4. SOC Analytics & Decision Support:
   - Executive KPIs: Total flows, malicious attack counts, attack ratio, and mean confidence.
   - Color-coded telemetry table for rapid forensic inspection.
   - Interactive timeline of attack detections and distribution charts.
   - High-priority security alerts panel for immediate threat mitigation.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Page Configuration & Custom Styling
# ==============================================================================

st.set_page_config(
    page_title="DL-IDF — IIoT Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished industrial SOC theme
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge-ok {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-err {
        background-color: #f8d7da;
        color: #842029;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .alert-box {
        background-color: #fff3cd;
        color: #664d03;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration Constants
DEFAULT_API_URL = "http://localhost:5000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "processed" / "feature_names.json"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.npy"


# ==============================================================================
# 2. API Communication Layer (Strictly HTTP — Never Imports Model Directly)
# ==============================================================================

def check_api_health(base_url: str = DEFAULT_API_URL, timeout: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Queries the Flask API health endpoint (GET /health).

    Returns:
        Tuple of (is_online, response_data_or_error_dict).
    """
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return True, data
            return False, {"error": f"Unexpected HTTP status: {resp.status}"}
    except Exception as e:
        return False, {"error": str(e)}


def query_prediction(
    features_dict: Dict[str, float],
    base_url: str = DEFAULT_API_URL,
    timeout: float = 5.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Sends a single network flow record to POST /predict.

    Args:
        features_dict: Unscaled feature dictionary matching the expected schema.
        base_url: Base endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (success, result_dict_or_error).
    """
    predict_url = f"{base_url.rstrip('/')}/predict"
    payload = json.dumps({"features": features_dict}).encode("utf-8")
    req = urllib.request.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return False, {"error": f"HTTP {e.code}", "details": err_body}
        except Exception:
            return False, {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return False, {"error": f"Network Error: {str(e)}"}


# ==============================================================================
# 3. Data Ingestion & Preprocessing Utilities
# ==============================================================================

@st.cache_data
def load_feature_schema() -> List[str]:
    """Loads expected feature names for validation and schema alignment."""
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_resource
def load_simulation_pool() -> Optional[Tuple[np.ndarray, np.ndarray, Any]]:
    """Loads holdout test data and scaler to generate authentic unscaled simulation samples."""
    import joblib
    if (
        os.path.exists(X_TEST_PATH)
        and os.path.exists(Y_TEST_PATH)
        and os.path.exists(SCALER_PATH)
    ):
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
        scaler = joblib.load(SCALER_PATH)
        return X_test, y_test, scaler
    return None


def align_dataframe_features(df: pd.DataFrame, expected_features: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Aligns an uploaded DataFrame with the 50 expected features.
    Handles both pre-encoded tabular features and raw TON_IoT CSV schemas.
    """
    aligned_rows: List[Dict[str, float]] = []
    warnings: List[str] = []

    # Case A: DataFrame already has all 50 features directly
    if set(expected_features).issubset(set(df.columns)):
        for _, row in df[expected_features].iterrows():
            aligned_rows.append({col: float(row[col]) for col in expected_features})
        return aligned_rows, warnings

    # Case B: Raw TON_IoT format with categorical columns ('proto', 'service', 'conn_state', etc.)
    df_clean = df.copy()
    # Drop identifiers and targets if present
    drop_candidates = ["ts", "src_ip", "src_port", "dst_ip", "label", "type"]
    cols_to_drop = [c for c in drop_candidates if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # Replace '-' with 'none'
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df_clean[col] = df_clean[col].replace("-", "none").fillna("none")

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False, dtype=float)

    # Fill any missing required feature columns with 0.0
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0

    df_aligned = df_encoded[expected_features]
    for _, row in df_aligned.iterrows():
        aligned_rows.append({col: float(row[col]) for col in expected_features})

    warnings.append(f"Auto-aligned {len(df)} records from raw schema to {len(expected_features)} model features.")
    return aligned_rows, warnings


# ==============================================================================
# 4. Results Presentation Component
# ==============================================================================

def render_results_section(results_df: pd.DataFrame):
    """
    Renders the executive summary KPIs, alert panel, timeline/pie charts,
    and color-coded predictions table.
    """
    if results_df.empty:
        st.info("No prediction records to display.")
        return

    total_flows = len(results_df)
    attacks_df = results_df[results_df["prediction"] == "attack"]
    normal_df = results_df[results_df["prediction"] == "normal"]
    total_attacks = len(attacks_df)
    attack_pct = (total_attacks / total_flows) * 100 if total_flows > 0 else 0.0
    avg_conf = results_df["confidence"].mean() * 100

    # 1. Executive Stat Cards
    st.subheader("📊 Executive Threat Telemetry Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Network Flows", value=f"{total_flows:,}")
    with kpi2:
        st.metric(
            label="Intrusions Detected",
            value=f"{total_attacks:,}",
            delta=f"{attack_pct:.1f}% Threat Ratio",
            delta_color="inverse"
        )
    with kpi3:
        st.metric(label="Normal Traffic Flows", value=f"{len(normal_df):,}")
    with kpi4:
        st.metric(label="Mean Model Confidence", value=f"{avg_conf:.2f}%")

    st.markdown("---")

    # 2. Charts Section
    st.subheader("📈 Threat Distribution & Temporal Dynamics")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Traffic Classification Breakdown**")
        fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.5))
        labels = ["Normal Traffic", "Attacks / Intrusions"]
        counts = [len(normal_df), total_attacks]
        colors = ["#28a745", "#dc3545"]
        explode = (0, 0.08) if total_attacks > 0 else (0, 0)

        # Handle zero total cases
        if sum(counts) > 0:
            ax_pie.pie(
                counts,
                labels=labels,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                explode=explode,
                textprops={"fontsize": 9}
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
        plt.close(fig_pie)

    with c2:
        st.markdown("**Flow-by-Flow Threat Trajectory**")
        # Line chart of continuous attack probabilities
        prob_chart_df = pd.DataFrame({
            "Flow Sequence Index": range(1, total_flows + 1),
            "Attack Probability": results_df["attack_probability"]
        }).set_index("Flow Sequence Index")
        st.line_chart(prob_chart_df, height=260)

    st.markdown("---")

    # 3. Security Alerts Panel
    st.subheader(f"🚨 Security Alerts Panel ({total_attacks} Active Alarms)")
    if total_attacks > 0:
        st.error(f"⚠️ High Priority: {total_attacks} anomalous or malicious flows detected by 1D-CNN detection engine.")
        with st.expander("Inspect Detected Security Incidents", expanded=True):
            alert_display = attacks_df[["flow_id", "prediction", "attack_probability", "confidence", "dst_port", "duration", "src_bytes", "dst_bytes"]].copy()
            st.dataframe(alert_display, use_container_width=True)
    else:
        st.success("✅ Zero active security threats detected. All analyzed flows are classified as benign.")

    st.markdown("---")

    # 4. Predictions Table (Color Coded)
    st.subheader("📋 Comprehensive Flow Predictions Log")

    def highlight_threat(row):
        is_att = row["prediction"] == "attack"
        bg = "#ffebee" if is_att else "#e8f5e9"
        color = "#c62828" if is_att else "#2e7d32"
        return [f"background-color: {bg}; color: {color}; font-weight: bold;" if col == "prediction" else "" for col in row.index]

    styled_table = results_df.style.apply(highlight_threat, axis=1)
    st.dataframe(styled_table, use_container_width=True, height=350)


# ==============================================================================
# 5. Main Application Workflow
# ==============================================================================

def main():
    st.title("🛡️ DL-IDF — IIoT Intrusion Detection Dashboard")
    st.caption("Deep Learning-Based Intrusion Detection Framework for Industrial IoT • 1D-CNN Microservice")

    expected_features = load_feature_schema()

    # --------------------------------------------------------------------------
    # SIDEBAR: Microservice Health & Mode Selection
    # --------------------------------------------------------------------------
    st.sidebar.header("🔌 System Status & Settings")
    api_url = st.sidebar.text_input("Flask API Base URL", value=DEFAULT_API_URL)

    # Health Check Probe
    is_online, health_data = check_api_health(api_url)
    if is_online:
        model_name = health_data.get("model", "unknown").upper()
        st.sidebar.markdown(f'<span class="status-badge-ok">🟢 API Online ({model_name})</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-badge-err">🔴 API Offline</span>', unsafe_allow_html=True)
        st.sidebar.error(f"Cannot reach Flask service at {api_url}:\n{health_data.get('error')}")

    st.sidebar.markdown("---")
    st.sidebar.header("🕹️ Operation Mode")
    mode = st.sidebar.radio(
        "Select Workflow Mode:",
        ["Upload CSV", "Simulate live traffic"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**College Viva Architecture Note:**\n\n"
        "Streamlit functions as a lightweight, stateless presentation layer. "
        "It communicates with the 1D-CNN model exclusively over HTTP/REST, "
        "ensuring strict separation of concerns and production scalability."
    )

    # Fail gracefully if API is offline
    if not is_online:
        st.error(
            f"### ⚠️ Deep Learning Inference Service Unavailable\n\n"
            f"The Streamlit dashboard cannot establish a connection with the Flask microservice at `{api_url}`.\n\n"
            f"**Troubleshooting Steps:**\n"
            f"1. Verify the Flask daemon is running: `python3 api/app.py`\n"
            f"2. Confirm port 5000 is open: `curl http://localhost:5000/health`\n"
            f"3. Check for firewall or port restrictions."
        )
        return

    # --------------------------------------------------------------------------
    # MODE 1: Upload CSV Analysis
    # --------------------------------------------------------------------------
    if mode == "Upload CSV":
        st.header("📂 Batch Network Traffic Analysis (Upload Mode)")
        st.write(
            "Upload captured industrial IoT network traffic in CSV format. "
            "The microservice normalizes features on the fly and streams predictions through the 1D-CNN model."
        )

        col_up, col_sample = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("Choose a CSV file of network flows", type=["csv"])
        with col_sample:
            st.write("**Need test data?**")
            sample_path = PROJECT_ROOT / "data" / "sample_traffic_test.csv"
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Sample CSV",
                        data=f,
                        file_name="sample_traffic_test.csv",
                        mime="text/csv"
                    )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(raw_df):,}** flow records across **{raw_df.shape[1]}** attributes.")

                if st.button("🚀 Analyze Traffic Flows", type="primary"):
                    aligned_samples, warnings = align_dataframe_features(raw_df, expected_features)
                    for w in warnings:
                        st.info(w)

                    results: List[Dict[str, Any]] = []
                    prog_bar = st.progress(0, text="Sending flows to 1D-CNN detection microservice...")
                    total_count = len(aligned_samples)

                    t0 = time.time()
                    for idx, sample in enumerate(aligned_samples):
                        success, resp = query_prediction(sample, base_url=api_url)
                        if success:
                            results.append({
                                "flow_id": idx + 1,
                                "prediction": resp["prediction"],
                                "confidence": resp["confidence"],
                                "attack_probability": resp["attack_probability"],
                                "dst_port": sample.get("dst_port", 0.0),
                                "duration": sample.get("duration", 0.0),
                                "src_bytes": sample.get("src_bytes", 0.0),
                                "dst_bytes": sample.get("dst_bytes", 0.0)
                            })
                        else:
                            st.warning(f"Flow #{idx+1} rejected by API: {resp.get('error')}")

                        prog_bar.progress((idx + 1) / total_count, text=f"Analyzing flow {idx + 1} of {total_count}...")

                    elapsed = time.time() - t0
                    prog_bar.empty()
                    st.success(f"Analysis complete: processed {len(results)} flows in {elapsed:.2f}s ({len(results)/elapsed:.1f} flows/sec).")

                    res_df = pd.DataFrame(results)
                    render_results_section(res_df)

            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

    # --------------------------------------------------------------------------
    # MODE 2: Live Traffic Simulation
    # --------------------------------------------------------------------------
    elif mode == "Simulate live traffic":
        st.header("⚡ Live Industrial IoT Traffic Simulation")
        st.write(
            "Simulates real-time packet telemetry by streaming random genuine flows from the "
            "TON_IoT holdout test pool to the inference microservice with synthetic delay."
        )

        sim_pool = load_simulation_pool()
        if sim_pool is None:
            st.error("Holdout test split data not found in `data/processed/`. Complete Phase 2 preprocessing first.")
            return

        X_test, y_test, scaler = sim_pool

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            num_samples = st.slider("Number of flows to stream (N):", min_value=5, max_value=60, value=20, step=5)
        with col_cfg2:
            stream_delay = st.slider("Simulated telemetry delay (seconds):", min_value=0.02, max_value=0.5, value=0.08, step=0.02)

        if st.button("▶️ Start Simulation", type="primary"):
            st.write(f"Initiating stream of **{num_samples}** live flows...")

            # Select random indices from test split
            np.random.seed(int(time.time()) % 10000)
            sample_indices = np.random.choice(len(X_test), size=num_samples, replace=False)

            # Invert standard scaling to produce true unscaled features
            unscaled_batch = scaler.inverse_transform(X_test[sample_indices])
            ground_truths = y_test[sample_indices]

            live_results: List[Dict[str, Any]] = []
            progress_bar = st.progress(0, text="Streaming flows...")
            status_text = st.empty()

            for i, (unscaled_vals, gt) in enumerate(zip(unscaled_batch, ground_truths)):
                sample_dict = {col: float(val) for col, val in zip(expected_features, unscaled_vals)}
                success, resp = query_prediction(sample_dict, base_url=api_url)

                if success:
                    live_results.append({
                        "flow_id": i + 1,
                        "prediction": resp["prediction"],
                        "confidence": resp["confidence"],
                        "attack_probability": resp["attack_probability"],
                        "ground_truth": "attack" if gt == 1 else "normal",
                        "dst_port": round(sample_dict.get("dst_port", 0.0), 1),
                        "duration": round(sample_dict.get("duration", 0.0), 4),
                        "src_bytes": round(sample_dict.get("src_bytes", 0.0), 1),
                        "dst_bytes": round(sample_dict.get("dst_bytes", 0.0), 1)
                    })
                    status_text.markdown(
                        f"**Live Packet #{i+1}:** Classified as `{resp['prediction'].upper()}` "
                        f"(Prob: `{resp['attack_probability']:.4f}`, Conf: `{resp['confidence']*100:.1f}%`)"
                    )
                else:
                    st.warning(f"Live flow #{i+1} failed: {resp.get('error')}")

                progress_bar.progress((i + 1) / num_samples)
                time.sleep(stream_delay)

            progress_bar.empty()
            status_text.empty()
            st.success(f"Simulation completed! Ingested {len(live_results)} live flows.")

            res_df = pd.DataFrame(live_results)
            render_results_section(res_df)


if __name__ == "__main__":
    main()
