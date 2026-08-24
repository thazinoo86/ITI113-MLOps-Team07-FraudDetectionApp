
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    bundle = joblib.load(Path(model_dir) / "model_bundle.joblib")
    if not bundle.get("risk_score_excluded"):
        raise RuntimeError("Unsafe model bundle: Risk_Score policy missing.")
    return bundle


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected one JSON object or a non-empty list.")
    frame = pd.DataFrame(payload)
    forbidden = {"Risk_Score", "Fraud_Label", "Transaction_ID", "User_ID", "Is_Weekend"}
    supplied_forbidden = forbidden & set(frame.columns)
    if supplied_forbidden:
        raise ValueError(f"Forbidden input fields supplied: {sorted(supplied_forbidden)}")
    return frame


def prepare_features(frame, bundle):
    missing = [c for c in bundle["raw_inference_columns"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    result = frame[bundle["raw_inference_columns"]].copy()
    timestamp = pd.to_datetime(result["Timestamp"], errors="raise")
    result["Transaction_Hour"] = timestamp.dt.hour
    result["Transaction_DayOfWeek"] = timestamp.dt.dayofweek
    result["Transaction_Month"] = timestamp.dt.month
    result["Derived_Is_Weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    result["Transaction_Hour_Sin"] = np.sin(2 * np.pi * result["Transaction_Hour"] / 24)
    result["Transaction_Hour_Cos"] = np.cos(2 * np.pi * result["Transaction_Hour"] / 24)
    result = result.drop(columns=["Timestamp"])
    return result[bundle["model_feature_columns"]]


def predict_fn(data, bundle):
    features = prepare_features(data, bundle)
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    threshold = float(bundle["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    return [{
        "prediction": int(prediction),
        "label": "Fraud" if prediction else "Non-fraud",
        "fraud_probability": float(probability),
        "decision_threshold": threshold,
        "algorithm": bundle["algorithm"],
    } for prediction, probability in zip(predictions, probabilities)]


def output_fn(prediction, accept):
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
