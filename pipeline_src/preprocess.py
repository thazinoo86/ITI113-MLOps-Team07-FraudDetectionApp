
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "Fraud_Label"
IDENTIFIERS = ["Transaction_ID", "User_ID"]
LEAKAGE_EXCLUDED = ["Risk_Score"]
SUPPLIED_QUALITY_EXCLUDED = ["Is_Weekend"]


def prepare_features(frame):
    result = frame.copy()
    timestamp = pd.to_datetime(result["Timestamp"], errors="raise")
    result["Transaction_Hour"] = timestamp.dt.hour
    result["Transaction_DayOfWeek"] = timestamp.dt.dayofweek
    result["Transaction_Month"] = timestamp.dt.month
    result["Derived_Is_Weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    result["Transaction_Hour_Sin"] = np.sin(2 * np.pi * result["Transaction_Hour"] / 24)
    result["Transaction_Hour_Cos"] = np.cos(2 * np.pi * result["Transaction_Hour"] / 24)
    return result.drop(columns=IDENTIFIERS + LEAKAGE_EXCLUDED + SUPPLIED_QUALITY_EXCLUDED + ["Timestamp", TARGET])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    files = list(Path("/opt/ml/processing/input").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No dataset CSV supplied.")
    data = pd.read_csv(files[0])
    required = {
        "Transaction_ID", "User_ID", "Timestamp", "Risk_Score", "Is_Weekend",
        "Failed_Transaction_Count_7d", "Fraud_Label",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.isna().sum().sum() > 0:
        raise ValueError("Dataset contains missing values; review before pipeline execution.")
    if data.duplicated().sum() > 0:
        raise ValueError("Dataset contains duplicate rows; review before pipeline execution.")

    timestamp = pd.to_datetime(data["Timestamp"], errors="raise")
    derived_weekend = (timestamp.dt.dayofweek >= 5).astype(int)
    weekend_mismatch = int((derived_weekend != data["Is_Weekend"]).sum())
    suspected_rule = ((data["Risk_Score"] > 0.85) | (data["Failed_Transaction_Count_7d"] >= 4)).astype(int)
    rule_match_rate = float((suspected_rule == data[TARGET]).mean())
    print(f"Explicit rule match rate: {rule_match_rate:.8f}")
    print(f"Is_Weekend mismatch rows: {weekend_mismatch}")
    print("Risk_Score policy: EXCLUDED")

    X = prepare_features(data)
    y = data[TARGET].astype(int)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        stratify=y_train_val, random_state=args.random_state
    )

    output = Path("/opt/ml/processing/output")
    output.mkdir(parents=True, exist_ok=True)
    for name, features, labels in [
        ("train", X_train, y_train),
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ]:
        frame = features.copy(); frame[TARGET] = labels.to_numpy()
        frame.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: rows={len(frame)}, fraud_rate={frame[TARGET].mean():.8f}")

    audit = {
        "rows": len(data), "duplicate_rows": 0, "missing_values": 0,
        "weekend_mismatch_rows": weekend_mismatch,
        "explicit_rule_match_rate": rule_match_rate,
        "risk_score_excluded": True,
        "split_strategy": "stratified_random_60_20_20",
    }
    (output / "pipeline_data_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
