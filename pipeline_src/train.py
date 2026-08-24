
import argparse
import base64
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def decode(value):
    return json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))


def make_classifier(algorithm, params):
    params = dict(params)
    if algorithm == "LogisticRegression":
        return LogisticRegression(**params), True
    if algorithm == "XGBoost":
        from xgboost import XGBClassifier
        params["n_jobs"] = -1
        return XGBClassifier(**params), False
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "test_accuracy": float(accuracy_score(y_true, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "test_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "test_fpr": float(fp / (fp + tn) if fp + tn else 0.0),
        "test_alert_rate": float(predictions.mean()),
        "test_auc_roc": float(roc_auc_score(y_true, probabilities)),
        "test_auc_pr": float(average_precision_score(y_true, probabilities)),
        "test_mcc": float(matthews_corrcoef(y_true, predictions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-params-b64", required=True)
    parser.add_argument("--schema-b64", required=True)
    parser.add_argument("--decision-threshold", type=float, required=True)
    parser.add_argument("--source-run-id", required=True)
    args = parser.parse_args()

    params, schema = decode(args.model_params_b64), decode(args.schema_b64)
    train = pd.read_csv("/opt/ml/input/data/processed/train.csv")
    test = pd.read_csv("/opt/ml/input/data/processed/test.csv")
    X_train, y_train = train.drop(columns=["Fraud_Label"]), train["Fraud_Label"].astype(int)
    X_test, y_test = test.drop(columns=["Fraud_Label"]), test["Fraud_Label"].astype(int)
    if X_train.columns.tolist() != schema["model_feature_columns"]:
        raise ValueError("Processed training schema does not match Notebook 02 champion schema.")
    if "Risk_Score" in X_train.columns:
        raise RuntimeError("Risk_Score must never enter training.")

    classifier, scale = make_classifier(args.algorithm, params)
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", schema["numeric_columns"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), schema["categorical_columns"]),
    ])
    model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, probabilities, args.decision_threshold)
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")

    model_dir = Path(os.environ["SM_MODEL_DIR"]); model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "decision_threshold": args.decision_threshold,
        "algorithm": args.algorithm,
        "model_family": args.model_family,
        "source_mlflow_run_id": args.source_run_id,
        "raw_inference_columns": schema["raw_inference_columns"],
        "model_feature_columns": schema["model_feature_columns"],
        "risk_score_excluded": True,
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    (model_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
