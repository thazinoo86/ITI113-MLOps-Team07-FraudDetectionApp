"""
SageMaker Training Job
----------------------
Trains a Random Forest classifier and saves the SageMaker model artefact.

MLflow logging is intentionally performed outside the SageMaker training
container by the notebook after a successful pipeline execution.
"""
import os
import argparse
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=6)
parser.add_argument("--min-samples-leaf", type=int, default=4)
parser.add_argument("--random-state", type=int, default=42)

# Metadata is retained in the SageMaker training job hyperparameters.
parser.add_argument("--team-id", type=str, default=os.environ.get("TEAM_ID", "unknown-team"))
parser.add_argument("--student-id", type=str, default=os.environ.get("STUDENT_ID", "s000"))
parser.add_argument("--semester", type=str, default=os.environ.get("SEMESTER", "26S1"))
parser.add_argument("--run-name", type=str, default="sagemaker_pipeline_run")

# SageMaker supplies these paths automatically.
parser.add_argument(
    "--model-dir",
    type=str,
    default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
)
parser.add_argument(
    "--train",
    type=str,
    default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
)
parser.add_argument(
    "--test",
    type=str,
    default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
)
args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)

print("=== SageMaker Training Environment ===")
print(f"Train channel: {args.train}")
print(f"Test channel: {args.test}")
print(f"Model directory: {args.model_dir}")

X_train = pd.read_csv(os.path.join(args.train, "train_features.csv"))
y_train = pd.read_csv(os.path.join(args.train, "train_labels.csv")).squeeze("columns")
X_test = pd.read_csv(os.path.join(args.test, "test_features.csv"))
y_test = pd.read_csv(os.path.join(args.test, "test_labels.csv")).squeeze("columns")

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

if len(pd.Series(y_train).unique()) < 2:
    raise ValueError("Training labels contain fewer than two classes.")

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_leaf=args.min_samples_leaf,
    class_weight="balanced",
    random_state=args.random_state,
    n_jobs=-1,
)
model.fit(X_train, y_train)

all_metrics = {}
for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    all_metrics.update({
        f"{split}_accuracy": round(accuracy_score(y, predictions), 4),
        f"{split}_f1": round(f1_score(y, predictions, zero_division=0), 4),
        f"{split}_precision": round(
            precision_score(y, predictions, zero_division=0), 4
        ),
        f"{split}_recall": round(
            recall_score(y, predictions, zero_division=0), 4
        ),
    })

    # AUC needs both classes in the evaluated split.
    if len(pd.Series(y).unique()) >= 2:
        all_metrics[f"{split}_auc_roc"] = round(
            roc_auc_score(y, probabilities), 4
        )
    else:
        all_metrics[f"{split}_auc_roc"] = None
        print(f"Warning: {split} split has only one class; AUC-ROC unavailable.")

print("=== Metrics ===")
for metric_name, metric_value in all_metrics.items():
    print(f"{metric_name}: {metric_value}")

with open(os.path.join(args.model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print(f"Model saved: {os.path.join(args.model_dir, 'model.pkl')}")

# These exact printed labels are captured by SageMaker metric_definitions
# and used by the Pipeline ConditionStep.
if all_metrics["test_auc_roc"] is None:
    raise ValueError("Test AUC-ROC is unavailable; cannot evaluate the quality gate.")

print(f"Test AUC-ROC: {all_metrics['test_auc_roc']}")
print(f"test_accuracy: {all_metrics['test_accuracy']}")
print(f"test_f1: {all_metrics['test_f1']}")
