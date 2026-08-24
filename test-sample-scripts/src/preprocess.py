"""SageMaker Processing Job — replicates Notebook 01 preprocessing."""
import os, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--test-size',    type=float, default=0.20)
parser.add_argument('--random-state', type=int,   default=42)
args = parser.parse_args()

input_path = '/opt/ml/processing/input/heart.csv'
output_dir = '/opt/ml/processing/output'
os.makedirs(output_dir, exist_ok=True)

COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
NUMERIC     = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

df = pd.read_csv(
    input_path,
    names=COLUMNS,
    header=None,
    na_values=["?", "", "NA", "N/A"]
)

# Convert every expected dataset column to numeric.
# This also turns an accidental header row or invalid text into NaN.
for col in COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Target must exist before binary conversion.
df = df.dropna(subset=["target"]).copy()
df["target"] = (df["target"] > 0).astype(int)

# Fill missing numeric values using the training-data median.
for col in NUMERIC:
    median_value = df[col].median()

    if pd.isna(median_value):
        raise ValueError(f"Column '{col}' has no valid numeric values.")

    df[col] = df[col].fillna(median_value)

# Fill missing categorical values using the most frequent category.
for col in CATEGORICAL:
    mode_values = df[col].mode(dropna=True)

    if mode_values.empty:
        raise ValueError(f"Column '{col}' has no valid categorical values.")

    df[col] = df[col].fillna(mode_values.iloc[0])

# Remove any remaining invalid records before feature engineering.
df = df.dropna(subset=NUMERIC + CATEGORICAL + ["target"]).copy()

print(f"Loaded {df.shape[0]} valid rows")
print("Column data types:")
print(df.dtypes)

# Use open-ended bins so an unexpected age does not produce NaN.
df["age_group"] = pd.cut(
    df["age"],
    bins=[float("-inf"), 45, 60, float("inf")],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

df["high_risk_count"] = (
    (df["exang"] == 1).astype(int)
    + (df["fbs"] == 1).astype(int)
    + (df["ca"] > 0).astype(int)
)

FEATURES = NUMERIC + CATEGORICAL + ["age_group", "high_risk_count"]
X, y = df[FEATURES], df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

scaler = StandardScaler()
X_train[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])
X_test[NUMERIC]  = scaler.transform(X_test[NUMERIC])

X_train.to_csv(f'{output_dir}/train_features.csv', index=False)
y_train.to_csv(f'{output_dir}/train_labels.csv',   index=False, header=True)
X_test.to_csv( f'{output_dir}/test_features.csv',  index=False)
y_test.to_csv( f'{output_dir}/test_labels.csv',    index=False, header=True)
print('Preprocessing complete. Saved 4 output files.')
