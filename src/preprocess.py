"""SageMaker Processing Job — replicates Notebook 01 preprocessing."""
import os, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


parser = argparse.ArgumentParser()
parser.add_argument('--test-size',    type=float, default=0.20)
parser.add_argument('--random-state', type=int,   default=42)
args = parser.parse_args()

input_path = '/opt/ml/processing/input/credit_card_fraud_10K.csv'
output_dir = '/opt/ml/processing/output'
os.makedirs(output_dir, exist_ok=True)

categorical_features = ["merchant_category"]
numeric_features = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]
feature_columns = categorical_features + numeric_features
label_column = "is_fraud"

df = pd.read_csv(input_path)

print(f"Loaded {df.shape[0]} valid rows")
print("Column data types:")
print(df.dtypes)

df = df.dropna(subset=feature_columns + [label_column])
X = df[feature_columns]
y = df[label_column].astype(float)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

# ColumnTransformer: One-Hot Encode strings + Scale numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# Extract transformed feature names (480 feature column names)
encoded_feature_names = list(preprocessor.get_feature_names_out())
    
train_dataset = pd.concat([pd.DataFrame(x_train, columns=encoded_feature_names), y_train.reset_index(drop=True)], axis=1)
test_dataset = pd.concat([pd.DataFrame(x_test, columns=encoded_feature_names), y_test.reset_index(drop=True)], axis=1)

# Write output CSVs
train_dataset.to_csv(f'{output_dir}/train.csv', index=False, header=True)
test_dataset.to_csv(f'{output_dir}/test.csv', index=False, header=True)
print('Preprocessing complete. Saved train.csv and test.csv')
