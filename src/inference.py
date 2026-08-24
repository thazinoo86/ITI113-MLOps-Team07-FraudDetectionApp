"""SageMaker inference handler for the deployed endpoint."""
import os, json, pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["merchant_category"]
numeric_features = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]
FEATURE_COLUMNS = categorical_features + numeric_features

def model_fn(model_dir):
    with open(os.path.join(model_dir,'model.pkl'),'rb') as f:
        return pickle.load(f)

def input_fn(body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported content type: {content_type}')
    payload = json.loads(body)
    if isinstance(payload, dict): payload = [payload]
    df = pd.DataFrame(payload)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing: raise ValueError(f'Missing features: {missing}')
        
    # Apply preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    
    # Fit and transform (in production, load fitted preprocessor from model artifacts)
    X = df[FEATURE_COLUMNS]
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed

def predict_fn(data, model):
    return model.predict(data), model.predict_proba(data)[:,1]

def output_fn(prediction, accept='application/json'):
    preds, probas = prediction
    response = [{'prediction':int(p),
                 'label':'Credit Card Fraud' if p==1 else 'Not Credit Card Fraud',
                 'probability':round(float(b),4)}
                for p,b in zip(preds,probas)]
    return json.dumps(response), accept
