"""SageMaker inference handler for the deployed endpoint."""
import os, json, pickle
import pandas as pd

FEATURE_COLUMNS = [
    'age','trestbps','chol','thalach','oldpeak',
    'sex','cp','fbs','restecg','exang','slope','ca','thal',
    'age_group','high_risk_count'
]

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
    return df[FEATURE_COLUMNS]

def predict_fn(data, model):
    return model.predict(data), model.predict_proba(data)[:,1]

def output_fn(prediction, accept='application/json'):
    preds, probas = prediction
    response = [{'prediction':int(p),
                 'label':'Heart disease present' if p==1 else 'No heart disease',
                 'probability':round(float(b),4)}
                for p,b in zip(preds,probas)]
    return json.dumps(response), accept
