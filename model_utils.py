# model_utils.py
import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join('models', 'best_model.pkl')


def load_model(path=MODEL_PATH):
    """Load the saved model and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    meta = joblib.load(path)
    return meta['model'], meta['features'], meta.get('model_name', 'model')


def predict_df(df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """
    Input: DataFrame containing the feature columns (case-insensitive).
    Output: DataFrame with Predicted_Score, Risk (boolean), Risk_Label, Model_Used.
    """
    model, features, model_name = load_model()

    # Work on a copy
    out = df.copy()

    # Map columns case-insensitively
    cols_map = {c.lower(): c for c in out.columns}
    missing = [f for f in features if f.lower() not in cols_map]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected columns: {features}")

    # Build X in correct order
    X = out[[cols_map[f.lower()] for f in features]]

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().any().any():
        raise ValueError('Found non-numeric or missing values in required feature columns.')

    preds = model.predict(X)
    preds = np.clip(preds, 0, 100)

    out['Predicted_Score'] = np.round(preds, 2)
    out['Risk'] = out['Predicted_Score'] < threshold
    out['Risk_Label'] = out['Risk'].map({True: 'HIGH', False: 'LOW'})
    out['Model_Used'] = model_name

    return out
