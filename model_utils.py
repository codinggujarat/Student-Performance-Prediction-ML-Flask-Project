import os
import joblib
import numpy as np
import pandas as pd

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = os.path.join("models", "best_model.pkl")


# ===========================
# LOAD MODEL
# ===========================
def load_model(path=MODEL_PATH):
    """
    Loads the saved model, features, and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Model not found at {path}. Please run train_model.py first."
        )

    meta = joblib.load(path)
    model = meta.get("model")
    features = meta.get("features")
    model_name = meta.get("model_name", "UnknownModel")

    if model is None or features is None:
        raise ValueError("Invalid model file. Retrain the model.")

    return model, features, model_name


# ===========================
# PREDICT FUNCTION
# ===========================
def predict_df(df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """
    Generates predictions and risk classification.
    Input:
        df → DataFrame containing required features
        threshold → Minimum score threshold for risk classification
    Output:
        DataFrame with columns:
            - Predicted_Score
            - Risk (True/False)
            - Risk_Label (HIGH/LOW)
            - Model_Used
    """
    model, features, model_name = load_model()

    # Make a copy to avoid modifying original data
    out = df.copy()

    # ----------------------------
    # HANDLE CASE-INSENSITIVE COLUMN MATCHING
    # ----------------------------
    cols_map = {c.lower(): c for c in out.columns}
    missing = [f for f in features if f.lower() not in cols_map]

    if missing:
        raise ValueError(
            f"❌ Missing required columns: {missing}. "
            f"Expected: {features}. "
            f"Provided: {list(out.columns)}"
        )

    # Ensure correct feature order
    X = out[[cols_map[f.lower()] for f in features]]

    # ----------------------------
    # HANDLE NON-NUMERIC DATA
    # ----------------------------
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"❌ Found missing or non-numeric values in columns: {bad_cols}. "
            f"Please clean your dataset."
        )

    # ----------------------------
    # MAKE PREDICTIONS
    # ----------------------------
    preds = model.predict(X)
    preds = np.clip(preds, 0, 100)  # Ensure valid score range

    # ----------------------------
    # ADD OUTPUT COLUMNS
    # ----------------------------
    out["Predicted_Score"] = np.round(preds, 2)
    out["Risk"] = out["Predicted_Score"] < threshold
    out["Risk_Label"] = out["Risk"].map({True: "HIGH", False: "LOW"})
    out["Model_Used"] = model_name

    return out


# ===========================
# TEST LOCALLY
# ===========================
if __name__ == "__main__":
    sample_path = os.path.join("data", "students_sample.csv")

    if not os.path.exists(sample_path):
        print(f"⚠️ Sample file missing. Run train_model.py to generate data.")
    else:
        df = pd.read_csv(sample_path)
        print("\n✅ Loaded sample data successfully!")
        predictions = predict_df(df)
        print("\n📌 Predictions Preview:")
        print(predictions.head())
