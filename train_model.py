import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ========================
# Configurations
# ========================
DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_PATH = os.path.join(DATA_DIR, 'students_sample.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')

FEATURES = ['Attendance', 'Study_Hours', 'Past_Result']
TARGET = 'Exam_Score'

# ========================
# Generate Sample Data
# ========================
def generate_sample_data(n=300, path=DATA_PATH):
    """Generate a sample student dataset if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.random.seed(42)

    df = pd.DataFrame({
        'Student_ID': [f'S{i+1:04d}' for i in range(n)],
        'Attendance': np.random.randint(50, 100, n),
        'Study_Hours': np.random.randint(1, 10, n),
        'Past_Result': np.random.randint(35, 100, n),
    })

    # Generate Exam Score = weighted formula + random noise
    df['Exam_Score'] = (
        0.4 * df['Attendance'] +
        3 * df['Study_Hours'] +
        0.3 * df['Past_Result'] +
        np.random.normal(0, 5, n)
    )
    df['Exam_Score'] = df['Exam_Score'].round(2).clip(0, 100)

    df.to_csv(path, index=False)
    print(f"✅ Sample data generated → {path}")
    return df

# ========================
# Train & Save Model
# ========================
def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Train the model and save the best one automatically."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load or create dataset
    if not os.path.exists(data_path):
        print("⚠️ Dataset missing → Generating sample dataset...")
        df = generate_sample_data()
    else:
        df = pd.read_csv(data_path)

    # Check required columns
    required_columns = FEATURES + [TARGET]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Dataset missing required columns: {missing_cols}")

    # Prepare data
    X = df[FEATURES]
    y = df[TARGET]

    # Split dataset → 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    dt = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)

    # ========================
    # Evaluate Model Function
    # ========================
    def eval_model(model):
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    # Evaluate both models
    res_lr = eval_model(lr)
    res_dt = eval_model(dt)

    print("\n📊 Model Performance:")
    print(f"🔹 Linear Regression → {res_lr}")
    print(f"🔹 Decision Tree     → {res_dt}")

    # Select the best model based on RMSE
    best_model = lr if res_lr['rmse'] <= res_dt['rmse'] else dt
    best_name = 'LinearRegression' if best_model is lr else 'DecisionTree'

    # Save the model
    joblib.dump({'model': best_model, 'features': FEATURES, 'model_name': best_name}, model_path)
    print(f"\n✅ Best model ({best_name}) saved → {model_path}")

    return best_model, best_name

# ========================
# Auto-Train Helper for Deployments
# ========================
def train_model_if_missing():
    """Auto-train the model if missing (for first-time deployments)."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found → Training now...")
        train_and_save()
    else:
        print("✅ Model already exists → Skipping training.")

# ========================
# Script Execution
# ========================
if __name__ == '__main__':
    print("🚀 Training model...")
    train_and_save()
