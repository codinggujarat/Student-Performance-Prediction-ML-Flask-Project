import numpy as np
import pandas as pd
import os

# Save path
DATA_PATH = "data/students_sample.csv"
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n = 100

# Generate student dataset
df = pd.DataFrame({
    "Student_ID": [f"S{i+1:04d}" for i in range(n)],
    "Attendance": np.random.randint(40, 100, n),       # Attendance between 40% - 100%
    "Study_Hours": np.random.randint(1, 10, n),        # Study hours 1-9 per day
    "Past_Result": np.random.randint(30, 100, n),      # Last exam score between 30-100
})

# Generate Exam Scores using formula + random noise
df["Exam_Score"] = (
    0.4 * df["Attendance"] +
    3 * df["Study_Hours"] +
    0.3 * df["Past_Result"] +
    np.random.normal(0, 5, n)
).round(2)

# Clip scores to keep within 0-100
df["Exam_Score"] = df["Exam_Score"].clip(0, 100)

# Save dataset to CSV
df.to_csv(DATA_PATH, index=False)

print(f"✅ 100 Student sample dataset created successfully → {DATA_PATH}")
print(df.head(10))  # Show first 10 rows