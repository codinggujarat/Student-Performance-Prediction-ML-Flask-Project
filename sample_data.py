import numpy as np
import pandas as pd
import os

# Save path
DATA_PATH = "data/students_sample.csv"
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Total students
n = 120

# Generate student dataset
df = pd.DataFrame({
    "Student_ID": [f"S{i+1:04d}" for i in range(n)],
    "Semester": np.random.choice([1, 2, 3, 4, 5, 6], n),  # Random semester 1-6
    "Attendance": np.random.randint(40, 100, n),          # Attendance %
    "Study_Hours": np.random.randint(1, 10, n),           # Study hours/day
    "Past_Result": np.random.randint(30, 100, n)          # Last exam score
})

# Calculate Exam Scores using formula + noise
df["Exam_Score"] = (
    0.4 * df["Attendance"] +
    3 * df["Study_Hours"] +
    0.3 * df["Past_Result"] +
    np.random.normal(0, 5, n)
).round(2)

# Clip exam score between 0 and 100
df["Exam_Score"] = df["Exam_Score"].clip(0, 100)

# Save dataset to CSV
df.to_csv(DATA_PATH, index=False)

print(f"✅ Student dataset created successfully → {DATA_PATH}")
print(df.head(10))  # Show sample rows
