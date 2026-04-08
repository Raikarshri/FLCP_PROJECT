import mlflow
import mlflow.keras
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mlflow.set_experiment("DL Experiment")
BASE_DIR = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(BASE_DIR / "DATA" / "lungcancer_clean.csv")

# Fix column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.upper()
df.columns = df.columns.str.replace(" ", "_")

# Convert values
df = df.replace({
    "YES": 1, "NO": 0,
    "Yes": 1, "No": 0,
    "M": 1, "F": 0
})

# Split
X = df.drop("LUNG_CANCER", axis=1).values
y = df["LUNG_CANCER"].values

# Scale
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
with mlflow.start_run(run_name="DL_Model"):

    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test)

    print("DL Accuracy:", accuracy)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.keras.log_model(model, name="model")

# Save result
with (MODULE_DIR / "dl_results.json").open("w", encoding="utf-8") as f:
    json.dump({"accuracy": float(accuracy)}, f)

print("✅ DL results saved")
